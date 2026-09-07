# Architecture and deviations from the specification

The specification's design principle DP-1 is "mirror OpenMC's architecture,
not just its file formats." This build follows that closely on the parts that
matter to a reader — the language split, the build system, the I/O conventions
and the API idioms — and departs from it in three places where following it
literally would have cost more than it bought. Each departure is recorded here
with the reason, so a reviewer can overrule it on the evidence rather than
rediscover the argument.

---

## Module layout

```
include/openndm/        public C++ headers
  constants.h           enumerations and sentinels
  error.h               the exception hierarchy
  geometry.h            node/surface graph and the Cartesian builder
  xslib.h               group constants, ADFs, branch tables
  matrix.h              sparse pattern, ILU0, BiCGSTAB, deterministic reductions
  kernel.h              the abstract two-node kernel interface
  cmfd.h                the coarse-mesh system and the nonlinear update
  settings.h            run-time configuration
  solver.h              the outer iteration and the top-level entry points

src/                    implementation
  kernel_common.h       internal: leakage fit, analytic basis, dense solve
  kernel_fdm.cpp        finite difference
  kernel_nem.cpp        polynomial nodal
  kernel_sanm.cpp       semi-analytic nodal
  bindings.cpp          pybind11 bridge

python/openndm/         the Python package
  gc/                   OpenMC group constant generation
```

The dependency direction is strictly one way: `geometry` and `xslib` know
nothing about solvers, `kernel` knows nothing about `cmfd`, and `cmfd` knows
nothing about `solver`. A kernel receives a `TwoNodeProblem` value and returns
currents; it cannot reach back into the system that called it.

---

## Deviation 1: pybind11 instead of a ctypes-called C API

**Specification** (§3.1): OpenMC exposes a hand-written `extern "C"` API called
from Python through ctypes. The specification already flags this as an open
question and leans toward pybind11.

**What this build does**: pybind11, in `src/bindings.cpp`.

**Why**: the hand-written path costs a marshalling function per entry point,
forever, and gives nothing back that pybind11 does not. The concrete
requirements decided it:

- FR-OPT-5 wants zero-copy numpy results. pybind11 gives that directly through
  `py::array_t` with a keep-alive base object; ctypes needs an explicit buffer
  protocol implementation per array.
- FR-OPT-6 wants typed Python exceptions. The translator in `bindings.cpp`
  maps each C++ exception onto the matching class in `openndm.exceptions`, so
  `except openndm.InputError` catches what the core raises and users can
  subclass the hierarchy. Through ctypes this is error codes and manual
  raising at every call site.
- FR-OPT-2 wants the GIL released during a solve. That is
  `py::call_guard<py::gil_scoped_release>()`, one line.

**Cost**: pybind11 becomes a build dependency, and a C++ developer reading
OpenMC's `capi.cpp` will not find its counterpart here.

**Still owed**: FR-NFR-EXT-5 asks for a public `extern "C"` layer for coupling
from Fortran, Julia and MATLAB. That is not implemented. It is a thin wrapper
over the same core, independent of the Python bridge, and nothing about this
choice blocks it.

## Deviation 2: a purpose-built sparse structure instead of Eigen

**Specification** (§3.1): Eigen for sparse linear algebra, optionally PETSc.

**What this build does**: a CSR pattern built once from the node adjacency
graph and shared by every energy group, with ILU0 and BiCGSTAB written against
it in `src/matrix.cpp`.

**Why**:

- The matrices are one diagonal plus one entry per node face and are
  re-assembled every nonlinear update. Only the values change, never the
  pattern, so building the pattern once and refreshing values is both simpler
  and faster than reconstructing an Eigen sparse matrix each time.
- FR-OPT-4 requires bit-identical results at any thread count. That needs
  control over the reduction order in every inner product. A general-purpose
  library's reductions are not specified to that level.
- NFR-EXT-4 requires building from source with no dependency outside
  conda-forge or apt. As it stands the build fetches nothing at configure
  time and needs only a C++17 compiler; Eigen through `FetchContent` would
  have made a network connection part of the build.

**Cost**: roughly 250 lines of linear algebra that a library would have
provided, and no access to Eigen's direct solvers or its other preconditioners.

**When to revisit**: if a future kernel needs a matrix structure that is not a
stencil — SP3 with its coupled second moment, or an implicit transient
operator with a different sparsity — the argument weakens considerably.

## Deviation 3: numpy views through pybind11 instead of xtensor

**Specification** (§3.1): xtensor for arrays, `xtensor-python` for zero-copy
numpy interop.

**What this build does**: `std::vector<double>` internally, with flat indexing
laid out group-major per node, exposed to Python as numpy arrays that wrap the
C++ buffer with the owning object as their base.

**Why**: the zero-copy requirement (FR-OPT-5) is met either way. xtensor's
value is expression templates over multidimensional arrays, and this code's
hot loops are hand-written stencil sweeps that do not use them. The
`xtl → xtensor → xtensor-python` chain would have added three build
dependencies for indexing sugar.

**Cost**: index arithmetic is explicit, which is more error-prone. It is
concentrated in `cmfd.cpp` and follows one convention throughout: `node * G +
g` for per-node group data, `(node * G + g) * G + g'` for scattering, and
`surface * G + g` for surface data.

---

## Choices worth explaining that are not deviations

### Discontinuity factors are folded into the coupling, not the kernel

`build_coupling()` derives both `D̃` and the initial `D̂` from the
factor-weighted continuity conditions, so the ADF path is exercised even when
the finite difference kernel is selected and cannot be silently skipped by a
kernel that forgets to apply it. See [`theory.md`](theory.md) §2.3.

### Out-of-core positions have their own boundary condition

`CartesianSpec::inactive_bc` is separate from the six face conditions. On a
quarter-core map the mesh edges carry the symmetry conditions while a face
looking at an out-of-core position is a real outer boundary. Conflating them
reflects neutrons back into the core from outside it, which inflates `k_eff` by
hundreds of pcm with no other symptom. This was a real bug during development,
caught by the fine-mesh consistency check.

### Reading a composition returns a snapshot

`XSLibrary::composition` has a mutable overload that marks the library
unfinalized, because the caller can invalidate the cached removal cross
sections through it, and a const overload that does not. The Python bindings
expose the const one as `composition()` and the mutable one as
`mutable_composition()`, used only by `set_composition`. Without the split,
merely *inspecting* a library invalidated it.

### The adjoint reuses the forward coupling coefficients

The two-node kernels solve the forward transverse-integrated problem. Running
them against an adjoint flux would produce meaningless corrections, so the
adjoint keeps the `D̂` converged by a forward solve, and a cold adjoint request
runs the forward problem first. The transpose of the corrected forward
operator is the corrected adjoint operator, so this is the correct thing to do
and not merely the convenient one.

### The statepoint is written from Python

FR-OUT-1 asks for an HDF5 statepoint on OpenMC's conventions. It is written
through h5py rather than from C++, which keeps libhdf5 off the C++ build's
dependency list. The consequence is that the statepoint is unavailable to a
non-Python caller — which matters only once FR-IN-2's standalone executable
exists, and that will need an HDF5 dependency for its own reasons.

---

## Extension points that were designed in

**Hexagonal geometry (FR-GEO-6, NFR-EXT-1).** Kernels walk `Surface` records
and the per-node `face` table; nothing in `cmfd.cpp`, `kernel_sanm.cpp` or
`kernel_nem.cpp` refers to `(i,j,k)`. A `Surface` carries its own `h_lo`,
`h_hi` and transverse-integration axis, which is exactly the information a hex
node's six lateral faces need. Adding hexagonal geometry means a second
builder in `geometry.cpp` and a kernel that understands three lateral axes;
the CMFD layer is unchanged.

**A new kernel (FR-SOL-8).** Implement `Kernel`, register it in
`Kernel::create`, name it in `settings.py`, add it to `ALL_KERNELS` in the test
conftest. The verification suite is parameterised over that list, so a new
kernel is immediately held to the analytic solution, the convergence order and
the fine-mesh consistency check.

**External thermal hydraulics (FR-TH-6, FR-TH-7).** Nothing in the solver
refers to temperature or density. Cross sections reach it only through
`XSLibrary`, which already interpolates over arbitrary state axes. A
`ThermalSolver` interface and a Picard driver can sit entirely above the
current code, with field transfer through numpy arrays as FR-TH-7 requires.

**Run-time sizing (NFR-EXT-2).** Group count, precursor count and branch axes
are all run-time values. Nothing is a template parameter and nothing is a
compile-time constant except `MAX_DELAYED_GROUPS`, which is a validation
bound rather than a storage bound.
