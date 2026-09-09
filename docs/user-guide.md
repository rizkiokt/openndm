# OpenNDM user guide

How to drive the solver. For the equations it implements see
[`theory.md`](theory.md); for what is and is not built yet see
[`status.md`](status.md); for runnable examples see [`../examples/`](../examples/).

---

## Contents

1. [Installing](#1-installing)
2. [The five objects](#2-the-five-objects)
3. [Cross sections](#3-cross-sections)
4. [Geometry](#4-geometry)
5. [Settings](#5-settings)
6. [Solving](#6-solving)
7. [Reading results](#7-reading-results)
8. [Checking your answer](#8-checking-your-answer)
9. [Group constants from OpenMC](#9-group-constants-from-openmc)
10. [Branch libraries and feedback](#10-branch-libraries-and-feedback)
11. [Embedding and performance](#11-embedding-and-performance)
12. [Files](#12-files)
13. [When something goes wrong](#13-when-something-goes-wrong)

---

## 1. Installing

```bash
pip install .            # needs a C++17 compiler; CMake and ninja come from pip
pip install '.[plot]'    # adds matplotlib for openndm.plots
pip install '.[openmc]'  # adds OpenMC, needed only by openndm.gc
pip install '.[dev]'     # pytest, ruff and the rest of the toolchain
```

OpenMC is **optional**. The solver imports and runs without it; only
`openndm.gc` needs it, and `import openndm` never pulls it in.

```python
import openndm
print(openndm.__version__)
```

---

## 2. The five objects

| Object | Holds |
|---|---|
| `XSLibrary` | group constants, one row per composition |
| `Geometry` | which composition sits where, and the boundary conditions |
| `Settings` | kernel choice and convergence criteria |
| `Model` | the three above, plus a persistent solver |
| `Result` | eigenvalue, flux, power, iteration history |

A complete calculation:

```python
import numpy as np
import openndm

lib = openndm.XSLibrary(n_groups=2, n_compositions=2)
lib.set_composition(0, D=[1.5, 0.4], absorption=[0.010, 0.085],
                    nu_fission=[0.0, 0.135], kappa_fission=[0.0, 0.135],
                    chi=[1.0, 0.0], scatter=[[0.0, 0.020], [0.0, 0.0]])
lib.set_composition(1, D=[2.0, 0.3], absorption=[0.0, 0.010],
                    scatter=[[0.0, 0.040], [0.0, 0.0]])
lib.finalize()

core = np.zeros((10, 9, 9), dtype=int)
core[:, 8, :] = core[:, :, 8] = 1

geom = openndm.Geometry.from_lattice(
    core, pitch=20.0,
    boundaries={"x_min": "reflective", "y_min": "reflective",
                "x_max": "zero_flux", "y_max": "zero_flux",
                "z_min": "vacuum", "z_max": "vacuum"})

result = openndm.Model(geom, lib, openndm.Settings(kernel="sanm")).solve()
print(result.k_eff, result.f_q)
```

The `Model` owns its solver, so re-solving a perturbed model is cheap and can
warm start. Building a `Model` is the expensive part; calling `solve()` again
is not.

---

## 3. Cross sections

### Conventions

These three cost the most time when they are wrong, and none of them produces
an error message on its own.

**Group 1 is the highest energy.** This matches OpenMC's multi-group
numbering, so index 0 of every array is the fast group.

**`scatter[from_group][to_group]`.** For two groups the down-scatter term is
`scatter[0][1]`. Transposing the matrix does not raise; it silently reverses
the spectrum.

**Removal is derived, not given.** `Σr = Σa + Σ_{g'≠g} Σs_{g→g'}`, computed by
`finalize()`. Within-group scattering never leaves the node and cancels, so
you may leave the diagonal at zero or fill it in; the answer is the same.

### Filling a composition

```python
lib.set_composition(
    index,
    state=0,                 # branch grid point; 0 unless you have axes
    D=[1.5, 0.4],            # or transport=[...], giving D = 1/(3 Σtr)
    absorption=[0.010, 0.085],
    nu_fission=[0.0, 0.135],
    kappa_fission=[0.0, 0.135],
    chi=[1.0, 0.0],
    inv_velocity=[...],      # only needed by a transient, once implemented
    scatter=[[0.0, 0.020], [0.0, 0.0]],
    std={"absorption": [1e-4, 8e-4]},   # 1-sigma, carried through
)
```

Anything omitted stays zero. **`kappa_fission` is what makes power**: a
composition with `nu_fission` but no `kappa_fission` contributes to `k_eff`
and to nothing else, and the power distribution comes back as zeros. That is
one of the warnings below.

### Validating

```python
warnings_raised = lib.finalize()   # returns a list, also issues them
```

`finalize()` **raises `LibraryError`** on data that cannot be physical:

* a non-positive diffusion coefficient,
* a negative absorption cross section,
* a fission spectrum that does not sum to one in a fissile composition,
* a non-monotonic branch axis.

It **warns** on data that is merely suspicious:

* negative scattering transfers, which Monte Carlo noise produces routinely
  and which are deliberately not an error,
* a non-positive removal cross section,
* a fissile composition with no `kappa_fission`,
* a fission spectrum in a composition with no fission source.

Reading a composition back returns a **snapshot**, so inspecting a library
cannot invalidate it:

```python
comp = lib.composition(0)          # a copy; mutating it does nothing
np.asarray(comp.scatter).reshape(lib.n_groups, lib.n_groups)
lib.array("absorption")            # (n_compositions, n_groups)
lib.array("scatter")               # (n_compositions, n_groups, n_groups)
```

### Discontinuity factors

Per composition, per face, per group, defaulting to 1.0. Face order is
`-x, +x, -y, +y, -z, +z`.

```python
lib.set_adf(0, values)        # (6, n_groups), or (n_groups,) to broadcast
lib.adf(0)                    # read them back as (6, n_groups)
```

They are folded into the coupling coefficient before any nonlinear
correction, so they act even with the `fdm` kernel.

### Delayed neutrons

```python
lib.set_delayed(beta=[...], decay_constant=[...], chi_delayed=None)
```

Up to eight precursor groups. Stored and round-tripped today; nothing consumes
them until the transient solver exists.

---

## 4. Geometry

### From a composition map

```python
geom = openndm.Geometry.from_lattice(
    composition,                    # (nz, ny, nx) int array; 2D is promoted
    pitch=20.0,                     # or (dx, dy, dz)
    dz=[20.0, 15.0, ...],           # explicit widths override pitch per axis
    boundaries={...},
    outside="vacuum",
    subdivide=1,                    # or (nx, ny, nz)
)
```

`openndm.INACTIVE` (-1) marks a position outside the core; those nodes are not
created.

### Boundary conditions

Per face, keyed `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max`:

| Value | Meaning |
|---|---|
| `"reflective"` | zero net current, i.e. a symmetry plane |
| `"vacuum"` | zero incoming partial current (Marshak) |
| `"zero_flux"` | zero surface flux |
| `"albedo"` | user supplied `β = J⁻/J⁺` per group |

```python
geom = openndm.Geometry.from_lattice(
    core, pitch=20.0,
    boundaries={"x_max": "albedo"},
    albedo={"x_max": [0.5, 0.3]},    # one value per group
)
```

The limits behave as you would expect and are tested to machine precision:
`β = 1` is reflective, `β = 0` is vacuum, `β = -1` is zero flux.

### `outside` is not `boundaries`

This is the one geometry trap worth stating twice.

`boundaries` applies to the **edges of the mesh**. `outside` applies to a face
looking at an `INACTIVE` position **inside** the mesh. They are separate
because on a quarter-core map the mesh edges carry the *symmetry* conditions,
while a face looking at an out-of-core position is a real outer boundary.

Conflating them reflects neutrons back into the core from outside it. It
inflates `k_eff` by hundreds of pcm and produces no other symptom.

```python
boundaries={"x_min": "reflective", "y_min": "reflective",   # symmetry planes
            "x_max": "zero_flux",  "y_max": "zero_flux"},   # real edges
outside="zero_flux",                                        # out-of-core faces
```

### Subdivision

`subdivide=(nx, ny, nz)` splits each lattice cell without changing the
material layout. Discontinuity factors follow the composition, so a subdivided
assembly keeps its parent's factors.

```python
geom = openndm.Geometry.from_lattice(core, pitch=20.0, subdivide=(4, 4, 1))
```

### Inspecting and mutating

```python
geom.n_nodes, geom.n_surfaces, geom.shape      # shape is (nz, ny, nx)
geom.volumes                                   # (n_nodes,) cm^3
geom.compositions                              # (n_nodes,) int
geom.expand(values, fill=np.nan)               # scatter onto (nz, ny, nx)
geom.set_composition(node, composition)        # in place
```

---

## 5. Settings

```python
settings = openndm.Settings(kernel="sanm", verbosity=1)
```

Every option is a keyword. Unknown names raise rather than being ignored.

### Choosing a kernel

| Kernel | Use it for |
|---|---|
| `"sanm"` | the default; most accurate on a coarse mesh |
| `"nem"` | quartic polynomial nodal, for cross-checking SANM |
| `"fdm"` | plain finite difference; the reference, and fastest per outer |

The kernel is a run-time choice, so one build runs all three and you can
compare them on the same `Model`.

### Convergence

| Option | Default | Notes |
|---|---|---|
| `k_tolerance` | `1e-9` | change in `k_eff` between outers |
| `fission_source_tolerance` | `1e-8` | node-wise fission source change |
| `max_outer` | `500` | raises `ConvergenceError` when hit |
| `inner_tolerance` | `1e-5` | BiCGSTAB relative residual |
| `max_inner` | `50` | |
| `group_sweeps` | `50` | maximum Gauss-Seidel sweeps over groups |
| `group_sweep_tolerance` | `1e-8` | when the sweep stops early |

The defaults leave about 0.1 pcm of iteration error. Loosening
`k_tolerance` past `1e-7` starts to be visible in the fifth decimal.

### Acceleration

| Option | Default | Notes |
|---|---|---|
| `wielandt_shift` | `0.05` | additive; 0 disables. Capped internally so the shifted operator cannot go singular |
| `wielandt_start` | `3` | outer at which the shift begins |
| `nodal_update_interval` | `1` | outers between nonlinear updates |
| `two_node_sweeps` | `2` | group sweeps inside one two-node problem |
| `dhat_limit` | `10.0` | clamp on the corrected coupling coefficient |
| `warm_start` | `False` | reuse the previous flux and coupling |
| `threads` | `0` | OpenMP threads; 0 leaves the environment default |

Results are **bit-identical at any thread count**, so `threads` is purely a
speed knob.

---

## 6. Solving

### Eigenvalue

```python
result = model.solve()                       # uses the model's settings
result = model.solve(kernel="nem")           # override for this call only
result = model.solve(settings=other)         # or replace them wholesale
```

Per-call overrides do not mutate `model.settings`.

### Adjoint

```python
adjoint = model.solve_adjoint()
```

The adjoint reuses the nonlinear coupling coefficients converged by a forward
solve, so a cold adjoint request runs the forward problem first. Both
eigenvalues agree; the flux shapes differ, which is the point.

### Fixed source

```python
source = np.zeros((geom.n_nodes, lib.n_groups))
source[...] = ...                            # source *density*
result = model.solve_fixed_source(source)
```

Fission is included at `k = 1`, so this is subcritical multiplication rather
than a pure absorber.

### Critical boron search

```python
def apply_boron(library, ppm):
    library.set_composition(0, absorption=[0.010, 0.085 + 1e-5 * ppm])
    library.finalize(warn=False)

search = model.search_boron(apply_boron, target_k=1.0, guess=800.0,
                            bracket=(0.0, 3000.0))
print(search.boron, search.k_eff, search.history)
```

You supply the callback, so the boron model stays yours. If the target cannot
be reached the error says so and names the range that *was* reachable.

### Sweeps

```python
def mutate(model, case):
    model.swap_assemblies(*case)

results = model.sweep(mutate, [(0, 6), (2, 50)], warm_start=True)
```

`sweep` mutates the model in place, so cases compound. Call `model.refresh()`
after mutating the geometry or library outside a sweep.

---

## 7. Reading results

```python
result.k_eff                 # float
result.converged             # bool
result.outer_iterations      # int
result.runtime               # seconds
result.kernel                # which kernel produced it

result.flux                  # (n_nodes, n_groups), zero-copy view
result.power                 # (n_nodes,), mean 1.0 over powered nodes
result.power_lattice()       # (nz, ny, nx)
result.radial_power()        # (ny, nx), volume weighted
result.axial_power()         # (nz,)
result.f_q                   # peak node power
result.f_dh                  # peak radial power
result.history               # structured array of the outer iteration
```

`flux` and `power` are views onto the C++ buffers and stay valid as long as
the `Result` does. Copy them if you intend to outlive it.

### Plots

```python
from openndm import plots
plots.plot_radial(result)
plots.plot_axial(result)
plots.plot_convergence(result)
```

Each takes an optional `ax` and returns it.

---

## 8. Checking your answer

Four habits, cheapest first.

**Neutron balance.** Every node should conserve neutrons.

```python
residual = np.abs(model.neutron_balance())
print(residual.max())
```

The residual is bounded by whichever convergence criterion is *looser*, the
inner linear solve or the outer iteration, so compare it against those rather
than against a fixed number. Measured on the model above:

| `inner_tolerance` | `fission_source_tolerance` | worst residual |
|---|---|---|
| `1e-5` (default) | `1e-8` (default) | `4.6e-5` |
| `1e-8` | `1e-8` | `3.6e-8` |
| `1e-11` | `1e-8` | `1.0e-8` |
| `1e-11` | `1e-10` | `8.7e-11` |

Tighten both when you want a sharp check:

```python
strict = openndm.Model(geom, lib, openndm.Settings(
    verbosity=0, inner_tolerance=1e-11, max_inner=600,
    fission_source_tolerance=1e-10, k_tolerance=1e-11))
strict.solve()
assert np.abs(strict.neutron_balance()).max() < 1e-9
```

It fails on a wrong coupling coefficient, a mis-assembled scattering term, or
a boundary condition applied to the wrong face — none of which need move
`k_eff` far enough to notice.

**Symmetry.** A symmetric core map must give a symmetric power.

```python
radial = result.radial_power()
assert np.allclose(radial, radial.T)          # for a diagonally symmetric map
```

The cheapest possible check on the x and y indexing paths.

**Mesh refinement.** Re-solve with `subdivide=2` and `subdivide=4`. A nodal
kernel should barely move; if it moves a lot, the nodal correction is not
doing its job.

**Kernel agreement.** SANM and NEM close their two-node problems by entirely
different routes and should agree to a fraction of a pcm on a refined mesh.
FDM converges to the same limit more slowly, and non-monotonically on a core
with a thin reflector.

---

## 9. Group constants from OpenMC

`openndm.gc` is the only part of the package that needs OpenMC. The full
worked example is
[`examples/02_openmc_to_openndm.ipynb`](../examples/02_openmc_to_openndm.ipynb).

```python
from openndm.gc import from_mgxs_library
xslib = from_mgxs_library(mgxs_library)
```

Also available: `from_mgxs_file` for an `mgxs.h5`, and `from_statepoint`.

### Which MGXS scores to tally

```python
mgxs_lib.mgxs_types = [
    "total", "absorption", "nu-fission", "kappa-fission", "chi",
    "transport", "diffusion-coefficient", "inverse-velocity",
    "consistent nu-scatter matrix",
    "consistent scatter matrix",
]
```

**Tally both scattering matrices.** Their row sums differ by the neutrons that
(n,2n) and (n,3n) create. OpenMC's `absorption` score does not count those
reactions, and a diffusion operator built from a single scattering matrix
cannot see that production at all, because in-scatter and out-scatter are the
same double sum and cancel. OpenNDM subtracts the difference from absorption,
which is exact group by group, and warns when it cannot. On a UO2 and water
mixture the effect is 230 pcm, and the eigenvalue is the only symptom.

Pass `scattering_multiplicity="ignore"` to disable it deliberately.

### The diffusion coefficient

`transport` and `diffusion-coefficient` are different estimators. OpenNDM
prefers the latter, takes `prefer="transport"` to swap, and warns when the two
disagree by more than `d_tolerance` — which they will in a strongly
heterogeneous node.

### Discontinuity factors

```python
from openndm.gc import add_adf_tallies, compute_adf
add_adf_tallies(model, lattice, energy_groups, slab_fraction=0.05)
# ... run OpenMC ...
adf = compute_adf(statepoint, slab_fraction=0.05)
adf.apply_to(xslib, composition=0)
```

The surface flux is estimated from a thin track-length slab rather than a true
surface tally, because a surface tally converges slowly and a discontinuity
factor is a *ratio* that amplifies the noisier of its two terms. `compute_adf`
warns when the resulting uncertainty exceeds a threshold, and that warning is
worth heeding: noisy factors can leave the solution worse than no factors.

Sanity check: a homogeneous assembly must give factors of 1.0, and a pin
lattice must give a thermal factor above 1.0 and a fast factor below it,
because the assembly surface sits in water.

### Critical spectrum

```python
from openndm.gc import critical_spectrum
result = critical_spectrum(total, scatter, nu_fission, chi, method="b1")
result.buckling, result.k_infinity, result.spectrum
result.apply_to(xslib, composition=0)
```

`b1` and `p1` give measurably different answers, so the convention is recorded
on the result. Record it in your statepoint too.

### Branch generation

```python
from openndm.gc import BranchGrid, BranchDriver
grid = BranchGrid(fuel_temperature=[560, 900, 1200], boron=[0, 800, 1600])
driver = BranchDriver(model_factory=..., grid=grid,
                      library_factory=..., workdir="branches")
library = driver.run(processes=4)             # or resume a partial run
driver.write_job_array("submit.sh", scheduler="slurm")
```

Checkpoints after every branch point, so an interrupted run resumes.

---

## 10. Branch libraries and feedback

Declare the axes first — doing so resizes the storage to one composition set
per grid point and discards anything already stored.

```python
lib.set_axes([("fuel_temperature", [560.0, 900.0, 1200.0]),
              ("boron", [0.0, 800.0, 1600.0])])
for state, (temperature, boron) in enumerate(
        itertools.product([560.0, 900.0, 1200.0], [0.0, 800.0, 1600.0])):
    lib.set_composition(0, state=state, ...)
lib.finalize()
```

The state index is row-major over the axes in declaration order, matching what
`BranchGrid` produces.

```python
state = lib.interpolate(fuel_temperature=900.0, boron=800.0)
```

Interpolation is multilinear and exact on data linear in the axes. Outside the
grid, `lib.extrapolation` selects `"clamp"` (default), `"linear"` or
`"error"`.

A branch library cannot be solved directly; collapse it first. That is
enforced, because silently solving at some default state is an easy way to get
a plausible wrong answer.

---

## 11. Embedding and performance

The whole workflow is in memory. Nothing touches the filesystem unless you ask
for a statepoint.

```python
model = openndm.Model(geom, lib, settings)
for case in cases:
    mutate(model, case)
    model.refresh()
    result = model.solve(warm_start=True)
```

* `solve()` releases the GIL, so several models can run from Python threads.
* `flux` and `power` are zero-copy numpy views.
* Results are bit-identical at any thread count, because reductions use a
  fixed-size chunked summation rather than a thread-count-dependent tree.
* A failed solve raises `ConvergenceError` carrying `.iterations` and
  `.residual`. Nothing aborts the process.

---

## 12. Files

### Cross section library

```python
lib.to_hdf5("xslib.h5")
lib = openndm.XSLibrary.from_hdf5("xslib.h5")
```

Round-trips compositions, discontinuity factors, delayed data, uncertainties
and branch axes. The layout is versioned; an unknown version raises rather
than being read as if it were understood.

### Statepoint

```python
openndm.write_statepoint("statepoint.h5", result, model,
                         extra={"leakage_correction": "b1"})

with openndm.StatePoint("statepoint.h5") as sp:
    sp.k_eff, sp.power, sp.radial_power, sp.history
```

The layout follows OpenMC's conventions. Use `extra` for the conventions a
downstream comparison needs — which leakage correction produced the group
constants, which data library, which branch state.

---

## 13. When something goes wrong

### Exceptions

All derive from `openndm.OpenNDMError`.

| Exception | Means |
|---|---|
| `InputError` | inconsistent geometry, settings or library |
| `LibraryError` | a `finalize()` validation check failed |
| `ConvergenceError` | an iteration exhausted its budget; carries `.iterations` and `.residual` |

### Common messages

**"library must be finalized before use"** — call `lib.finalize()`. If you
mutated the library after finalizing, call it again.

**"a branch-parameterised library must be collapsed first"** — call
`lib.interpolate(**state)` and pass the result to `Model`.

**"the model has no fission source"** — every composition has zero
`nu_fission`. Use `solve_fixed_source` instead, or check the map is pointing
at the compositions you think.

**"produces neutrons but has no kappa-fission"** — a warning. `k_eff` will be
right and the power distribution will be all zeros.

**"outer iteration did not converge"** — first raise `max_outer`. If it still
will not converge, try `wielandt_shift=0` to rule out the acceleration, then
`kernel="fdm"` to rule out the nonlinear update. A model that converges with
FDM but not with SANM usually has an extreme coupling coefficient somewhere;
lower `dhat_limit`.

**"cannot correct for scattering multiplicity"** — a warning from
`from_mgxs_library`. Tally both scattering matrices, or pass
`scattering_multiplicity="ignore"` if you mean it.

**"disagree by N%"** — a warning that the two diffusion coefficient estimators
differ. Expected in a strongly heterogeneous node; pick one with `prefer=`.

### The eigenvalue is wrong and nothing else looks wrong

That is the hard case, and the reason for the checks in
[section 8](#8-checking-your-answer). In order:

1. Run `model.neutron_balance()`.
2. Check the symmetry of the power against the symmetry of the map.
3. Check the `outside` boundary condition, which is easy to conflate with the
   face conditions and worth hundreds of pcm on a quarter core.
4. Check the scattering matrix orientation, `scatter[from][to]`.
5. If the constants came from OpenMC, check that both scattering matrices were
   tallied.
6. Refine the mesh, and see whether the kernels converge together.

---

## Where things are

```
python/openndm/       the package
  gc/                 OpenMC group constant generation
examples/             worked notebooks
benchmarks/           runnable benchmark decks
tests/python/         the test suite, which doubles as usage examples
tests/validation/     the OpenMC coupling validation cases
docs/theory.md        every equation, with its discretisation
docs/status.md        requirement-by-requirement implementation state
docs/architecture.md  module layout and the deviations from the spec
```
