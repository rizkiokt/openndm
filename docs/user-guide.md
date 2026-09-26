# OpenNDM user guide

How to drive the solver. For the equations it implements see
[`theory.md`](theory.md); for what is and is not built yet see
[`status.md`](status.md); for runnable examples see [`examples/`](https://github.com/rizkiokt/openndm/tree/main/examples).

---

## Contents

1. [Installing](#installing)
2. [The five objects](#the-five-objects)
3. [Cross sections](#cross-sections)
4. [Geometry](#geometry)
5. [Settings](#settings)
6. [Solving](#solving)
7. [Reading results](#reading-results)
8. [Checking your answer](#checking-your-answer)
9. [Group constants from OpenMC](#group-constants-from-openmc)
10. [Branch libraries and feedback](#branch-libraries-and-feedback)
11. [Control rods](#control-rods)
12. [Transients](#transients)
13. [Thermal-hydraulic coupling](#thermal-hydraulic-coupling)
14. [Embedding and performance](#embedding-and-performance)
15. [Files](#files)
16. [When something goes wrong](#when-something-goes-wrong)

---

## Installing

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

## The five objects

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

## Cross sections

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

#### Rotated assemblies

OpenMC gives you an assembly's factors in one orientation. The same assembly
loaded at a position turned 90 degrees presents different faces to its
neighbours, so the factors have to be permuted to match.

Rotation belongs to the **position**, not to the composition, because one
assembly type is loaded at many positions in different orientations:

```python
rotation = np.zeros_like(core)          # quarter turns counter-clockwise
rotation[core == MOX] = 1               # 90 degrees
geom = openndm.Geometry.from_lattice(core, pitch=21.5, rotation=rotation)

geom.set_rotation(node, 2)              # or per node, absolute not cumulative
geom.rotations                          # quarter turns per node
```

0 to 3 quarter turns counter-clockwise about +z, following KOMODO's `%ADF`
`ROT` convention so a deck translates without re-deriving anything. Only the
discontinuity factors turn; a rotation costs nothing when they are all one.

To permute a set by hand instead, for example to build a pre-rotated
composition:

```python
openndm.rotate_adf(values, quarter_turns)   # returns a new (6, G) array
lib.rotated_adf(0, quarter_turns)           # the same, read from the library
```

### Delayed neutrons

```python
lib.set_delayed(beta=[...], decay_constant=[...], chi_delayed=None)
```

Up to eight precursor groups. Stored and round-tripped today; nothing consumes
them until the transient solver exists.

---

## Geometry

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

## Settings

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

## Solving

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

**Re-finalize the library after writing a composition.** `removal` is derived
from absorption and the scattering matrix when you call `finalize()`, and the
kernels read it rather than recomputing it. Writing a composition marks the
library unfinalized; `refresh()` and every solve refuse until you finalize
again, because otherwise the solve would quietly use the old absorption while
every accessor reported the new one.

```python
lib.set_composition(2, D=..., absorption=..., ...)
lib.finalize(warn=False)     # without this the next solve raises
model.refresh()
```

---

## Reading results

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

## Checking your answer

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

## Group constants from OpenMC

`openndm.gc` is the only part of the package that needs OpenMC. The full
worked example is
[`examples/02_openmc_to_openndm.ipynb`](https://github.com/rizkiokt/openndm/blob/main/examples/02_openmc_to_openndm.ipynb).

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

## Branch libraries and feedback

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

## Control rods

A `ControlRodBank` is a set of radial lattice columns that share one axial
position, and a substitution from each unrodded composition to its rodded
counterpart. Positions are given in steps, following the convention KOMODO's
`%CROD` card uses, so an existing deck translates without arithmetic:
`zero_position` is the tip height above the bottom of the mesh at step 0 and
`step_size` is the travel per step. **Step 0 is fully inserted**; increasing
steps withdraw the bank upward.

```python
import numpy as np
import openndm

banked = np.zeros((9, 9), dtype=bool)
banked[0, 0] = True

bank = openndm.ControlRodBank(
    "A",
    columns=banked,
    rodded={FUEL: FUEL_RODDED, REFLECTOR: REFLECTOR_RODDED},
    step_size=1.0,
    zero_position=20.0,      # the bottom axial reflector
)
rods = openndm.ControlRods(geometry, [bank])

rods.insert(A=80.0)          # tip 80 cm above the bottom of the active core
model.refresh()              # the solver is holding this geometry
print(model.solve().k_eff)
```

Build the geometry with the banks **withdrawn** — every node carrying its
unrodded composition. `ControlRods` snapshots that state, which is what makes
positions absolute rather than incremental: moving a bank twice gives the same
core as setting its final position once, and `rods.withdraw()` restores
exactly what was there before.

### Worth curves

`worth_curve` sweeps a bank and returns integral and differential worth:

```python
curve = rods.worth_curve(model, "A", range(0, 229, 12))

curve.steps           # positions swept
curve.tip_height      # cm above the bottom of the mesh
curve.k_eff
curve.integral        # pcm, relative to fully withdrawn
curve.differential    # d(integral)/d(step)
```

Worth is a **reactivity** difference, `1/k(p) − 1/k_ref`, signed so that an
inserted bank has positive worth. `Δk/k` is a common shortcut that drifts from
this by of order its own square, which matters once a bank is worth several
thousand pcm.

Mind the sign of `differential`: steps *withdraw* the bank, so it is negative
for a normal bank. It is a plain derivative of `integral`, so integrating it
returns `integral`; a conventional differential-worth plot is its negative.

Every other bank stays where it is, which is how an overlapping sequence is
modelled — set the others first. For a sequence that moves several banks, pass
mappings instead:

```python
curve = rods.worth_curve(
    model,
    positions=[{"A": None, "B": None}, {"A": 0.0, "B": None}, {"A": 0.0, "B": 0.0}],
    reference={"A": None, "B": None},
)
```

Points warm start from each other by default, which is the case warm starting
exists for: neighbouring positions differ in one node. It is an accelerator,
not an approximation — the curve is the same either way, and a test asserts it.
The bank is put back where it was when the sweep finishes, so measuring worth
does not move the rods.

Every composition a bank can reach must appear in `rodded`, including
reflector compositions if the rods travel through an axial reflector. A
missing key raises rather than passing the node through unchanged, because an
unsubstituted node is indistinguishable from a correctly withdrawn one.

### Partial insertion, and the cusping correction

A tip that lands between plane boundaries leaves one node partly rodded.
Without a correction that node is rounded to whichever state covers its
centre, so `k_eff` is a staircase in rod position: on a ten-plane test core
the largest single step is about 1561 pcm.

Give the bank a `cusp` slot and the partial node gets a homogenised mixture
instead:

```python
bank = openndm.ControlRodBank(
    "A", columns=banked, rodded={FUEL: FUEL_RODDED},
    cusp={FUEL: MIXTURE},          # a spare composition index
)
rods = openndm.ControlRods(geometry, [bank], library=lib)
```

`cusp` maps each unrodded composition to a **spare composition in the
library**, which the mixture is written into each time the bank moves. Reserve
those slots when you build the library.

`insert` weights the mixture by volume alone. That is the flat-flux limit and
it is biased: the flux is depressed on the rodded side, so volume weighting
over-counts the rodded absorption and puts `k_eff` low. Correcting it needs a
flux, and a flux needs a solve, so it is an iteration:

```python
rods.insert(A=32.5)
model.refresh()
result = rods.converge_cusping(model)   # solves, re-weights, repeats
```

Measured against a 0.5 cm reference mesh on which the tip always falls on a
boundary:

| | largest error | mean bias | largest step |
|---|---|---|---|
| no cusping | 784 pcm | +165 pcm | 1561 pcm |
| volume-weighted (`insert`) | 259 pcm | −103 pcm | 625 pcm |
| flux-weighted (`converge_cusping`) | **55 pcm** | **−13 pcm** | 466 pcm |
| the reference itself | — | — | 410 pcm |

Flux weighting brings the error down to the size of the coarse mesh's own
discretisation error: at positions where the tip *does* land on a boundary,
this core is already 11–42 pcm from the reference, and cusping does not touch
those. Putting the tip on a plane boundary, as `benchmarks/iaea3d` does,
remains the most accurate option.

---

## Transients

A transient starts from a converged static solution and is advanced step by
step:

```python
model.solve()                       # the initial state
transient = model.start_transient(theta=0.5)

for _ in range(200):
    step = transient.step(1.0e-3)
    print(step.time, step.total_power)
```

`theta` weights the time integration: 1 is fully implicit and the default,
0.5 is Crank-Nicolson. The observed order is 1.00 and 2.00 respectively, so
0.5 is worth using wherever the solution is smooth — on the test case in
`docs/theory.md` §11 it is about 800× more accurate at the same step size.

**The library needs two things a static solve does not:** inverse velocities
on every composition, and delayed data.

```python
lib.set_composition(0, ..., inv_velocity=[1/1.8e7, 1/2.2e5])
lib.set_delayed(beta=[...], decay_constant=[...], chi_delayed=[...])
```

Both raise if missing rather than defaulting, because a plausible default
here is a wrong answer that looks right.

### Driving it

Anything you change between steps belongs to the step that follows. Rod
banks, cross sections and compositions all work:

```python
rods.insert(A=0.0)                  # scram
step = transient.step(1.0e-3)
```

**Do not call `model.refresh()` between steps.** A step re-reads the cross
sections, the node compositions and the coupling coefficients by itself, so
`refresh()` raises during a transient.

Re-finalize the library after writing a composition, as for any other
mutation — see [Solving](#solving). That is still required; it is `refresh`
that is not.

A static `model.solve()` ends the transient, because it overwrites the flux
and the eigenvalue. Stepping the old `Transient` afterwards raises rather than
continuing from a static solution.

### What a step gives you

`TransientStep` carries `time`, `dt`, `total_power`, `peak_power`,
`iterations` and `converged`. Power is **not** renormalised, since the point
of a transient is that it moves. `transient.flux` and `transient.precursors`
give the current state.

### Temperature feedback

`DopplerFeedback` applies a square-root law to a set of compositions:

```python
doppler = openndm.DopplerFeedback(
    lib, compositions=[FUEL_1, FUEL_2],
    gamma=3.034e-3, reference_temperature=300.0,
    groups=[1],                 # thermal group only, as LRA specifies
)
doppler.apply([900.0, 1100.0])  # one temperature per composition
model.refresh()
```

`Sigma(T) = Sigma_0 [1 + gamma (sqrt(T) - sqrt(T_0))]`. Doppler broadening
widens a capture resonance as the square root of temperature, so a per-Kelvin
coefficient is a linearisation of this — fine near `T_0`, less so across the
hundreds of Kelvin a transient covers.

Base data is snapshotted, so temperatures are absolute: applying twice
matches applying once, and `apply(T_0)` restores the library bit-for-bit.
`apply` re-finalizes, so you only need `model.refresh()`.

**Cross sections live per composition, not per node.** A temperature
*distribution* therefore needs one composition per region that can hold its
own temperature — give each such region its own index in the core map.

A branch library from OpenMC carries the real temperature dependence of every
cross section and is strictly better where you have one. This is for when you
do not.

### What is not implemented

No exponential transformation, no adaptive time stepping, no decay heat, and
no feedback — FR-MODE-7 needs a thermal-hydraulics model, and there is no
built-in one yet. There *is* somewhere to attach your own; see the next
section. The nonlinear nodal coupling coefficients are held at their static
values inside a step, so the nodal kernels drift from consistency as the flux
shape moves; FDM has nothing to freeze.

---

## Thermal-hydraulic coupling

A closed-channel coolant model and radial pin conduction are built in, and
`Model.solve_coupled` runs the Picard loop over them. Everything behind the
loop is replaceable: the interface, the driver and the mesh mapping know no
physics, so an external solver attaches in the same place.

### A coupled steady state

```python
pins = openndm.PinGeometry(
    fuel_radius=4.1195e-3, gap_thickness=6.8e-5, clad_thickness=5.71e-4,
    pin_pitch=1.2655e-2, n_pins=264, n_guide_tubes=25,
)
conduction = openndm.PinConduction(
    pins, fuel_conductivity=3.0, clad_conductivity=15.0,
    gap_conductance=1.0e4, film_coefficient=3.0e4, doppler_weight=0.7,
)
channel = openndm.ChannelModel(
    geom, pins, mass_flow=82.12, inlet_temperature=559.15,
    pressure=15.5e6, direct_heating=0.019, conduction=conduction,
)

mapping = openndm.CompositionMapping(geom)

def apply_state(temperatures, densities):
    branch.interpolate_by_composition(
        {
            "fuel_temperature": mapping.average(temperatures["doppler_temperature"]),
            "moderator_temperature": mapping.average(temperatures["moderator_temperature"]),
            "coolant_density": mapping.average(densities["moderator_density"]),
        },
        out=model.library,
    )

coupled = model.solve_coupled(channel, apply_state, total_power=693.75e6)
print(coupled.k_eff, coupled.iterations)
print(coupled.temperatures["fuel_temperature"])
```

`total_power` is the thermal power of the geometry you modelled, so a quarter
core carries a quarter of the core's power. `percent` scales it. The relative
power the solver reports is turned into watts per node by volume weighting;
`openndm.absolute_power` does that on its own if you need it elsewhere.

### The channel

One channel per radial column of the core map, at fixed mass flow and constant
pressure, so energy is the only conservation law left. Enthalpy integrates up
the channel and inverts through the water backend for temperature and density.

`mass_flow` and the pin counts are **per channel, not per assembly**. If you
used `subdivide`, one assembly is several columns and both must be scaled.

`direct_heating` does **not** change the outlet temperature. In steady state
every watt reaches the coolant whichever route it takes; the fraction splits
the power between coolant and pin and so sets the linear heat rate the pin
sees. Raising it lowers the fuel temperature and leaves the outlet alone.

### The pin

`PinConduction` meshes the pellet radially and puts the gap, the cladding and
the film in series behind it. It reports `fuel_temperature`, the volume-average
pellet temperature, and `doppler_temperature`, weighted `doppler_weight` on the
surface and the rest on the centreline. The default 0.7/0.3 is a convention,
not a law; at 0.5 you get the volume average exactly.

Conductivities are yours to supply, as a constant or a callable `k(T)`:

```python
conduction = openndm.PinConduction(pins, fuel_conductivity=lambda t: 3.0 + 0.0,
                                   clad_conductivity=15.0, gap_conductance=1.0e4,
                                   film_coefficient=3.0e4, n_rings=20)
```

**No conductivity correlation ships with OpenNDM.** The published ones are
temperature-dependent and citing one is your call, not ours. A constant is
exact at any ring count; a callable is evaluated at each ring's outer boundary,
which is first order in `n_rings`, so mesh accordingly.

### Boiling

A PWR channel stays liquid. A BWR one does not, and `two_phase=True` lets it
boil:

```python
channel = openndm.ChannelModel(
    geom, pins, mass_flow=15.0, inlet_temperature=550.0,
    pressure=7.0e6, two_phase=True,
)
channel.quality          # equilibrium steam quality per node
channel.void_fraction    # void fraction per node
```

Nothing about the enthalpy integration changes. Only its inversion does: above
the saturated liquid enthalpy the temperature stops at the boiling point and
the surplus becomes quality and void, and `moderator_density` becomes the
mixture density. **A channel that stays subcooled gives bit-identical answers
with and without the flag**, so turning it on extends the range of validity
rather than switching model.

The backend has to know both sides of the saturation line, so `ConstantWater`
is refused here and `IF97Water` is not. A channel driven past the saturated
vapour enthalpy raises rather than reporting superheated steam, which is not
modelled.

`slip_ratio` defaults to 1, which is the homogeneous equilibrium model proper
and makes the mixture density exactly the inverse of the mass-weighted
specific volume. Anything else is a correlation you are choosing; **no
published slip correlation ships with OpenNDM**, for the same reason no
conductivity correlation does.

### Cross sections live per composition

This is the constraint the whole coupled mode is shaped around. A temperature
varies node to node; cross sections do not. A distribution you want resolved
therefore needs one composition per region that can differ, which is a property
of your core map and nothing the code can invent for you.

`CompositionMapping` is where the two meet, explicitly:

```python
mapping = openndm.CompositionMapping(geom)              # volume-weighted
by_power = openndm.CompositionMapping(geom, weights=result.power)
states = mapping.average(temperatures["doppler_temperature"])   # (n_compositions,)
```

Weighting by power rather than volume is often the better choice for a Doppler
temperature: the one that matters is where the fissions are.

`XSLibrary.interpolate_by_composition` then collapses the branch grid with a
different state per composition, which `interpolate` cannot do. Pass
`out=model.library` so it writes in place: the solver holds a reference to that
library object, and replacing it would strand the solver on the old one.
`solve_coupled` raises if you replace it.

The grid is interpolated once per *distinct* state row, so compositions sitting
at the same condition cost one interpolation between them.

### Attaching your own solver

A thermal solver is anything with four methods:

```python
class Channel:
    def set_heat_source(self, q): ...      # node power, W
    def solve(self): ...
    def get_temperatures(self): ...        # {name: array}, K
    def get_densities(self): ...           # {name: array}, kg/m^3
```

`isinstance(channel, openndm.ThermalSolver)` checks it. The field names are
yours, and the point of that is that they can be the axis names of a branch
library, so the state the solver reports feeds the library unchanged.

`PicardCoupling` is the loop on its own, if you want to drive the neutronics
yourself rather than through `solve_coupled`:

```python
coupling = openndm.PicardCoupling(channel, apply_state, relaxation=0.7)
result = coupling.solve(node_power, k_eff_source=lambda: last["result"].k_eff)
```

The loop stops when the node power moves by less than `tolerance` relative to
the mean power, and raises `ConvergenceError` if it is still moving after
`max_iterations`. It does not return an unconverged state. Lower `relaxation`
if it oscillates: a coupling whose power falls off temperature steeply enough
to diverge at `relaxation=1.0` usually converges in a few iterations at 0.25.

`coupled.history` holds one `CouplingStep` per iteration with the power change
and the eigenvalue.

Every iteration calls `model.refresh()`, which keeps the flux, so passing
`warm_start=True` starts each solve from the previous iteration's.

### Water properties

The channel model reads its properties through a protocol, so you can swap
the backend:

```python
water = openndm.IF97Water()
water.density(15.5e6, 583.0)               # kg/m^3, ~705 at PWR conditions
water.enthalpy(15.5e6, 583.0)              # J/kg
water.temperature(15.5e6, 1.35e6)          # K, the inverse the channel needs
water.saturation_temperature(15.5e6)       # K, ~618
```

Pressures are Pa, temperatures K, enthalpies J/kg and densities kg/m^3, and
everything broadcasts over arrays.

`IF97Water` is IAPWS-IF97 region 1 (compressed liquid), region 2 (vapour) and
region 4 (the saturation line). Steam comes from the `vapour_` methods:

```python
water.vapour_density(7.0e6, 600.0)         # kg/m^3
water.vapour_enthalpy(7.0e6, 600.0)        # J/kg
water.region23_pressure(700.0)             # Pa, the region 2/3 boundary
```

Both sides of the saturation line, which is what a two-phase model needs:

```python
water.saturated_liquid_enthalpy(7.0e6)     # J/kg, ~1.267e6
water.saturated_vapour_enthalpy(7.0e6)     # J/kg, ~2.773e6
water.saturated_liquid_density(7.0e6)      # kg/m^3, ~740
water.saturated_vapour_density(7.0e6)      # kg/m^3, ~36.5
water.latent_heat(7.0e6)                   # J/kg, the gap between them
```

Those four are `SaturationProperties`, a protocol of their own rather than
part of `WaterProperties`. A single-phase channel never asks for them, and
`ConstantWater` does not provide them, so a two-phase model checks
`isinstance(water, openndm.SaturationProperties)` before it starts.

**Region 3 is not implemented**, the dense fluid around the critical point.
The practical consequence is that the saturation line is reachable only up to
`openndm.water.SATURATION_REGION3_PRESSURE`, 16.529 MPa, where it runs into
the region 2/3 boundary. A PWR at 15.5 MPa is below that and a BWR at 7 MPa
far below. Above it the saturated accessors raise.

A state outside the implemented regions raises rather than returning a number
the equation does not stand behind.

`ConstantWater` has fixed density and specific heat. Use it to verify a
channel, not to run one: with constant properties the axial enthalpy rise is
exactly the integral of the heat input, so there is a closed-form answer to
check against. That is how the channel model here is verified, to 1e-12.

`openndm.water.external_backend()` adapts the `iapws` package or CoolProp if
you have one installed, which is how you make properties agree with an
external thermal-hydraulics code. Neither is a dependency; `iapws` is
GPL-licensed, so installing it is your decision rather than ours.

### Mapping onto another mesh

An external solver rarely uses the neutronics mesh. `AxialMapping` transfers
between two axial meshes over the same span, conserving volume:

```python
mapping = openndm.AxialMapping(source_edges=[0.0, 30.0, 70.0, 100.0],
                               target_edges=np.linspace(0.0, 100.0, 21))
channel_power = mapping.distribute(watts_per_node)   # keeps the total
node_temperature = mapping.reverse().average(channel_temperature)
```

The two directions are different operations and choosing the wrong one
conserves nothing while looking entirely plausible:

| Field | Method | What is preserved |
|---|---|---|
| power, heat rate — **extensive** | `distribute` | the total |
| temperature, density — **intensive** | `average` | the volume-weighted mean |

Both take a trailing mesh axis and leave the leading ones alone, so a whole
core of channels maps in one call: pass an array of shape
`(n_channels, n_source)` and get back `(n_channels, n_target)`.

Everything is numpy arrays in memory. Nothing in this module writes a file.

---

## Embedding and performance

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

## Files

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

### VTK, for ParaView

```python
openndm.write_vtk("core.vtu", result, model)
```

One hexahedral cell per node, on the real mesh including non-uniform widths
and whatever `subdivide` produced. Cell data is `power`, `flux_g1` to
`flux_gG` and `composition`; the eigenvalue rides along as field data. Groups
are numbered from one in the file, the way you read them off a legend, though
the arrays they come from are indexed from zero.

**An out-of-core position gets no cell at all**, rather than a cell carrying
zero. In a plot those two are indistinguishable, and the first is how a wrong
core map goes unnoticed. A reflector node is *not* out of core: it makes no
power but it is there, and it is written.

`composition` is what makes a loading pattern or a rod position visible, so a
dump before solving and one after a bank moves are worth comparing.

Use `extra` for a per-node field the solver does not produce:

```python
openndm.write_vtk("core.vtu", result, model,
                  extra={"fuel_temperature": temperatures})
```

The writer has no dependency — the format is XML and is written directly, so
nothing needs VTK installed.

---

## When something goes wrong

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
[Checking your answer](#checking-your-answer). In order:

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
