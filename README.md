# OpenNDM

**Open Nodal Diffusion Method** — a three-dimensional, multi-group nodal
diffusion solver for reactor core analysis, with a C++17 core, a pybind11
Python API, and a group-constant generation path built natively on the OpenMC
stack.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

OpenNDM is the second stage of a two-step lattice → core calculation scheme in
which OpenMC performs the lattice-physics stage. Its distinguishing goal:

> A user who already has an `openmc.mgxs.Library` object should be able to run
> a full-core nodal calculation without writing a single line of
> format-conversion code.

See [`docs/requirements.md`](docs/requirements.md) for the full software
requirements specification, and [`docs/status.md`](docs/status.md) for exactly
which of those requirements are implemented today.

---

## Status

**Pre-alpha (v0.1.0).** The static solver is implemented and verified against
both the analytic bare-cuboid solution and the published IAEA-2D/3D PWR
benchmarks; the transient, thermal-hydraulics and pin-power chapters of the
specification are not yet written. `docs/status.md` maps every requirement ID
to its state — read it before assuming a feature exists.

Implemented and tested:

- 3D Cartesian geometry on an abstract node/surface graph, non-uniform mesh,
  inactive core-map positions, per-face zero-flux / vacuum / reflective /
  albedo boundaries
- Arbitrary group count with full `G×G` scattering including upscattering,
  assembly discontinuity factors, branch-parameterised libraries with
  multilinear interpolation
- Three run-time-selectable kernels: FDM, polynomial nodal (NEM) and
  semi-analytic nodal (SANM), coupled through a nonlinear two-node CMFD
  iteration
- Power iteration with Wielandt shift, ILU0-preconditioned BiCGSTAB inners,
  forward / adjoint / fixed-source modes, critical boron search
- `openndm.gc` ingestion of `openmc.mgxs.Library`, `mgxs.h5` files and
  statepoints, with OpenMC as an optional dependency, validated end to end
  against a real OpenMC run (C-1)
- HDF5 statepoint output and an `openndm.StatePoint` reader

## Installation

```bash
pip install .            # needs a C++17 compiler; CMake and ninja come from pip
pip install '.[dev]'     # plus pytest, ruff and the docs toolchain
```

OpenMC is optional and only needed for `openndm.gc`.

## Examples

Four executed notebooks live in [`examples/`](examples/). The one the package
exists for is
[`02_openmc_to_openndm.ipynb`](examples/02_openmc_to_openndm.ipynb): an OpenMC
lattice calculation becomes a nodal core calculation with no format conversion
written by hand. On the run committed there it reproduces OpenMC's `k_inf` to
12 pcm, 0.2σ.

[`docs/user-guide.md`](docs/user-guide.md) is the reference.

## Quick start

```python
import numpy as np
import openndm

# A 2-group, 3D quarter-core model.
lib = openndm.XSLibrary(n_groups=2, n_compositions=2)
lib.set_composition(
    0,                                                 # fuel
    D=[1.5, 0.4], absorption=[0.010, 0.085],
    nu_fission=[0.0, 0.135], kappa_fission=[0.0, 0.135],
    chi=[1.0, 0.0], scatter=[[0.0, 0.020], [0.0, 0.0]],
)
lib.set_composition(
    1,                                                 # reflector
    D=[2.0, 0.3], absorption=[0.0, 0.010],
    scatter=[[0.0, 0.040], [0.0, 0.0]],
)
lib.finalize()                                         # validates, returns warnings

core = np.zeros((10, 9, 9), dtype=int)
core[:, 8, :] = 1                                      # radial reflector
core[:, :, 8] = 1

geom = openndm.Geometry.from_lattice(
    core, pitch=(20.0, 20.0, 20.0),
    boundaries={"x_min": "reflective", "y_min": "reflective",
                "x_max": "zero_flux", "y_max": "zero_flux",
                "z_min": "vacuum", "z_max": "vacuum"},
)

model = openndm.Model(geom, lib, openndm.Settings(kernel="sanm", verbosity=0))
result = model.solve()

print(result.k_eff)              # 1.036880
print(result.radial_power())     # assembly-wise relative power, (9, 9)
print(result.f_q, result.f_dh)   # peaking factors
```

## Verification

`pytest tests/python` runs the verification suite, and
`ctest --test-dir build` the C++ one. The checks that pin down correctness:

- **V-1, analytic buckling.** A bare homogeneous cuboid against
  `k = νΣf / (Σa + D B²)`. Every kernel converges second order; at 64 nodes per
  side the error is 0.1 pcm.
- **V-1, reflected slab.** A core plus reflector against the transcendental
  criticality condition `D₁B tan(B a) = D₂κ coth(κb)`, which exercises a
  material interface, a reflector and two boundary conditions at once.
- **Albedo boundary.** Against its own analytic condition `D B tan(B a) = γ`
  across five albedo values, plus all three limits: β=1 is reflective, β=0 is
  vacuum, β=−1 is zero flux.
- **V-2, manufactured solutions.** The full multi-group operator, scattering
  and fission included, against a solution chosen in advance, with the
  observed order of accuracy of each kernel.
- **V-3, kernel consistency.** Coarse-mesh SANM and NEM against a refined FDM
  solution of the same problem. On the IAEA-2D core map, SANM at one node per
  assembly lands 18 pcm from the fine-mesh converged eigenvalue, and all three
  kernels agree to within 25 pcm under refinement. This is the check that
  catches a wrong transverse leakage, a wrong discontinuity factor convention
  or a broken two-node closure; mesh refinement alone would not, because a
  kernel with any of those defects still converges smoothly, just to the wrong
  answer.
- **V-4, adjoint reciprocity.** Forward and adjoint eigenvalues agree to under
  1 pcm while the flux shapes differ.
- **V-4, perturbation theory.** First-order adjoint-weighted reactivity
  against a direct re-solve. This tests the adjoint flux *shape*; eigenvalue
  equality alone only confirms the operator was transposed.
- **Neutron balance.** Every node conserves neutrons to 3e-11, and the
  core-wide leakage-plus-absorption-equals-production identity closes to 1e-9.
- **Symmetry.** A symmetric core map gives a symmetric power; rotating or
  mirroring a lopsided core is a pure relabelling.
- **Determinism.** Bit-identical results on 1, 2, 4 and 8 threads.

### OpenMC coupling

`tests/validation/` runs the specification's coupling validation cases.

**C-1** is a
uniform infinite medium in OpenMC with continuous-energy physics, whose
multi-group cross sections are then solved by OpenNDM. It isolates the
translation path from every other source of error, and reproduces OpenMC's
`k_inf` to **12 pcm, 1.2σ** over 90M histories.

**C-2** is a single reflected assembly, which must give discontinuity factors
of exactly 1.0 when homogeneous and a thermal factor above 1.0 when it is a
pin lattice. Measured: **1.0 to 1.2σ**, and **1.045 thermal / 0.993 fast**.

Between them these found three defects that stand-in test doubles could not
have caught, two of which fail silently — see
[`tests/validation/README.md`](tests/validation/README.md).

### Benchmarks

| Deck | Reference | Result |
|---|---|---|
| Bare cuboid, analytic | `k = νΣf / (Σa + D B²)` | 0.05 pcm at 64 nodes/side |
| IAEA-2D PWR | published `k_eff = 1.02959` | SANM **−3.7 pcm** at one node per assembly |
| IAEA-3D PWR | published `k_eff = 1.02903` | SANM **+44.7 pcm** at a 20 cm axial mesh |

All three meet the specification's 100 pcm acceptance criterion. See
[`benchmarks/README.md`](benchmarks/README.md) for the full convergence
tables and for the two transcription errors the structural invariants now
catch.

## Documentation

| Document | Contents |
|---|---|
| [`docs/user-guide.md`](docs/user-guide.md) | **How to drive the solver.** Start here |
| [`examples/`](examples/) | Four executed notebooks, including OpenMC to OpenNDM end to end |
| [`docs/requirements.md`](docs/requirements.md) | The full specification |
| [`docs/status.md`](docs/status.md) | Requirement-by-requirement implementation state |
| [`docs/theory.md`](docs/theory.md) | Every equation the code solves, with the discretisation written out |
| [`docs/architecture.md`](docs/architecture.md) | Module layout and the deviations from the specification, with rationale |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Build, test and style workflow |

## License

MIT — see [LICENSE](LICENSE).
