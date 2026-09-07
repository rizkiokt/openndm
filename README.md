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

**Pre-alpha (v0.1.0).** The static solver is implemented and verified; the
transient, thermal-hydraulics and pin-power chapters of the specification are
not yet written. `docs/status.md` maps every requirement ID to its state —
read it before assuming a feature exists.

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
  statepoints, with OpenMC as an optional dependency
- HDF5 statepoint output and an `openndm.StatePoint` reader

## Installation

```bash
pip install .            # needs a C++17 compiler; CMake and ninja come from pip
pip install '.[dev]'     # plus pytest, ruff and the docs toolchain
```

OpenMC is optional and only needed for `openndm.gc`.

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
- **V-2, convergence order.** The observed spatial order of each kernel,
  measured over three mesh refinements.
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
- **Determinism.** Bit-identical results on 1, 2, 4 and 8 threads.

`benchmarks/` holds runnable decks; see `benchmarks/README.md` for what each
one currently reproduces, including where a deck does *not* yet match its
published reference.

## Documentation

| Document | Contents |
|---|---|
| [`docs/requirements.md`](docs/requirements.md) | The full specification |
| [`docs/status.md`](docs/status.md) | Requirement-by-requirement implementation state |
| [`docs/theory.md`](docs/theory.md) | Every equation the code solves, with the discretisation written out |
| [`docs/architecture.md`](docs/architecture.md) | Module layout and the deviations from the specification, with rationale |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Build, test and style workflow |

## License

MIT — see [LICENSE](LICENSE).
