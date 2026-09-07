# Changelog

All notable changes to OpenNDM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html) (NFR-QA-5).

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-09-07

First release. Covers milestones M0 and M1 of the specification, the M3 solver
kernels, and the M2 OpenMC ingestion path. `docs/status.md` maps every
requirement ID to its state.

### Added

**Geometry (FR-GEO-1..7)**

- 3D Cartesian node/surface connectivity graph with non-uniform mesh spacing,
  built from an `(nz, ny, nx)` composition map.
- Out-of-core lattice positions, radial subdivision with automatic
  discontinuity factor inheritance, and per-face zero-flux, vacuum, reflective
  and per-group albedo boundaries.
- Faces looking at an out-of-core position are distinguished from faces on the
  mesh edge, so a quarter-core map cannot inherit a reflective symmetry
  condition on its real outer boundary.

**Cross sections (FR-XS-1..9)**

- Arbitrary group count with full `G×G` scattering including upscattering,
  delayed neutron data for up to 8 precursor groups, per-face per-group
  discontinuity factors and per-value standard deviations.
- Branch-parameterised libraries over arbitrary state variables with
  multilinear interpolation and a configurable extrapolation policy.
- Load-time validation that fails on unphysical data but only warns on
  negative scattering transfers, which Monte Carlo noise produces routinely,
  and on a fissile composition with no kappa-fission, whose only symptom would
  otherwise be a silently zero power distribution.
- Versioned `xslib.h5` round-tripping.

**Solver (FR-SOL-1..8, FR-MODE-1..4, FR-MODE-8)**

- Three run-time-selectable kernels: finite difference, polynomial nodal (NEM,
  quartic expansion) and semi-analytic nodal (SANM, the default), coupled
  through a nonlinear two-node CMFD iteration with a quadratic transverse
  leakage fit.
- Power iteration with a Wielandt shift over ILU0-preconditioned BiCGSTAB
  inner solves, with the shift capped so the shifted operator cannot turn
  singular.
- Forward, adjoint and fixed-source modes; critical boron search; parametric
  sweeps; loading pattern mutation.

**Embedding (FR-OPT-1..7)**

- Fully in-memory workflow, zero-copy numpy result views, GIL released during
  the solve, warm start from a previous solution, and typed exceptions on
  every failure path.
- Bit-identical results regardless of thread count, through fixed-size chunked
  reductions.

**OpenMC interface (FR-OMC-1..11, FR-OMC-14)**

- `openndm.gc.from_mgxs_library`, `from_mgxs_file` and `from_statepoint`
  ingestion, with uncertainty propagation and a documented preference order
  between the `diffusion-coefficient` and `transport` sources of D.
- ADF tally instrumentation and surface-to-volume ratio extraction, using a
  thin track-length slab rather than a surface tally.
- B1 and P1 critical spectrum with a buckling search that stays inside the
  physical region, recording which convention was used.
- Adjoint-weighted kinetics parameters with a k-ratio fallback, pin form
  functions, and a branch driver with checkpointing, resume and job-array
  emission.
- OpenMC is an optional dependency; `import openndm` never pulls it in.

**Output (FR-OUT-1..3, FR-OUT-5, FR-OUT-7)**

- Versioned `statepoint.h5` on OpenMC's conventions, an `openndm.StatePoint`
  reader, node/radial/axial power distributions, `F_q` and `F_ΔH`, iteration
  history and matplotlib plotting helpers.

### Verified

- **V-1**: bare homogeneous cuboid against `k = νΣf / (Σa + D B²)`. All three
  kernels converge second order; 0.05 pcm at 64 nodes per side.
- **V-2**: observed spatial convergence order of every kernel.
- **V-3**: coarse-mesh SANM and NEM against refined FDM. On the IAEA-2D core
  map, SANM at one node per assembly lands 18 pcm from the fine-mesh limit and
  all three kernels agree to within 25 pcm under refinement.
- **V-4**: adjoint and forward eigenvalues agree to under 1 pcm while the flux
  shapes differ.
- Infinite-medium `k_∞` reproduced exactly, and leakage-free fixed-source
  multiplication reproduced exactly.

### Known limitations

- The IAEA-2D and IAEA-3D decks do **not** reproduce their published
  eigenvalues; see `benchmarks/README.md` for the numbers and the likely
  causes. They are shipped as regression baselines.
- The nonlinear nodal correction is applied on interior surfaces only.
  Boundary faces keep their finite difference coupling, which is what limits
  the nodal kernels to second order overall on the analytic case.
- Not implemented, and documented as such in `docs/status.md`: transients and
  kinetics (FR-KIN), thermal hydraulics (FR-TH), pin power reconstruction
  (FR-OUT-4), VTK export (FR-OUT-6), XML input and the standalone executable
  (FR-IN-2), and the KOMODO input reader (FR-IN-4).

### Deviations from the specification

Recorded with rationale in `docs/architecture.md`. The substantive ones:
pybind11 instead of a ctypes-called C API; a purpose-built 7-point sparse
structure instead of Eigen; numpy views through pybind11 instead of xtensor.

[Unreleased]: https://github.com/rizkiokt/openndm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rizkiokt/openndm/releases/tag/v0.1.0
