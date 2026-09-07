# Changelog

All notable changes to OpenNDM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html) (NFR-QA-5).

## [Unreleased]

### Added

- **C-1 OpenMC coupling validation** (`tests/validation/c1_homogeneous.py`),
  the first of the specification's §6.3 cases. A uniform infinite medium runs
  in OpenMC with continuous-energy physics and the multi-group cross sections
  from that same run are solved by OpenNDM, isolating the translation path
  from every other source of error. It agrees to 12 pcm, 1.2 sigma.
- Opt-in OpenMC integration tests (`tests/python/test_openmc_integration.py`),
  skipped unless OpenMC, its executable and a nuclear data library are all
  available.
- `scattering_multiplicity` option on `from_mgxs_library`.
- **Analytic verification suite** (`tests/python/test_analytic.py`), the
  standard set a nodal diffusion code is verified against: a reflected slab
  closed by a transcendental criticality condition, an albedo boundary against
  its own analytic condition plus all three of its limits, and a method of
  manufactured solutions for the full multi-group operator (V-2).
- **First-order perturbation theory** against a direct re-solve (V-2's
  companion, V-4). Checking that the adjoint eigenvalue equals the forward one
  only confirms the operator was transposed; weighting a localised
  perturbation with the adjoint flux tests its *shape*, and the error must
  fall linearly with the perturbation rather than hitting a floor.
- **Symmetry invariance**: a diagonally symmetric core map must give a
  diagonally symmetric power, and rotating or mirroring a lopsided core must
  be a pure relabelling.
- `Model.surface_currents()` and `Model.neutron_balance()`. The node-wise
  balance is the standard internal consistency check for a nodal code, and
  closes to 3e-11 on all three kernels.

### Fixed

- **The external source was missing from the two-node problem**, so a
  fixed-source solve with a nodal kernel reconstructed the within-node shape
  from the scattering and fission sources alone. Found by the manufactured
  solution, where it made SANM and NEM nine times *worse* than plain finite
  difference: a nodal kernel losing to FDM on a smooth problem is the
  signature of a term missing from its local problem. The external source is
  now expanded on the same quadratic basis as the transverse leakage, which
  puts both nodal kernels ahead of FDM and raises their observed order of
  accuracy above two. The eigenvalue path is untouched, since there is no
  external source in it.
- **(n,2n) and (n,3n) production was lost in translation**, worth 230 pcm on a
  UO2 and water mixture. OpenMC's `absorption` score excludes those reactions;
  the extra neutrons appear only as an excess in the `nu-scatter matrix` row
  sums. A diffusion operator built from a single scattering matrix cannot see
  them, because in-scatter and out-scatter are the same double sum and cancel
  identically. `from_mgxs_library` now subtracts the multiplicity excess from
  absorption, which is algebraically exact group by group, and warns when the
  library carries only one of the two matrices. Found by C-1; it fails
  silently, with the scattering orientation, the computed spectrum and the
  solver's internal consistency all looking correct.
- **`MGXS.get_xs` takes integer domain ids, not domain objects.**
  `Library.get_mgxs` takes the object, so passing it straight through is the
  natural mistake; real OpenMC raises a `TypeError` from inside its argument
  checking. The stand-in test doubles now enforce the same contract.

- **IAEA-2D core map.** The transcribed quarter-core map was asymmetric about
  the diagonal at one position pair, and was missing one cell of the
  peripheral fuel-1 band at `(5,5)`, which broke that band into two
  disconnected arcs. Restoring it moved `k_eff` by +75 pcm and brought the
  deck from 81 pcm below the published reference to 6 pcm below it. The map
  now carries two structural invariants — diagonal symmetry and an
  edge-connected fuel-1 band — asserted by `check_radial_map()` and enforced
  by the test suite, because neither violation had any symptom other than the
  eigenvalue.
- **IAEA-3D rod orientation.** The 80 cm in the benchmark specification is the
  height of the rod tips above the bottom of the active core, not an insertion
  depth measured down from the top. The deck had it inverted, putting rods in
  the upper 80 cm instead of the upper 260 cm, which left `k_eff` 1700 pcm
  high with no other visible symptom: the axial profile was still a plausible
  cosine and the power distribution still physical. A test now asserts the
  orientation directly against the node compositions.
- **IAEA-3D axial mesh.** The rod tip is now placed exactly on a plane
  boundary for any requested node height. Smearing it across a node was worth
  tens of pcm and showed up as an axial mesh that would not converge
  monotonically.

### Changed

- Both IAEA decks now assert against their published eigenvalues rather than
  against recorded baselines. All three benchmarks meet the specification's
  100 pcm acceptance criterion for static problems.
- The V-3 verification adds a direct SANM-against-NEM comparison. The two
  kernels share only the transverse leakage fit and agree to 0.08 pcm on the
  IAEA map, which makes it the sharpest single check in the suite.

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
  map, SANM at one node per assembly lands 2.6 pcm from the mesh-converged
  eigenvalue, SANM and NEM agree with each other to 0.08 pcm, and all three
  kernels agree to within 25 pcm under refinement.
- **V-4**: adjoint and forward eigenvalues agree to under 1 pcm while the flux
  shapes differ.
- Infinite-medium `k_∞` reproduced exactly, and leakage-free fixed-source
  multiplication reproduced exactly.

### Known limitations

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
