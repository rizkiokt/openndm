# Changelog

All notable changes to OpenNDM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html) (NFR-QA-5).

## [Unreleased]

### Added

- **Delayed neutron precursor integration** (C++ `PrecursorState`), the first
  part of FR-KIN-1. The precursor equation is linear in the concentration
  once the fission source is known, so it is integrated in closed form across
  a step with the fission source taken linear, rather than with whatever
  scheme the flux uses. Two properties follow and are asserted: a genuinely
  linear fission source is reproduced exactly for any step size, and
  equilibrium is a fixed point for any step size -- a scheme that misses the
  second starts every transient with a jump that looks like physics.

  The decay integrals are evaluated by series below `lambda*dt = 1e-4`. Both
  closed forms are catastrophic cancellations there, and the second loses
  every significant digit long before the argument underflows, which would
  quietly corrupt the delayed source for a long-lived precursor group on a
  short step.

  Nothing is exposed to Python yet: there is no flux time integration for it
  to drive, and a public API for a component that cannot be used alone is one
  that has to change when it can. The equations are in `docs/theory.md` §9.
- **Rod worth curves** (`ControlRods.worth_curve`, `RodWorth`), completing
  FR-MODE-5. Integral and differential worth over a sequence of static
  solves, for one bank or a prepared multi-bank sequence so that overlap can
  be modelled. Worth is a reactivity difference, `1/k(p) - 1/k_ref`, signed
  so an inserted bank has positive worth; it agrees with worth computed from
  two direct solves to 0.01 pcm. Points warm start from each other, which is
  the case warm starting exists for, and the sweep restores the bank
  position it started from.

  This waited for cusping deliberately. Sampling worth between plane
  boundaries without it returned a staircase, so a differential curve would
  have been a sequence of spikes and flats -- an artefact of the axial mesh
  rather than a property of the reactor.
- **The control rod cusping correction** (`ControlRodBank(cusp=...)`,
  `ControlRods.converge_cusping`). A rod tip between plane boundaries leaves
  one node partly rodded; rounding it to whichever state covers its centre
  made `k_eff` a staircase in rod position, with a largest single step of
  1561 pcm on a ten-plane test core. The partial node now takes a homogenised
  mixture, written into a spare library composition each time the bank moves.

  `insert` weights that mixture by volume, which is the flat-flux limit and
  biased: the flux is depressed on the rodded side, so volume weighting
  over-counts the rodded absorption and puts `k_eff` low.
  `converge_cusping` iterates solve and re-weight to get flux-volume weights
  instead. Against a 0.5 cm reference mesh on which the tip always falls on a
  boundary, the largest error goes 784 -> 259 -> 55 pcm and the mean bias
  +165 -> -103 -> -13 pcm across no cusping, volume weighting and flux
  weighting. The remaining 55 pcm is the size of the coarse mesh's own
  discretisation error: at tip positions that *do* land on a boundary, where
  cusping does nothing, this core is already 11-42 pcm from the reference.

  KOMODO documents no cusping correction, so this has no counterpart in the
  reference implementation.

### Fixed

- **Writing a composition after `finalize()` was silently ignored by the
  solver.** `removal` is derived from absorption and the scattering matrix at
  finalize time and the kernels read it rather than recomputing it, so a
  mutated library solved with the *old* absorption while `composition()`
  reported the new one. The C++ contract was already right -- taking mutable
  access marks the library unfinalized -- and `Model.__init__` checked it, but
  `Model.refresh()` did not, and `refresh()` is exactly what the documentation
  tells you to call after changing cross sections. On an 8x8x8 test case the
  discarded change was worth 39000 pcm. `refresh()`, `solve()` and
  `solve_fixed_source()` now refuse a library that has been modified since it
  was finalized, and say to call `finalize()`.

### Added

- **Control rod banks** (`ControlRodBank`, `ControlRods`), addressing rods by
  bank and position rather than by editing compositions plane by plane
  (FR-MODE-5). A bank carries a radial column mask, a substitution from each
  unrodded composition to its rodded counterpart, and a position in steps on
  the convention KOMODO's `%CROD` card uses -- `zero_position` and
  `step_size`, with step 0 fully inserted -- so an existing deck translates
  without arithmetic. The unrodded core is snapshotted at construction, which
  makes positions absolute: moving a bank twice matches setting its final
  position once, and withdrawing restores exactly what was there before.
  Verified against `benchmarks/iaea3d`, which places the same rods by hand:
  with the tip on a plane boundary the bank builds that core node for node.

  A composition a bank reaches but cannot substitute raises rather than
  passing the node through unchanged, because an unsubstituted node is
  indistinguishable from a correctly withdrawn one.

  There is no cusping correction yet. A node is rodded when its centre lies
  above the tip, which is exact on a plane boundary and rounds to the nearest
  plane in between, so `k_eff` is a staircase in rod position -- largest
  single step about 1500 pcm on a ten-plane test core. Put the tip on a plane
  boundary where the answer matters, as `benchmarks/iaea3d` does.
- `Geometry.dz`, the axial node widths after subdivision, which a bank needs
  to place a tip at a continuous position.

## [0.2.0] - 2026-09-19

### Added

- **A roadmap** (`docs/roadmap.md`), recording what is planned and why in that
  order: control rod banks, then kinetics, then thermal hydraulics, with the
  independent items listed separately. `status.md` records what exists;
  nothing recorded what comes next.
- **The documentation builds as a Sphinx site** with an autodoc API reference
  (#3). CI builds it with warnings as errors against the installed extension,
  so an API page that renders empty fails the build rather than passing
  quietly. This completes NFR-QA-6.
- **A release pipeline** (#7): tag-triggered, using trusted publishing, and
  refusing to build unless the tag, `pyproject.toml` and `CMakeLists.txt`
  agree on the version. A conda-forge recipe ships alongside it. Nothing is
  published yet, because no tag has been cut.
- **A user guide** (`docs/user-guide.md`), covering the whole API from
  installation to troubleshooting, with every code snippet executed against
  the build.
- **Four worked notebooks** in `examples/`, each executed end to end with its
  outputs committed. `02_openmc_to_openndm.ipynb` is the workflow the package
  exists for: an OpenMC lattice run becomes a nodal core calculation with no
  format conversion in between, reproducing OpenMC's k_inf to 12 pcm.
- **C-1 OpenMC coupling validation** (`tests/validation/c1_homogeneous.py`),
  the first of the specification's §6.3 cases. A uniform infinite medium runs
  in OpenMC with continuous-energy physics and the multi-group cross sections
  from that same run are solved by OpenNDM, isolating the translation path
  from every other source of error. It agrees to 12 pcm, 1.2 sigma.
- Opt-in OpenMC integration tests (`tests/python/test_openmc_integration.py`)
  covering C-1 and C-2, skipped unless OpenMC, its executable and a nuclear
  data library are all available.
- `scattering_multiplicity` option on `from_mgxs_library`.
- **C-2 assembly discontinuity factor validation**
  (`tests/validation/c2_assembly_adf.py`). A single reflected assembly is an
  infinite lattice, so a homogeneous one must return factors of exactly 1.0
  and a pin lattice must return a thermal factor above 1.0, its surface being
  water. Measured 1.0 to 1.2 sigma and 1.045 thermal against 0.993 fast.
- **Discontinuity factor verification against the equivalence theorem**
  (`tests/python/test_discontinuity_factors.py`). With flux-volume
  homogenised cross sections and the factors implied by a reference solution,
  the coarse solve reproduces the reference eigenvalue and node-average fluxes
  to 0.00000 pcm, for any homogenised diffusion coefficient. Dropping the
  factors costs 3000 pcm on the same problem, which is the control. Until now
  this path was tested only for defaulting to 1.0 and surviving a round trip,
  neither of which says anything about whether the coupling coefficient built
  from a factor is right.
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

- **Subdividing an axis that had explicit node widths enlarged the core.**
  `Geometry.from_lattice` gives `dx`/`dy`/`dz` per lattice cell and
  `subdivide` splits each cell, but the explicit-width path repeated the
  parent width once per child instead of sharing it, scaling that axis by the
  subdivision factor. A 60 cm axial stack became a 120 cm one and `k_eff`
  moved 2751 pcm, with no error raised and nothing else out of place: the
  node count was right, the composition map was right, and the solution was
  a perfectly good answer to a core twice the intended height. The
  pitch-only path was always correct, which is why every shipped deck
  escaped -- `iaea3d` supplies an explicit `dz` but only ever subdivides
  radially, and `iaea2d` is uniform.

- **The V-3 coarse-mesh test measured against a reference that was not
  converged.** It compared SANM and NEM at one node per assembly to an FDM
  solve at eight nodes per assembly and called that the fine-mesh converged
  eigenvalue. FDM on the IAEA-2D map is still 19.6 pcm from its own limit at
  that mesh, approaching from below at second order, so the test charged
  FDM's discretisation error to whichever kernel was under test: SANM
  measures 22.2 pcm against that reference and 3.7 pcm against the limit, and
  the 25 pcm tolerance was 89% consumed by an artifact unrelated to SANM. The
  reference is now a Richardson extrapolation over two refinements, which
  borrows nothing from either nodal kernel, and SANM's tolerance drops to 10
  pcm. The extrapolated limit lands 1.1 pcm from the eigenvalue SANM and NEM
  converge to, so three kernels sharing no spatial machinery now agree on one
  answer with one of them arriving at a verified order; that is asserted in
  its own test. The same wrong number had been copied into the README and
  `docs/status.md`, where it made a boundary-face nodal correction look worth
  scheduling: the real headroom on a core problem is 2.6 pcm, not 18.

- **The critical boron search reported the wrong cause on failure.** An
  unreachable target and a callback that does nothing both surfaced as "k_eff
  is insensitive to boron", which sends the reader looking in the wrong place.
  The two are now distinguished, and the unreachable case names the range of
  k_eff that was actually attainable. Found while writing the notebooks.
- **`compute_adf` returned its groups in increasing-energy order**, while
  `from_mgxs_library` and the solver both put group 1 at the highest energy,
  so every discontinuity factor was applied to the wrong group. An
  `openmc.EnergyFilter` orders its bins by increasing energy and
  `MGXS.get_xs` by decreasing: two OpenMC APIs, two conventions, and only a
  physical argument distinguishes them. Found by C-2, and silent by nature —
  the values stayed plausible, symmetric across opposite faces and sensibly
  converged, with only their sense inverted.
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

- CI linters are pinned, the wheel job builds against `manylinux_2_28` because
  h5py ships nothing older, and the OpenMC coupling job runs on manual
  dispatch with an explicit nuclear data URL rather than a hard-coded one. See
  `tests/validation/README.md` for why that job cannot run on every push.
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

[Unreleased]: https://github.com/rizkiokt/openndm/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rizkiokt/openndm/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rizkiokt/openndm/releases/tag/v0.1.0
