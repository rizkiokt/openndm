# Changelog

All notable changes to OpenNDM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html) (NFR-QA-5).

## [Unreleased]

### Added

- **Radial fuel pin conduction and the Doppler temperature**
  (`openndm.PinConduction`, FR-TH-2, FR-TH-4). A radial mesh across the pellet,
  then the pellet-to-clad gap, the cladding and the film in series. Attach one
  to a `ChannelModel` with `conduction=` and it reports `fuel_temperature`, the
  volume-average pellet temperature, and `doppler_temperature` alongside the
  coolant.

  The conduction equation is integrated exactly across each ring, so a constant
  conductivity reproduces the analytic parabola `T(r) = T_s + q'''(R^2 -
  r^2)/(4k)` at every ring boundary, to 4e-16 relative at ring counts from 1 to
  200. The volume average is integrated on `r^2`, on which that profile is
  linear, so it is exact as well and lands exactly midway between centre and
  surface. Gap, cladding and film each carry their closed-form drop to 1e-13.

  `doppler_weight` is the weight on the pellet surface, defaulting to the 0.7
  surface / 0.3 centre convention FR-TH-4 names. At 0.5 it reproduces the
  volume average exactly.

  **No conductivity correlation is written here.** Fuel and cladding
  conductivities, the gap conductance and the film coefficient are all
  injected, as a constant or a callable, the way `WaterProperties` backends
  are. The published correlations are temperature-dependent and a correlation
  written from memory is a failure this project has already paid for once. A
  callable is evaluated at each ring's outer boundary, which converges first
  order in the ring count -- 3.02 K, 1.49 K and 0.74 K at 25, 50 and 100 rings
  on a test correlation -- where a constant is exact at any count.

  This also closes the other half of the direct-heating check from #85: raising
  the fraction deposited straight in the coolant lowers the fuel temperature
  while the outlet temperature does not move.

- **Single-phase closed-channel coolant model** (`openndm.ChannelModel`,
  FR-TH-1, FR-TH-6). One channel per radial column of the core map, carrying a
  fixed mass flow at constant pressure, so the only conservation law left is
  energy. Enthalpy is integrated up the channel and inverted through the
  `WaterProperties` backend for temperature and density, reported per node as
  `moderator_temperature` and `moderator_density`. It is a `ThermalSolver`, so
  it drives `PicardCoupling` exactly as an external solver would.

  Against `ConstantWater` the channel has a closed-form answer and matches it
  to 1e-12 relative, for uniform power and for a cosine shape. Against
  `IF97Water` the enthalpy rise carries the power put in to 1e-9 relative, and
  the outlet temperature is identical at 5, 10 and 40 axial nodes.

  The fraction of heat deposited directly in the coolant does **not** change
  the outlet temperature: in steady state every watt reaches the coolant
  whichever route it takes. It splits the power between coolant and pin, and so
  sets `linear_heat_rate`, which is what a conduction model consumes.

  `PinGeometry` holds the fuel radius, gap, cladding, pitch and rod counts and
  derives the flow area, the heated and wetted perimeters and the hydraulic
  diameter. `absolute_power` turns `Result.power`, which is normalised to a
  mean of one, into watts per node.

- **The performance acceptance cases, run for the first time**
  (`benchmarks/performance/`, NFR-PERF-1 to NFR-PERF-7). The specification has
  stated numeric targets since the beginning and none of them had ever been
  measured. The deck names the hardware, governor, build type and compiler in
  its output, because a target without a machine is not a claim.

  On an AMD Ryzen 9 6900HX, 8 physical cores, Release with OpenMP: NFR-PERF-1
  181 ms against 1 s, NFR-PERF-2 1.9 s against 10 s, NFR-PERF-5 70 MB against
  500 MB, NFR-PERF-6 104 ms against 0.3 s. All four pass with at least a factor
  of five in hand.

  **Two fail, and both were previously carried as unmeasured `partial`.**

  NFR-PERF-7 asks for 60% parallel efficiency on eight threads and gets **32%**
  -- 1.79x on two threads, 2.36x on four, 2.55x on eight. Amdahl implies a 31%
  serial fraction, which is the price of the serial ILU0 triangular solves that
  FR-SOL-7 has been calling `partial` without a number. Measured on 16200 nodes
  at eight groups, and 25% at 28800 nodes, so it is not a small-problem
  artefact. The eigenvalue is bit-identical on every thread count, so FR-OPT-4
  holds.

  FR-OPT-3 asks for a 2x warm-start speedup on a shuffled loading pattern and
  gets **0.99x**. Warm start itself works, and dramatically -- re-solving an
  unchanged model takes 2 outers instead of 24 -- but every way of changing the
  model requires `Model.refresh()`, and `refresh()` clears the flux, so there
  is nothing left to start from. Skipping the refresh is 2x faster and 366 pcm
  wrong.

  NFR-PERF-3 and NFR-PERF-4 cannot be run: one needs the coupled steady state,
  the other names adaptive time stepping, and neither exists.

  Not wired into CI. Timing on a shared runner is noisy enough to turn a
  performance gate into a disabled performance gate; report first.

### Fixed

- **A boundary Dhat update outside the trusted band is now discarded rather
  than clamped**, which is what actually fixes the multi-group convergence the
  under-relaxation of #79 only partly addressed. Damping removed the two-cycle
  at one node per assembly; on a 2x2 radially refined eight-group core it did
  not, and no damping factor above 0.2 converged at all.

  The clamp was the cause. A one-node correction that lands outside
  `dhat_limit` has been asked for a current the coarse mesh cannot express as
  Dhat times a node flux, so saturating it at the limit replaces an
  untrustworthy number with a large wrong one. Discarding it keeps the value
  that face already had, which on the first update is the finite difference
  coupling -- the right thing to fall back to.

  With it the refined eight-group core converges in 25 outers undamped and 16
  damped, against not at all before. Interior faces still clamp, as PARCS and
  KOMODO do: their Dhat divides by the sum of two node fluxes and does not
  reach the band in the first place.

  Every eigenvalue, error and convergence order reported for #77 is unchanged.

- **The boundary Dhat update no longer oscillates on a multi-group deck.**
  The one-node boundary problem added in #77 writes a face current as `Dhat`
  times the node-average flux. In an intermediate group of a long down-scatter
  chain, at a zero flux face, the flux there is small next to the within-node
  source driving it, so that ratio is badly conditioned: the undamped update
  overshot, saturated against `dhat_limit` and settled into a two-cycle,
  alternating by 1e-4 in k forever instead of converging.

  `Settings.boundary_relaxation` damps the step and defaults to 0.5. On an
  eight-group deck both nodal kernels go from not converging in 3000 outers to
  converging in 18 and 20, which is what they took before #77. Interior faces
  are not damped: their `Dhat` divides by the sum of two node fluxes and is
  tied to a neighbour by continuity, so it does not have the same
  conditioning.

  Every eigenvalue, error and convergence order reported for #77 is unchanged,
  because damping alters the path to the fixed point and not the fixed point.

  This went unnoticed because **nothing in the test suite had more than two
  groups**. `conftest.chain_library` now builds an eight-group deck by
  resolving the IAEA fast group into a chain, and the suite solves it with all
  three kernels.

### Changed

- **The nonlinear nodal correction now covers boundary faces** (FR-SOL-4), and
  the transverse leakage of a boundary node is fitted linearly rather than by
  repeating its own average. Both nodal kernels gain roughly two orders of
  accuracy on every problem with an analytic answer.

  `Kernel` gains an optional `solve_boundary`, a one-node problem in which the
  node-average constraint and the boundary condition determine the within-node
  shape. SANM has one free coefficient and NEM two, the second closed by the
  coarse-mesh current at the node's interior face. Both are exactly determined,
  with none of the two-node problem's continuity rows.

  The leakage fit is the other half and neither works without it. A boundary
  node has only two of the three averages the quadratic needs; repeating its
  own asserts a mirror symmetry that only a reflective face actually has. It is
  now a linear fit through the two real averages on any other face, which is
  what KOMODO does. The one-node problem *on its own* made every
  three-dimensional case worse, because it made the boundary current depend on
  a leakage shape that had been wrong all along without anything reading it.

  | | before | after |
  |---|---|---|
  | 1D slab, SANM | 231 to 4.0 pcm | exact |
  | Reflected slab, SANM | 1.8e-5 | round-off on any mesh |
  | Bare cuboid, 32 per side | 3.45 pcm | 0.014 pcm |
  | Bare cuboid observed order | 2.0 | 4.2 |
  | Manufactured solution, 32 per side | 1.13e-4 | 4.2e-6 |
  | IAEA-2D, NEM, 2 nodes per assembly | -20.7 pcm | +1.4 pcm |
  | IAEA-2D, SANM, 1 node per assembly | +2.5 pcm | +27.8 pcm |

  That last row is the one regression and it is a lost cancellation, not lost
  accuracy: SANM's boundary error used to run against its coarse-mesh error on
  that one problem. The converged eigenvalue is unchanged at 1.029527, and both
  kernels still reach it.

  A node that spans the core along an axis keeps the finite difference coupling
  on both of its faces there: with no interior surface on that axis there is
  nothing to carry information into the one-node problem.

### Added

- **Water properties from IAPWS-IF97, behind a pluggable backend** (FR-TH-3,
  FR-TH-8), in `openndm.water`. Three backends, all satisfying the
  `WaterProperties` protocol:

  `IF97Water` implements region 1 (compressed liquid) and region 4 (the
  saturation line) from the coefficient tables of IAPWS R7-97(2012). Region 2
  (vapour) is not implemented, so the steam side of a BWR is not covered yet.

  `ConstantWater` has fixed density and specific heat. No physics, which is
  the point: a channel model run against it has a closed-form answer, so the
  channel model can be verified before the properties are trusted.

  `external_backend()` adapts the `iapws` package or CoolProp when one is
  installed, so property consistency with an external thermal-hydraulics code
  can be enforced. Neither is a dependency and neither is imported until it is
  asked for.

  **The coefficients are transcribed from the release, and the release's own
  program-verification values are asserted in the test suite** -- Tables 5, 35
  and 36, to the nine significant figures they are printed to. That is what
  makes a mistranscribed digit visible rather than plausible. As a second,
  independent check the built-in formulation agrees with the `iapws` package
  to 4e-16 over the PWR range.

  `temperature(pressure, enthalpy)` inverts the basic equation by Newton
  iteration rather than using the release's backward equation. The release
  permits those two to disagree by up to 25 mK; an inverse that is exact
  against the equation it inverts has no such gap and no second coefficient
  table to keep in step. It lands 6.5 to 16.8 mK from the backward equation's
  published values, inside that allowance.

  A state outside region 1, or a saturation state above the critical point, is
  an `InputError` rather than a quietly wrong number.

### Added

- **VTK export for 3D visualisation** (FR-OUT-6). `openndm.write_vtk` writes
  the serial XML unstructured grid ParaView reads, one hexahedral cell per
  active node on the real Cartesian mesh — non-uniform widths and whatever
  `subdivide` produced, not the lattice. Cell data is `power`, `flux_g1` to
  `flux_gG` and `composition`, with `k_eff` as field data.

  **An out-of-core position gets no cell**, rather than a cell carrying zero.
  In a plot those two are indistinguishable, which is exactly how a wrong core
  map goes unnoticed. A reflector node is not out of core and is written,
  powerless or not.

  No new dependency. The format is plain XML and is written directly, so the
  package still imports without VTK, the way FR-OMC-14 requires of OpenMC.

- `Geometry.dx` and `Geometry.dy`, alongside the existing `Geometry.dz`. All
  three are recorded by `from_lattice` and recovered from the node graph
  otherwise, so a caller needing the mesh gets the post-subdivision widths
  rather than reconstructing them from its own input.

- **Thermal-hydraulic coupling interface, Picard driver and field mapping**
  (FR-TH-6, FR-TH-7), in `openndm.thermal`. No physics: the shape of the
  coupling only, so that a v1.1 external solver attaches without touching the
  neutronics.

  `ThermalSolver` is a runtime-checkable protocol over `set_heat_source`,
  `solve`, `get_temperatures` and `get_densities`. Temperatures and densities
  are keyed by name rather than fixed, because the names are the axis names
  of a branch library and so feed `XSLibrary.interpolate` unchanged.

  `PicardCoupling` owns the loop, the under-relaxation and the convergence
  test, and reaches the thermal model only through the protocol. The
  neutronics step is a callable the caller supplies, the way `search_boron`
  takes the boron model. It raises `ConvergenceError` rather than returning an
  unconverged state.

  `AxialMapping` maps volume-conservatively in both directions. `distribute`
  keeps the total of an extensive field, `average` keeps the volume-weighted
  mean of an intensive one; they are separate methods rather than a flag
  because using one where the other belongs conserves nothing and looks
  plausible. Leading axes are free, so a whole core of channels maps in one
  call.

  The interface comes before the channel model deliberately: built the other
  way round, the channel model becomes the shape of the coupling.

- **Assembly rotation for discontinuity factors** (FR-OPT-7).
  `Geometry.from_lattice(rotation=...)` and `Geometry.set_rotation` turn a
  core position 0 to 3 quarter turns counter-clockwise, following KOMODO's
  `%ADF` `ROT` convention so a deck translates without re-deriving anything.

  Rotation belongs to the position rather than to the composition, because one
  assembly type is loaded at many positions in different orientations --
  KOMODO's own example rotates a checkerboard of positions through all four
  orientations of a single diagonally symmetric assembly. Every ADF lookup
  follows it, in the coupling coefficients and in the two-node problem alike,
  so all three kernels see it.

  `rotate_adf` and `rotated_face` expose the permutation for callers who would
  rather build a pre-rotated composition. `rotated_face` is the binding of the
  function the solver itself uses, not a second copy that could drift from it.

  An unrotated core is bit-identical to before.

### Fixed

- **A transient now sees a change in the diffusion coefficient.** A step
  refreshed the cached cross sections but not the coupling coefficients
  derived from them, and `Dtilde` is derived from `D`. Nothing else in the
  operator reads `D`, so changing it mid-transient had *no effect at all*:
  halving it across a whole core reproduced the unchanged power history bit
  for bit. `Dhat` is preserved across the rebuild, so the nodal correction
  stays frozen at its static value as documented.
- **`Model.refresh()` during a transient is refused** rather than emptying the
  flux the transient is advancing. The next step then read past the end of the
  emptied vector and returned plausible-looking power. The user guide
  documented that exact sequence; it does not any more. A step re-reads the
  cross sections, the compositions and the coupling by itself, so nothing
  needs refreshing between steps. A static `solve()` ends the transient and
  makes `refresh()` available again.

### Added

- **The BIBLIS-2D benchmark deck** (`benchmarks/biblis2d/`). A 9x9 quarter
  core over eight compositions, published `k_eff = 1.02511`. SANM lands 2.1
  pcm out at one node per assembly; at four nodes per assembly all three
  kernels agree with the published value to 0.1 pcm and FDM converges at
  second order.

  This deck was attempted once before and deliberately not shipped, because
  the map written from memory used five of eight compositions and could not
  be verified. It is now **generated by parsing the KOMODO sample deck**
  rather than transcribed -- `smpl/static/BIBLIS`, commit `b70d4ee`, MIT,
  the underlying data being the published BIBLIS PWR benchmark.

  The earlier attempt was wrong twice: the map, and the reference itself,
  which was taken as 1.02513 against the 1.02511 the source states. Tuning
  the one to reproduce the other would have fitted a wrong core to a wrong
  number and looked exactly like a pass. Nothing was tuned here.
- **LMW transient benchmark** (`benchmarks/lmw/`), the first deck to run
  control rod banks, the cusping correction, delayed precursors and
  theta-weighted integration together. Two banks move against each other over
  60 seconds at 3 steps per second, against a 5 cm axial mesh, so a rod tip
  sits inside a node four steps out of five.

  `data.py` is generated by parsing the KOMODO sample deck, not hand-written;
  the provenance is recorded in the module header. **The specification states
  the scenario and not the answer** — the published LMW reference is a power
  curve that is not in the file — so the power history is reported rather than
  scored, and the deck's tests assert the conventions it depends on and the
  shape it produces.

  The internal verification found that the mesh is converged and the time step
  is not: eight times the nodes moves the peak power by 0.06%, while the
  deck's own 0.25 s step sits about 3.4% below the extrapolated peak and
  converges first order. Crank-Nicolson is second order here, measured at
  2.05, but only for steps resolving the prompt time constant of about a
  millisecond; an operational transient runs two hundred times coarser, which
  is the stiff regime where the theta method loses an order. This is the first
  measurement of what the unimplemented exponential flux transformation costs.
- **`ControlRods.reweight`**, which re-weights a partially rodded node's
  mixture against a flux already in hand. `converge_cusping` reaches the same
  mixture by iterating static solves, which a transient cannot do: the static
  solve would discard the time-dependent flux and the precursors with it.

- **Square-root temperature feedback** (`DopplerFeedback`), completing
  FR-XS-7. `Sigma(T) = Sigma_0 [1 + gamma (sqrt(T) - sqrt(T_0))]`, applied to
  a chosen field, group set and composition set. Doppler broadening widens a
  capture resonance as the square root of temperature, so a per-Kelvin
  coefficient is a linearisation of this that parts company with it over the
  hundreds of Kelvin a transient covers. The LRA BWR specification defines
  its feedback in exactly this form.

  Base cross sections are snapshotted, so temperatures are absolute rather
  than incremental and the reference temperature restores the library
  bit-for-bit. A temperature that would drive a cross section negative raises
  rather than writing a physically impossible library. Verified against the
  analytic bare cuboid at temperature.

  A branch library generated from OpenMC carries the real temperature
  dependence and is strictly better where it exists; this is for the case
  where there is none.
- **Time-dependent solves** (`Model.start_transient`, `Transient.step`),
  completing FR-KIN-1 and most of FR-KIN-2. Theta-weighted integration with
  `0 < theta <= 1`, defaulting to fully implicit, on KOMODO's `%THET`
  convention.

  The analytic precursor solution is linear in the new fission source, so its
  implicit part folds into an effective fission spectrum and a step is a
  single fixed-source solve rather than an iteration between flux and
  precursors.

  The static eigenvalue is generally not one, so the fission source is
  divided by it for the whole transient. Without that criticality
  normalisation a core at k = 1.03 ramps from the first step and the ramp
  looks like physics.

  Verified against exact point kinetics in a leakage-free box, which needs no
  other code and no transcribed deck: the null transient is flat to 1 part in
  1e10, the prompt jump matches `beta/(beta - rho)` to 0.5%, the asymptotic
  period matches the inhour root to 0.2%, and the observed order in time is
  1.00 at theta=1 and 2.00 at theta=0.5.

  The order test found a defect worth recording. The explicit half of the
  scheme was evaluated against the *previous* step's cross sections, which
  puts an O(1) error into the one step where a perturbation lands -- O(dt)
  overall. It dragged Crank-Nicolson down to first order and left theta=1
  untouched, because theta=1 never reads that term. It also has to be
  measured inside the prompt layer: later the amplitude is set by the
  precursor equation, which is integrated exactly, and that masks the flux
  scheme's order entirely.

  Not implemented: the exponential transformation, adaptive time stepping,
  decay heat, and feedback (FR-MODE-7, which needs FR-TH). The nonlinear
  nodal coupling coefficients are held at their static values inside a step.
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
