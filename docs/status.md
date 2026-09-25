# Implementation status

Every requirement ID from [`requirements.md`](requirements.md), mapped to what
actually exists in this build. Read this before assuming a feature is present.

Legend: **done** — implemented and covered by a test. **partial** —
implemented with a stated limitation. **not started** — the API may name it,
but nothing behind it works.

Version 0.1.0. Roughly: M0, M1, M2 and M3 of the specification's phase plan.

---

## 4.1 Geometry (FR-GEO)

| ID | State | Notes |
|---|---|---|
| FR-GEO-1 | done | 3D Cartesian grid, per-direction non-uniform spacing. |
| FR-GEO-2 | done | `INACTIVE` marks out-of-core positions. |
| FR-GEO-3 | partial | Quarter and half cores work through reflective symmetry boundaries. Explicit expansion of a symmetric map into a full core, and rotational symmetry, are not implemented. |
| FR-GEO-4 | done | `subdivide=` splits a lattice cell; discontinuity factors follow the composition. |
| FR-GEO-5 | done | Zero flux, vacuum, reflective and per-group albedo, per face. `outside=` sets the condition on faces looking at an inactive position. |
| FR-GEO-6 | done | Kernels see only the node/surface graph. No solver code indexes `(i,j,k)`. |
| FR-GEO-7 | not started | Construction from an `openmc.RectLattice`. |

## 4.2 Cross sections (FR-XS)

| ID | State | Notes |
|---|---|---|
| FR-XS-1 | done | Any `G ≥ 1`, full `G×G` scattering including upscattering. |
| FR-XS-2 | done | D or transport, absorption, νΣf, κΣf, χ, scattering matrix, 1/v. |
| FR-XS-3 | done | Up to 8 precursor groups with β, λ and delayed χ. Stored and round-tripped; nothing consumes them yet, because FR-KIN is not implemented. |
| FR-XS-4 | done | Per node face per group, defaulting to 1.0. Verified against the equivalence theorem: with flux-volume homogenised cross sections and the factors implied by a reference solution, the coarse solve reproduces the reference eigenvalue and node-average fluxes exactly, for any homogenised diffusion coefficient. On the test problem the factors are worth 3000 pcm. |
| FR-XS-5 | partial | Arbitrary branch axes with arbitrary names, so fuel temperature, moderator density and temperature, boron and rod state all work. Burnup as a *history* axis, distinct from an instantaneous axis, is not modelled. |
| FR-XS-6 | done | Multilinear interpolation; `clamp`, `linear` and `error` extrapolation. |
| FR-XS-7 | done | `DopplerFeedback` applies Sigma(T) = Sigma_0 [1 + gamma(sqrt(T) - sqrt(T_0))] to a chosen field, groups and compositions. Base data is snapshotted, so temperatures are absolute and the reference temperature restores the library exactly. Verified against the analytic bare cuboid at temperature. Cross sections live per composition, so a temperature distribution needs one composition per region that can differ. |
| FR-XS-8 | done | Fails on unphysical data, warns on negative scattering. |
| FR-XS-9 | done | Standard deviations stored and carried through ingestion. |

## 4.3 OpenMC interface (FR-OMC)

| ID | State | Notes |
|---|---|---|
| FR-OMC-1 | done | `from_mgxs_library`, validated end to end against a real OpenMC run (C-1, `tests/validation/`). Reproduces OpenMC's k_inf to 12 pcm, 1.2 sigma. Handles (n,xn) scattering multiplicity, which is worth 230 pcm and which the stand-in tests could not have caught. |
| FR-OMC-2 | partial | Any domain type is accepted and each domain becomes a composition. Mapping mesh elements onto nodes is left to the caller ordering the domains to match the core map. |
| FR-OMC-3 | partial | `from_mgxs_file` reads `mgxs.h5`. Untested against a real file. |
| FR-OMC-4 | done | `from_statepoint`. |
| FR-OMC-5 | done | Both D sources accepted, preference configurable, warning above a configurable disagreement tolerance. C-1 confirms D does not affect an infinite medium, as it must not. |
| FR-OMC-6 | done | Standard deviations propagated where OpenMC provides them. |
| FR-OMC-7 | done | `add_adf_tallies` and `compute_adf`, using a thin track-length slab rather than a surface tally, validated against a real OpenMC run (C-2). A homogeneous assembly returns factors of 1.0 to 1.2 sigma; a pin lattice returns 1.045 thermal and 0.993 fast, the right sense and a typical magnitude. |
| FR-OMC-8 | partial | B1 and P1 buckling search, verified in all three criticality regimes. It reports the buckling and the corrected D; it does **not** re-condense the group constants, which needs the fine-group data inside the lattice calculation. |
| FR-OMC-9 | partial | IFP results are read when present, with the k-ratio fallback. Untested against a real statepoint. |
| FR-OMC-10 | partial | `compute_form_functions` extracts and normalises. Nothing consumes them, because FR-OUT-4 is not implemented. |
| FR-OMC-11 | done | `BranchDriver` with serial and multiprocessing execution, checkpointing, resume, and Slurm/PBS job arrays. The assembly path is tested; the execution path needs OpenMC. |
| FR-OMC-12 | not started | The small full-core self-consistency test (C-4). C-1 is passing; C-2, C-3, C-5 and C-6 are not written. See `tests/validation/README.md`. |
| FR-OMC-13 | not started | Exposing the CMFD kernel to `openmc.lib`. |
| FR-OMC-14 | done | Enforced by a dedicated CI job that installs without OpenMC and solves. |

## 4.4 Solver kernels (FR-SOL)

| ID | State | Notes |
|---|---|---|
| FR-SOL-1 | done | Finite difference, also the CMFD base. |
| FR-SOL-2 | done | NEM, quartic expansion, quadratic transverse leakage. |
| FR-SOL-3 | done | SANM, the default. |
| FR-SOL-4 | done | Nonlinear two-node iteration on interior surfaces and a one-node problem on boundary faces, so every surface carries a corrected coupling. The boundary update is under-relaxed by `Settings.boundary_relaxation`, without which a group whose boundary flux is small next to its within-node source drives the update into a two-cycle. A node spanning the core along an axis keeps the finite difference coupling on both of its faces there, having no interior surface to take information from. |
| FR-SOL-5 | done | Power iteration with a capped Wielandt shift; BiCGSTAB with ILU0. |
| FR-SOL-6 | done | Criteria on k, node-wise fission source and iteration count; `ConvergenceError` carries the count and the residual. |
| FR-SOL-7 | partial | OpenMP over nodes and surfaces in the two-node update, the matrix-vector product and the source assembly. The triangular solves in ILU0 are serial, and that is now measured: 31% of the runtime is effectively serial by Amdahl on eight threads, so parallel efficiency is 32% where NFR-PERF-7 asks for 60%. The eigenvalue is bit-identical on every thread count. See `benchmarks/performance/`. |
| FR-SOL-8 | done | Run-time selection; one build runs all three. |

## 4.5 Calculation modes (FR-MODE)

| ID | State | Notes |
|---|---|---|
| FR-MODE-1 | done | |
| FR-MODE-2 | done | Reuses the coupling coefficients converged by a forward solve. |
| FR-MODE-3 | done | Verified against exact algebra in a leakage-free box, and against a method of manufactured solutions for the full multi-group operator. |
| FR-MODE-4 | done | Secant with a bisection fallback; the boron model is supplied by the caller. |
| FR-MODE-5 | done | `ControlRodBank` and `ControlRods` give banks a radial map, a composition substitution and a position in steps on KOMODO's `%CROD` convention. Verified against `benchmarks/iaea3d`, which places the same rods by hand, node for node. `worth_curve` returns integral and differential worth over a sequence of static solves, for one bank or a prepared multi-bank sequence, warm starting between points; it agrees with worth computed from two direct solves to 0.01 pcm. A partially rodded node takes a homogenised mixture written into a spare composition: volume-weighted from `insert`, flux-weighted through `converge_cusping`, which iterates solve and re-weight. Against a fine-mesh reference the largest error falls 784 -> 259 -> 55 pcm across no cusping, volume and flux weighting, the last being the size of the coarse mesh's own discretisation error. |
| FR-MODE-6 | done | `Model.solve_coupled`: solve, hand the power to a `ThermalSolver`, push what comes back into the cross sections, repeat until the node power stops moving. The feedback model stays in the caller's `apply_state`, the way `search_boron` keeps the boron model in the caller's hands. A library with no temperature dependence returns the uncoupled eigenvalue to 1e-12 in one iteration; on a test slab with a 5% Doppler swing the loop converges in 10 iterations and moves `k_eff` by -3590 pcm, monotonically in power. Every iteration calls `refresh()`, which clears the flux, so each solve starts cold -- the cost issue #82 records. |
| FR-MODE-7 | not started | Needs FR-KIN. |
| FR-MODE-8 | done | `Model.sweep`, reusing geometry and library, warm starting by default. |

## 4.6 Kinetics (FR-KIN)

| ID | State | Notes |
|---|---|---|
| FR-KIN-1 | done | Time-dependent multi-group diffusion with up to 8 precursor groups. Precursors are integrated in closed form over a step; the analytic solution is linear in the new fission source, so its implicit part folds into an effective fission spectrum rather than needing an iteration. |
| FR-KIN-2 | partial | Theta-weighted integration, 0 < theta <= 1, default fully implicit. Observed order 1.00 at theta=1 and 2.00 at theta=0.5, measured against exact point kinetics inside the prompt layer. Those orders are asymptotic: at the quarter-second steps an operational transient uses, two hundred times the prompt time constant, both weightings converge first order, which on `benchmarks/lmw` leaves the deck's own step about 3.4% from the extrapolated peak power. The exponential transformation, which is the standard remedy, is not implemented. The nonlinear nodal coupling coefficients are held at their static values inside a step. |
| FR-KIN-3 | not started | Adaptive time stepping. |
| FR-KIN-4 | partial | Any change the caller makes between steps is picked up: cross sections, compositions, rod bank positions, and the coupling coefficients derived from the diffusion coefficient. `Model.refresh` is refused during a transient, because the retained flux is the transient's state rather than a cache, and a step re-reads all of the above by itself. `ControlRods.reweight` re-weights a partially rodded node against a flux already in hand, which is how cusping works in a transient, where the static solve `converge_cusping` runs would discard the time-dependent flux. `benchmarks/lmw` schedules two banks against time; there is no driver in the library that does. |
| FR-KIN-5 | not started | Decay heat. |
| FR-KIN-6 | partial | Each step returns time, total and peak power and iteration counts; the flux and precursors are readable. No time-series writer. |

Verified against exact point kinetics in a leakage-free box: null transient
flat to 1 part in 1e10, prompt jump to 0.5%, asymptotic period against the
inhour equation to 0.2%, and the observed order in time. See
[`theory.md`](theory.md) §10 and §11.

**FR-MODE-7** (transient with feedback) still needs FR-TH.

## 4.7 Thermal hydraulics (FR-TH)

| ID | State | Notes |
|---|---|---|
| FR-TH-1 | done | `ChannelModel`, one closed channel per radial column of the core map, at constant mass flow and constant pressure. Enthalpy is integrated up the channel and inverted through the property backend for temperature and density. Against `ConstantWater` the answer is closed form and the tests match it to 1e-12 relative for uniform and for cosine power; against `IF97Water` the enthalpy rise carries the power put in to 1e-9, and the outlet is identical at 5, 10 and 40 axial nodes. `PinGeometry` holds the pin dimensions and derives the flow area, the heated and wetted perimeters and the hydraulic diameter. Cross-flow, a momentum equation and boiling are all absent, the last being FR-TH-5. |
| FR-TH-2 | done | `PinConduction`: a radial mesh across the pellet, then gap conductance, cladding conduction and the film in series. The conduction equation is integrated exactly across each ring, so a constant conductivity reproduces the analytic parabola at every ring boundary to 4e-16 relative for ring counts from 1 to 200, not merely at the ends. Volume-average pellet temperature is integrated on r^2, on which a constant-conductivity profile is linear, so that integral is exact too. Conductivities are injected as a constant or a callable and **no correlation is written here**; a callable is evaluated at each ring's outer boundary, which is first order in the ring count -- measured 3.02 K, 1.49 K, 0.74 K at 25, 50 and 100 rings on a test correlation. |
| FR-TH-3 | done | `IF97Water` implements IAPWS-IF97 region 1 (compressed liquid), region 2 (vapour, via the `vapour_` methods) and region 4 (the saturation line), from the coefficient tables of R7-97(2012). Verified against the release's own program-verification values, Tables 5, 15, 35 and 36 and the B23 point of Section 4, to the nine significant figures they are printed to, and cross-checked against the independent `iapws` package to 1e-9 over 60 pressures along the saturation line. `saturated_liquid_enthalpy` and its three companions give both sides of the line, which is what FR-TH-5 needs. **Region 3 is not implemented**, so the saturation line is reachable only up to 16.529 MPa, where it meets the B23 boundary; a PWR at 15.5 MPa is below that and a BWR at 7 MPa far below, but the near-critical fluid is out of reach. `ConstantWater` gives fixed properties, which is what makes a channel model analytically verifiable. |
| FR-TH-4 | done | `PinConduction(doppler_weight=...)`, the weight on the pellet surface with the rest on the centreline. The default is the 0.7 surface, 0.3 centre convention the requirement names as an example, and it is a parameter rather than a constant in the source. At 0.5 it reproduces the volume average exactly, the parabola's average sitting midway between centre and surface. |
| FR-TH-5 | done | `ChannelModel(two_phase=True)`. The enthalpy integration is unchanged and only its inversion differs: above the saturated liquid enthalpy the temperature stops at the boiling point and the surplus becomes equilibrium quality, void fraction and a mixture density. A channel that stays subcooled is **bit-identical** to the single-phase one, which is the check that keeps the two honest. At unit slip the void fraction is the homogeneous relation and the mixture density is exactly the inverse of the mass-weighted specific volume, to 1e-12. `slip_ratio` is injectable for a caller who wants a drift-flux correction; **no published slip correlation is shipped**, on the same grounds as the conductivities. A channel driven past the saturated vapour enthalpy raises rather than reporting superheated steam, which is not modelled. The backend must satisfy `SaturationProperties`, so `ConstantWater` is refused. |
| FR-TH-6 | done | `ThermalSolver` is a runtime-checkable protocol over `set_heat_source`, `solve`, `get_temperatures` and `get_densities`, and no kernel calls anything behind it: nothing in `src/` refers to temperature or density. `ChannelModel` is one implementation of it and reaches the neutronics only through `PicardCoupling`, the same way an external solver would. |
| FR-TH-7 | done | `PicardCoupling` owns the iteration, the under-relaxation and the convergence test, and reaches the thermal model only through the protocol. `AxialMapping` maps volume-conservatively in both directions: `distribute` for an extensive field, `average` for an intensive one. Transfer is numpy arrays throughout; nothing in the module touches the filesystem. |
| FR-TH-8 | done | `WaterProperties` is a runtime-checkable protocol and `external_backend()` adapts the `iapws` package or CoolProp when one is installed, neither being a dependency. The `iapws` adapter is verified against the built-in formulation; the CoolProp one is written to `PropsSI` and is untested here, since CoolProp is not installed. |

The interface and the driver were built before any physics, deliberately. Had
the channel model come first, the channel model would have become the shape of
the coupling, and FR-TH-7's v1.1 external solver would have had to be shaped
like it.

## 4.8 Optimisation and embedding (FR-OPT)

| ID | State | Notes |
|---|---|---|
| FR-OPT-1 | done | No filesystem access anywhere in build/solve/read/mutate/re-solve. |
| FR-OPT-2 | done | The GIL is released for the whole solve. |
| FR-OPT-3 | partial | Warm start works and reduces the outer count dramatically where it applies: re-solving an unchanged model takes 2 outers instead of 24. It gives **nothing** on a perturbed solve, which is what the requirement asks for, because any change to the model needs `Model.refresh()` and `refresh()` clears the flux. The two do not compose, so the measured speedup on a shuffled loading pattern is 0.99×, not ≥2×. Skipping the refresh is 2× faster and 366 pcm wrong. |
| FR-OPT-4 | done | Fixed-size chunked reductions; tested bit-identical on 1, 2, 4 and 8 threads. |
| FR-OPT-5 | done | Zero-copy numpy views with keep-alive, through pybind11 rather than xtensor. |
| FR-OPT-6 | done | Typed exceptions on every path; nothing aborts. |
| FR-OPT-7 | partial | `swap_assemblies` and `set_composition`. Assembly rotation is implemented for discontinuity factors: `from_lattice(rotation=...)` and `Geometry.set_rotation` turn a position 0 to 3 quarter turns counter-clockwise on KOMODO's `%ADF` `ROT` convention, and every ADF lookup in the coupling and in the two-node problem follows it. Nothing else about an assembly is orientation-dependent today, so nothing else turns. Positions are still addressed by index rather than by label. |

## 4.9 Output (FR-OUT)

| ID | State | Notes |
|---|---|---|
| FR-OUT-1 | done | Versioned `statepoint.h5` on OpenMC's conventions, written from Python through h5py. |
| FR-OUT-2 | done | `openndm.StatePoint`. |
| FR-OUT-3 | done | k, node power, radial and axial profiles, `F_q`, `F_ΔH`. |
| FR-OUT-4 | not started | Pin power reconstruction. |
| FR-OUT-5 | done | Radial map, axial profile and convergence history. |
| FR-OUT-6 | done | `write_vtk` writes the serial XML unstructured grid, one hexahedral cell per active node on the real mesh, with `power`, `flux_g1..G` and `composition` as cell data and `k_eff` as field data. An out-of-core position gets no cell rather than a cell carrying zero. No dependency: the format is XML and is written directly. |
| FR-OUT-7 | done | Verbosity-controlled iteration history to stdout, and the history array on every result. |

## 4.10 Input (FR-IN)

| ID | State | Notes |
|---|---|---|
| FR-IN-1 | partial | The Python API is the primary interface and validates its input. It does not generate XML, because FR-IN-2 does not exist. |
| FR-IN-2 | not started | XML input, pugixml parsing, the schema, and the standalone `openndm` executable. |
| FR-IN-3 | done | Versioned `xslib.h5`. |
| FR-IN-4 | not started | Reader for an external nodal deck format. |
| FR-IN-5 | partial | Benchmarks ship a Python script; there is no XML to ship alongside it. |

## 5 Non-functional

| ID | State | Notes |
|---|---|---|
| NFR-PERF-1 | done | 181 ms against < 1 s, single-threaded. |
| NFR-PERF-2 | done | 1.9 s against < 10 s, single-threaded, on a synthetic eight-group library. |
| NFR-PERF-3 | not started | Needs the coupled steady state, which needs FR-TH and FR-MODE-7. |
| NFR-PERF-4 | not started | Names adaptive time stepping, which is not implemented. A fixed-step rod ejection can be timed but is not the stated case. |
| NFR-PERF-5 | done | 70 MB peak resident against < 500 MB, interpreter included. |
| NFR-PERF-6 | done | 104 ms against < 0.3 s. Passes on the clock, but see FR-OPT-3: it is not actually warm started. |
| NFR-PERF-7 | **failed** | 32% parallel efficiency on eight threads against a 60% target, on 16200 nodes at eight groups. 25% at 28800 nodes, so not a small-problem artefact. The serial ILU0 triangular solves are the cause; see FR-SOL-7. |
| NFR-QA-1 | partial | Catch2 for C++ and pytest for Python. Coverage is collected in CI but the 80% line coverage gate is not enforced. |
| NFR-QA-2 | done | Every deck runs in CI. The four static decks meet the 100 pcm acceptance criterion: IAEA-2D, IAEA-3D, BIBLIS-2D and the analytic cuboid, the last against exact algebra rather than a published reference. `benchmarks/lmw` is a transient whose specification states the scenario and not the answer, so it is tested for the conventions it depends on and the shape it produces, and its power history is reported rather than scored. |
| NFR-QA-3 | done | Linux gcc and clang, macOS clang, Python 3.10 to 3.13, all green. Windows is not built and is not documented as WSL-only. The OpenMC coupling job runs on manual dispatch only, because no stable public nuclear data URL exists to hard-code; see `tests/validation/README.md`. |
| NFR-QA-4 | done | clang-format and ruff, both enforced. |
| NFR-QA-5 | done | Semantic versioning and a changelog. |
| NFR-QA-6 | done | A user guide, a theory manual, architecture notes, this status map, and four worked notebooks in `examples/`, all executed with their outputs committed. Built as a Sphinx site with an autodoc API reference; CI builds it with warnings as errors against the installed extension, so an API page that renders empty fails the build. |
| NFR-QA-7 | partial | [`theory.md`](theory.md) writes out every equation the code currently solves, with the discretisation. It covers only what is implemented. |
| V-1 | done | Analytic bare cuboid, and a reflected slab against its transcendental criticality condition. |
| V-2 | done | Method of manufactured solutions for the multi-group operator, with the observed order of accuracy of each kernel. |
| V-3 | done | Coarse nodal against refined finite difference; SANM and NEM agree to the last digit printed from four nodes per assembly onward. |
| V-4 | done | Adjoint eigenvalue equality, and first-order perturbation theory against a direct re-solve, which tests the adjoint flux shape rather than only the operator transpose. |
| NFR-EXT-1 | done | See FR-GEO-6. |
| NFR-EXT-2 | done | Group count, precursor count and branch axes are all run-time. |
| NFR-EXT-3 | partial | cp310-cp313 wheels build in CI on manylinux_2_28 and macOS arm64, with x86_64 macOS cross-compiled from the arm64 runner because GitHub is retiring the Intel one. The release pipeline is tag-triggered and uses trusted publishing, refusing to build unless the tag, `pyproject.toml` and `CMakeLists.txt` agree on the version, and there is a conda-forge recipe. No tag has been cut, so nothing is published yet. |
| NFR-EXT-4 | done | No dependency beyond a C++17 compiler; the build fetches nothing at configure time. |
| NFR-EXT-5 | not started | The public `extern "C"` API. |

Measured on an AMD Ryzen 9 6900HX, 8 physical cores, `powersave` governor,
Release build with OpenMP, g++. A target without a machine is not a claim, so
any of these numbers travels with that line. `benchmarks/performance/run.py`
re-runs them all.


---

## What the OpenMC coupling establishes

C-1 from the specification's §6.3 runs a uniform infinite medium in OpenMC
with continuous-energy physics, hands the multi-group cross sections tallied
from that same run to OpenNDM, and compares eigenvalues. There is no spatial
discretisation, no leakage and no homogenisation error, so the case isolates
the translation and nothing else. It agrees to 12 pcm, 1.2 sigma, at 90M histories,
after fixing the two defects it found: `MGXS.get_xs` takes integer domain ids
rather than domain objects, and OpenMC's absorption score excludes (n,2n).
The second was worth 230 pcm and had no symptom but the eigenvalue.

C-2 runs a single reflected assembly and checks the discontinuity factor
tallies two ways: a homogeneous assembly must give factors of exactly 1.0,
and a pin lattice must give a thermal factor above 1.0 and a fast factor
below it, because the assembly surface sits in water. It found the group
ordering reversed, which had applied every factor to the wrong group without
changing anything else about the numbers.

C-3 through C-6 are not written.

## What the benchmarks establish

The nodal kernels are verified two ways that do not share machinery. Against
the analytic bare cuboid, every kernel converges to the closed-form
eigenvalue at second order. Against the IAEA-2D core map, SANM at one node
per assembly lands 2.6 pcm from the mesh-converged eigenvalue, and SANM and
NEM — which close their two-node problems by entirely different routes —
agree with each other to 0.08 pcm. Both IAEA decks reproduce their published
eigenvalues inside the 100 pcm acceptance criterion.

The LMW deck establishes something different, because it has no reference to
reproduce: it is the first problem to run rod banks, cusping, precursors and
theta integration together, and the first measurement of what the missing
exponential transformation costs. Its mesh is converged — eight times the
nodes moves the peak power by 0.06% — while its time step is not, and the
gap between those two is the finding. See `benchmarks/README.md`.

## What the boundary treatment costs and buys

Every surface now carries a corrected coupling: interior faces from the
two-node problem, boundary faces from the one-node problem of
[`theory.md`](theory.md) §3.6. Alongside it the transverse leakage of a
boundary node drops to a linear fit on a non-reflective face instead of
repeating its own average.

The two changes belong together. The one-node problem alone makes the boundary
current depend on the nodal shape, which makes it sensitive to a leakage fit
that the old flat extrapolation had got wrong all along; on its own it made
every three-dimensional case worse. With the leakage fit corrected:

| | before | after |
|---|---|---|
| 1D slab, SANM, 2 to 8 nodes | 231 to 4.0 pcm | exact |
| Reflected slab, SANM | 1.8e-5 to 3.3e-7 | round-off, any mesh |
| Bare cuboid V-1, 32 per side | 3.45 pcm | 0.014 pcm |
| Bare cuboid observed order | 2.0 | 4.2 |
| Manufactured solution, 32 per side | 1.13e-4 | 4.2e-6 |
| Manufactured solution order | 2.8 | 3.8 to 4.7 |
| IAEA-2D, NEM, 2 nodes per assembly | −20.7 pcm | +1.4 pcm |
| IAEA-2D, SANM, 1 node per assembly | +2.5 pcm | +27.8 pcm |

The last row is the price. SANM at one node per assembly was that close to the
limit by cancellation rather than by accuracy — the boundary error ran against
the coarse-mesh error on that particular problem — and the cancellation is
gone. Every refinement of it, both kernels at every other mesh, and every
problem with an analytic answer are better. The converged eigenvalue is
unchanged at 1.029527.
