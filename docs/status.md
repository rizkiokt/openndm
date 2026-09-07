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
| FR-XS-7 | not started | Analytic √T Doppler feedback model. |
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
| FR-SOL-4 | done | Nonlinear two-node iteration on interior surfaces. Boundary faces keep their finite difference coupling; see the limitation below. |
| FR-SOL-5 | done | Power iteration with a capped Wielandt shift; BiCGSTAB with ILU0. |
| FR-SOL-6 | done | Criteria on k, node-wise fission source and iteration count; `ConvergenceError` carries the count and the residual. |
| FR-SOL-7 | partial | OpenMP over nodes and surfaces in the two-node update, the matrix-vector product and the source assembly. The triangular solves in ILU0 are serial. |
| FR-SOL-8 | done | Run-time selection; one build runs all three. |

## 4.5 Calculation modes (FR-MODE)

| ID | State | Notes |
|---|---|---|
| FR-MODE-1 | done | |
| FR-MODE-2 | done | Reuses the coupling coefficients converged by a forward solve. |
| FR-MODE-3 | done | Verified against exact algebra in a leakage-free box, and against a method of manufactured solutions for the full multi-group operator. |
| FR-MODE-4 | done | Secant with a bisection fallback; the boron model is supplied by the caller. |
| FR-MODE-5 | partial | Achievable through `Model.sweep`, but there is no dedicated rod-worth API. |
| FR-MODE-6 | not started | Needs FR-TH. |
| FR-MODE-7 | not started | Needs FR-KIN. |
| FR-MODE-8 | done | `Model.sweep`, reusing geometry and library, warm starting by default. |

## 4.6 Kinetics (FR-KIN)

**Not started.** FR-KIN-1 through FR-KIN-6. Delayed neutron data is stored and
survives a round trip, so the library format will not need to change, but
there is no time integration.

## 4.7 Thermal hydraulics (FR-TH)

**Not started.** FR-TH-1 through FR-TH-8. No `ThermalSolver` interface exists
yet. FR-TH-6 and FR-TH-7 constrain the *shape* of that work rather than the
neutronics, and nothing in the current solver blocks them: no kernel refers to
temperature or density, and cross sections reach the solver only through
`XSLibrary`, which already interpolates over a temperature axis.

## 4.8 Optimisation and embedding (FR-OPT)

| ID | State | Notes |
|---|---|---|
| FR-OPT-1 | done | No filesystem access anywhere in build/solve/read/mutate/re-solve. |
| FR-OPT-2 | done | The GIL is released for the whole solve. |
| FR-OPT-3 | partial | Warm start works and reduces the outer count. The ≥2× speedup on a shuffled loading pattern is not measured. |
| FR-OPT-4 | done | Fixed-size chunked reductions; tested bit-identical on 1, 2, 4 and 8 threads. |
| FR-OPT-5 | done | Zero-copy numpy views with keep-alive, through pybind11 rather than xtensor. |
| FR-OPT-6 | done | Typed exceptions on every path; nothing aborts. |
| FR-OPT-7 | partial | `swap_assemblies` and `set_composition`. Rotation is not implemented, and positions are addressed by index rather than by label. |

## 4.9 Output (FR-OUT)

| ID | State | Notes |
|---|---|---|
| FR-OUT-1 | done | Versioned `statepoint.h5` on OpenMC's conventions, written from Python through h5py. |
| FR-OUT-2 | done | `openndm.StatePoint`. |
| FR-OUT-3 | done | k, node power, radial and axial profiles, `F_q`, `F_ΔH`. |
| FR-OUT-4 | not started | Pin power reconstruction. |
| FR-OUT-5 | done | Radial map, axial profile and convergence history. |
| FR-OUT-6 | not started | VTK export. |
| FR-OUT-7 | done | Verbosity-controlled iteration history to stdout, and the history array on every result. |

## 4.10 Input (FR-IN)

| ID | State | Notes |
|---|---|---|
| FR-IN-1 | partial | The Python API is the primary interface and validates its input. It does not generate XML, because FR-IN-2 does not exist. |
| FR-IN-2 | not started | XML input, pugixml parsing, the schema, and the standalone `openndm` executable. |
| FR-IN-3 | done | Versioned `xslib.h5`. |
| FR-IN-4 | not started | KOMODO input reader. |
| FR-IN-5 | partial | Benchmarks ship a Python script; there is no XML to ship alongside it. |

## 5 Non-functional

| ID | State | Notes |
|---|---|---|
| NFR-PERF-1..7 | not measured | No performance acceptance runs have been done on the reference hardware. The IAEA-2D quarter core at one node per assembly solves in about 15 ms and the IAEA-3D core with 19 axial planes in about 150 ms, both single-threaded, which suggests the targets are reachable, but that is an observation and not an acceptance test. |
| NFR-QA-1 | partial | Catch2 for C++ and pytest for Python. Coverage is collected in CI but the 80% line coverage gate is not enforced. |
| NFR-QA-2 | done | Every deck runs in CI against its published reference with an explicit tolerance. All three meet the 100 pcm acceptance criterion for static benchmarks. |
| NFR-QA-3 | done | Linux gcc and clang, macOS clang, Python 3.10 to 3.13. Windows is not built and is not documented as WSL-only. |
| NFR-QA-4 | done | clang-format and ruff, both enforced. |
| NFR-QA-5 | done | Semantic versioning and a changelog. |
| NFR-QA-6 | partial | Markdown documentation covering theory, architecture, status and contribution. No Sphinx build and no notebooks. |
| NFR-QA-7 | partial | [`theory.md`](theory.md) writes out every equation the code currently solves, with the discretisation. It covers only what is implemented. |
| V-1 | done | Analytic bare cuboid, and a reflected slab against its transcendental criticality condition. |
| V-2 | done | Method of manufactured solutions for the multi-group operator, with the observed order of accuracy of each kernel. |
| V-3 | done | Coarse nodal against refined finite difference; SANM against NEM to 0.08 pcm. |
| V-4 | done | Adjoint eigenvalue equality, and first-order perturbation theory against a direct re-solve, which tests the adjoint flux shape rather than only the operator transpose. |
| NFR-EXT-1 | done | See FR-GEO-6. |
| NFR-EXT-2 | done | Group count, precursor count and branch axes are all run-time. |
| NFR-EXT-3 | partial | The wheel job is configured for cp310–cp313 on manylinux and macOS but has not been run. |
| NFR-EXT-4 | done | No dependency beyond a C++17 compiler; the build fetches nothing at configure time. |
| NFR-EXT-5 | not started | The public `extern "C"` API. |

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

## The one physics limitation worth knowing

The nonlinear nodal correction is applied to **interior surfaces only**.
Boundary faces keep the finite difference coupling derived from the boundary
condition. This is why the nodal kernels are only about 4× more accurate than
FDM on the analytic bare cuboid rather than the order of magnitude they achieve
inside a core: on that problem the entire remaining error lives at the
boundary. It matters much less on a real core, where the outer boundary sits
in a reflector far from the fuel, which is why SANM at one node per assembly
still lands 18 pcm from the fine-mesh limit on the IAEA map.

Closing it means adding a one-node boundary problem to the `Kernel` interface:
the node-average constraint fixes one coefficient and the boundary condition
fixes the other, so it is exactly determined and structurally simpler than the
two-node problem already implemented.
