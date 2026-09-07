# OpenNDM — Software Requirements Specification

**Open Nodal Diffusion Method**
A pure-Python-fronted, C++-cored nodal diffusion solver built natively on the OpenMC stack.

| Field | Value |
|---|---|
| Document version | 0.1 (draft) |
| Status | For review |
| Owner | rizkiokt |
| Target license | MIT (matching OpenMC) |
| Repository | `github.com/rizkiokt/openndm` |
| Distribution | PyPI `openndm`, conda-forge `openndm` |

---

## 1. Purpose and Scope

### 1.1 Purpose

OpenNDM is a three-dimensional, multi-group nodal diffusion code for reactor core analysis. It provides the second stage of a two-step (lattice → core) calculation scheme in which OpenMC performs the lattice-physics stage.

### 1.2 The gap being filled

Existing open-source nodal codes are all Fortran and all assume a lattice code that is not OpenMC:

| Code | Language | Group constants from | Status |
|---|---|---|---|
| KOMODO (ADPRES) | Fortran | External, custom text format | Mature, well-benchmarked |
| OpenNode | Fortran 90 + PyQt5 | External | Active, educational focus |
| OpenNodal | Fortran | External | Active |
| DYN3D | Fortran | HELIOS/CASMO/Serpent | Mature, restricted |
| PARCS | Fortran | PMAXS via GenPMAXS | Mature, RSICC-restricted |

Meanwhile, OpenMC exposes `openmc.mgxs` for automated multi-group cross section generation, but the module was designed to feed **fine-mesh multigroup transport**, not coarse-mesh nodal diffusion. The pieces a nodal code needs — assembly discontinuity factors, critical-spectrum leakage correction, pin form functions, and a branch-parameterized library — do not exist in the OpenMC ecosystem today. Every user who wants OpenMC → nodal rebuilds that glue from scratch.

**OpenNDM's central design goal is that a user who already has an `openmc.mgxs.Library` object should be able to run a full-core nodal calculation without writing a single line of format-conversion code.**

### 1.3 In scope (v1.0)

- 3D Cartesian geometry, arbitrary number of energy groups
- FDM, polynomial nodal (NEM), and semi-analytic nodal (SANM) kernels
- Nonlinear two-node CMFD iteration
- Static eigenvalue, adjoint, fixed-source, critical boron search
- Point-kinetics-consistent transient solver with delayed neutron precursors
- Single-channel thermal-hydraulic feedback with a radial fuel conduction model
- Group-constant generation driver that runs OpenMC lattice calculations and produces a branch-parameterized library including ADFs
- Pin power reconstruction

### 1.4 Deferred to v1.1

Not in the v1.0 release, but on the roadmap. The v1.0 architecture must not preclude any of these — see NFR-EXT-1 and FR-TH-7.

| Item | Enabling v1.0 requirement |
|---|---|
| Hexagonal and triangular-Z geometry | FR-GEO-6, NFR-EXT-1 |
| SP3 / simplified transport correction | FR-SOL-8 (runtime kernel selection) |
| Nodal depletion and burnup history tracking | FR-XS-5 (burnup as a history axis) |
| Multi-channel and subchannel TH; drift-flux two-phase | FR-TH-6, FR-TH-7 |
| Coupling to external open-source TH packages | FR-TH-7 |
| MPI domain decomposition | §3.1 (MPI listed optional) |

**Note on external TH coupling.** There is no plan to couple to RELAP5, RELAP-7, TRACE, or other restricted-distribution system codes. The target is open-source thermal-hydraulics, so that an OpenMC → OpenNDM → TH chain can be assembled and redistributed by anyone without a license request. Candidate targets, in rough order of expected effort:

- **CTF (COBRA-TF)** — subchannel, source-available through the NEAMS/CASL ecosystem, the natural pairing for pin-resolved PWR/BWR work.
- **MOOSE Thermal Hydraulics Module** — LGPL, 1D system TH, gives a path to plant-level transients.
- **GeN-Foam** — GPL, OpenFOAM-based porous-medium reactor TH.
- **NekRS / Nek5000** — BSD, for high-fidelity CFD reference cases rather than production coupling.
- **COBRA-EN** — an older subchannel code with published OpenMC coupling precedent (Mylonakis et al., 2015).

The architectural model to follow is **ENRICO**, the open-source code-agnostic coupling driver from ANL/ORNL. ENRICO already couples OpenMC (and Shift) to Nek5000 and to a simplified heat-diffusion/subchannel surrogate, with solver-agnostic control flow, domain mapping, Picard iteration, and convergence checks, and it transfers solution fields **in memory rather than through filesystem I/O**. OpenNDM should expose a coupling interface shaped so that it can be dropped in as an ENRICO neutronics solver, rather than inventing a parallel coupling framework. `OFELIA` (OpenMC–FEniCSx) is a second useful reference for what a lightweight Python-side coupling looks like.

---

## 2. Design Principles

**DP-1 — Mirror OpenMC's architecture, not just its file formats.** A developer fluent in OpenMC should be able to read OpenNDM's source and immediately know where things live. Same language split, same build system, same I/O conventions, same API idioms.

**DP-2 — OpenMC objects are first-class inputs.** Not "export to a text file, then import." An `openmc.mgxs.Library`, an `openmc.StatePoint`, or an `openmc.Model` should be accepted directly by the Python API.

**DP-3 — The Python API is the primary interface.** The XML input path exists for reproducibility and for running without Python, exactly as in OpenMC, but the Python API is what users are expected to touch.

**DP-4 — Fast enough to sit inside an optimization loop.** Loading pattern optimization, control rod programming, and ML-driven core design all require thousands to millions of core evaluations. Steady-state solves must be sub-second and callable in-memory without file I/O.

**DP-5 — Statistical uncertainty is carried, not discarded.** Group constants from Monte Carlo have error bars. OpenNDM must propagate them rather than silently truncating to a mean value.

**DP-6 — Verifiable against KOMODO.** Every benchmark KOMODO publishes should be runnable in OpenNDM with a documented comparison. KOMODO is the reference implementation for feature parity.

---

## 3. Architecture

### 3.1 Stack

Deliberately parallel to OpenMC:

| Layer | OpenMC | OpenNDM |
|---|---|---|
| Compute core | C++17 | C++17 |
| Build | CMake | CMake ≥ 3.16 |
| Arrays | xtensor | xtensor |
| XML parsing | pugixml | pugixml |
| Binary I/O | HDF5 (via HighFive/native) | HDF5 |
| Formatting | fmt | fmt |
| Shared-memory parallelism | OpenMP | OpenMP |
| Distributed | MPI | MPI (optional, v2) |
| Linear algebra | — | Eigen (sparse), optional PETSc |
| Python bridge | ctypes → `libopenmc` C API | **pybind11 → `libopenndm`** |
| Python package | `openmc` | `openndm` |
| In-memory module | `openmc.lib` | `openndm.lib` |

**Note on the Python bridge (deviation from OpenMC):** OpenMC uses a hand-written C-interoperable API (`openmc_*` functions) called from Python via `ctypes`. That choice predates the maturity of pybind11 and carries significant maintenance burden. OpenNDM should use **pybind11** instead, while still exporting a thin `extern "C"` layer for non-Python coupling. This gives automatic numpy ↔ xtensor zero-copy via `xtensor-python`, which matters for DP-4.

### 3.2 Module decomposition

```
openndm/
├── src/                        # C++ core
│   ├── geometry.cpp            # node grid, symmetry, albedo boundaries
│   ├── xslib.cpp               # macroscopic XS storage, branch interpolation
│   ├── kernel_fdm.cpp          # finite difference
│   ├── kernel_pnm.cpp          # polynomial nodal (NEM)
│   ├── kernel_sanm.cpp         # semi-analytic nodal
│   ├── cmfd.cpp                # nonlinear two-node coupling
│   ├── eigen.cpp               # power iteration, Wielandt shift, BiCGSTAB
│   ├── adjoint.cpp
│   ├── transient.cpp           # theta method, precursor integration
│   ├── th.cpp                  # channel + conduction
│   ├── pinpower.cpp
│   └── output.cpp              # HDF5 statepoint writer
├── include/openndm/
├── python/openndm/
│   ├── __init__.py
│   ├── lib/                    # pybind11 bindings
│   ├── model.py                # Model, Settings, Geometry
│   ├── xslib.py                # XSLibrary, Branch, BranchGrid
│   ├── gc/                     # ← group-constant generation from OpenMC
│   │   ├── driver.py
│   │   ├── adf.py
│   │   ├── leakage.py          # B1 / critical buckling
│   │   ├── kinetics.py         # beta_eff, Lambda
│   │   └── formfunc.py
│   ├── statepoint.py
│   └── plots.py
├── tests/
├── benchmarks/
└── docs/
```

---

## 4. Functional Requirements

Requirements are labelled `FR-<area>-<n>` and are written to be individually testable.

### 4.1 Geometry (FR-GEO)

| ID | Requirement | Priority |
|---|---|---|
| FR-GEO-1 | Represent the core as a structured 3D Cartesian grid of nodes with per-direction, non-uniform mesh spacing. | Must |
| FR-GEO-2 | Accept a 2D radial core map assigning a material/composition index to each assembly position, with an inactive marker for out-of-core positions. | Must |
| FR-GEO-3 | Support quarter-core, half-core, and full-core symmetry with rotational and mirror options; internally expand or apply symmetry BCs. | Must |
| FR-GEO-4 | Support multiple nodes per assembly radially (e.g. 2×2 subdivision) with automatic ADF assignment. | Must |
| FR-GEO-5 | Boundary conditions specified per face as zero-flux, zero-incoming-current (vacuum), reflective, or user-supplied albedo (per group). | Must |
| FR-GEO-6 | Internal representation must be a generic node/surface connectivity graph so that hexagonal geometry can be added without rewriting solver kernels. | Should |
| FR-GEO-7 | Construct geometry directly from an `openmc.RectLattice` used in the lattice model, inferring pitch and assembly map. | Should |

### 4.2 Cross Sections and Group Constants (FR-XS)

| ID | Requirement | Priority |
|---|---|---|
| FR-XS-1 | Support arbitrary number of energy groups G ≥ 1, with full G×G scattering matrices including upscattering. | Must |
| FR-XS-2 | Store, per composition: transport XS or diffusion coefficient, absorption XS, ν-fission XS, κ-fission XS, fission spectrum χ, scattering matrix, and inverse neutron speed. | Must |
| FR-XS-3 | Support delayed neutron data: up to 8 precursor groups with β_i, λ_i, and delayed χ_d,i. | Must |
| FR-XS-4 | Support assembly discontinuity factors, defined per node face per group, defaulting to 1.0 when absent. | Must |
| FR-XS-5 | Support branch-parameterized libraries over an arbitrary set of state variables. Minimum required axes: fuel temperature, moderator density, moderator temperature, boron concentration, control rod insertion state. Burnup as a history axis. | Must |
| FR-XS-6 | Multilinear interpolation across branch axes, with configurable extrapolation policy (clamp / linear / error). | Must |
| FR-XS-7 | Optional analytic feedback model (√T_fuel Doppler coefficient, linear moderator density coefficient) as a lightweight alternative to a full branch table, for verification cases and benchmark problems. | Should |
| FR-XS-8 | Validate libraries on load: non-negative XS, χ sums to 1, scattering matrix consistency, monotonic branch axes. Warn on negative scattering elements rather than failing (Monte Carlo noise produces these). | Must |
| FR-XS-9 | Carry per-value standard deviations through the library when the source data provides them. | Should |

### 4.3 OpenMC-Native Interface (FR-OMC)

**This is the differentiating capability of the project.**

| ID | Requirement | Priority |
|---|---|---|
| FR-OMC-1 | `openndm.XSLibrary.from_mgxs_library(lib)` shall accept an in-memory `openmc.mgxs.Library` and construct a complete single-branch OpenNDM library, mapping MGXS types to OpenNDM fields with no user-supplied translation. | Must |
| FR-OMC-2 | Accept `openmc.mgxs.Library` objects with domain type `material`, `cell`, `universe`, or `mesh`. For `mesh` domains, map mesh elements onto OpenNDM nodes directly. | Must |
| FR-OMC-3 | `openndm.XSLibrary.from_mgxs_file(path)` shall read the `openmc.MGXSLibrary` HDF5 format (`mgxs.h5`) produced by `Library.create_mg_mode()`, so libraries already prepared for OpenMC's multi-group mode work unchanged. | Must |
| FR-OMC-4 | `openndm.XSLibrary.from_statepoint(sp, ...)` shall reconstruct group constants directly from an `openmc.StatePoint`, given the tally specification used. | Should |
| FR-OMC-5 | Accept both `TransportXS` (with transport correction) and the native `DiffusionCoefficient` MGXS score as the source of D, with a documented preference order and a warning when the two disagree by more than a configurable tolerance. | Must |
| FR-OMC-6 | Propagate `uncertainties.ufloat` standard deviations from `openmc.mgxs` into the OpenNDM library (FR-XS-9). | Should |
| FR-OMC-7 | **ADF generation.** Provide `openndm.gc.add_adf_tallies(model, lattice)` that instruments an `openmc.Model` with the surface flux tallies required for ADFs, and `openndm.gc.compute_adf(sp)` that forms the surface-to-volume flux ratios. | Must |
| FR-OMC-8 | **Leakage correction.** Provide a B1 or P1 critical-spectrum correction with a buckling search, applied to infinite-lattice OpenMC results, with the option to disable it. | Must |
| FR-OMC-9 | **Kinetics parameters.** Provide adjoint-weighted β_eff and Λ from OpenMC using the iterated fission probability method, falling back to the k-ratio method when IFP data is unavailable. | Should |
| FR-OMC-10 | **Pin form functions.** Extract normalized pin-wise fission rate distributions from OpenMC lattice tallies for use in pin power reconstruction (FR-OUT-4). | Should |
| FR-OMC-11 | **Branch driver.** `openndm.gc.BranchDriver` shall take a parameterized `openmc.Model` factory and a `BranchGrid`, execute the OpenMC runs (serially, via multiprocessing, or by emitting a job array script), and assemble the resulting branch library. Must support resume-from-partial. | Must |
| FR-OMC-12 | **Self-consistency test.** Provide a documented workflow and regression test in which a small full-core OpenMC model produces mesh-domain MGXS that OpenNDM then solves, and the resulting k_eff agrees with the OpenMC k_eff to within 3σ plus a documented homogenization bias. | Must |
| FR-OMC-13 | The OpenNDM CMFD kernel shall be exposable as an acceleration backend for OpenMC via `openmc.lib`, allowing OpenNDM to serve as an alternative to OpenMC's built-in CMFD. | Could |
| FR-OMC-14 | OpenMC shall be an optional runtime dependency. The core solver must import and run without OpenMC installed; only `openndm.gc` requires it. | Must |

### 4.4 Solver Kernels (FR-SOL)

| ID | Requirement | Priority |
|---|---|---|
| FR-SOL-1 | Finite difference method kernel, for verification and as the CMFD base. | Must |
| FR-SOL-2 | Polynomial nodal method (NEM) with fourth-order transverse-integrated flux expansion and quadratic transverse leakage. | Must |
| FR-SOL-3 | Semi-analytic nodal method (SANM) — the default kernel, matching KOMODO's default. | Must |
| FR-SOL-4 | Nonlinear two-node iteration producing corrected coupling coefficients (D̂) folded into the CMFD system. | Must |
| FR-SOL-5 | Outer eigenvalue iteration via power iteration with Wielandt shift; inner linear solves via BiCGSTAB with ILU or block-Jacobi preconditioning. | Must |
| FR-SOL-6 | Configurable convergence criteria on k_eff, node-wise fission source, and outer iteration count, with clear non-convergence diagnostics. | Must |
| FR-SOL-7 | OpenMP parallelization over nodes for the two-node updates and over planes/groups where the sweep structure allows. | Should |
| FR-SOL-8 | Kernel selection at runtime, not compile time, so a single build can run all three. | Must |

### 4.5 Calculation Modes (FR-MODE)

| ID | Requirement | Priority |
|---|---|---|
| FR-MODE-1 | Forward static eigenvalue (k_eff and flux distribution). | Must |
| FR-MODE-2 | Adjoint static eigenvalue. | Must |
| FR-MODE-3 | Fixed external source, including subcritical multiplication. | Must |
| FR-MODE-4 | Critical boron search: iterate boron concentration to a target k_eff with a configurable tolerance and bracketing. | Must |
| FR-MODE-5 | Control rod worth: differential and integral worth curves from a sequence of static solves. | Must |
| FR-MODE-6 | Steady state with thermal-hydraulic feedback (coupled Picard iteration). | Must |
| FR-MODE-7 | Transient with feedback (see FR-KIN). | Must |
| FR-MODE-8 | Batch/parametric mode: execute N static solves over a set of perturbed inputs from a single in-memory model, reusing the geometry and library. | Should |

### 4.6 Kinetics and Transients (FR-KIN)

| ID | Requirement | Priority |
|---|---|---|
| FR-KIN-1 | Time-dependent multi-group diffusion with up to 8 delayed precursor groups. | Must |
| FR-KIN-2 | Fully implicit or θ-weighted time integration with exponential transformation of the precursor equations. | Must |
| FR-KIN-3 | Adaptive time stepping with user-controllable min/max Δt and an error criterion. | Should |
| FR-KIN-4 | Transient drivers: control rod motion (position vs. time), boron injection, inlet temperature/flow ramps, and direct XS perturbation vs. time. | Must |
| FR-KIN-5 | Decay heat model (ANS-5.1 or a simple multi-group exponential fit). | Should |
| FR-KIN-6 | Output time-series of total power, peak node power, reactivity, and TH state at user-specified intervals. | Must |

### 4.7 Thermal Hydraulics (FR-TH)

| ID | Requirement | Priority |
|---|---|---|
| FR-TH-1 | One-dimensional single-phase closed-channel model per assembly with axial enthalpy rise. | Must |
| FR-TH-2 | Radial fuel pin conduction: fuel pellet mesh, gap conductance, cladding, producing volume-average and effective Doppler fuel temperatures. | Must |
| FR-TH-3 | Water properties from IAPWS-IF97 (or a documented tabulated equivalent) covering PWR and BWR operating ranges. | Must |
| FR-TH-4 | Configurable Doppler temperature weighting (e.g. the 0.7·T_surface + 0.3·T_center convention) exposed as a user parameter. | Should |
| FR-TH-5 | Homogeneous equilibrium two-phase model with a void fraction correlation for BWR applications. | Should |
| FR-TH-6 | The built-in TH solver shall sit behind an abstract `ThermalSolver` interface — `set_heat_source(q)` / `solve()` / `get_temperatures()` / `get_densities()` — with the built-in channel model as one implementation among several. No solver kernel shall call the TH module directly. | Must |
| FR-TH-7 | **Coupling readiness.** The Picard iteration driver, field mapping between the node grid and an external solver's mesh, and convergence checking shall be separated from the built-in TH implementation, so that a v1.1 external solver can be attached without touching the neutronics. Field transfer shall be in-memory (numpy arrays / xtensor views), never via files. Volume-conservative mapping between the nodal grid and a finer external mesh shall be provided. | Must |
| FR-TH-8 | Water properties shall be obtainable from a pluggable backend (built-in IAPWS-IF97 table, or CoolProp / the `iapws` package when installed) so that property consistency with an external TH code can be enforced. | Should |

### 4.8 Optimization and Embedding (FR-OPT)

Driven by DP-4. These requirements are what distinguish OpenNDM from a teaching code.

| ID | Requirement | Priority |
|---|---|---|
| FR-OPT-1 | Complete in-memory workflow: build model, solve, extract results, mutate model, re-solve — with zero filesystem access. | Must |
| FR-OPT-2 | `Model.solve()` shall release the GIL during the C++ solve so multiple models can be driven from Python threads. | Must |
| FR-OPT-3 | Warm-start: reuse a converged flux and D̂ from a previous solve as the initial guess for a perturbed model. Must demonstrate ≥2× speedup on a shuffled loading pattern. | Should |
| FR-OPT-4 | Deterministic and reproducible: identical inputs produce bit-identical outputs regardless of thread count. | Must |
| FR-OPT-5 | Results exposed as numpy arrays with zero copy where possible (`xtensor-python`). | Must |
| FR-OPT-6 | A solve that fails to converge shall raise a typed Python exception, never abort the process. | Must |
| FR-OPT-7 | Loading pattern mutation API: swap, rotate, and reassign assemblies by position label. | Should |

### 4.9 Output and Post-processing (FR-OUT)

| ID | Requirement | Priority |
|---|---|---|
| FR-OUT-1 | Write an HDF5 statepoint (`statepoint.h5`) using OpenMC's conventions for attribute naming, versioning, and dataset layout. | Must |
| FR-OUT-2 | `openndm.StatePoint` reader class mirroring `openmc.StatePoint` idioms. | Must |
| FR-OUT-3 | Standard derived quantities: k_eff, node power distribution, radial (2D) and axial (1D) power profiles, F_ΔH, F_q, assembly-wise and core-average burnup-independent peaking factors. | Must |
| FR-OUT-4 | Pin power reconstruction combining nodal flux, corner-point flux, and pin form functions (FR-OMC-10). | Should |
| FR-OUT-5 | Plotting helpers: radial core map, axial profile, convergence history, transient time-series. Matplotlib-based, matching `openmc.plots` conventions. | Should |
| FR-OUT-6 | VTK export for 3D visualization. | Could |
| FR-OUT-7 | Structured, level-controlled logging of iteration history to stdout and optionally to file. | Must |

### 4.10 Input Formats (FR-IN)

| ID | Requirement | Priority |
|---|---|---|
| FR-IN-1 | Python API is the primary interface; it generates and validates the XML. | Must |
| FR-IN-2 | XML input files (`model.xml` or split `geometry.xml`/`settings.xml`/`xslib.xml`) readable by the standalone `openndm` executable, parsed with pugixml, with an XSD or RelaxNG schema shipped for validation. | Must |
| FR-IN-3 | Binary XS library in HDF5 (`xslib.h5`), with a documented and versioned layout. | Must |
| FR-IN-4 | KOMODO input reader (best-effort) to allow direct migration of existing KOMODO benchmark decks. | Should |
| FR-IN-5 | Every published benchmark in `benchmarks/` shall ship both a Python script and the generated XML. | Should |

---

## 5. Non-Functional Requirements

### 5.1 Performance (NFR-PERF)

Reference hardware for acceptance: AMD Ryzen 9 6900HX (8C/16T), 16 GB usable RAM.

| ID | Requirement |
|---|---|
| NFR-PERF-1 | 3D quarter-core PWR (≈ 60 assemblies, 1 node/assembly radial, 24 axial planes, 2 groups), static SANM with nonlinear CMFD: **< 1 s** single-threaded. |
| NFR-PERF-2 | Same problem with 2×2 radial subdivision and 8 groups: **< 10 s** single-threaded. |
| NFR-PERF-3 | Coupled neutronics/TH steady state on the NFR-PERF-1 problem: **< 5 s**. |
| NFR-PERF-4 | 5-second rod ejection transient on the NFR-PERF-1 problem with adaptive stepping: **< 5 min**. |
| NFR-PERF-5 | Memory for NFR-PERF-2: **< 500 MB**. |
| NFR-PERF-6 | Warm-started perturbed solve (FR-OPT-3): **< 0.3 s**. |
| NFR-PERF-7 | Parallel efficiency ≥ 60% on 8 threads for problems above 10⁵ nodes×groups. |

### 5.2 Quality and Process (NFR-QA)

| ID | Requirement |
|---|---|
| NFR-QA-1 | Unit tests: Catch2 or GoogleTest for C++, pytest for Python. Line coverage ≥ 80% on the C++ core. |
| NFR-QA-2 | Regression tests: every benchmark in §6 runs in CI with tolerances on k_eff and power distribution. |
| NFR-QA-3 | CI on GitHub Actions: Linux (gcc, clang), macOS, and a Windows build or a documented WSL-only statement. |
| NFR-QA-4 | `clang-format` on C++ (matching OpenMC's `.clang-format`), `ruff` on Python, enforced in CI. |
| NFR-QA-5 | Semantic versioning; a changelog; no breaking Python API change without a deprecation cycle. |
| NFR-QA-6 | Sphinx documentation with a theory manual, user guide, API reference, and at least three worked examples as Jupyter notebooks. |
| NFR-QA-7 | The theory manual shall state every equation the code implements, with the discretization written out, sufficient for independent reimplementation. |

### 5.3 Portability and Extensibility (NFR-EXT)

| ID | Requirement |
|---|---|
| NFR-EXT-1 | Solver kernels shall operate on the abstract node/surface connectivity (FR-GEO-6), not on `(i,j,k)` indices, so hexagonal geometry is a geometry-module addition only. |
| NFR-EXT-2 | Adding an energy-group count, precursor count, or branch axis shall require no recompilation. |
| NFR-EXT-3 | Binary wheels on PyPI for cp310–cp313 on manylinux and macOS (arm64 + x86_64). |
| NFR-EXT-4 | Build from source with no dependency outside conda-forge / apt. |
| NFR-EXT-5 | Public C API (`extern "C"`, `openndm_*` symbols) for coupling from Fortran, Julia, or MATLAB. |

---

## 6. Verification and Validation Plan

### 6.1 Code verification (analytic and method-of-manufactured-solutions)

- V-1: 1D and 3D homogeneous bare slab/cuboid against the analytic buckling solution — must converge to machine precision as the mesh refines.
- V-2: Method of manufactured solutions for the multi-group diffusion operator; verify the expected spatial order of accuracy for each kernel.
- V-3: FDM kernel with a very fine mesh vs. SANM/NEM with a coarse mesh — must agree within a documented tolerance.
- V-4: Adjoint solve reciprocity check: ⟨φ†, Fφ⟩ consistency.

### 6.2 Benchmark suite

| Benchmark | Type | Purpose |
|---|---|---|
| IAEA 2D/3D PWR | Static, 2-group | The canonical nodal verification problem |
| BIBLIS 2D | Static, 2-group | Coarse-mesh accuracy |
| KOEBERG / Zion | Static | Realistic PWR core map |
| LMW | Transient without feedback | Rod withdrawal kinetics |
| TWIGL 2D | Transient, seed-blanket | Step and ramp perturbation |
| LRA BWR | Transient with adiabatic feedback | Superprompt-critical rod ejection |
| NEACRP-L336 3D PWR (A1–C2) | Transient with full TH feedback | The reference rod ejection validation |
| PWR MOX/UO₂ (C5G7-derived) | Static, 7-group | Multi-group and strong flux gradients |
| V1000CT-1 (if hex added, v2) | Static + transient | VVER geometry |
| APR1400 initial core | Static | Cross-comparison against KOMODO, PARCS, SIMULATE-3 |

### 6.3 OpenMC-coupling validation

This is validation of the *interface*, not just the solver, and has no precedent in the existing nodal codes.

- C-1: **Homogeneous consistency.** A uniform infinite medium, MGXS from OpenMC, solved by OpenNDM — k_eff must match OpenMC's k_∞ to within 1σ. Isolates the XS translation path.
- C-2: **Single assembly, reflective.** Verifies ADF machinery returns ≈ 1.0 for an infinite lattice.
- C-3: **Colorset.** 2×2 assembly colorset with a strong spectral gradient (UO₂/MOX or fuel/reflector), solved by both OpenMC continuous-energy and OpenNDM with ADFs. Quantify the homogenization error with and without ADFs; the improvement is the headline result for the interface paper.
- C-4: **Small full core.** A reduced-size 3D core run in OpenMC continuous-energy as reference, with node-wise MGXS from a mesh-domain library. Report k_eff difference and node power RMS error. This is FR-OMC-12.
- C-5: **Branch library round trip.** Generate a small branch table, interpolate at an off-node state point, and compare against a directly-computed OpenMC case at the same state.
- C-6: **Uncertainty propagation.** Sample group constants from their MC uncertainties and report the induced σ on k_eff and peak power.

### 6.4 Acceptance criteria

- k_eff within 100 pcm of reference for static benchmarks with published solutions.
- Assembly power RMS error within 2%, maximum error within 5%.
- Transient peak power and time-to-peak within 5% of the reference solution.

---

## 7. Development Phases

| Phase | Deliverable | Key requirements | Rough effort |
|---|---|---|---|
| M0 | Repo scaffolding, CMake + pybind11 build, CI, empty-but-importable package | NFR-QA-3, NFR-EXT-3 | 1–2 weeks |
| M1 | 3D Cartesian geometry, XS library object, FDM kernel, power iteration | FR-GEO-1..5, FR-XS-1..3, FR-SOL-1, FR-SOL-5 | 3–4 weeks |
| M2 | **`from_mgxs_library` ingestion path + C-1, C-2 passing** | FR-OMC-1..6, FR-OMC-14 | 2–3 weeks |
| M3 | SANM and NEM kernels, nonlinear two-node CMFD; IAEA-3D and BIBLIS passing | FR-SOL-2..4, FR-SOL-8 | 4–6 weeks |
| M4 | ADF and leakage-correction generation; C-3 quantified | FR-OMC-7, FR-OMC-8, FR-XS-4 | 3–4 weeks |
| M5 | Branch library, driver, interpolation; TH feedback; boron search | FR-XS-5..7, FR-OMC-11, FR-TH-1..4, FR-MODE-4, FR-MODE-6 | 5–6 weeks |
| M6 | Transient solver; LMW, TWIGL, LRA, NEACRP passing | FR-KIN-1..6 | 5–6 weeks |
| M7 | Optimization/embedding API, warm start, batch mode | FR-OPT-1..7 | 2–3 weeks |
| M8 | Pin power reconstruction, plotting, VTK | FR-OUT-4..6, FR-OMC-10 | 3 weeks |
| M9 | Documentation, theory manual, full V&V report, JOSS/ANS paper | NFR-QA-6, NFR-QA-7 | 3–4 weeks |

**v1.0 released after M9.**

### v1.1 phases

| Phase | Deliverable | Notes |
|---|---|---|
| M10 | Hexagonal geometry; VVER and SFR benchmarks | Should be a geometry-module change only if FR-GEO-6 held |
| M11 | External TH coupling: first target CTF, driven through the FR-TH-7 interface | Validate against the built-in channel model before trusting the external result |
| M12 | ENRICO integration — register OpenNDM as an ENRICO neutronics solver | Gets NekRS and the ENRICO surrogate solvers for free |
| M13 | Nodal depletion with burnup history axis | Requires FR-XS-5 history support to have been designed in |
| M14 | SP3 kernel | Runtime-selectable alongside SANM/NEM/FDM |
| M15 | MPI domain decomposition | Only if problem sizes justify it |

M2 is the milestone that makes the project distinctive. Consider it the point at which the repository is worth announcing publicly, even in an incomplete state.

---

## 8. Risks and Open Questions

| Risk | Impact | Mitigation |
|---|---|---|
| **ADF statistical noise.** ADFs are ratios of surface flux to volume flux; surface tallies converge slowly and the ratio amplifies relative error. Noisy ADFs can degrade the solution below the no-ADF baseline. | High | Use a track-length-estimated thin-slab volume tally in place of a true surface tally; document the tally-count requirement; provide a smoothing/regularization option and a warning when ADF σ exceeds a threshold. |
| **Negative scattering matrix elements** from MC noise in weakly-coupled group transfers. | Medium | Detect, warn, and offer a documented fix-up (zero-and-renormalize) rather than silently proceeding. |
| **Definition ambiguity in the leakage correction.** B1 vs. P1 vs. a simple buckling-corrected D produce measurably different results; different lattice codes make different choices, so cross-comparison against KOMODO decks generated with Serpent may not be apples-to-apples. | Medium | Implement more than one, make it explicit in the input and in the statepoint metadata, and document which convention each benchmark uses. |
| **Branch library generation cost.** A 5-axis branch table over 30 compositions is thousands of OpenMC runs. | High | Design `BranchDriver` for HPC job arrays and resume-from-partial from day one; provide a "coarse grid + analytic feedback" mode for development work. |
| **Spectral history effects** are not captured by instantaneous branch interpolation. | Medium | Out of scope for v1; document as a known limitation. Design the branch axis system so a history axis can be added. |
| **OpenMC API churn.** `openmc.mgxs` internals may change between releases. | Medium | Pin a minimum OpenMC version, test against both `stable` and `develop` in CI, and keep translation logic in one module. |
| **Scope creep toward being a full lattice code.** | Medium | The `gc` subpackage orchestrates OpenMC; it does not reimplement lattice physics. Hold that line. |

### Open questions for the design review

1. Should the primary D source be `TransportXS` with an out-scatter correction, or the newer `DiffusionCoefficient` estimator? Behavior in strongly heterogeneous nodes needs testing before the default is fixed (FR-OMC-5).
2. Is pybind11 the right call given that OpenMC uses ctypes, or is matching OpenMC exactly worth the maintenance cost for the sake of DP-1?
3. Should the XS library format be a new HDF5 layout, or an extension of `openmc.MGXSLibrary` with ADF and branch data added as extra groups? The latter is more "native" but risks confusing users about what OpenMC itself can consume.
4. Is a KOMODO input reader (FR-IN-4) worth the effort, or is a one-time manual conversion of the benchmark decks sufficient?

---

## 9. References

- OpenMC documentation — Python API, `openmc.mgxs`, C/C++ API, `openmc.lib`
- KOMODO / ADPRES — Imron, *Development and verification of open reactor simulator ADPRES*
- OpenNode — Satti et al., nodal expansion method with PyQt5 interface
- OpenNodal — `github.com/nfherrin/opennodal`
- Smith, K.S. — *Assembly homogenization techniques for light water reactor analysis*
- Lawrence, R.D. — *Progress in nodal methods for the solution of the neutron diffusion and transport equations*
- Downar, T. et al. — PARCS theory manual
- Romano, P. et al. — *A code-agnostic driver application for coupled neutronics and thermal-hydraulic simulations* (ENRICO), Nuclear Science and Engineering
- *OFELIA: An OpenMC–FEniCSx coupling for neutronic calculation with temperature feedback*
- Mylonakis, A. et al. — OpenMC / COBRA-EN coupling for PWR and BWR systems
