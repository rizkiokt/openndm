# Roadmap

What is planned, in what order, and why that order. For what already exists,
read [`status.md`](status.md) — this document does not duplicate it.

Version 0.1.0 covers M0–M3 of the specification's phase plan: geometry, cross
sections, the three kernels, static calculation modes, the OpenMC coupling,
and output. What follows is M4 onward.

## The ordering principle

Two things decide what comes first.

**A feature is scheduled by what it unblocks, not by how visible it is.**
Control rod banks come before kinetics because every rod-driven transient
needs them, and because building them for a transient means building them
twice.

**Nothing ships against a reference that cannot be verified.** This is not a
general principle about testing; it is a specific lesson from BIBLIS-2D,
recorded in `benchmarks/README.md`. A deck
transcribed from memory can be tuned until it reproduces a published
eigenvalue, and the result looks exactly like verification while being
fitting. So every phase below leads with a reference that is analytic,
exact, or independently derived, and treats published benchmark decks as
confirmation rather than as the primary evidence.

## Phase 1 — control rod banks — complete

The last substantial static-scope capability the solver lacked, and its
largest usability gap: rods today are separate compositions
assigned per axial plane, which is how `benchmarks/iaea3d/run.py` expresses
them, and there is no way to ask for a bank at a position.

| Branch | Delivers | Verified by | State |
|---|---|---|---|
| `feat/rod-banks` | Bank definitions, radial bank map, continuous position | Reproducing the existing IAEA-3D deck exactly whenever the tip lands on a plane boundary | **done** — `ControlRodBank`, `ControlRods` |
| `feat/rod-cusping` | Flux-weighted homogenisation of the partially rodded node | A continuous position sweep against a fine-mesh reference: exact at every plane boundary, and the staircase gone between them | **done** — `cusp=`, `converge_cusping` |
| `feat/rod-worth` | FR-MODE-5 proper — differential and integral worth curves | Worth from a sweep against worth from two direct solves | **done** — `worth_curve` |

Cusping took the largest error against a fine-mesh reference from 784 pcm to
55 pcm, and the mean bias from +165 pcm to -13 pcm, on a ten-plane test core.
Volume weighting alone reaches 259 pcm and is biased low by 103 pcm, because
it ignores the flux depression on the rodded side; the flux-weighted iteration
is what closes the rest.

The library half already exists: FR-XS-5 accepts rod state as a branch axis.
Only the geometry half — mapping a bank at a position onto node compositions
— is missing.

Position follows KOMODO's convention (`POS0` zero-step position in cm,
`SSIZE` cm per step, position in steps, zero fully inserted) so existing
decks translate without arithmetic.

**Cusping is the part worth having.** The common workaround for a
partially inserted bank is to declare a separate rodded planar type, which is
the same workaround this project's
IAEA-3D deck uses. It is also a problem the OpenMC coupling is unusually
well placed to solve: rather than smearing rodded and unrodded constants,
the partially rodded node's constants can be generated directly.

## Phase 2 — kinetics (FR-KIN)

The largest capability gap. Delayed neutron data already round-trips through
the library (FR-XS-3), so the storage format does not change.

1. `feat/kinetics-precursors` — precursor equations integrated analytically
   over the step, which is more accurate than lagging them.
   **Verified against exact point kinetics** in a leakage-free box: the
   prompt jump on a step insertion, and the asymptotic period from the inhour
   equation. Neither needs a published deck.
2. `feat/kinetics-theta` — theta-method time integration of the full spatial
   operator, plus the exponential flux transformation. Verified by measuring
   the observed order in Δt: first order at θ=1, second at θ=0.5.
3. `feat/doppler-sqrt-t` — FR-XS-7, the √T adiabatic Doppler model. A
   √T dependence follows the physics of Doppler broadening, where a linear
   per-Kelvin coefficient is an approximation to it. Needed for LRA.
4. `feat/kinetics-benchmarks` — LMW (needs Phase 1), TWIGL ramp, LRA.

Steps 1 and 2 stand on their own evidence. Step 4 confirms.

## Phase 3 — thermal hydraulics (FR-TH)

A closed-channel mass and energy solve with one-dimensional radial
conduction through fuel, gap and cladding. The input set this needs is
well established: percent power, thermal power, inlet temperature, mass flow,
pin geometry, pins and guide tubes per assembly, and the fraction of heat
deposited directly in the coolant.

Unblocks FR-MODE-6 and the NEACRP benchmarks. Nothing in the current solver
obstructs it: no kernel refers to temperature or density, and cross sections
reach the solver only through `XSLibrary`, which already interpolates over a
temperature axis.

## Unscheduled, and independent of the above

| Item | Why it is worth doing |
|---|---|
| `feat/adf-rotation` | FR-OPT-7. Directly coupled-scope: OpenMC gives discontinuity factors in one orientation, and a rotated assembly needs its faces permuted. 90/180/270 over assembly ranges |
| `feat/perf-acceptance` | NFR-PERF-1..7 have **never been measured**. The specification states numeric targets and no acceptance run has ever been made against them |
| `feat/vtk-export` | FR-OUT-6. Mesh output for external visualisation |
| Cut the first release | NFR-EXT-3. The release pipeline landed in #7: tag-triggered, trusted publishing, version agreement enforced before it builds. No tag has been cut, so nothing is on PyPI or conda-forge yet |
| Boundary nodal correction | FR-SOL-4 completeness. See below — small value, listed so the reasoning is not lost |

### On the boundary nodal correction

The nonlinear correction is applied to interior surfaces only. It is
tempting to schedule this because it is cheap and closes a documented
limitation, and an earlier version of `status.md` justified it by claiming
SANM at one node per assembly sits 18 pcm from the mesh-converged
eigenvalue on the IAEA-2D map.

It sits 2.6 pcm from it. On a core problem there is essentially nothing for
a boundary correction to recover, because the outer boundary sits in a
reflector far from the fuel. It remains worth doing for the analytic bare
cuboid, where the entire remaining error genuinely does live at the
boundary, and for completeness — but not for core accuracy.

## Deferred on purpose

- **FR-IN-2** (XML input, pugixml, the standalone executable) and
  **FR-IN-4** (external deck reader). A milestone of plumbing that adds no
  physics while the Python API is serving as the primary interface.
- **NFR-EXT-5**, the public `extern "C"` API. No consumer yet.
- **FR-OUT-4**, pin power reconstruction. Half built already, since
  `compute_form_functions` extracts and normalises them, but downstream of
  everything above.

## Blocked

Further static benchmarks — BIBLIS, LRA static, TWIGL static — need
reference specifications. They are not blocked on solver capability. See the
BIBLIS record in `benchmarks/README.md` for why a transcribed map cannot be
closed by searching for one that reproduces the published eigenvalue.
