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

1. **done** — `feat/kinetics-precursors` — precursor equations integrated analytically
   over the step, which is more accurate than lagging them.
   **Verified against exact point kinetics** in a leakage-free box: the
   prompt jump on a step insertion, and the asymptotic period from the inhour
   equation. Neither needs a published deck.
2. **done** — `feat/kinetics-theta` — theta-method time integration of the
   full spatial operator. Observed order in Δt measured: 1.00 at θ=1, 2.00 at
   θ=0.5. The exponential flux transformation is *not* done and remains open.
   Nor is the nonlinear nodal coupling re-converged inside a step.
3. **done** — `feat/doppler-sqrt-t` — FR-XS-7, `DopplerFeedback`. A √T
   dependence follows the physics of Doppler broadening, where a linear
   per-Kelvin coefficient is an approximation to it. Needed for LRA, whose
   specification defines its feedback in exactly this form. The adiabatic
   heat-up that drives it belongs with the LRA deck.
4. **done** — `feat/lmw-transient` — the LMW operational transient, parsed
   from its specification. Two banks moving against each other for 60
   seconds: the first problem to run rod banks, cusping, precursors and theta
   integration at once. Its specification states the scenario and **not** the
   answer, so it is reported rather than scored; the verification is internal
   and is what produced the measurement below. TWIGL and LRA remain.

Steps 1 and 2 stand on their own evidence: null transient flat to 1 part in
1e10, prompt jump to 0.5%, inhour period to 0.2%.

**What LMW changed about the ordering.** The measured second order at
theta = 1/2 is asymptotic and only visible for steps that resolve the prompt
time constant. At the quarter-second an operational transient runs at, both
weightings converge first order, and the deck's own step sits about 3.4% from
the extrapolated peak while the mesh is converged to 0.06%. That promotes the
exponential flux transformation from a line in "what is not done" to the
single change that would most improve transient accuracy — it is the standard
remedy for exactly this, and nothing else on this list competes with a 3%
error at the step size the problem is meant to be run at.

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

### On the boundary nodal correction, now done

Worth keeping, because the reasoning that scheduled it was wrong twice over
and the record is more useful than the conclusion.

An early `status.md` justified the work by claiming SANM at one node per
assembly sits 18 pcm from the mesh-converged IAEA-2D eigenvalue. It sat 2.6
pcm from it, so the issue was rewritten to say there was essentially nothing
for a boundary correction to recover on a core problem, and that it was worth
doing only for the analytic cuboid and for completeness.

That was wrong in the other direction. The one-node boundary problem on its
own made *every* three-dimensional case worse, because it makes the boundary
current depend on the nodal shape and so exposed a transverse leakage fit
that had been wrong at boundary nodes all along — flat extrapolation asserts a
mirror symmetry only a reflective face has. Fixing both together took the
nodal kernels from third to fourth order and cut the manufactured-solution
error by 27×.

The lesson is not about boundaries. It is that a measurement which says a
change is not worth making can be measuring a cancellation, and that the
2.6 pcm which retired the issue was itself the cancellation.

## Deferred on purpose

- **FR-IN-2** (XML input, pugixml, the standalone executable) and
  **FR-IN-4** (external deck reader). A milestone of plumbing that adds no
  physics while the Python API is serving as the primary interface.
- **NFR-EXT-5**, the public `extern "C"` API. No consumer yet.
- **FR-OUT-4**, pin power reconstruction. Half built already, since
  `compute_form_functions` extracts and normalises them, but downstream of
  everything above.

## Blocked

**BIBLIS is done, and LMW with it.** Both were blocked on a reference
specification and the specification existed all along: KOMODO ships them, and
NEACRP, KOEBERG and MOX, as runnable sample decks under an MIT licence. The
underlying data is the published OECD/NEA and ANL benchmark material; KOMODO
is a machine-readable transcription of it.

Parsed rather than transcribed, BIBLIS landed 2.1 pcm from its published
eigenvalue on the first run. The deck written from memory had been wrong in
both the map and the reference -- see `benchmarks/README.md`.

LMW came with a caveat BIBLIS did not, and it is worth stating on its own: a
specification can unblock the *deck* without supplying the *answer*. LMW
gives the scenario exactly and leaves the published power curve outside the
file. It ships anyway, labelled as reported rather than compared, because a
scenario that runs is worth having while the comparison stays open. Closing
that comparison needs the published curve from a citable source, not one
written down from recollection.

Still open: LRA and TWIGL, static and transient, which KOMODO does not ship.
The same lesson applies -- look for a specification before writing a deck
from recollection.
