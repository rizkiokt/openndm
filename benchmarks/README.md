# OpenNDM benchmark decks

Each directory holds a runnable deck. Run one with `python run.py` from its
own directory.

| Deck | Type | Reference | Result | Status |
|---|---|---|---|---|
| `analytic/` | 1-group bare cuboid, 3D | Exact: `k = νΣf / (Σa + D B²)` | 0.05 pcm at 64 nodes/side | **Verified** |
| `iaea2d/` | 2-group PWR, quarter core, 2D | Published `k_eff = 1.02959` | SANM −3.7 pcm at one node per assembly | **Verified** |
| `iaea3d/` | 2-group PWR, quarter core, 3D | Published `k_eff = 1.02903` | SANM +44.7 pcm at 20 cm axial mesh | **Verified** |
| `biblis2d/` | 2-group PWR, quarter core, 2D, 8 compositions | Published `k_eff = 1.02511` | SANM −2.1 pcm at one node per assembly | **Verified** |
| `lmw/` | 2-group PWR rod transient, 3D | None in the specification | Power history, converged in mesh, not in time step | **Reported, not compared** |

The specification's acceptance criterion is 100 pcm on `k_eff` for a static
benchmark with a published solution. All four static decks meet it. LMW is a
transient and has no reference in the specification the deck was parsed from;
what it is good for, and what it is not, is set out below.

## analytic — exact reference

The only reference here that needs no other code and no transcription: the
eigenvalue of a bare homogeneous cuboid with zero-flux faces is known in
closed form. It fixes the spatial discretisation, the boundary treatment and
the eigenvalue iteration together.

```
nodes/side                     FDM                     NEM                    SANM
         4  1.20755774 (+217.04 pcm)  1.20743391 (+204.65 pcm)  1.20743465 (+204.73 pcm)
         8  1.20593766 ( +55.03 pcm)  1.20566957 ( +28.22 pcm)  1.20566959 ( +28.22 pcm)
        16  1.20552544 ( +13.81 pcm)  1.20542229 (  +3.49 pcm)  1.20542229 (  +3.49 pcm)
        32  1.20542193 (  +3.45 pcm)  1.20539167 (  +0.43 pcm)  1.20539167 (  +0.43 pcm)
        64  1.20539603 (  +0.86 pcm)  1.20538792 (  +0.05 pcm)  1.20538792 (  +0.05 pcm)
```

FDM converges at exactly second order, the error falling by 4× per
refinement. NEM and SANM agree with each other to 1e-8 throughout and are
about 4× more accurate than FDM at the same mesh, limited by the finite
difference treatment of the outer boundary faces rather than by the node
interior.

## iaea2d — published reference 1.02959

```
nodes/asm  nodes                       FDM                     NEM                    SANM
        1     75   1.033324 ( +373.4 pcm) 1.028412 ( -117.8 pcm) 1.029553 (   -3.7 pcm)
        2    300   1.029580 (   -1.0 pcm) 1.029507 (   -8.3 pcm) 1.029560 (   -3.0 pcm)
        4   1200   1.029088 (  -50.2 pcm) 1.029527 (   -6.3 pcm) 1.029528 (   -6.2 pcm)
        8   4800   1.029331 (  -25.9 pcm) 1.029528 (   -6.2 pcm) 1.029528 (   -6.2 pcm)
       16  19200   1.029470 (  -12.0 pcm) 1.029527 (   -6.3 pcm) 1.029527 (   -6.3 pcm)
       24  43200   1.029501 (   -8.9 pcm) 1.029527 (   -6.3 pcm) 1.029527 (   -6.3 pcm)
```

The headline nodal result is the top-right cell. **SANM at one node per
assembly — a 20 cm mesh — lands 2.6 pcm from the mesh-converged eigenvalue
and 3.7 pcm from the published reference.** That is what a nodal method is
for, and it is what a broken transverse leakage, discontinuity factor
convention or two-node closure would destroy.

Two things in this table are worth reading carefully.

**SANM and NEM converge to 1.0295271 and stay there** from four nodes per
assembly onward, agreeing with each other to 0.08 pcm. The two kernels share
only the transverse leakage fit: NEM closes its two-node problem with quartic
polynomials, two moment equations and the coarse-mesh outer-face currents,
while SANM uses analytic basis functions and the node-average constraint
alone. They have no reason to agree to that precision unless both are right.

**FDM is non-monotone, and that is not a defect.** It starts 380 pcm high at
20 cm, crosses the converged value between 10 and 5 cm, and then approaches
it from below, reaching second order asymptotically (the error ratio over the
last two refinements gives an observed order of 1.9). The crossover is two
error terms of opposite sign: the 20 cm reflector is a single node on the
coarsest mesh and badly under-resolved. Filling the out-of-core corner
positions with reflector, which makes the domain a convex square, reproduces
the same curve, so the re-entrant corners of the staircase boundary are not
the cause.

## iaea3d — published reference 1.02903

```
 radial     dz   nodes                       FDM                     NEM                    SANM
      1   20.0    1425   1.031928 ( +289.8 pcm) 1.028038 (  -99.2 pcm) 1.029477 (  +44.7 pcm)
      1   10.0    2700   1.031776 ( +274.6 pcm) 1.028439 (  -59.1 pcm) 1.029347 (  +31.7 pcm)
      1    5.0    5250   1.031719 ( +268.9 pcm) 1.028453 (  -57.7 pcm) 1.029341 (  +31.1 pcm)
      2   10.0   10800   1.029138 (  +10.8 pcm) 1.029317 (  +28.7 pcm) 1.029390 (  +36.0 pcm)
      2    5.0   21000   1.029062 (   +3.2 pcm) 1.029330 (  +30.0 pcm) 1.029383 (  +35.3 pcm)
```

FDM stays near +270 pcm down the first three rows because those refine only
axially: the radial mesh is still one node per assembly, and no amount of
axial refinement fixes a radial discretisation error. Subdividing radially
brings it to +3 pcm.

### Two things this deck gets right that are easy to get wrong

**The 80 cm is the height of the rod tips above the bottom of the core, not
an insertion depth measured down from the top.** The rods occupy the upper
260 cm of the 340 cm core, from z = 100 cm to z = 360 cm. Reading it the
other way round — rods only in the top 80 cm — leaves `k_eff` about 1700 pcm
high, and nothing else about the solution looks wrong: the axial profile is
still a plausible cosine, the power distribution is still physical, and the
eigenvalue is the only symptom. `tests/python/test_benchmarks.py` asserts the
orientation directly against the node compositions.

**The axial mesh puts the rod tip exactly on a plane boundary.** Smearing a
rod tip across a node is worth tens of pcm and shows up as an axial mesh that
refuses to converge monotonically, which is easy to mistake for a solver
problem. `axial_mesh()` chooses the plane counts for the rodded and unrodded
sections separately so the tip always lands on a boundary, for any requested
node height.

## The core map, and how it was checked

The quarter-core radial map lives in `benchmarks/common.py`. It has two
structural invariants that `check_radial_map()` asserts and the test suite
enforces:

1. It is symmetric about the diagonal.
2. The peripheral fuel-1 band is **edge-connected** — a continuous one-cell
   ring following the core outline.

Both exist because the first transcription of this deck violated them, and
neither violation had any symptom other than the eigenvalue:

- One position pair disagreed across the diagonal, `(5,8)` against `(8,5)`.
- One cell of the fuel-1 band was missing at `(5,5)`, which broke the ring
  into two disconnected arcs and left rows 4, 5 and 6 with 2, 1 and 2 band
  cells where the taper requires 2, 2 and 2.

Restoring the missing cell moved `k_eff` by +75 pcm, from 81 pcm below the
published reference to 6 pcm below it. A single assembly is worth roughly
100 pcm in a core this size, which is exactly the scale of a dropped cell —
and exactly small enough to be mistaken for a solver problem.

## BIBLIS-2D — published reference 1.02511

A 9x9 quarter core of 23.1226 cm assemblies over eight compositions, with
half-width assemblies on the two symmetry faces. The deck's assembly
divisions make the node mesh uniform at 11.5613 cm, so it runs at 17x17.

```
nodes/asm  nodes                       FDM                     NEM                    SANM
        1    514   1.028516 ( +340.6 pcm) 1.024987 (  -12.3 pcm) 1.025089 (   -2.1 pcm)
        2   2056   1.025826 (  +71.6 pcm) 1.025104 (   -0.6 pcm) 1.025107 (   -0.3 pcm)
        4   8224   1.025243 (  +13.3 pcm) 1.025109 (   -0.1 pcm) 1.025109 (   -0.1 pcm)
```

All three kernels converge on the published eigenvalue to 0.1 pcm, and FDM
gets there at second order — the error falls by roughly a factor of five per
refinement. SANM is 2.1 pcm out at one node per assembly.

### This deck was attempted once before and not shipped

The first attempt was written from memory. It was wrong twice over, and the
record is worth keeping because it is the case for not fitting.

The map used five of the eight defined compositions, and put fuel where the
reflector belongs. Relabelling the peripheral band moved the eigenvalue from
+253 pcm to -194 pcm without fixing the unused compositions, so neither
reading was right. All three kernels agreed with each other throughout, so
the solver was never implicated — the deck was wrong.

**And the reference was wrong too.** It was written against 1.02513. The
source deck states 1.02511.

So searching for a map that reproduced 1.02513 would have fitted a wrong
core to a wrong number, and the result would have been indistinguishable
from a pass. An 8x8 map over eight compositions has far too many degrees of
freedom for that search to mean anything — unlike IAEA-2D, where the
candidate was a single cell, the alternatives were physically incoherent,
and the fix restored a visible structural regularity.

What closed it was the specification, not more effort: the deck is now
parsed from its source rather than transcribed, and lands 2.1 pcm out on the
first run with nothing tuned.

### Provenance

`biblis_data.py` is generated, not hand-written, from the KOMODO sample deck:

| | |
|---|---|
| Source | `smpl/static/BIBLIS` in [imronuke/KOMODO](https://github.com/imronuke/KOMODO) |
| Commit | `b70d4ee262dcbd2547993302f6f86bef6d4dca00` |
| sha256 | `0f1e275579fd13a22b944fd33207e3fe0bf442696715bc0260d3b8ac11442c6f` |
| Licence | MIT |

KOMODO is a machine-readable transcription; the underlying data is the
published BIBLIS PWR benchmark. Two conventions had to be resolved when
reading it, both checked by tests rather than assumed:

**The map is printed north row first.** Its y indices run south to north and
the half-width assembly is the last y entry, so the printed order is
reversed on the way in. Getting this wrong puts the half-width assemblies on
the outer faces instead of the symmetry cuts, which changes the core size
without changing the node count.

**Boundary codes are 0 zero flux, 1 zero incoming current, 2 reflective.**
The deck's `1 2 2 1 2 2` makes east and south the outer faces, west and
north the symmetry cuts, and both axial faces reflective, which is what
makes the problem two-dimensional.

## lmw — a scenario without an answer

The LMW operational transient is the first deck that runs rod banks, the
cusping correction, delayed precursors and θ-weighted integration at once.
Two banks move against each other over 60 seconds: bank 2 withdraws from 100
steps at t = 0, bank 1 inserts from 180 steps at t = 7.5 s, both at 3 steps
per second. One step is 1 cm against a 5 cm axial mesh, so a rod tip sits
inside a node four steps out of five and cusping is active almost throughout.

The data is parsed from the KOMODO sample deck; see `lmw/lmw_data.py` for the
provenance. **The deck states the scenario in full and does not state its
answer.** The published LMW reference is a power-versus-time curve, and that
is not in the file. So this deck is not a comparison, and the numbers below
are reported rather than scored. The temptation is to write the reference
curve down from memory and call the agreement verification; the BIBLIS record
below is what that looks like when it goes wrong.

Steady state at the initial bank positions, SANM, one node per 10 cm:
`k_eff = 0.999553`, which is 45 pcm from critical — an operating core, as the
scenario requires.

Power relative to the steady state, SANM, θ = 0.5, banks placed at each
interval's midpoint:

```
      dt         5s        10s        20s        26s        30s        45s        60s
  1.0000   1.069233   1.212338   1.519106   1.504421   1.371651   0.676743   0.411153
  0.5000   1.091074   1.263382   1.608773   1.555924   1.385818   0.652353   0.401617
  0.2500   1.106658   1.297768   1.662396   1.581240   1.389587   0.641001   0.397705
  0.1250   1.116069   1.317963   1.691344   1.593226   1.390426   0.635640   0.396002
  0.0625   1.121255   1.328937   1.706300   1.598944   1.390572   0.633043   0.395213
```

The shape is the one the scenario is built to produce: the power rises while
only bank 2 is moving, peaks near 20 s as bank 1 overtakes it, and ends below
where it started, with the banks worth −299 pcm net.

### The time step is the error, and the mesh is not

Eight times the nodes — 4 680 to 37 440 — moves `k_eff` by 1 pcm and the peak
by 0.06%:

```
 sub   nodes     k_eff        5s       10s       20s       30s       45s       60s
   1    4680  0.999553  1.106658  1.297768  1.662396  1.389587  0.641001  0.397705
   2   37440  0.999563  1.107148  1.299033  1.663423  1.385590  0.635054  0.392932
```

The time step is a different matter. The successive differences at the peak
are 0.0897, 0.0536, 0.0289, 0.0150 — halving, which is **first order** — so
the extrapolated peak is about 1.721 and the deck's own 0.25 s step sits
about 3.4% below it.

This is not a defect in the scheme. θ = 1/2 is second order, and measured at
2.05 on this core when the step resolves the prompt time constant of about a
millisecond. An operational transient runs two hundred times coarser than
that, which is the stiff regime where the θ method loses an order; §11.1 of
`docs/theory.md` has the measurement. It is the quantitative case for the
exponential flux transformation, which is the piece of FR-KIN-2 that is not
implemented, and this is the first number that says what its absence costs.

Two approximations sit underneath and are smaller than the above: the
nonlinear coupling coefficient D-hat is frozen at its static value for the whole
transient, and the cusping mixture is re-weighted against the previous step's
flux rather than the current one. Turning the flux weighting off entirely, or
switching from SANM to FDM, changes the observed convergence not at all.

## Adding a deck

A deck is a directory with a `run.py` that imports shared data from
`benchmarks/common.py`, prints one summary line per kernel through
`common.report`, and states its reference value as a module constant. Add a
regression test in `tests/python/test_benchmarks.py` asserting against that
reference with an explicit tolerance. If a deck does not reproduce its
reference, say so in the table above with the number you actually get, rather
than quietly loosening the tolerance.

**Give any generated data module a name of its own** -- `biblis_data.py`,
`lmw_data.py` -- rather than `data.py`. Each deck puts its own directory on
`sys.path`, so two decks sharing a module name means the second one silently
imports the first one's data. Both of these were called `data.py` until they
existed at the same time, at which point the second deck to be imported got
the first deck's core map. `tests/python/test_benchmarks.py` already loads
each `run.py` under a unique name for the same reason.
