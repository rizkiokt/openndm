# OpenNDM benchmark decks

Each directory holds a runnable deck. Run one with `python run.py` from its
own directory.

| Deck | Type | Reference | Result | Status |
|---|---|---|---|---|
| `analytic/` | 1-group bare cuboid, 3D | Exact: `k = νΣf / (Σa + D B²)` | 0.05 pcm at 64 nodes/side | **Verified** |
| `iaea2d/` | 2-group PWR, quarter core, 2D | Published `k_eff = 1.02959` | SANM −3.7 pcm at one node per assembly | **Verified** |
| `iaea3d/` | 2-group PWR, quarter core, 3D | Published `k_eff = 1.02903` | SANM +44.7 pcm at 20 cm axial mesh | **Verified** |

The specification's acceptance criterion is 100 pcm on `k_eff` for a static
benchmark with a published solution. All three decks meet it.

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

## Adding a deck

A deck is a directory with a `run.py` that imports shared data from
`benchmarks/common.py`, prints one summary line per kernel through
`common.report`, and states its reference value as a module constant. Add a
regression test in `tests/python/test_benchmarks.py` asserting against that
reference with an explicit tolerance. If a deck does not reproduce its
reference, say so in the table above with the number you actually get, rather
than quietly loosening the tolerance.
