# OpenNDM benchmark decks

Each directory holds a runnable deck. Run one with `python run.py` from its
own directory.

**Read the status column before quoting a result.** One of these decks is
verified against a reference that is exact. The IAEA decks are transcribed
from the published problem and do *not* yet reproduce their published
eigenvalues; they ship because they exercise the code and because the
discrepancy is worth closing, not because they pass.

| Deck | Type | Reference | Status |
|---|---|---|---|
| `analytic/` | 1-group bare cuboid, 3D | Exact: `k = νΣf / (Σa + D B²)` | **Verified.** 0.05 pcm at 64 nodes/side, second-order convergence on all three kernels. |
| `iaea2d/` | 2-group PWR, quarter core, 2D | Published `k_eff = 1.02959` | **Not reproduced.** SANM gives 1.028753, −84 pcm. See below. |
| `iaea3d/` | 2-group PWR, quarter core, 3D | Published `k_eff = 1.02903` | **Not reproduced.** SANM gives 1.045729, +1670 pcm. See below. |

## What the analytic deck establishes

This is the only reference here that needs no other code and no transcription:
the eigenvalue of a bare homogeneous cuboid with zero-flux faces is known in
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

FDM converges at exactly second order: the error falls by 4× per refinement.
NEM and SANM agree with each other to 1e-8 throughout and are roughly 4× more
accurate than FDM at the same mesh, limited by the finite difference treatment
of the outer boundary faces rather than by the node interior.

## What the IAEA decks do and do not establish

The solver's internal consistency on the IAEA-2D core map *is* verified, and
that is a stronger statement about the nodal kernels than the eigenvalue
comparison would be:

| Mesh | FDM | SANM | NEM |
|---|---|---|---|
| 1 node/assembly | 1.032610 | 1.028753 | 1.027552 |
| 8 nodes/assembly | 1.028451 | 1.028581 | 1.028581 |

All three kernels converge to the same fine-mesh limit, 1.02858, and SANM on
**one node per assembly** lands 18 pcm from it. That is precisely what a nodal
method is for, and a broken transverse leakage, discontinuity factor
convention or two-node closure would show up here immediately. This comparison
is what `tests/python/test_verification.py::test_v3_*` enforces in CI.

What is *not* established is that the deck is the published problem. The
converged eigenvalue sits about 100 pcm below the published 1.02959, and the
3D extrusion is 1670 pcm above its published value. The likeliest causes, in
order:

1. **The radial core map.** It is transcribed from the published problem
   (ANL-7416, Problem 11-A2). A scan over plausible alternative maps put every
   other candidate 1000–5000 pcm away, so this one is close to right, but one
   or two assembly assignments may still be wrong.
2. **The 3D axial rod model.** `iaea3d/run.py` inserts every rodded position
   to a uniform 80 cm from the top of the core. The published problem uses
   more than one rod group with different insertions, which the deck does not
   currently represent. This is the most likely single cause of the 3D gap.
3. **The outer boundary convention.** Zero flux at the reflector surface
   versus zero incoming current moves `k_eff` by about 20 pcm. Both are
   available through `boundaries=` and `outside=`; zero flux is used here.

Closing these needs the reference specification, not more computation. Until
then, treat the IAEA numbers as regression baselines for this code, not as
benchmark agreement.

## Adding a deck

A deck is a directory with a `run.py` that imports shared data from
`benchmarks/common.py`, prints one summary line per kernel through
`common.report`, and states its reference value as a module constant. If the
deck reproduces its reference, add a regression test in
`tests/python/test_benchmarks.py` with an explicit tolerance. If it does not,
say so in the table above with the number you actually get.
