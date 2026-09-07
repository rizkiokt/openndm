# OpenMC coupling validation

The cases in §6.3 of the specification. Unlike `tests/python/`, these need a
working OpenMC installation *and* a nuclear data library, so they are opt-in
rather than part of the default suite.

**These do not run on every push.** There is no stable public URL for a
nuclear data library to hard-code into CI: the official OpenMC libraries are
served from rotating Box links, and the page listing them is currently a 404.
The `openmc-coupling` job therefore runs only on a manual workflow dispatch,
and only when given a `cross_sections_url` to fetch from. A job that silently
skipped when it could not find data would be worse than no job at all,
because the silence would be indistinguishable from success.

What *does* run on every push is `tests/python/test_gc.py`, 45 tests covering
the translation logic against stand-ins that enforce OpenMC's real API
contracts, and `tests/python/test_openmc_integration.py`, which runs a small
C-1 and C-2 whenever OpenMC, its executable and a data library all happen to
be present.

```bash
export OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml
export PATH=/path/to/openmc/bin:$PATH        # the `openmc` executable
python c1_homogeneous.py --particles 100000 --batches 200
```

| Case | What it isolates | State |
|---|---|---|
| C-1 `c1_homogeneous.py` | The cross section translation path, and nothing else | **Passing** |
| C-2 `c2_assembly_adf.py` | The ADF tallies and their ratio arithmetic | **Passing** |
| C-3 colorset | Homogenisation error with and without ADFs | not written |
| C-4 small full core | Node-wise mesh-domain MGXS, FR-OMC-12 | not written |
| C-5 branch round trip | Interpolation against a directly computed state point | not written |
| C-6 uncertainty propagation | Induced σ on k_eff and peak power | not written |

## C-1, and what it found

A uniform infinite medium is run in OpenMC with continuous-energy physics.
The multi-group cross sections tallied from that same run are handed to
OpenNDM, which solves a single fully reflected node. There is no spatial
discretisation, no leakage, no homogenisation error and no discontinuity
factor, so any disagreement beyond the Monte Carlo uncertainty is a defect in
the translation.

Result on a homogenised UO2 and water mixture, ENDF/B-VIII.1, 90M active
histories:

```
  OpenMC continuous energy   k_inf = 1.191328 +/- 10.1 pcm

  D from diffusion-coefficient       k_ndm=1.191211  diff=   -11.7 pcm  =  1.16 sigma
  D from transport                   k_ndm=1.191211  diff=   -11.7 pcm  =  1.16 sigma
  multiplicity correction disabled   k_ndm=1.188532  diff=  -279.6 pcm  = 27.80 sigma
```

Writing this case found two defects that the stand-in-based unit tests could
not have found, because both are properties of OpenMC's actual API and
physics conventions rather than of any interface I could have guessed.

**`MGXS.get_xs` takes integer domain ids, not domain objects.**
`Library.get_mgxs` takes the object, which makes it natural to pass the same
object straight through. Real OpenMC then raises a `TypeError` from deep
inside its argument checking. This one fails loudly, so it cost only time.

**OpenMC's `absorption` score excludes (n,2n) and (n,3n).** Those reactions
destroy one neutron and create two or three; the extra neutrons appear only in
the row sums of the `nu-scatter matrix`, as an excess over the plain `scatter
matrix`. A diffusion operator built from a single scattering matrix cannot see
that production at all: summed over groups, the in-scatter and out-scatter
terms are the same double sum and cancel exactly, leaving
`k = Σ νΣf φ / Σ Σa φ` with the extra neutrons nowhere in it.

On this mixture the (n,2n) rate is 0.2258% of the absorption rate — 226 pcm of
production — against a measured bias of 234 pcm. Subtracting the multiplicity
excess from the absorption cross section, which is algebraically exact group
by group rather than merely in the global balance, closes it. OpenMC's own
tallied `nu-fission`/`absorption` ratio shows the same 244 pcm gap, confirming
this is physics the translation was dropping and not an artefact of the
solver.

This defect fails **silently**. Every other diagnostic looks healthy: the
scattering matrix orientation is right (transposing it costs 15700 pcm), the
computed spectrum matches OpenMC's to 0.02%, and the solver reproduces its own
input data to 0.00 pcm. Only the eigenvalue moves, by an amount that is easy
to attribute to statistics if the run is small enough. It is exactly the
category of error the specification's C-1 exists to catch.

## C-2, and what it found

A single assembly with reflective boundaries is an infinite lattice, so the
homogeneous solution of the homogenised assembly is flat and equal to its
volume average. That makes two things checkable with no reference calculation
at all:

```
homogeneous assembly: every factor must be 1.0
  face                  fast            thermal
  x_min    0.99922 +/-0.00062  1.00102 +/-0.00179
  x_max    1.00041 +/-0.00065  0.99989 +/-0.00178
  y_min    0.99976 +/-0.00064  0.99857 +/-0.00207
  y_max    1.00022 +/-0.00069  0.99757 +/-0.00202
  worst deviation from 1.0 = 0.00243 (1.18 sigma)

heterogeneous pin lattice: thermal factors must exceed 1.0
  x_min    0.99415 +/-0.00066  1.04603 +/-0.00194
  x_max    0.99345 +/-0.00068  1.04151 +/-0.00191
  y_min    0.99265 +/-0.00065  1.04689 +/-0.00190
  y_max    0.99389 +/-0.00065  1.04290 +/-0.00176
```

The homogeneous case verifies the slab geometry, the width normalisation and
the group ordering: a flat flux must give factors of exactly one, and it does,
to 1.2σ.

The heterogeneous case is the one that found a defect. **`compute_adf` was
returning its groups in increasing-energy order**, while
`from_mgxs_library` and the solver both put group 1 at the highest energy.
Every discontinuity factor was therefore being applied to the wrong group.

Nothing about the numbers looked wrong. They were plausible, symmetric across
opposite faces, and had sensible uncertainties. Only their *sense* was
inverted, and seeing that requires knowing what the answer should be: the
surface of a pin lattice sits in water, where the thermal flux peaks and the
fast flux does not, so the thermal factor must exceed one and the fast must
fall below it. The measured 1.045 thermal is a typical PWR assembly value.

The cause is that an `openmc.EnergyFilter` orders its bins by increasing
energy, while `MGXS.get_xs` returns decreasing. Two OpenMC APIs, two
conventions, and only a physical argument distinguishes them. The deck now
asserts the sense directly, and unit tests cover both orderings.

## The residual is statistical

At 18.5M histories the agreement was +34.9 pcm against σ = 23.9 pcm, close
enough to the 1σ bar to be worth settling. Raising the count to 90M shrank σ
to 10.1 pcm and moved the difference to **−11.7 pcm** — it changed sign. A
systematic bias does not do that. The residual is Monte Carlo noise, and C-1
agrees within 1.2σ.

The same run at 8 groups gives +38.9 pcm against the 18.5M reference, matching
the 2-group value, so the residual is also group-structure independent and
therefore not multi-group condensation error.

For contrast, the multiplicity defect went the other way as statistics
improved: −234 pcm at 9.8σ became −280 pcm at **27.8σ**. That is what a real
bias looks like.
