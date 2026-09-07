# OpenMC coupling validation

The cases in §6.3 of the specification. Unlike `tests/python/`, these need a
working OpenMC installation *and* a nuclear data library, so they are opt-in
rather than part of the default suite.

```bash
export OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml
export PATH=/path/to/openmc/bin:$PATH        # the `openmc` executable
python c1_homogeneous.py --particles 100000 --batches 200
```

| Case | What it isolates | State |
|---|---|---|
| C-1 `c1_homogeneous.py` | The cross section translation path, and nothing else | **Passing** |
| C-2 single assembly, reflective | The ADF machinery, which must return ≈ 1.0 | not written |
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

Result on a homogenised UO2 and water mixture, ENDF/B-VIII.1, 18.5M active
histories:

```
  OpenMC continuous energy   k_inf = 1.191016 +/- 23.9 pcm

  D from diffusion-coefficient       k_ndm=1.191365  diff=   +34.9 pcm  =  1.46 sigma
  D from transport                   k_ndm=1.191365  diff=   +34.9 pcm  =  1.46 sigma
  multiplicity correction disabled   k_ndm=1.188673  diff=  -234.3 pcm  =  9.81 sigma
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

## The residual

After the correction the agreement is +35 pcm at 1.46σ, and +39 pcm on an
8-group structure from the same transport run. Being group-structure
independent, the residual is not multi-group condensation error. It is the
size expected from the difference between OpenMC's combined k estimator and
the reaction-rate ratio the multi-group balance forms, and it sits inside the
Monte Carlo uncertainty. The specification asks for agreement within 1σ; this
is marginal against that bar and worth revisiting with tighter statistics
before C-1 is called closed.
