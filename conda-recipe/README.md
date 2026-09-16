# conda-forge recipe

This is the recipe OpenNDM is packaged with on conda-forge, kept here so it is
reviewable alongside the code it builds. **conda-forge does not build from this
copy.** It builds from `recipe/meta.yaml` in `conda-forge/openndm-feedstock`.

Keep the two in step. This copy is what the next person reads to understand how
OpenNDM is packaged, and a stale copy is worse than no copy.

## Why conda-forge

OpenMC is distributed through conda-forge and not through PyPI. A user who
wants the `openndm.gc` coupling installs both from the same channel and gets
one consistent set of compilers, HDF5 and OpenMP runtimes. Mixing a pip-built
OpenNDM with a conda OpenMC is the configuration most likely to produce a
confusing linker or HDF5 version error.

## Submitting it the first time

The full procedure, including the checksum step, is in
[`docs/releasing.md`](../docs/releasing.md#conda-forge). In short: copy this
file into `recipes/openndm/meta.yaml` in a fork of
[conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes),
fill in the `sha256` of the sdist that PyPI is serving, and open a pull
request.

The `sha256` here is deliberately left as zeros. It must be the checksum of the
**published** sdist, and a placeholder that looks plausible is more dangerous
than one that obviously is not.

## Building it locally

```bash
conda install -n base conda-build
conda build conda-recipe/ -c conda-forge
```

That runs the same import and smoke-test commands conda-forge will run,
including the analytic bare-cuboid eigenvalue check, which is what catches a
wrongly built extension before it reaches a user.
