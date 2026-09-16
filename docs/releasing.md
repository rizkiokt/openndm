# Releasing

How a version of OpenNDM reaches users. There are three distribution channels
and they are strictly ordered: PyPI is published first, conda-forge builds from
what PyPI holds, and the GitHub release is the human-readable record of both.

Only the maintainer can complete a release: the `main` ruleset and the PyPI
trusted publisher are both bound to this repository, and the conda-forge
feedstock has its own maintainer list.

---

## Version numbering

Semantic versioning (NFR-QA-5). The Python API does not break without a
deprecation cycle.

The version lives in exactly two places and they must agree:

- `pyproject.toml` → `[project] version`
- `CMakeLists.txt` → `project(openndm VERSION ...)`

`openndm.__version__` comes from the C++ side at build time, so a mismatch
between the two shows up as a wheel whose metadata and runtime disagree. The
`release` workflow refuses to publish when they differ, which is the only
reason that check is not merely advisory.

---

## Cutting a release

1. **Land everything through pull requests.** `main` is protected; nothing is
   pushed to it directly. See [Contributing](contributing.rst).

2. **Bump the version** in the two files above, in one pull request, with the
   `CHANGELOG.md` entry in the same change. Move the `Unreleased` items under a
   new `## [X.Y.Z] - YYYY-MM-DD` heading.

3. **Check the verification numbers are the ones you mean to ship.** The
   suite passes on every pull request, but a release is the moment to read the
   numbers rather than the green tick:

   ```bash
   pytest tests/python -v
   cd benchmarks/analytic && python run.py
   ```

4. **Merge, then tag the merge commit.** The tag drives everything downstream;
   it is not a label applied afterwards.

   ```bash
   git switch main && git pull
   git tag -a v0.2.0 -m "OpenNDM 0.2.0"
   git push origin v0.2.0
   ```

5. **Watch the `release` workflow.** On a `v*` tag it builds the sdist and the
   wheels for CPython 3.10–3.13 on Linux x86-64/aarch64 and macOS
   x86-64/arm64, verifies the tag matches the packaged version, publishes to
   PyPI through trusted publishing, and opens the GitHub release with the
   artifacts attached.

6. **Update the conda-forge feedstock** once PyPI has the release. See below.

---

## PyPI

Publishing uses [trusted publishing](https://docs.pypi.org/trusted-publishers/),
so there is no API token anywhere in the repository or in the Actions secrets.
PyPI is configured to accept uploads that originate from this repository's
`release.yml` workflow running in the `pypi` environment, and from nothing else.

One-time setup, already done:

- On PyPI, under the project's *Publishing* settings, a GitHub publisher with
  owner `rizkiokt`, repository `openndm`, workflow `release.yml`, environment
  `pypi`.
- In this repository, an Actions environment named `pypi`. Restricting it to
  tags means a workflow run from a branch cannot publish even if it is made to
  run.

If a release fails partway, **do not delete and re-push the tag**. PyPI refuses
to accept a second upload of a version that already exists, even a deleted one.
Bump the patch version and release again.

---

## conda-forge

OpenNDM is distributed on conda-forge, which is also where OpenMC lives — a
user who wants the OpenMC coupling installs both from the same channel and
gets one consistent set of compilers and HDF5 libraries.

conda-forge builds from the **PyPI sdist**, not from the git tag, so step 5
above must have finished before any of this.

### The first submission

The recipe in [`conda-recipe/meta.yaml`](https://github.com/rizkiokt/openndm/blob/main/conda-recipe/meta.yaml)
is ready to submit; it has not yet been through staged-recipes.

```bash
# 1. Fork and clone conda-forge/staged-recipes
git clone git@github.com:<you>/staged-recipes.git
cd staged-recipes

# 2. Copy the recipe in
mkdir -p recipes/openndm
cp /path/to/openndm/conda-recipe/meta.yaml recipes/openndm/

# 3. Fill in the sha256 of the sdist PyPI now holds
python -c "
import hashlib, urllib.request
url = 'https://pypi.io/packages/source/o/openndm/openndm-0.1.0.tar.gz'
print(hashlib.sha256(urllib.request.urlopen(url).read()).hexdigest())"

# 4. Open a pull request against conda-forge/staged-recipes
```

Review typically takes a few days. Once merged, conda-forge creates
`conda-forge/openndm-feedstock` and grants the listed maintainers write access.

### Every release after that

The feedstock's bot notices the new PyPI sdist within a few hours and opens a
pull request that bumps `version` and `sha256` and resets the build number.
Merge it once its CI is green. Only intervene by hand when the dependencies or
the build itself changed, in which case edit `recipe/meta.yaml` on a branch of
the feedstock and open a pull request there.

Keep `conda-recipe/meta.yaml` in this repository in step with the feedstock.
It is not what conda-forge builds from, but it is what the next person reads
to understand how OpenNDM is packaged, and a stale copy is worse than none.

---

## Documentation

Read the Docs builds on every push to `main` and on every tag, from
`.readthedocs.yaml`. `latest` tracks `main` and `stable` tracks the most recent
tag. Nothing needs doing at release time beyond confirming the new version
appears in the version selector.

---

## Checklist

```
[ ] Version bumped in pyproject.toml and CMakeLists.txt, and they agree
[ ] CHANGELOG.md has a dated section for this version
[ ] pytest tests/python passes, and the verification numbers were read
[ ] Version bump merged to main through a pull request
[ ] Tag v<version> pushed, release workflow green
[ ] PyPI shows the sdist and every expected wheel
[ ] GitHub release published with notes
[ ] conda-forge feedstock pull request merged
[ ] Read the Docs shows the new version as stable
```
