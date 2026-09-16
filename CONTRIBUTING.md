# Contributing to OpenNDM

OpenNDM was initiated and is developed by **Rizki Oktavian, PhD**
(<rizkiokt@gmail.com>). Contributions are welcome — bug reports, benchmark
cases, kernels, documentation. If you are considering something substantial,
open an issue or email first, so the design discussion happens before you have
written the code rather than in review.

By contributing you agree that your contribution is licensed under the MIT
licence, and you agree to the [Code of Conduct](https://github.com/rizkiokt/openndm/blob/main/CODE_OF_CONDUCT.md).

---

## How a change gets merged

**`main` is protected. Nothing is pushed to it directly, including by the
maintainer.** Every change arrives as a pull request, passes CI, and is
approved by the maintainer before it can merge. This is enforced by a
repository ruleset, not by convention.

Concretely, a pull request cannot merge until:

| Requirement | What satisfies it |
|---|---|
| It is a pull request | Direct pushes to `main` are rejected |
| CI is green | The `ci` check — lint, the test matrix, C++ tests, the no-OpenMC import, docs |
| The maintainer approved it | Code owner review from [@rizkiokt](https://github.com/rizkiokt), per [`.github/CODEOWNERS`](https://github.com/rizkiokt/openndm/blob/main/.github/CODEOWNERS) |
| The approval is current | Pushing new commits dismisses stale approvals |
| History stays clean | Force pushes to `main` and deletion of `main` are blocked |

### The steps

1. **Fork**, or branch if you have write access. Branch names are free-form;
   `fix/leakage-sign` and `feat/hex-geometry` read well.

   ```bash
   git switch -c fix/leakage-sign
   ```

2. **Install the pre-commit hooks.** They run the same linters CI does, plus a
   secret scan and a check that no local filesystem path is baked into a
   notebook's committed output. Catching these locally is much cheaper than in
   review:

   ```bash
   pip install pre-commit
   pre-commit install
   ```

3. **Make the change, with a test.** See [Running the tests](#running-the-tests).

4. **Commit** using conventional-commit subjects — see [Commits](#commits).

5. **Open the pull request.** The template asks one question that matters more
   than the rest: *does this move any number?* Answer it honestly. A change to
   `k_eff`, a flux, a power distribution or a convergence order is a physics
   change, not a refactor, even when the diff looks like a cleanup.

6. **Respond to review.** Push follow-up commits rather than force-pushing, so
   the review history stays readable. The branch is squashed on merge, so the
   intermediate commits do not end up in `main`.

7. **Merge.** The maintainer merges once the checks and the approval are in
   place. Your branch is deleted automatically.

### For the maintainer

The ruleset grants repository admins a bypass, because GitHub does not let
anyone approve their own pull request and a sole maintainer would otherwise be
unable to merge their own work. The bypass exists for that case only: the
pull request, the CI run and the review still happen, and using the bypass to
skip a red CI run defeats the point of having the rule.

---

## Getting a development build

OpenNDM is a C++17 core with pybind11 bindings. You need a C++17 compiler;
CMake, ninja and pybind11 arrive through the build requirements.

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install cmake ninja pybind11 scikit-build-core
pip install --no-build-isolation -e '.[dev]'
```

`--no-build-isolation` with an editable install means a C++ change is picked up
by re-running the `pip install` line, without re-resolving the build
environment each time.

## Running the tests

```bash
# `python -m pytest` rather than bare `pytest`: on a machine with Anaconda on
# PATH, the bare command often resolves to Anaconda's pytest, which runs
# against a different interpreter and cannot see the venv's openndm.
python -m pytest tests/python                  # Python suite, about 11 s
python -m pytest tests/python -m "not slow"    # skip the mesh-refinement runs

cmake -S . -B build -DOPENNDM_BUILD_TESTS=ON -DOPENNDM_BUILD_PYTHON=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The verification suite in `tests/python/test_verification.py` is the part that
matters. It checks the analytic bare-cuboid eigenvalue, the spatial
convergence order of each kernel, agreement between the coarse-mesh nodal
kernels and a refined finite difference solve, and adjoint/forward
reciprocity. **A change that moves any of those numbers is a physics change,
not a refactor**, and needs an explanation in the commit message.

## Style

- C++ follows OpenMC's `.clang-format` (DP-1). Run
  `clang-format -i include/openndm/*.h src/*.cpp src/*.h`.
- Python follows `ruff`. Run `ruff check --fix python tests benchmarks`.
- Both are enforced in CI (NFR-QA-4).

## Where things live

```
include/openndm/   public C++ headers
src/               C++ implementation, plus the pybind11 bindings
python/openndm/    the Python package
  gc/              OpenMC group constant generation, the only part needing OpenMC
tests/cpp/         Catch2 unit tests
tests/python/      pytest suite
benchmarks/        runnable decks, one directory each
docs/              specification, status, theory, architecture
```

`docs/architecture.md` records every place the implementation departs from the
specification, with the reason. Read it before proposing a change that looks
like it contradicts `docs/requirements.md`.

## Adding a solver kernel

1. Implement `openndm::Kernel` in `src/kernel_<name>.cpp`. The only method that
   matters is `solve()`, which answers "given the coarse-mesh node averages,
   the eigenvalue and the transverse leakage, what is the net current across
   this interface?"
2. Register it in `Kernel::create` and add a `KernelType` value.
3. Expose the name in `python/openndm/settings.py`.
4. Add it to `ALL_KERNELS` in `tests/python/conftest.py`. The verification
   suite is parameterised over that list, so a new kernel is immediately held
   to the analytic solution, the convergence order and the fine-mesh
   consistency check.

Kernels must operate on the node/surface graph, never on `(i, j, k)` indices
(NFR-EXT-1). That constraint is what keeps hexagonal geometry a geometry-module
addition.

## Commits

Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`,
optionally scoped (`feat(core):`). Explain *why* in the body, especially for a
numerical change; the diff already says what.

## Versioning

Semantic versioning (NFR-QA-5). The Python API does not break without a
deprecation cycle. Bump the version in `pyproject.toml` and `CMakeLists.txt`
together and add a `CHANGELOG.md` entry.
