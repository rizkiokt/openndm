# Contributing to OpenNDM

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
pytest tests/python                  # Python suite, about 11 s
pytest tests/python -m "not slow"    # skip the mesh-refinement runs

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
