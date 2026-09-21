# CLAUDE.md

Conventions for this repository. They apply to every change, human or agent.

## The rule that differs from most projects

**Source code carries no comments.** No `#` in Python, no `//` or `/* */` in
C++. If a line needs explaining, the explanation belongs in a name, a smaller
function, or a docstring — not in a comment beside it.

Docstrings are not comments and are required. See *Docstrings* below.

Where the "why" goes instead:

| Kind of knowledge | Where it belongs |
|---|---|
| What this does, what it takes, what it returns | Docstring |
| Why the code is shaped this way | Commit message body |
| Why a numerical choice was made | `docs/theory.md`, and the commit |
| Why the implementation departs from the spec | `docs/architecture.md` |
| Why a change was accepted | Pull request description |

A comment is invisible to anyone reading the rendered docs and rots silently
when the code moves. A commit message is permanent, attached to the change
that made it true, and reachable from `git blame`.

This rule covers `python/`, `src/`, `include/`, `tests/` and `benchmarks/`.
Configuration files — `.github/workflows/*.yml`, `.pre-commit-config.yaml`,
`pyproject.toml`, `conda-recipe/meta.yaml`, `.clang-format` — may carry
comments, because a pinned version or a skipped platform has nowhere else to
record its reason.

### Making code self-explanatory

Replace a comment by extracting a named thing:

```python
if bank.position <= 0 or bank.position >= bank.n_steps:
    ...
```

becomes

```python
if bank.is_fully_inserted() or bank.is_fully_withdrawn():
    ...
```

Prefer a named constant to a literal with an explanation, a named intermediate
to a dense expression, and a small function to a block that needed a heading.

## Python

PEP 8, enforced by `ruff` (`line-length = 88`, target `py310`). Run
`ruff check --fix python tests benchmarks` before committing; CI runs the same
check and a pinned version.

- `snake_case` for functions, variables and modules; `CapWords` for classes;
  `UPPER_SNAKE` for module constants.
- A leading underscore marks the private surface. Everything public appears in
  the module's `__all__`.
- Type annotations on public signatures. `from __future__ import annotations`
  at the top of every module.
- Prefer a `@property` to a `get_*` method. Validate in the setter, so an
  invalid object cannot exist.
- Raise the project's own exceptions from `openndm.exceptions`, not bare
  `ValueError`, when the caller could plausibly catch it.
- NumPy arrays are the interchange type. Return views where the underlying
  buffer outlives the call, and say so in the docstring.

## C++

The Google C++ Style Guide, with the deviations recorded in `.clang-format`:
two-space indent, 80-column limit, pointers bound left, C++17. Run
`clang-format -i` on anything you touch; CI enforces it with a pinned version.

- `CamelCase` for types, `snake_case` for functions and variables, trailing
  underscore on private data members.
- Header guards use `#pragma once`.
- Anything that can be `const` is. Pass by `const&` unless the callee stores it.
- No raw owning pointers. `std::unique_ptr` for ownership, references or spans
  for borrowing.
- Public headers in `include/openndm/` carry Doxygen `//!` docstrings on every
  declaration. This is the one place `//` appears, and it is a docstring, not a
  comment.
- Keep the node/surface graph abstraction: kernels must never index `(i, j, k)`.

## Docstrings

NumPy style, on every public module, class, function and method. Rendered by
`sphinx.ext.napoleon`.

Keep them short. Document the contract, not the theory.

```python
def critical_spectrum(library, method="b1", max_iterations=50):
    """Search for the critical buckling of an infinite medium.

    Parameters
    ----------
    library : XSLibrary
        Finalised library holding a single composition.
    method : {'b1', 'p1'}, optional
        Leakage approximation.
    max_iterations : int, optional
        Iteration cap for the buckling search.

    Returns
    -------
    CriticalSpectrum
        Buckling in cm^-2, infinite-medium eigenvalue, and the
        leakage-corrected diffusion coefficient per group.

    Raises
    ------
    ConvergenceError
        If the search does not converge.
    """
```

- First line: one sentence, imperative, ending in a period.
- Every parameter gets its shape and units. `cm`, `cm^-1`, `pcm`, `K`.
- Say whether a returned array is a view onto a C++ buffer or a copy.
- Class attributes go in the class docstring, not `__init__`.
- Derivations go in `docs/theory.md`. Link to it, do not restate it.
- No multi-paragraph essays. If a docstring needs several paragraphs, the
  material belongs in `docs/`.

## Brevity

Shorter is better, but only after it is clear. A dense one-liner that needs a
second reading loses to three plain statements. Never sacrifice an explicit
name to fit a line limit, and never collapse a loop into a comprehension that
no longer reads as a sentence.

The test is whether a reader who knows reactor physics but not this codebase
can follow it at reading speed.

## Tests

- `pytest tests/python` via `python -m pytest`. Bare `pytest` may resolve to
  another interpreter's copy.
- A new kernel goes into `ALL_KERNELS` in `tests/python/conftest.py`, which
  parameterises the verification suite over it automatically.
- A change that moves `k_eff`, a flux, a power distribution or a convergence
  order is a physics change. It needs before/after numbers in the pull request,
  whatever the diff looks like.

## Development environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install cmake ninja pybind11 scikit-build-core
pip install --no-build-isolation -e '.[dev]'
```

Never `--system-site-packages`. It lets an Anaconda installation satisfy
imports the project never declared, so undeclared dependencies pass locally and
fail in CI.

## Writing

Commit messages, pull requests, issues and code review comments are written
plainly. Short, specific, no fluff.

- Say what changed and why. Skip the background the reader already has.
- One or two sentences is usually enough. Add detail only when it changes what
  someone does.
- No restating the diff, no summarising what you just wrote, no closing
  paragraph tying it together.
- Tables and headings only when they genuinely help. A three-line PR body is a
  good PR body.
- Numbers, file paths and error text over adjectives.

A commit body exists so someone running `git blame` learns something the code
cannot tell them. If there is nothing like that, the subject line alone is
fine.

## Commits and branches

- One branch, one commit, one concern. Amend rather than stacking fixups.
- Conventional commit subjects: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`,
  `chore:`, `build:`, `ci:`, optionally scoped (`feat(core):`).
- `main` is protected: pull request, green `ci` check, and code-owner approval.
