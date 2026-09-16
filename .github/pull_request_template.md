## What this changes

<!-- One or two sentences. The diff says what; say why. -->

## Type of change

- [ ] `fix` — corrects behaviour that was wrong
- [ ] `feat` — adds behaviour
- [ ] `docs` — documentation only
- [ ] `test` — tests only
- [ ] `refactor` — no behaviour change, no number changes
- [ ] `chore` / `ci` / `build` — tooling, packaging, workflows

## Does this move any number?

A change to `k_eff`, a flux, a power distribution, a convergence order or a
benchmark result is a **physics change**, not a refactor, even when it looks
like a cleanup. See `CONTRIBUTING.md`.

- [ ] No verification or benchmark result moves.
- [ ] Some result moves, and the body below says which, by how much, and why
      the new number is the correct one.

<!-- If a number moved, paste the before/after here:

      case                     before         after        delta
      analytic bare cuboid     1.00123456     1.00123401   -0.55 pcm
      IAEA-2D k_eff            1.02958        1.02958       0.00 pcm
-->

## Checklist

- [ ] `pytest tests/python` passes locally.
- [ ] `ruff check python tests benchmarks` is clean.
- [ ] `clang-format` is clean on any touched C++ (`pre-commit run -a` covers both).
- [ ] New behaviour has a test. A new kernel is registered in `ALL_KERNELS`
      so the verification suite picks it up automatically.
- [ ] Public API changes are reflected in `docs/` and `CHANGELOG.md`.
- [ ] Commits follow conventional-commit subjects (`feat:`, `fix:`, ...).

## Related issues

<!-- "Closes #123", or delete this section. -->
