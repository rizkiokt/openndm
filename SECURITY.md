# Security policy

## Supported versions

OpenNDM is pre-1.0. Fixes land on `main` and go out in the next release; there
is no backport branch. Report against the latest release or `main`.

| Version | Supported |
|---|---|
| 0.1.x | yes |
| < 0.1 | no |

## Reporting a vulnerability

Use GitHub's private reporting:
[**Report a vulnerability**](https://github.com/rizkiokt/openndm/security/advisories/new).
That opens a private advisory visible only to the maintainer. Do not open a
public issue for a security problem.

If private reporting is unavailable to you, email Rizki Oktavian at
<rizkiokt@gmail.com> with `OPENNDM SECURITY` in the subject.

Expect an acknowledgement within a week. If a report is confirmed, the advisory
is where the fix and the disclosure timeline get coordinated, and you will be
credited in it unless you ask otherwise.

## What counts as a vulnerability here

OpenNDM is a numerical library. It does not listen on a socket, authenticate
anyone, or run with elevated privileges. The realistic attack surface is the
parsing of files a user was handed by someone else:

- **Memory safety in the C++ core** reachable from Python input — an
  out-of-bounds read or write, a use-after-free, or an integer overflow in
  sizing, triggered by a crafted geometry, cross section library or statepoint.
  This is the category that matters most.
- **HDF5 statepoint and library loading** — a malformed or hostile `.h5` file
  causing anything worse than a clean exception.
- **Deserialisation of a branch library** that executes code or writes outside
  the intended path.
- **A dependency advisory** that OpenNDM actually exposes to input.

## What does not

- A wrong answer, a failure to converge, or a benchmark disagreement. Those are
  bugs, sometimes serious ones — file them as a
  [physics discrepancy](https://github.com/rizkiokt/openndm/issues/new?template=physics_discrepancy.yml)
  in the open.
- A crash on input the caller constructed themselves. Passing a negative
  diffusion coefficient is a usage error; a clean exception is the correct
  behaviour and an assertion failure is an ordinary bug.
- Resource exhaustion from a legitimately large model.
- Anything requiring the attacker to already be able to run code as you.

## A note on results

Nothing in this project is qualified for licensing or safety analysis. It
carries no nuclear quality assurance pedigree — no NQA-1 program, no design
control, no verification and validation package of the kind a regulator
expects. The verification suite in `tests/python/test_verification.py` and the
benchmarks establish that the equations in `docs/theory.md` are solved
correctly; they establish nothing about fitness for a licensing basis. See the
disclaimer in the `LICENSE` and use accordingly.
