#!/usr/bin/env python3
"""Reject notebooks whose committed outputs carry a local absolute path.

The examples are committed with their outputs so they read without being run.
That convenience means every cell's captured stdout, every warning and every
traceback is published verbatim, and those routinely contain the author's home
directory, conda prefix and data paths. None of it is a credential, but it is
noise at best and an unwanted disclosure of a machine's layout at worst, and
it is invisible in review because the diff of an executed notebook is mostly
opaque JSON.

Run by pre-commit over any staged .ipynb. Exits non-zero and names the cell.
"""

from __future__ import annotations

import json
import re
import sys

# Home directories and Windows user profiles. `/home/runner` is what GitHub
# Actions uses, so a notebook executed in CI is allowed to say it.
PATTERNS = [
    re.compile(r"/home/(?!runner\b)[A-Za-z0-9._-]+"),
    re.compile(r"/Users/[A-Za-z0-9._-]+"),
    re.compile(r"[Cc]:\\\\Users\\\\[A-Za-z0-9._-]+"),
]


def _outputs_text(cell: dict) -> list[tuple[str, str]]:
    """Every string an output cell will render, tagged with where it came from."""
    found = []
    for output in cell.get("outputs", []):
        for key in ("text",):
            value = output.get(key)
            if value is not None:
                found.append((key, "".join(value)))
        for mime, value in (output.get("data") or {}).items():
            if isinstance(value, list):
                value = "".join(value)
            if isinstance(value, str):
                found.append((mime, value))
        if "traceback" in output:
            found.append(("traceback", "".join(output["traceback"])))
    return found


def check(path: str) -> list[str]:
    try:
        notebook = json.load(open(path, encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return [f"{path}: could not be parsed as a notebook: {exc}"]

    problems = []
    for index, cell in enumerate(notebook.get("cells", [])):
        sources = [("source", "".join(cell.get("source", [])))]
        sources += _outputs_text(cell)
        for where, text in sources:
            for pattern in PATTERNS:
                for hit in set(pattern.findall(text)):
                    problems.append(
                        f"{path}: cell {index} ({where}): local path {hit!r}"
                    )
    return problems


def main(argv: list[str]) -> int:
    problems = [p for path in argv for p in check(path)]
    if not problems:
        return 0
    print("Local absolute paths found in notebook content:\n", file=sys.stderr)
    for problem in sorted(set(problems)):
        print(f"  {problem}", file=sys.stderr)
    print(
        "\nReplace them with a placeholder such as /path/to/... before "
        "committing.\nThese notebooks are published in a public repository "
        "with their outputs intact.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
