#!/usr/bin/env python
"""Fail the check if a Python file contains a trailing (end-of-line) comment.

Definition
---------
A "trailing comment" is a COMMENT token whose column index > 0 (i.e., not starting
at the first column), excluding the following whitelisted cases:
- lines that contain '# noqa' or '# nosec'
- lines that contain 'http://' or 'https://'
- the very first line shebang '#!'

Rationale
---------
We use the standard `tokenize` module so that '#' inside strings/docstrings
does not get misclassified as a comment.
"""

from __future__ import annotations

import io
import pathlib
import re
import sys
import tokenize

# ⛔ Skip checking this script itself to prevent recursion
SELF_NAME = pathlib.Path(__file__).name

ALLOW = re.compile(r"#\s*(noqa|nosec)|https?://", re.IGNORECASE)


def has_forbidden_trailing_comment(path: pathlib.Path) -> list[str]:
    """Return a list of human-readable violations for the given Python file.

    Args:
        path: Path to the Python source file.

    Returns:
        A list of violation strings; empty list means no violation.
    """
    violations: list[str] = []
    try:
        data = path.read_bytes()
    except Exception:
        return violations

    try:
        tokens = tokenize.tokenize(io.BytesIO(data).readline)
    except tokenize.TokenError:
        # Skip unreadable files
        return violations

    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue

        text = tok.string
        line_no, col = tok.start

        # Allow shebang only on first line
        if line_no == 1 and col == 0 and text.startswith("#!"):
            continue

        # Allow pure comment line even if indented (only whitespace before '#').
        prefix = tok.line[:col] if tok.line is not None else ""
        if prefix.strip() == "":
            continue

        # Whitelisted patterns
        if ALLOW.search(text):
            continue

        # This is a trailing comment
        preview = text[:60].replace("\n", "")
        violations.append(f"{path}:{line_no}:{col}: trailing comment found -> {preview}")

    return violations


def main(files: list[str]) -> int:
    """CLI entry: check a list of files; return non-zero on violation."""
    bad: list[str] = []
    for f in files:
        p = pathlib.Path(f)

        # Skip this script itself
        if p.name == SELF_NAME:
            continue

        bad.extend(has_forbidden_trailing_comment(p))

    if bad:
        print("\n".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
