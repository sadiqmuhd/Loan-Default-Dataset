"""Fail if any text file carries a byte order mark or is not valid UTF-8.

Run directly, or via `make encoding`:

    python scripts/check_encoding.py

This lives in a real file rather than inline in the CI workflow. An earlier
version embedded the same logic as a YAML -> shell heredoc -> Python string, and
the byte escapes were resolved one layer too early, so the workflow shipped
literal 0xFF characters and died on a SyntaxError. A script file has no escaping
stack, runs identically on a laptop and in CI, and is covered by the test suite.

Exit status is 0 when clean and 1 when something is wrong, so it works as a CI
gate. It imports nothing outside the standard library, which matters: it has to
run BEFORE `pip install`, because a BOM on README.md breaks the install itself.
"""

from __future__ import annotations

import pathlib
import sys

# Built with bytes([...]) rather than escape sequences so that no amount of
# copying this file between shells, editors or YAML can corrupt the literals.
BOMS: dict[bytes, str] = {
    bytes([0xFF, 0xFE]): "UTF-16 LE",
    bytes([0xFE, 0xFF]): "UTF-16 BE",
    bytes([0xEF, 0xBB, 0xBF]): "UTF-8 with BOM",
}

TEXT_SUFFIXES = {
    ".cfg",
    ".example",
    ".ini",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

TEXT_NAMES = {"Makefile", "Procfile", "Dockerfile", ".gitignore", ".dockerignore"}

SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "data",
    "node_modules",
    "venv",
}


def text_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Every file we expect to be readable UTF-8 text."""
    found = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES:
            found.append(path)
    return sorted(found)


def problems(root: pathlib.Path) -> list[str]:
    """Human-readable description of every encoding problem found."""
    issues = []
    for path in text_files(root):
        raw = path.read_bytes()
        relative = path.relative_to(root)

        for bom, name in BOMS.items():
            if raw.startswith(bom):
                issues.append(
                    f"{relative}: starts with a {name} byte order mark. "
                    "Rewrite the file as UTF-8 without a BOM."
                )
                break
        else:
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                issues.append(f"{relative}: not valid UTF-8 ({exc}).")
    return issues


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    checked = len(text_files(root))
    issues = problems(root)

    if issues:
        print(f"Encoding problems in {len(issues)} of {checked} text files:")
        for issue in issues:
            print(f"  {issue}")
        print(
            "\nA byte order mark on README.md breaks `pip install -e .`, because "
            "pyproject.toml reads it as long_description. That fails before any "
            "code runs, which is why this check comes first."
        )
        return 1

    print(f"{checked} text files are clean UTF-8 with no BOM.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
