"""Every text file in the repository must be clean UTF-8 with no BOM.

This exists because README.md once acquired a UTF-16 LE byte order mark (FF FE)
in front of otherwise UTF-8 content - the signature of a PowerShell redirect,
which defaults to UTF-16 on Windows. The body was never actually UTF-16, but
GitHub reads the BOM, believes it, decodes the UTF-8 bytes two at a time and
renders the whole page as CJK garbage. The file looked fine in most local
editors, which is what made it worth a test rather than a one-off fix.
"""

from __future__ import annotations

import pathlib

from loan_default.config import PROJECT_ROOT

BOMS = {
    b"\xff\xfe": "UTF-16 LE",
    b"\xfe\xff": "UTF-16 BE",
    b"\xef\xbb\xbf": "UTF-8 with BOM",
}

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".json",
    ".cfg",
    ".ini",
    ".example",
    ".ipynb",
}
TEXT_NAMES = {"Makefile", "Procfile", "Dockerfile", ".gitignore", ".dockerignore"}

SKIP_DIRS = {
    "venv",
    ".venv",
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "data",
}


def text_files() -> list[pathlib.Path]:
    files = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES:
            files.append(path)
    return sorted(files)


FILES = text_files()


def test_the_scan_actually_finds_files():
    """A guard that silently checks nothing is worse than no guard at all."""
    assert len(FILES) > 20, f"Only found {len(FILES)} text files; the scan is misconfigured"


def test_no_text_file_has_a_byte_order_mark():
    """Reported in one test rather than one per file, so the suite count stays
    a count of behaviours rather than a count of files."""
    offenders = []
    for path in FILES:
        head = path.read_bytes()[:3]
        for bom, name in BOMS.items():
            if head.startswith(bom):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)} ({name} BOM)")
                break
    assert not offenders, (
        "These files start with a byte order mark. GitHub will trust it, decode "
        "the file accordingly and render it as garbage:" + "".join(f"\n  {o}" for o in offenders)
    )


def test_every_text_file_decodes_as_utf8():
    offenders = []
    for path in FILES:
        try:
            path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {exc}")
    assert not offenders, "Not valid UTF-8:" + "".join(f"\n  {o}" for o in offenders)


def test_readme_starts_with_its_heading():
    """Catches anything prepended ahead of the first character."""
    text = (PROJECT_ROOT / "README.md").read_bytes().decode("utf-8")
    assert text.startswith("# Loan Default Prediction"), (
        f"README.md starts with {text[:40]!r} rather than its heading"
    )
