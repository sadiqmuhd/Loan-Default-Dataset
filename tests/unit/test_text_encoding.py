"""Every text file in the repository must be clean UTF-8 with no BOM.

This exists because README.md once acquired a UTF-16 LE byte order mark (FF FE)
in front of otherwise UTF-8 content. The body was never actually UTF-16, but
GitHub reads the BOM, believes it, and renders the page as CJK garbage - and
worse, `pip install -e .` fails outright, because pyproject.toml reads README.md
as long_description. That took down CI and the deploy before any code ran.

The checking logic lives in scripts/check_encoding.py rather than here, because
CI has to run it before installing anything. These tests exercise that same
script so the local suite and the CI gate can never disagree.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

from loan_default.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from check_encoding import BOMS, problems, text_files  # noqa: E402


def test_the_scan_actually_finds_files():
    """A guard that silently checks nothing is worse than no guard at all."""
    found = text_files(PROJECT_ROOT)
    assert len(found) > 20, f"Only found {len(found)} text files; the scan is misconfigured"


def test_no_encoding_problems_anywhere():
    issues = problems(PROJECT_ROOT)
    assert not issues, "Encoding problems found:" + "".join(f"\n  {i}" for i in issues)


def test_readme_starts_with_its_heading():
    """Catches anything prepended ahead of the first character."""
    text = (PROJECT_ROOT / "README.md").read_bytes().decode("utf-8")
    assert text.startswith("# Loan Default Prediction"), (
        f"README.md starts with {text[:40]!r} rather than its heading"
    )


def test_the_checker_detects_a_bom_it_is_given(tmp_path: pathlib.Path):
    """Proves the gate can fail, not just that it passes on a clean tree."""
    (tmp_path / "clean.md").write_bytes(b"# fine\n")
    for bom in BOMS:
        offender = tmp_path / f"bad_{bom.hex()}.md"
        offender.write_bytes(bom + b"# not fine\n")

    issues = problems(tmp_path)
    assert len(issues) == len(BOMS), f"Expected {len(BOMS)} problems, got: {issues}"
    assert not any("clean.md" in issue for issue in issues)


def test_the_checker_detects_invalid_utf8(tmp_path: pathlib.Path):
    (tmp_path / "broken.md").write_bytes(bytes([0x48, 0x69, 0x80, 0x81]))
    issues = problems(tmp_path)
    assert len(issues) == 1
    assert "not valid UTF-8" in issues[0]


def test_the_script_runs_standalone_and_exits_zero():
    """CI invokes it as a subprocess with no dependencies installed."""
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "check_encoding.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean UTF-8" in result.stdout
