"""Deployment manifest consistency.

pyproject.toml is the source of truth for dependencies; requirements.txt is what
the Railway/Nixpacks build actually installs. If they drift, the deployed
environment differs from the tested one - which is exactly how the original
project ended up serving a model pickled under scikit-learn 1.7.2 from a
runtime with 1.8.0.
"""

from __future__ import annotations

import re
import tomllib

import pytest

from loan_default.config import PROJECT_ROOT

PYPROJECT = PROJECT_ROOT / "pyproject.toml"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

_SPEC = re.compile(r"^([A-Za-z0-9_.\-]+)(\[[^\]]+\])?==(.+)$")


def _parse(lines: list[str]) -> dict[str, str]:
    """Map normalised package name -> pinned version."""
    parsed = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SPEC.match(line)
        if match:
            name = match.group(1).lower().replace("_", "-")
            parsed[name] = match.group(3).strip()
    return parsed


@pytest.fixture(scope="module")
def pyproject_pins() -> dict[str, str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return _parse(data["project"]["dependencies"])


@pytest.fixture(scope="module")
def requirements_pins() -> dict[str, str]:
    return _parse(REQUIREMENTS.read_text(encoding="utf-8").splitlines())


def test_requirements_file_exists():
    """Nixpacks keys its Python detection off this file."""
    assert REQUIREMENTS.exists()


def test_same_packages_in_both_files(pyproject_pins, requirements_pins):
    assert set(pyproject_pins) == set(requirements_pins), (
        f"only in pyproject: {sorted(set(pyproject_pins) - set(requirements_pins))}; "
        f"only in requirements: {sorted(set(requirements_pins) - set(pyproject_pins))}"
    )


def test_versions_agree(pyproject_pins, requirements_pins):
    mismatched = {
        name: (pyproject_pins[name], requirements_pins[name])
        for name in pyproject_pins
        if name in requirements_pins and pyproject_pins[name] != requirements_pins[name]
    }
    assert not mismatched, f"pinned versions differ (pyproject, requirements): {mismatched}"


def test_every_runtime_dependency_is_pinned(pyproject_pins):
    """Unpinned dependencies are not reproducible."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    for spec in data["project"]["dependencies"]:
        assert "==" in spec, f"dependency is not pinned to an exact version: {spec!r}"


def test_installed_versions_match_the_pins(pyproject_pins):
    """The environment running the tests must match what will be deployed."""
    import importlib.metadata as metadata

    mismatched = {}
    for name, pinned in pyproject_pins.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            pytest.fail(f"pinned dependency {name} is not installed")
        if installed != pinned:
            mismatched[name] = (pinned, installed)
    assert not mismatched, f"installed versions differ from pins (pinned, installed): {mismatched}"


def test_dev_dependencies_are_not_in_the_deployment_manifest(requirements_pins):
    for dev_only in ("pytest", "pytest-cov", "ruff", "httpx", "matplotlib"):
        assert dev_only not in requirements_pins, (
            f"{dev_only} is a dev dependency and must not ship to the deployed environment"
        )
