"""The role markers `pyproject.toml` registers are exactly the roles `tests/conftest.py` knows."""

import tomllib
from pathlib import Path

from conftest import ROLES


def test_the_registered_markers_are_the_roles():
    pyproject = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())
    markers = pyproject["tool"]["pytest"]["ini_options"]["markers"]
    assert {line.split(":", 1)[0].strip() for line in markers} == set(ROLES)
