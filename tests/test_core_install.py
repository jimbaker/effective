"""What the substrate imports without its extras.

Every module under `src/effective` imports, or refuses with an `ImportError` naming the extra it
needs, and `EXTRA_OF` lists the modules that need one.

The census runs in whatever environment pytest runs in: the dev environment holds every extra, and
`just race-free-threaded` runs it on the core and the judge extra. The second test takes each
extra away in turn, in a subprocess, so the refusals run on every pass of the gate. The census
sees module-level imports; the lint's lazy-import rule keeps a parser import out of a function
body.

    python tests/test_core_install.py ast_grep_py    # the census with that package absent
"""

import importlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"

EXTRA_OF: dict[str, str] = {
    "effective.lint": "coding",
    "effective.coding.gate": "coding",
    "effective.coding.runners": "coding",
    "effective.coding.edits.structural": "coding",
    "effective.interpreters.jev": "judge",
}
"""Each module that needs an extra, and the extra."""

PACKAGES_OF: dict[str, tuple[str, ...]] = {
    "coding": ("ast_grep_py",),
    "judge": ("typesafe_sdk",),
}
"""The top-level packages each extra brings, which the subprocess census takes away."""


def modules() -> list[str]:
    """Every module under `src/effective`, by its dotted name."""
    names = []
    for path in sorted((SRC / "effective").rglob("*.py")):
        parts = path.relative_to(SRC).with_suffix("").parts
        names.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return names


def census() -> dict[str, str]:
    """Each module that did not import, and the message it refused with."""
    refused = {}
    for module in modules():
        try:
            importlib.import_module(module)
        except ImportError as missing:
            refused[module] = str(missing)
    return refused


def test_every_module_imports_or_names_its_extra():
    wrong = {
        module: message
        for module, message in census().items()
        if (extra := EXTRA_OF.get(module)) is None or f"effective[{extra}]" not in message
    }
    assert wrong == {}


@pytest.mark.parametrize("extra", sorted(PACKAGES_OF))
def test_without_an_extra_exactly_its_modules_refuse(extra):
    absent = subprocess.run(
        [sys.executable, __file__, *PACKAGES_OF[extra]],
        capture_output=True,
        text=True,
        check=True,
    )
    refused = json.loads(absent.stdout)
    already = census()
    needing = {module for module, needs in EXTRA_OF.items() if needs == extra}
    assert set(refused) == needing | set(already)
    assert all(f"effective[{extra}]" in refused[module] for module in needing)


def test_psycopg_and_its_binary_build_lock_to_one_release():
    """psycopg runs a binary build of another release without a check."""
    locked = tomllib.loads((SRC.parent / "uv.lock").read_text())["package"]
    versions = {
        p["name"]: p["version"] for p in locked if p["name"] in {"psycopg", "psycopg-binary"}
    }
    assert versions["psycopg"] == versions["psycopg-binary"]


class _Absent:
    """A finder that makes the named top-level packages unimportable."""

    def __init__(self, packages: list[str]):
        self.packages = frozenset(packages)

    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in self.packages:
            raise ModuleNotFoundError(name, name=name)
        return None


if __name__ == "__main__":
    sys.path.insert(0, str(SRC))
    sys.meta_path.insert(0, _Absent(sys.argv[1:]))
    print(json.dumps(census()))
