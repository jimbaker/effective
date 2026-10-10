"""A SQLite-only process loads the engines, the handler and the SQLite readers without psycopg or
the Absurd SDK."""

import subprocess
import sys
import textwrap

import pytest

SQLITE_SIDE = (
    "effective.engines",
    "effective.engines.sqlite",
    "effective.engines.absurd",
    "effective.handlers.durable",
    "effective.fork",
    "effective.dashboard",
    "effective.bridge_sqlite",
    "effective.runread",
)

BLOCKED = textwrap.dedent(
    """
    import importlib, sys

    class Absent:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("psycopg", "psycopg_binary", "psycopg_c", "absurd_sdk"):
                raise ImportError(f"absent: {name}")

    sys.meta_path.insert(0, Absent())
    importlib.import_module(sys.argv[1])
    """
)


@pytest.mark.parametrize("module", SQLITE_SIDE)
def test_a_sqlite_side_module_imports_with_psycopg_and_the_sdk_absent(module):
    ran = subprocess.run(
        [sys.executable, "-c", BLOCKED, module], capture_output=True, text=True, check=False
    )
    assert ran.returncode == 0, ran.stderr
