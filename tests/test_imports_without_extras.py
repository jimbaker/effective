"""A consumer's module imports with every vendor blocked, or refuses by naming its extra."""

import subprocess
import sys

import pytest

BLOCKED = ("typesafe_sdk", "openai", "textual", "redis")

IMPORT_BLOCKED = """
import importlib, sys
module, *blocked = sys.argv[1:]
for name in blocked:
    sys.modules[name] = None
importlib.import_module(module)
"""

REFUSALS = {
    "effective.interpreters": None,
    "effective.interpreters.cli": None,
    "effective.interpreters.openai": None,
    "effective.interpreters.tool_catalog": None,
    "effective.interpreters.jev": (
        "effective.interpreters.jev needs the judge extra: effective[judge]"
    ),
    "effective.spend": None,
    "effective.runread": None,
}


@pytest.mark.parametrize(("module", "refusal"), REFUSALS.items(), ids=list(REFUSALS))
def test_a_module_imports_with_every_vendor_blocked(module, refusal):
    run = subprocess.run(
        [sys.executable, "-c", IMPORT_BLOCKED, module, *BLOCKED], capture_output=True, text=True
    )
    match refusal:
        case None:
            assert run.returncode == 0, run.stderr
        case str():
            assert (run.returncode, run.stderr.splitlines()[-1]) == (1, "ImportError: " + refusal)
