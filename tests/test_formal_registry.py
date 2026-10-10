"""The formal gate's registry: every Quint model is checked, and the states it checks are reached.

An invariant holds vacuously on a model whose steps never fire, so each model also registers a
property expected to be violated, whose counterexample is a run reaching the states its
invariants quantify over.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = sorted(p.name for p in (ROOT / "formal" / "quint").glob("*.qnt"))


def _registered(array: str) -> list[str]:
    script = 'source scripts/formal_checks.sh; declare -n e=$1; printf "%s\\n" "${e[@]}"'
    out = subprocess.run(
        ["bash", "-c", script, "_", array], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.splitlines()


def _model(entry: str) -> str:
    """A check is quint's argv; a guard and a tooth lead with the model and a `|`."""
    return entry.split("|")[0] if "|" in entry else entry.split()[1]


def test_every_model_has_a_check_that_must_pass() -> None:
    checked = {_model(entry) for entry in _registered("FORMAL_CHECKS")}
    assert MODELS
    assert [m for m in MODELS if m not in checked] == []


def test_every_model_has_a_run_that_reaches_its_states() -> None:
    witnessed = {_model(entry) for entry in _registered("FORMAL_EXPECT_VIOLATION")}
    assert [m for m in MODELS if m not in witnessed] == []


ARRAYS = ["FORMAL_CHECKS", "FORMAL_EXPECT_VIOLATION", "FORMAL_TEETH"]
PROPERTY = re.compile(r"--(invariant|temporal)=\S+")


@pytest.mark.parametrize("array", ARRAYS)
def test_every_registered_entry_names_a_model(array: str) -> None:
    assert [e for e in _registered(array) if _model(e) not in MODELS] == []


@pytest.mark.parametrize("array", ARRAYS)
def test_every_registered_entry_checks_a_property_on_a_backend_that_runs_it(array: str) -> None:
    """With no property quint checks nothing and passes. A temporal property needs TLC: Apalache
    stops at a `y/N` prompt and, with no stdin, exits 0 having checked nothing."""
    unchecked = [
        e
        for e in _registered(array)
        if not PROPERTY.search(e) or ("--temporal=" in e and "--backend=tlc" not in e)
    ]
    assert unchecked == []
