"""Pins for examples/first_workflow.py, the program `docs/first-workflow.md` tells a reader to run.

The module is loaded by path, as a reader runs it; it is deliberately not a package.
"""

import ast
import contextlib
import importlib
import importlib.util
import io
import re
import runpy
import sys
from pathlib import Path

import pytest

from effective import RecordingHandler, ReplayHandler, ReplayMismatch
from effective.channels import Repair

EXAMPLE = Path(__file__).parent.parent / "examples" / "first_workflow.py"
PAGE = Path(__file__).parent.parent / "docs" / "first-workflow.md"

_SPEC = importlib.util.spec_from_file_location("first_workflow", EXAMPLE)
assert _SPEC is not None
assert _SPEC.loader is not None
first = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(first)


def warmer(ceiling: float = 30):
    return lambda: first.set_temperature("r1", "make it warmer", ceiling)


def section(title: str) -> str:
    return PAGE.read_text(encoding="utf-8").split(f"## {title}\n", 1)[1].split("\n## ", 1)[0]


def imported_names() -> list[tuple[str, str]]:
    """(module, name) for every name in the page's "What you imported" table."""
    pairs: list[tuple[str, str]] = []
    for row in section("What you imported").splitlines():
        cells = row.split("|")
        if len(cells) > 3 and cells[2].strip().startswith("`"):
            module = cells[2].strip().strip("`")
            pairs += [(module, name) for name in re.findall(r"`(\w+)`", cells[1])]
    return pairs


def test_recording_asks_the_model_sets_the_thermostat_and_appends_one_row():
    recorder = RecordingHandler(first.RESPONSES)
    assert recorder.run(warmer()) == 22.0
    assert [entry.key.stored() for entry in recorder.trace] == [
        "step:setpoint",
        "step;tool:thermostat",
        "ledger;setpoint:r1",
    ]
    assert [(row.kind, row.get("celsius")) for row in recorder.ledger] == [("setpoint", 22.0)]


def test_replay_takes_no_responses_and_returns_the_recorded_answer():
    recorder = RecordingHandler(first.RESPONSES)
    recorder.run(warmer())
    assert ReplayHandler(recorder.trace).run(warmer()) == 22.0


def test_the_guardrail_refuses_before_the_thermostat_runs():
    hot = RecordingHandler({"setpoint": {"celsius": 45}})
    refused = hot.run(lambda: first.set_temperature("r2", "make it much warmer"))
    assert isinstance(refused, Repair)
    assert [entry.key.stored() for entry in hot.trace] == ["step:setpoint"]
    assert hot.ledger == []


def test_past_the_guardrail_an_uncanned_thermostat_raises():
    hot = RecordingHandler({"setpoint": {"celsius": 45}})
    with pytest.raises(KeyError, match="tool:thermostat"):
        hot.run(lambda: first.set_temperature("r2", "make it much warmer", ceiling=50))


def test_lowering_the_ceiling_makes_the_recorded_history_refuse_to_replay():
    recorder = RecordingHandler(first.RESPONSES)
    recorder.run(warmer())
    with pytest.raises(ReplayMismatch, match="ended after 1 ops but 3 were recorded"):
        ReplayHandler(recorder.trace).run(warmer(ceiling=20))


def _run_as_script() -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        runpy.run_path(str(EXAMPLE), run_name="__main__")
    return out.getvalue()


def test_the_page_shows_exactly_what_the_script_prints():
    shown = section("Run it").split("```text\n", 1)[1].split("```", 1)[0]
    assert shown == _run_as_script()


@pytest.mark.parametrize(("module", "name"), imported_names())
def test_every_name_the_page_lists_imports_from_the_module_it_names(module, name):
    assert hasattr(importlib.import_module(module), name)


def test_the_imported_table_reaches_the_step_payloads():
    assert ("effective.domain", "AskLLM") in imported_names()


def test_it_imports_only_the_public_substrate():
    tree = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
    roots = {
        name.split(".")[0]
        for node in ast.walk(tree)
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    }
    assert roots - sys.stdlib_module_names == {"effective", "pydantic"}
