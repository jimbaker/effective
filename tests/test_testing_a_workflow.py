"""Pins for examples/testing_a_workflow.py, which `wiki/concepts/testing.md` cites.

The module is loaded by path, as a reader runs it, beside the first workflow it imports.
"""

import contextlib
import importlib
import io
import runpy
import sys
from pathlib import Path

import pytest

from effective import RecordingHandler, ReplayHandler, ReplayMismatch
from effective.channels import Repair

EXAMPLES = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))
set_temperature = importlib.import_module("first_workflow").set_temperature
RESPONSES = importlib.import_module("testing_a_workflow").RESPONSES


def warmer(ceiling: float = 30):
    return lambda: set_temperature("r1", "make it warmer", ceiling)


def test_recording_asks_the_model_sets_the_thermostat_and_appends_one_row():
    recorder = RecordingHandler(RESPONSES)
    assert recorder.run(warmer()) == 22.0
    assert [entry.key.stored() for entry in recorder.trace] == [
        "step:setpoint",
        "step;tool:thermostat",
        "ledger;setpoint:r1",
    ]
    assert [(row.kind, row.get("celsius")) for row in recorder.ledger] == [("setpoint", 22.0)]


def test_replay_takes_no_responses_and_returns_the_recorded_answer():
    recorder = RecordingHandler(RESPONSES)
    recorder.run(warmer())
    assert ReplayHandler(recorder.trace).run(warmer()) == 22.0


def test_the_guardrail_refuses_before_the_thermostat_runs():
    hot = RecordingHandler({"setpoint": {"celsius": 45}})
    refused = hot.run(lambda: set_temperature("r2", "make it much warmer"))
    assert isinstance(refused, Repair)
    assert [entry.key.stored() for entry in hot.trace] == ["step:setpoint"]
    assert hot.ledger == []


def test_past_the_guardrail_an_uncanned_thermostat_raises():
    hot = RecordingHandler({"setpoint": {"celsius": 45}})
    with pytest.raises(KeyError, match="tool:thermostat"):
        hot.run(lambda: set_temperature("r2", "make it much warmer", ceiling=50))


def test_lowering_the_ceiling_makes_the_recorded_history_refuse_to_replay():
    recorder = RecordingHandler(RESPONSES)
    recorder.run(warmer())
    with pytest.raises(ReplayMismatch, match="ended after 1 ops but 3 were recorded"):
        ReplayHandler(recorder.trace).run(warmer(ceiling=20))


def test_the_script_prints_one_line_per_demonstration():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        runpy.run_path(str(EXAMPLES / "testing_a_workflow.py"), run_name="__main__")
    assert [line.split(":", 1)[0] for line in out.getvalue().splitlines()] == [
        "record",
        "replay",
        "guardrail",
        "drift",
    ]
