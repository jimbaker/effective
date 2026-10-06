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

pytestmark = pytest.mark.journey

EXAMPLE = Path(__file__).parent.parent / "examples" / "first_workflow.py"
PAGE = Path(__file__).parent.parent / "docs" / "first-workflow.md"

_SPEC = importlib.util.spec_from_file_location("first_workflow", EXAMPLE)
assert _SPEC is not None
assert _SPEC.loader is not None
first = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(first)


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


def test_a_durable_run_asks_the_model_sets_the_thermostat_and_appends_one_row():
    house = first.House()
    done, steps = first.run_durably(house, "make it warmer")
    assert (done.state, done.result) == ("completed", 22.0)
    assert steps == ("step:setpoint", "step;tool:thermostat", "ledger;setpoint:r1")
    assert house.asked == 1


def test_a_retry_after_an_outage_replays_the_answer_it_already_has():
    house = first.House(outages=1)
    done, steps = first.run_durably(house, "make it warmer")
    assert (done.state, done.result, house.outages) == ("completed", 22.0, 0)
    assert house.attempts == 2, "the outage failed the first attempt and the task ran again"
    assert house.asked == 1, "the retry was served the setpoint from the file"
    assert steps == ("step:setpoint", "step;tool:thermostat", "ledger;setpoint:r1")


def test_an_outage_past_the_last_attempt_fails_the_run_and_still_asks_once():
    house = first.House(outages=10)
    done, steps = first.run_durably(house, "make it warmer")
    assert done.state == "failed"
    assert house.asked == 1
    assert steps == ("step:setpoint",)


def test_the_guardrail_refuses_before_the_thermostat_runs():
    house = first.House(celsius=45)
    done, steps = first.run_durably(house, "make it much warmer")
    assert done.result == {"reason": "celsius outside the allowed range"}
    assert steps == ("step:setpoint",)


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
