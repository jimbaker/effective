"""Pins for the SkillsBench harness's pure core (audit item 5 — the flagship
M1 harness shipped with zero tests). Offline: no container, no API, no network.
Covers the turn resolver, the arm-parameterized system prompt, and the
aggregation — the parsing/prompt/summary spine campaigns rode on unpinned.
"""

import json
from pathlib import Path

import pytest

from agent.skillsbench import (
    ARMS,
    TaskSpec,
    TrialResult,
    _resolve_turn,
    measurements_record,
    summarize,
    system_prompt,
)
from effective.keys import Key
from effective.skills import SkillRegistry
from effective.telemetry import Span


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="fjsp-1",
        path=Path("/tmp/fjsp-1"),
        prompt="Schedule the jobs.",
        agent_timeout_sec=60.0,
        verifier_timeout_sec=30.0,
        cpus=1.0,
        memory_mb=512,
        workdir="/work",
    )


def _registry() -> SkillRegistry:
    # a plain (interpolation-free) body: disclosable and pinnable, which a
    # channel-bearing one is not
    return SkillRegistry.in_memory({"fjsp": ("Flexible job-shop scheduling", t"Use OR-Tools.")})


# --- _resolve_turn: nullish -> None, tool normalized --------------------------


def test_resolve_turn_normalizes_nullish_and_lowercases_tool():
    turn = _resolve_turn(
        {"thought": "run it", "tool": "BASH ", "command": "ls", "skill": "-", "answer": "null"}
    )
    assert turn.tool == "bash"  # stripped + lowercased
    assert turn.command == "ls"
    assert turn.skill is None  # "-" is nullish
    assert turn.answer is None  # "null" is nullish
    assert turn.thought == "run it"


def test_resolve_turn_absent_sections_fill_as_none():
    turn = _resolve_turn({"answer": "42"})  # no tool/command/skill keys
    assert turn.answer == "42"
    assert turn.tool is None
    assert turn.command is None


# --- system_prompt: one per arm ----------------------------------------------


def test_system_prompt_absent_omits_skills():
    base = system_prompt("absent", _task(), _registry())
    assert "/work" in base  # the workdir is threaded into the protocol
    assert "Use OR-Tools" not in base  # no skill body
    assert "fjsp" not in base  # no catalog either


def test_system_prompt_inline_embeds_the_body():
    out = system_prompt("inline", _task(), _registry())
    assert "Use OR-Tools" in out  # the disclosed body is inlined


def test_system_prompt_catalog_embeds_the_index_not_the_body():
    out = system_prompt("catalog", _task(), _registry())
    assert "fjsp: Flexible job-shop scheduling" in out  # the catalog line
    assert "Use OR-Tools" not in out  # the body stays behind the tool


def test_system_prompt_none_registry_is_treated_as_absent():
    assert system_prompt("inline", _task(), None) == system_prompt("absent", _task(), None)


def test_system_prompt_unknown_arm_raises():
    with pytest.raises(ValueError, match="unknown arm"):
        system_prompt("wat", _task(), _registry())
    assert set(ARMS) == {"absent", "inline", "catalog"}


# --- summarize: per-(task, arm) aggregation ----------------------------------


def _trial(task_id: str, arm: str, reward: float, *, pins=(), error=None) -> TrialResult:
    return TrialResult(
        task_id=task_id,
        arm=arm,
        attempt=1,
        reward=reward,
        stop_reason="finish",
        steps=3,
        prompt_tokens=100,
        completion_tokens=20,
        cache_read_tokens=0,
        cost_usd=0.001,
        elapsed_sec=1.0,
        pins=list(pins),
        error=error,
    )


def test_summarize_groups_by_task_arm_and_computes_pass_rate():
    trials = [
        _trial("fjsp-1", "inline", 1.0, pins=[{"name": "fjsp"}]),
        _trial("fjsp-1", "inline", 0.0),
        _trial("fjsp-1", "absent", 0.0, error="boom"),
    ]
    rows = summarize(trials)["rows"]
    by = {(r["task"], r["arm"]): r for r in rows}
    assert by[("fjsp-1", "inline")]["pass_rate"] == 0.5  # 1 of 2
    assert by[("fjsp-1", "inline")]["activations"] == 1  # one pin
    assert by[("fjsp-1", "absent")]["errors"] == 1
    assert by[("fjsp-1", "absent")]["pass_rate"] == 0.0


# --- the span list finally has a reader ------------------------------------------------------


def test_measurements_record_attributes_cost_and_duration_to_the_placed_key():
    """`run_trial` collected spans into a list nothing read — two references, both writes.

    This is the reader. What it adds over `TrialResult.cost_usd` is ATTRIBUTION: the meter is a
    scalar total, so a trial that cost twice as much could not say where, and this maps every
    dollar and nanosecond onto the op address that spent it."""
    spans = [
        Span(
            name="LM.0",
            kind="LLM",
            session_id="inline:t1:0",
            duration_ns=1_000,
            usage_attributes={"effective.cost.usd": 0.25},
            key=Key.parse("d:0;step;react:turn"),
        ),
        Span(
            name="tool.run.1",
            kind="TOOL",
            session_id="inline:t1:0",
            duration_ns=2_000,
            key=Key.parse("step;tool:run"),
        ),
    ]
    record = measurements_record(spans)
    assert record == {
        "d:0;step;react:turn": {"cost_usd": 0.25, "duration_ns": 1_000},
        # `None`, NOT 0.0 — a tool span carries no cost attribute, and a zero here would read as
        # "this node was free", which is a different and false claim.
        "step;tool:run": {"cost_usd": None, "duration_ns": 2_000},
    }
    assert json.loads(json.dumps(record)) == record  # it is written with json.dumps


def test_measurements_record_drops_a_span_with_no_address():
    """A span with `key=None` has no address to attribute to — the recording/replay core publishes
    no placement. Dropping it is a left-outer miss, not an orphan, and inventing a placeholder key
    would put a row in the record that joins to nothing on the tape."""
    unplaced = Span(name="LM.0", kind="LLM", session_id="s", duration_ns=7, key=None)
    assert measurements_record([unplaced]) == {}
