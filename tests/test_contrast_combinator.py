"""Pins for the contrast `combinator` arm (amendment C1) — the author-fixed
decomposition: route's recorded classify -> typed param extraction -> pinned
script on the durable engine.

The load-bearing pin is the first one: every pinned script answers every
generated task's ground truth exactly, on the real Monty engine, with zero
model involvement — so extraction is the ONLY thing a live model can get
wrong (the registered architectural claim)."""

import re
from typing import Any

from agent.contrastbench import (
    _PINNED_OUT,
    _PINNED_SCRIPTS,
    ContrastTask,
    _QtypeLabel,
    _ThresholdParams,
    combinator_wf,
    make_log,
    make_tasks,
    replay_combinator,
    run_combinator,
)
from effective.api import scoped
from effective.code import CodeOutcome
from effective.cost import Usage
from effective.handlers.recording import RecordingHandler
from effective.keys import Segment, compose_key
from effective.monty import MontyEngine


def _params_from(question: str) -> dict[str, Any]:
    """Derive the correct extraction from the question text (what a perfect
    extractor would return) — keeps the fixtures honest across generator drift."""
    action = re.search(r"action=(\w+)", question)
    assert action is not None
    params: dict[str, Any] = {"action": action.group(1)}
    if m := re.search(r"user=(\w+)", question):
        params["user"] = m.group(1)
    if m := re.search(r"amount > (\d+)", question):
        params["min_amount"] = int(m.group(1))
    return params


def test_every_pinned_script_answers_every_generated_task():
    engine = MontyEngine()
    tasks = make_tasks(seed=7)
    assert len(tasks) == 16  # 4 qtypes x 2 questions x 2 sizes
    for task in tasks:
        rows, _ = make_log(task.seed, task.records)
        outcome = engine.execute(
            {
                "code": _PINNED_SCRIPTS[task.qtype],
                "inputs": {"records": rows, "params": _params_from(task.question)},
            }
        )
        assert isinstance(outcome.output, _PINNED_OUT[task.qtype])
        assert str(outcome.output).strip().lower() == task.truth, task.task_id


def test_combinator_wf_routes_on_the_recorded_label():
    """The recorder pin: classify and extract are sealed ops; the taken path is
    in the tape (`classify` -> `extract` -> the pinned code key), placed by the caller's
    `scoped(...)` rather than spliced into each name."""
    handler = RecordingHandler(
        responses={
            "task:t0;classify": _QtypeLabel(qtype="filter-count"),
            "task:t0;extract": _ThresholdParams(action="refund", min_amount=250),
            "task:t0;code:seg,0,answer": CodeOutcome(status="complete", output=7),
        }
    )

    def wf():
        return (
            yield from scoped(
                compose_key(t"task:{Segment('t0')}"), lambda: combinator_wf("How many...?", [])
            )
        )

    assert handler.run(wf) == "7"
    assert [e.key.stored() for e in handler.trace] == [
        "task:t0;step:classify",
        "task:t0;step:extract",
        "task:t0;step;code:seg,0,answer",
    ]


_QTYPE_BY_PREFIX = {
    "How many records have": "filter-count",
    "How many distinct": "count-distinct",
    "Which user has": "group-argmax",
    "What is the total": "sum",
}


class _PerfectExtractor:
    """A scripted schema-honoring caller: returns exactly what a perfect
    extraction model would (meter-compatible with SchemaCaller)."""

    def __init__(self, question: str) -> None:
        self._question = question
        self.calls = 0
        self.meter = Usage()
        self.wire_failures = 0

    def __call__(self, op: Any) -> tuple[Any, Usage]:
        self.calls += 1
        schema = op.response_schema
        if schema is _QtypeLabel:
            qtype = next(q for p, q in _QTYPE_BY_PREFIX.items() if self._question.startswith(p))
            return _QtypeLabel(qtype=qtype), Usage()
        return schema(**_params_from(self._question)), Usage()


def test_combinator_arm_completes_durably_and_replays_with_zero_model(tmp_path):
    """HC3's shape, offline: the trial completes on the SQLite engine and the
    poison replay (model RAISES, sandbox RAISES) re-binds to the same answer."""
    task = make_tasks(seed=7)[0]
    rows, _ = make_log(task.seed, task.records)
    db = tmp_path / "task.db"
    llm = _PerfectExtractor(task.question)
    outcome = run_combinator(llm, task, rows, db_path=db, budget_usd=0.10)
    assert outcome.answer is not None
    assert str(outcome.answer).strip().lower() == task.truth
    assert outcome.turns == 2  # classify + extract, nothing else
    assert replay_combinator(task, rows, db_path=db) == outcome.answer


def test_combinator_arm_rejects_the_semantic_family(tmp_path):
    """C1 is registered for aggregation only — the guard is loud."""
    import pytest

    from agent.contrastbench import run_trial

    sem = ContrastTask(
        task_id="sem-x",
        qtype="sem-filter-count",
        size="S",
        question="?",
        truth="1",
        seed=7,
        n_records=10,
        semantic=True,
    )
    with pytest.raises(ValueError, match="aggregation family only"):
        run_trial(sem, "combinator", 1, make_client=lambda: None, out_dir=tmp_path)
