"""debug_loop: a debugging objective loop, built on `improve`.

Debugging is not a new capability: it is the objective loop (`effective.improve.improve`)
with **propose = the precise-edit `Gated` channel** (`agent.precise_edit`), **score = run
the tests *and* rate the quality**, and **done = the hard objective is met, then the soft
one**. It composes existing seams with no new op, and delegates its control to `improve`
rather than hand-rolling a `while`.

**The Pareto tension, made concrete.** Passing the tests is not the only objective: the
code must also meet the project's constraints — in this repo, **CLAUDE.md is the quality
rubric** (determinism boundary, ty-clean, one-way seam, the idioms). So the objectives are
two — `tests` (correctness) and `quality` (conformance) — and the loop is genuinely MOO.
The common good practice, staged, falls out of `done`: **get it working, then raise
quality**. `done` gates on `tests` first; while that fails the frontier is driven by the
correctness signal, and once a candidate passes, later rounds refine quality on it.

**ASI is the gradient (GEPA).** Each round throws off exactly the diagnostic the proposer
needs — the test failure, the lint/ty error, the rubric note — and `score` returns it as
the `Measurement.asi`. `propose` reads the frontier *with its ASI* and reflects on *why*
before emitting the next edit. The scalar pass/fail alone would discard this; the loop
does not.

The control/data split:
- **Data axis:** the edit is emitted through the `Gated` channel — a malformed patch is a
  `Repair`, re-emitted inside the one recorded `ask_llm` Step; it never reaches apply.
- **Control axis:** `apply_edit` is the mutation. It re-checks the anchor, so a stale
  snapshot that passed the gate returns `ApplyResult(ok=False)` and the round yields no
  candidate. The loop also honours `Refused` from a cascade (denial-as-observation).

Durability is by replay: every score/propose is a sealed op, so a recorded debug run
replays with zero model calls, applies, or test executions; selection re-derives from
recorded measures. The `Workspace` is live-run state (mutated by `apply_edit`); replay
never touches it. Imports `effective` and `agent` only. Deferred (designed-for): a
`FixCommitted` ledger promotion of the winning edit chain, memory of what-was-tried, and
CLAUDE.md-conformance as a real `quality_fn` (lint/ty/determinism) rather than a
caller-supplied rubric.
"""

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel

from agent.precise_edit import Edit, make_precise_editor
from effective import Effect, Refused, ask_llm, step
from effective.cost import CostBudget, MeteredInterpreter
from effective.domain import CallTool
from effective.govern import routable
from effective.improve import Measurement, Reflection, Summarize, improve
from effective.pareto import Objective


class TestReport(BaseModel):
    """The correctness objective's reading: did the suite pass, and if not, why."""

    __test__ = False  # not a pytest test class despite the Test* name

    passed: bool
    summary: str = ""


class QualityReport(BaseModel):
    """The quality objective's reading: a [0,1] conformance score + the rubric notes (the
    ASI for quality — in this repo, derived from the CLAUDE.md constraints)."""

    score: float = 0.0
    notes: str = ""


class ApplyResult(BaseModel):
    """The outcome of applying one edit to the workspace."""

    ok: bool
    detail: str = ""


class DebugResult(BaseModel):
    """The loop's verdict: whether correctness was met, the quality reached, and the edit
    chain that got there."""

    passed: bool
    quality: float = 0.0
    iters: int = 0
    edits: list[Edit] = []


# --- the workspace: live-run backing state (a ToolRunner's store) -----------


class Workspace:
    """A mutable path -> content file store. `read` gives a snapshot; `apply` mutates
    (re-checking the anchor, so a stale edit fails cleanly rather than corrupting)."""

    def __init__(self, files: Mapping[str, str]) -> None:
        self.files: dict[str, str] = dict(files)

    def read(self, path: str) -> str:
        return self.files.get(path, "")

    def apply(self, path: str, old: str, new: str) -> ApplyResult:
        content = self.files.get(path, "")
        if content.count(old) != 1:  # stale snapshot: passed the gate, fails the apply
            return ApplyResult(ok=False, detail="anchor no longer unique at apply time")
        self.files[path] = content.replace(old, new, 1)
        return ApplyResult(ok=True, detail=f"applied to {path}")


# --- the loop (a workflow: an `improve` over an edit chain) ------------------

QUALITY = "quality"
TESTS = "tests"


def _edit_request(path: str, content: str, reflection: Reflection) -> list[dict[str, Any]]:
    # the reflection rides into the prompt: the PINNED rubric (constraints) + the ASI
    # history (prior diagnostics). The rubric is always present, even after a compaction.
    return [{"role": "user", "content": f"{reflection.text()}\n\nFile {path}:\n{content}"}]


def debug_loop(
    path: str,
    *,
    max_iters: int = 4,
    quality_floor: float = 0.0,
    rubric: str = "",
    compact: Callable[[str], bool] | None = None,
    summarize: Summarize | None = None,
) -> Effect[DebugResult]:
    """Drive a failing test to green — then to quality — with anchored edits. A thin
    `improve`: propose = read+edit+apply (one candidate per round, self-refine), score =
    run tests + rate quality, objectives = [tests, quality], done = the staged gate.

    ``rubric`` is the pinned quality signal (the CLAUDE.md constraints); it rides into every
    edit prompt and survives compaction. When ``compact``/``summarize`` are given, a long
    debug session folds its ASI history but never the rubric."""

    def propose(parents, reflection: Reflection) -> Effect[list[tuple[Edit, ...]]]:
        best = max(parents, key=lambda s: (s.measures[TESTS], s.measures[QUALITY]))
        content: str = yield from step(
            "tool:read_file",
            CallTool(name="read_file", result_schema=str, args={"path": path}),
        )
        edit: Edit | None = yield from ask_llm(
            "react:edit", _edit_request(path, content, reflection), Edit
        )
        if edit is None:  # the Gated channel could not produce a safe edit
            return []
        try:
            applied: ApplyResult = yield from step(
                "tool:apply_edit",
                CallTool(
                    name="apply_edit",
                    result_schema=ApplyResult,
                    args={"path": edit.path, "old": edit.old, "new": edit.new},
                ),
            )
        except Refused as refusal:  # a denied apply yields no candidate this round
            routable(refusal)
            return []
        if not applied.ok:  # stale anchor -> this round produced no usable candidate
            return []
        return [(*best.candidate, edit)]

    def score(cand: tuple[Edit, ...]) -> Effect[Measurement]:
        tr: TestReport = yield from step(
            "tool:run_tests", CallTool(name="run_tests", result_schema=TestReport, args={})
        )
        qr: QualityReport = yield from step(
            "tool:quality",
            CallTool(name="check_quality", result_schema=QualityReport, args={}),
        )
        asi = " | ".join(x for x in (tr.summary, qr.notes) if x)
        return Measurement(measures={TESTS: float(tr.passed), QUALITY: qr.score}, asi=asi)

    objectives = [Objective(TESTS, "max"), Objective(QUALITY, "max")]

    def done(front) -> bool:  # staged: correctness first, then the quality floor
        return any(s.measures[TESTS] >= 1 and s.measures[QUALITY] >= quality_floor for s in front)

    front = yield from improve(
        (),
        propose,
        score,
        objectives=objectives,
        rounds=max_iters,
        done=done,
        rubric=rubric,
        compact=compact,
        summarize=summarize,
    )
    if not front:
        return DebugResult(passed=False)
    best = max(front, key=lambda s: (s.measures[TESTS], s.measures[QUALITY]))
    return DebugResult(
        passed=best.measures[TESTS] >= 1,
        quality=best.measures[QUALITY],
        iters=len(best.candidate),
        edits=list(best.candidate),
    )


# --- the live interpreter: wire the editor + the code tools over a workspace --

type TestFn = Callable[[Mapping[str, str]], TestReport]
type QualityFn = Callable[[Mapping[str, str]], QualityReport]


def make_code_tools(
    ws: Workspace, test_fn: TestFn, quality_fn: QualityFn
) -> Callable[[CallTool[Any]], Any]:
    """A ToolRunner routing the loop's CallTool ops by name. ``run_tests`` and
    ``check_quality`` read the live workspace (post-apply); ``read_file`` is observational;
    ``apply_edit`` mutates — all recorded."""

    def run(op: CallTool[Any]) -> Any:
        match op.name:
            case "read_file":
                return ws.read(op.args["path"])
            case "apply_edit":
                return ws.apply(op.args["path"], op.args["old"], op.args["new"])
            case "run_tests":
                return test_fn(ws.files)
            case "check_quality":
                return quality_fn(ws.files)
        raise ValueError(f"unknown code tool: {op.name!r}")

    return run


def _no_quality(_files: Mapping[str, str]) -> QualityReport:
    """Default rubric: quality is satisfied (score 1.0) — a single-objective debug where
    only correctness matters. Pass a real ``quality_fn`` for the MOO/staged loop."""
    return QualityReport(score=1.0)


def make_debug_interpreter(
    client: Any,
    ws: Workspace,
    test_fn: TestFn,
    *,
    quality_fn: QualityFn = _no_quality,
    budget: float = 1.0,
    **editor_kwargs: Any,
) -> MeteredInterpreter:
    """Assemble the loop's DomainInterpreter: the precise-editor is the LLMCall (its gate
    closes over the *live* ``ws.files``, so each edit sees the current post-apply state),
    and the code-tools router is the ToolRunner. Metered, exactly like any other domain's path."""
    editor = make_precise_editor(client, snapshot=ws.files, **editor_kwargs)
    return MeteredInterpreter(
        llm=editor, tools=make_code_tools(ws, test_fn, quality_fn), budget=CostBudget(budget)
    )
