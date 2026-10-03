"""The debugging objective loop, now built on `improve` (design-space §3+§4.5).

Proofs:
- END TO END, a real fix: a live in-memory run (`DurableHandler` over `LocalCtx`) with the
  real precise-editor + real workspace tools drives a genuinely failing test to green.
- STAGED MOO: get it working, THEN raise quality — two objectives (tests,
  quality), `done` gating correctness before the quality floor; the loop fixes the bug in
  round 0 (quality still low) and cleans it up in round 1, driven by the ASI.
- CONTROL FLOW + REPLAY: a canned run pins the op sequence and replays with zero calls.
- SAFETY: a model that can't anchor an edit changes nothing; the apply re-check rejects a
  non-unique anchor without corrupting the file.
"""

import json
from types import SimpleNamespace

from agent.debug import (
    ApplyResult,
    DebugResult,
    QualityReport,
    TestReport,
    Workspace,
    debug_loop,
    make_debug_interpreter,
)
from agent.precise_edit import Edit
from effective import RecordingHandler, ReplayHandler
from effective.handlers.absurd import DurableHandler

BUGGY = "def add(a, b):\n    return a - b\n"


def add_test(files) -> TestReport:
    """Run the code and check it — a real objective over the (controlled) workspace."""
    ns: dict = {}
    exec(files["calc.py"], ns)  # test-owned content — running it IS the objective
    got = ns["add"](2, 3)
    return TestReport(passed=got == 5, summary="" if got == 5 else f"add(2,3)={got}, expected 5")


class _FakeCreate:
    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(
            prompt_tokens=90,
            completion_tokens=12,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        message = SimpleNamespace(content=self.contents.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _client(contents: list[str]):
    completions = _FakeCreate(contents)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def _edit_json(path: str, old: str, new: str) -> str:
    return json.dumps({"edit": {"path": path, "old": old, "new": new}})


# --- END TO END: a live in-memory run that actually fixes the bug -----------


def test_debug_loop_fixes_a_failing_test_end_to_end():
    from agent.runtime import LocalCtx

    ws = Workspace({"calc.py": BUGGY})
    assert not add_test(ws.files).passed  # the bug is real before we start

    client, completions = _client([_edit_json("calc.py", "return a - b", "return a + b")])
    interp = make_debug_interpreter(client, ws, add_test)
    handler = DurableHandler(ctx=LocalCtx(), domain=interp)

    # the durable handler round-trips its return through JSON (a dict); validate it back
    result = DebugResult.model_validate(handler.run(lambda: debug_loop("calc.py")))

    assert result.passed
    assert result.iters == 1
    assert result.edits == [Edit(path="calc.py", old="return a - b", new="return a + b")]
    assert ws.files["calc.py"] == "def add(a, b):\n    return a + b\n"  # actually edited
    assert add_test(ws.files).passed  # and genuinely green now
    assert len(completions.calls) == 1


# --- STAGED MOO: get it working, then raise quality -------------------------

BUGGY_UGLY = "def add(a, b):\n    return a - b  # TODO cleanup\n"


def add_test_ugly(files) -> TestReport:
    ns: dict = {}
    exec(files["calc.py"], ns)
    got = ns["add"](2, 3)
    return TestReport(passed=got == 5, summary="" if got == 5 else f"add(2,3)={got}, expected 5")


# lint: working-note — the tag IS this rubric's subject
def rubric_quality(files) -> QualityReport:
    """The CLAUDE.md-style rubric, in miniature: a leftover TODO is a quality breach."""
    c = files["calc.py"]
    if "TODO" in c:
        return QualityReport(score=0.3, notes="leftover TODO")
    return QualityReport(score=1.0)


def test_staged_moo_fixes_then_cleans_up():
    from agent.runtime import LocalCtx

    ws = Workspace({"calc.py": BUGGY_UGLY})
    client, completions = _client(
        [
            _edit_json("calc.py", "return a - b  # TODO cleanup", "return a + b  # TODO cleanup"),
            _edit_json("calc.py", "return a + b  # TODO cleanup", "return a + b"),
        ]
    )
    interp = make_debug_interpreter(client, ws, add_test_ugly, quality_fn=rubric_quality)
    handler = DurableHandler(ctx=LocalCtx(), domain=interp)

    # quality_floor forces a SECOND pass: correctness clears in round 0, the tag in round 1
    result = DebugResult.model_validate(
        handler.run(lambda: debug_loop("calc.py", quality_floor=0.8))
    )

    assert result.passed  # correctness met
    assert result.quality == 1.0  # AND the quality floor cleared
    assert result.iters == 2  # it took the staged second pass
    assert "TODO" not in ws.files["calc.py"]
    assert len(completions.calls) == 2  # one edit to fix, one to clean up


# --- CONTROL FLOW + REPLAY: canned, no interpreters -------------------------


def _canned_one_pass() -> dict:
    """seed (fail) -> propose fix -> score (pass): the op stream of a one-round fix."""
    return {
        "seed;tool:run_tests": TestReport(passed=False, summary="add(2,3)=-1, expected 5"),
        "seed;tool:quality": QualityReport(score=1.0),
        "gen:0;tool:read_file": BUGGY,
        "gen:0;react:edit": Edit(path="calc.py", old="return a - b", new="return a + b"),
        "gen:0;tool:apply_edit": ApplyResult(ok=True, detail="applied to calc.py"),
        "cand:0,0;tool:run_tests": TestReport(passed=True),
        "cand:0,0;tool:quality": QualityReport(score=1.0),
    }


CANNED_TRACE = [
    "seed;step;tool:run_tests",
    "seed;step;tool:quality",
    "gen:0;step;tool:read_file",
    "gen:0;step;react:edit",
    "gen:0;step;tool:apply_edit",
    "cand:0,0;step;tool:run_tests",
    "cand:0,0;step;tool:quality",
]


def test_debug_loop_op_stream_and_objective_termination():
    h = RecordingHandler(_canned_one_pass())
    result = h.run(lambda: debug_loop("calc.py"))
    assert isinstance(result, DebugResult)
    assert result.passed
    assert result.iters == 1
    assert [e.key.stored() for e in h.trace] == CANNED_TRACE


def test_debug_loop_replays_without_the_model_or_tools():
    rec = RecordingHandler(_canned_one_pass())
    live = rec.run(lambda: debug_loop("calc.py"))
    replayed = ReplayHandler(rec.trace).run(lambda: debug_loop("calc.py"))
    assert replayed == live


# --- SAFETY -----------------------------------------------------------------


def test_no_safe_edit_leaves_the_file_untouched():
    from agent.runtime import LocalCtx

    ws = Workspace({"calc.py": BUGGY})
    bad = _edit_json("calc.py", "ghost", "x")  # anchor absent -> gate repairs, gives up
    client, completions = _client([bad, bad, bad])  # max_repairs=2 -> 3 attempts
    interp = make_debug_interpreter(client, ws, add_test)

    result = DebugResult.model_validate(
        DurableHandler(ctx=LocalCtx(), domain=interp).run(
            lambda: debug_loop("calc.py", max_iters=1)
        )
    )
    assert not result.passed
    assert ws.files["calc.py"] == BUGGY  # never corrupted — no patch reached the world
    assert len(completions.calls) == 3


def test_workspace_apply_rejects_a_non_unique_anchor():
    ws = Workspace({"a.py": "x = 0\nx = 0\n"})  # `x = 0` is ambiguous
    result = ws.apply("a.py", "x = 0", "x = 1")
    assert result.ok is False
    assert ws.files["a.py"] == "x = 0\nx = 0\n"  # unchanged
