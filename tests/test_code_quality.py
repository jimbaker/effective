"""make_code_quality — the CLAUDE.md rubric, mechanized, and the debug loop driven by it.

Two proofs:
- UNIT: ruff over a candidate scores it — an unused variable drops the score below 1.0 and
  ruff's F841 message becomes the ASI; clean code scores 1.0 with no notes.
- SHOWCASE: a live debug_loop whose quality_fn IS ruff runs the staged MOO for real — round
  0 fixes the failing test (correctness), round 1 removes the unused variable the linter
  flagged (quality), driven by the recorded ruff output as the reflective signal.
"""

import json
from types import SimpleNamespace

from agent.code_quality import check_code, make_code_quality, ty_checker
from agent.debug import DebugResult, TestReport, Workspace, debug_loop, make_debug_interpreter
from effective.handlers.absurd import DurableHandler

# --- UNIT: the rubric scores a candidate and reports ruff's message as ASI ---


def test_ruff_rubric_flags_a_violation_and_scores_below_one():
    q = make_code_quality()
    report = q({"calc.py": "def add(a, b):\n    x = 1\n    return a + b\n"})  # x unused
    assert report.score < 1.0
    assert "F841" in report.notes  # the ASI is ruff's own diagnostic


def test_ruff_rubric_passes_clean_code():
    q = make_code_quality()
    report = q({"calc.py": "def add(a, b):\n    return a + b\n"})
    assert report.score == 1.0
    assert report.notes == ""


# --- the run_code connection: static quality/ASI on a code string -----------


def test_check_code_scores_a_run_code_style_script():
    # a code string of the kind run_code runs (Monty subset -> still valid Python)
    dirty = check_code("import os\nresult = 41\nresult + 1\n")  # os unused
    assert dirty.score < 1.0
    assert "F401" in dirty.notes  # ruff's message is the ASI a run_code loop reflects on
    clean = check_code("result = 41\nresult + 1\n")
    assert clean.score == 1.0


def test_ty_checker_catches_a_type_error():
    # pluggable + language-extensible: ty on self-contained code (a Monty-style script)
    q = make_code_quality((ty_checker(),))
    report = q({"m.py": "def f(x: int) -> int:\n    return x + 's'\n"})
    assert report.score < 1.0
    assert "error[" in report.notes  # ty's diagnostic is the ASI


# --- SHOWCASE: debug_loop with ruff as the quality gate, staged for real -----

# buggy AND unclean: it subtracts (test fails) and leaves an unused `x` (ruff fails)
BUGGY_UNCLEAN = "def add(a, b):\n    x = 1\n    return a - b\n"


def add_test(files) -> TestReport:
    ns: dict = {}
    exec(files["calc.py"], ns)  # test-owned content — running it IS the objective
    got = ns["add"](2, 3)
    return TestReport(passed=got == 5, summary="" if got == 5 else f"add(2,3)={got}, expected 5")


def _client(contents: list[str]):
    class _Create:
        def __init__(self, xs):
            self.xs, self.calls = list(xs), []

        def create(self, **kw):
            self.calls.append(kw)
            usage = SimpleNamespace(
                prompt_tokens=90,
                completion_tokens=12,
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            )
            msg = SimpleNamespace(content=self.xs.pop(0))
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)

    c = _Create(contents)
    return SimpleNamespace(chat=SimpleNamespace(completions=c)), c


def _edit(old: str, new: str) -> str:
    return json.dumps({"edit": {"path": "calc.py", "old": old, "new": new}})


def test_debug_loop_stages_fix_then_lint_cleanup_against_real_ruff():
    from agent.runtime import LocalCtx

    ws = Workspace({"calc.py": BUGGY_UNCLEAN})
    client, completions = _client(
        [
            _edit("return a - b", "return a + b"),  # round 0: fix correctness
            _edit("    x = 1\n", ""),  # round 1: remove the unused var ruff flagged
        ]
    )
    interp = make_debug_interpreter(client, ws, add_test, quality_fn=make_code_quality())
    handler = DurableHandler(ctx=LocalCtx(), domain=interp)

    # quality_floor=1.0 forces a second pass: tests clear in round 0, ruff clears in round 1
    result = DebugResult.model_validate(
        handler.run(lambda: debug_loop("calc.py", quality_floor=1.0))
    )

    assert result.passed  # correctness met
    assert result.quality == 1.0  # AND ruff is clean
    assert result.iters == 2  # the staged second pass happened
    assert ws.files["calc.py"] == "def add(a, b):\n    return a + b\n"
    assert add_test(ws.files).passed
    assert len(completions.calls) == 2
