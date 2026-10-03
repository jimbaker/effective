"""Two-books reconciliation tool: the telemetry cross-check.

Proves the reconciler's honesty logic on synthetic evidence dirs: an OVER-claim (actor
says closed, an independent read finds the result dirty) is flagged DIVERGE; a clean
close and a stall are both AGREE. Also the fresh control-char scan (the fidelity check
that flags a whole-file DEL over-claim). Shells out to `ruff`; skips if absent.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.reconcile_runs import _control_char, reconcile  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not available")

_CLEAN = "x = 1\n"
_DEL = "x = 1  \x7f\n"  # a control char real ruff/parse accept — the fidelity floor's target


def _rec(d: Path):
    """`reconcile` narrowed to non-None (the test dirs always have a `summary.json`)."""
    r = reconcile(d)
    assert r is not None
    return r


def _make_run(
    root: Path, name: str, *, closed: bool, content: str | None, mode: str | None = None
) -> Path:
    d = root / name
    (d / "copy_effective").mkdir(parents=True)
    result = {"closed": True} if closed else {"applied": False, "reason": "stalled"}
    summary = {"state": "completed", "result": result, "cost_usd": 0.005, "nano_calls": 5}
    if mode is not None:
        summary["mode"] = mode
    (d / "summary.json").write_text(json.dumps(summary))
    if content is not None:
        (d / "copy_effective" / "mod.py").write_text(content)
    return d


def test_control_char_catches_del_but_allows_unicode():
    assert _control_char(_CLEAN) is None
    assert _control_char('"""café — naïve 日本語."""\n') is None  # normal Unicode is printable
    hit = _control_char(_DEL)
    assert hit is not None
    assert "U+007F" in hit


def test_overclaim_closed_but_dirty_is_flagged_diverge(tmp_path):
    """The core two-books catch: the actor claims closed, the independent read finds a
    control-char-dirty result → DIVERGE (the whole-file DEL over-claim, automated)."""
    d = _make_run(tmp_path, "nano-run-x-writefile", closed=True, content=_DEL)
    actor, obs, agreement = _rec(d)
    assert actor.claimed_closed is True
    assert obs.verdict == "dirty"
    assert "U+007F" in obs.reason
    assert agreement == "DIVERGE"


def test_clean_close_agrees(tmp_path):
    d = _make_run(tmp_path, "nano-run-x-sequential", closed=True, content=_CLEAN)
    actor, obs, agreement = _rec(d)
    assert actor.claimed_closed is True
    assert obs.verdict == "clean"
    assert agreement == "AGREE"


def test_stall_leaving_a_clean_file_is_consistent_not_a_divergence(tmp_path):
    """A stalled run asserts nothing, so a clean-but-unchanged result is CONSISTENT — the
    reconciler must NOT flag it (the bug the first pass had)."""
    d = _make_run(tmp_path, "nano-run-x-batch", closed=False, content=_CLEAN)
    actor, obs, agreement = _rec(d)
    assert actor.claimed_closed is False
    assert obs.verdict == "clean"
    assert agreement == "AGREE"


def test_observer_flags_a_syntax_error(tmp_path):
    d = _make_run(tmp_path, "nano-run-x-sequential", closed=True, content="def f(:\n")
    _, obs, agreement = _rec(d)
    assert obs.verdict == "dirty"
    assert "syntax" in obs.reason
    assert agreement == "DIVERGE"


def test_mode_label_inferred_from_dir_suffix(tmp_path):
    cases = [
        ("writefile", "whole-file"),
        ("batch", "blind-batch"),
        ("sequential", "sequential-visibility"),
    ]
    for suffix, mode in cases:
        d = _make_run(tmp_path, f"nano-run-x-{suffix}", closed=True, content=_CLEAN)
        actor, _, _ = _rec(d)
        assert actor.mode == mode


def test_no_result_file_with_stall_agrees(tmp_path):
    d = tmp_path / "nano-run-x-r2"
    d.mkdir()
    summary = {"state": "completed", "result": {"applied": False}, "cost_usd": 0.001}
    (d / "summary.json").write_text(json.dumps(summary))
    _actor, obs, agreement = _rec(d)
    assert obs.verdict == "no-result"
    assert agreement == "AGREE"


# ── tree-fold observer, goal-met predicate, explicit mode ──
def test_observer_folds_over_the_whole_tree_not_just_the_first_file(tmp_path):
    """A multi-file task: a clean first file must NOT mask a dirty second file."""
    d = _make_run(tmp_path, "nano-run-x-sequential", closed=True, content=_CLEAN)
    (d / "copy_effective" / "other.py").write_text(_DEL)  # a second, DEL-dirty file
    _actor, obs, agreement = _rec(d)
    assert obs.verdict == "dirty"
    assert "other.py" in obs.reason
    assert agreement == "DIVERGE"


def test_goal_unmet_on_a_gate_clean_noop_is_not_a_true_close(tmp_path):
    """The load-bearing §4b gap: a no-op run leaves a gate-clean file, so `closed` must also
    require the goal met — else a cost-minimizing policy games close-rate. Actor claims closed,
    the file is clean, but the goal-met predicate fails → goal-unmet → DIVERGE."""
    d = _make_run(tmp_path, "nano-run-x-batch", closed=True, content=_CLEAN)

    def goal_met(_run_dir):
        return (False, "expected symbol absent")

    r = reconcile(d, goal_met=goal_met)
    assert r is not None
    _actor, obs, agreement = r
    assert obs.verdict == "goal-unmet"
    assert obs.goal_met is False
    assert agreement == "DIVERGE"


def test_goal_met_and_gate_clean_is_a_true_close(tmp_path):
    d = _make_run(tmp_path, "nano-run-x-batch", closed=True, content=_CLEAN)

    def goal_met(_run_dir):
        return (True, "symbol present")

    r = reconcile(d, goal_met=goal_met)
    assert r is not None
    _actor, obs, agreement = r
    assert obs.verdict == "clean"
    assert obs.goal_met is True
    assert agreement == "AGREE"


def test_explicit_mode_in_summary_overrides_dir_suffix(tmp_path):
    # dir suffix says "batch", but the matrix driver wrote an explicit mode → the explicit wins.
    d = _make_run(tmp_path, "nano-run-x-batch", closed=True, content=_CLEAN, mode="structural")
    actor, _obs, _ = _rec(d)
    assert actor.mode == "structural"


def test_the_actor_counts_ops_by_span_name_from_the_otlp_span_file(tmp_path):
    from effective.telemetry import Span, otlp_jsonl_sink
    from scripts.reconcile_runs import actor_account

    (tmp_path / "summary.json").write_text(json.dumps({"result": {"closed": True}}))
    sink = otlp_jsonl_sink(tmp_path / "spans.jsonl", clock=lambda: 1)
    sink(Span(name="LM.0", kind="LLM", session_id="s"))
    sink(Span(name="tool.read.0", kind="TOOL", session_id="s", tool_name="read"))
    sink(Span(name="tool.read.0", kind="TOOL", session_id="s", tool_name="read", attempt=1))

    actor = actor_account(tmp_path)
    assert actor is not None
    assert actor.ops == {"chat": 1, "execute_tool read": 2}
