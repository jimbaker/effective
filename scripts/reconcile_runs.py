"""Two-books reconciliation for edit-mode runs: a telemetry cross-check.

Reconciles the **actor's** self-telemetry (the run's `summary.json` and `spans.jsonl`,
Effective's own account of what it did and spent) against an **independent observer**: a
fresh read of the result file with real `ruff`, `compile` and a control-char scan. The
observer never runs the actor's own gate code, so an over-claim cannot hide. Emits:

| output                 | one row per run                                                   |
|------------------------|-------------------------------------------------------------------|
| reconciliation         | actor claim against observer verdict, AGREE or DIVERGE; a         |
|                        | divergence means the books disagree with what an outsider saw     |
| mode Pareto table      | the priced points an edit-mode policy selects over: cost, calls,  |
|                        | wall time, outcome                                                |

The observer is deterministic and so costs nothing to run.

Usage: ``uv run python scripts/reconcile_runs.py [build/nano-run-* ...]``
(no args → every ``build/nano-run-*`` dir).
"""

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from effective.telemetry import decode_otlp_line

REPO = Path(__file__).resolve().parent.parent

# Map an evidence-dir suffix to the edit mode it exercised (the Pareto axis label).
_MODE_BY_SUFFIX = {
    "writefile": "whole-file",
    "batch": "blind-batch",
    "sequential": "sequential-visibility",
    "r2": "small-anchor",
}
_ALLOWED_WS = frozenset("\n\t\r")


def _mode_of(name: str) -> str:
    for suffix, mode in _MODE_BY_SUFFIX.items():
        if name.endswith(suffix):
            return mode
    return "small-anchor"  # the early real-gate runs (no distinguishing suffix)


@dataclass
class Actor:
    """Effective's own account of the run (self-telemetry)."""

    mode: str
    claimed_closed: bool
    cost_usd: float
    calls: int
    wall_s: float
    ops: dict[str, int]  # span-name -> count (the op breakdown from spans.jsonl)


# A per-task goal-met predicate: given the run's result dir, was the intended change actually
# made? (`(dir) -> (ok, reason)`.) The static gate is necessary and insufficient: a no-op run
# leaves a gate-clean file, so a cost-minimizing policy would game close-rate. A caller supplies
# one per task; the generic CLI leaves it None (static verdict only).
GoalMet = Callable[[Path], "tuple[bool, str]"]


@dataclass
class Observer:
    """An independent read of the result — fresh tools, not the actor's gate. ``verdict`` is a
    TRUE close only when the static gate is clean AND (if a goal-met predicate was supplied) the
    intended change was actually made; `goal_met` records that second gate's outcome."""

    verdict: str  # "clean" | "dirty" | "goal-unmet" | "no-result"
    reason: str
    goal_met: bool | None = None


def actor_account(d: Path) -> Actor | None:
    """Parse the actor's `summary.json` (outcome/cost/calls) + `spans.jsonl` (op counts)."""
    summary = d / "summary.json"
    if not summary.exists():
        return None
    s = json.loads(summary.read_text())
    result = s.get("result") or {}
    # "closed" is expressed per-mode: sequential -> result.closed; others -> result.applied.
    claimed = bool(result.get("closed") if "closed" in result else result.get("applied"))
    # Prefer an EXPLICIT mode written by the matrix driver; fall back to the dir-name suffix.
    mode = str(s["mode"]) if s.get("mode") else _mode_of(d.name)
    ops: dict[str, int] = {}
    spans = d / "spans.jsonl"
    if spans.exists():
        for line in spans.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for span in decode_otlp_line(row):
                ops[span["name"]] = ops.get(span["name"], 0) + 1
    return Actor(
        mode=mode,
        claimed_closed=claimed,
        cost_usd=float(s.get("cost_usd", 0.0)),
        calls=int(s.get("nano_calls", 0)),
        wall_s=float(s.get("wall_s", 0.0)),
        ops=ops,
    )


def _control_char(content: str) -> str | None:
    """A FRESH control-char scan (independent of the actor's `_nonprintable_diagnostic`) — the
    fidelity check that caught the whole-file DEL over-claim by hand."""
    line = 1
    for ch in content:
        if ch == "\n":
            line += 1
        elif ch not in _ALLOWED_WS and not ch.isprintable():
            return f"non-printable U+{ord(ch):04X} at line {line}"
    return None


def _static_dirty(f: Path) -> str | None:
    """The independent static verdict for ONE file: control-char scan → `compile` → real `ruff`.
    Returns the first dirty reason, or ``None`` if clean. Fresh tools, not the actor's gate."""
    content = f.read_text()
    if (cc := _control_char(content)) is not None:
        return f"{f.name}: fidelity: {cc}"  # the DEL-char class — real ruff misses it
    try:
        compile(content, f.name, "exec")
    except SyntaxError as e:
        return f"{f.name}: syntax: {e.msg} (line {e.lineno})"
    proc = subprocess.run(
        [
            "ruff",
            "check",
            "--quiet",
            "--no-fix",
            "--output-format=concise",
            "--config",
            str(REPO / "pyproject.toml"),
            str(f),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        first = next((ln for ln in proc.stdout.splitlines() if ":" in ln), proc.stdout.strip())
        return f"ruff: {first.strip()[:80]}"
    return None


def observer_verdict(d: Path, goal_met: GoalMet | None = None) -> Observer:
    """Independently judge the result: `ruff` + `compile` + control-char scan folded over EVERY
    `*.py` in the copy dir (a task may be multi-file), then, if a `goal_met` predicate is
    supplied, whether the intended change was actually made. A TRUE close needs both. It never
    imports the actor's gate, which is what makes the observer independent."""
    copy = d / "copy_effective"
    results = sorted(copy.rglob("*.py")) if copy.exists() else []
    if not results:
        return Observer("no-result", "no result file to read")
    for f in results:  # fold over the whole tree — the first dirty file loses
        if (reason := _static_dirty(f)) is not None:
            return Observer("dirty", reason)
    if goal_met is not None:
        ok, why = goal_met(d)
        if not ok:
            return Observer("goal-unmet", f"goal not met: {why}", goal_met=False)
        return Observer("clean", f"gate-clean + goal met: {why}", goal_met=True)
    return Observer("clean", f"ruff + parse + fidelity clean ({len(results)} file(s))")


def reconcile(d: Path, goal_met: GoalMet | None = None) -> tuple[Actor, Observer, str] | None:
    """Actor vs observer. A TRUE close = the observer verdict is ``clean`` (static gate + — when a
    ``goal_met`` predicate is supplied — the intended change actually made). DIVERGE (a finding)
    when the actor asserted closure but the independent read is not a true close: the whole-file
    DEL-char case (dirty) OR a no-op that games the gate (goal-unmet). A STALLED run asserts
    nothing, so a clean-but-unchanged file is CONSISTENT, never a divergence."""
    actor = actor_account(d)
    if actor is None:
        return None
    obs = observer_verdict(d, goal_met)
    agreement = "DIVERGE" if (actor.claimed_closed and obs.verdict != "clean") else "AGREE"
    return actor, obs, agreement


def main(argv: list[str]) -> int:
    dirs = [Path(a) for a in argv] if argv else sorted((REPO / "build").glob("nano-run-*"))
    rows = [r for d in dirs if (r := reconcile(d)) is not None]
    if not rows:
        print("no reconcilable runs found (need build/nano-run-*/summary.json)", file=sys.stderr)
        return 1

    print("\n═══ Two-books reconciliation (actor self-telemetry vs independent observer) ═══")
    print(f"{'mode':<22}{'actor':<10}{'observer':<10}{'verdict':<9}reason")
    print("─" * 90)
    diverged = 0
    for actor, obs, agreement in rows:
        if agreement == "DIVERGE":
            diverged += 1
        claim = "closed" if actor.claimed_closed else "stalled"
        print(f"{actor.mode:<22}{claim:<10}{obs.verdict:<10}{agreement:<9}{obs.reason}")

    print("\n═══ Edit-mode Pareto table (priced points for an edit-mode policy) ═══")
    print(f"{'mode':<22}{'outcome':<9}{'cost $':<10}{'calls':<7}{'wall s':<8}op-spans")
    print("─" * 90)
    for actor, _obs, _ in sorted(rows, key=lambda r: r[0].cost_usd):
        outcome = "CLOSED" if actor.claimed_closed else "STALL"
        ops = ",".join(f"{k}:{v}" for k, v in sorted(actor.ops.items())) or "—"
        print(
            f"{actor.mode:<22}{outcome:<9}{actor.cost_usd:<10.4f}{actor.calls:<7}{actor.wall_s:<8.1f}{ops}"
        )

    print(
        f"\nreconciled {len(rows)} runs · {diverged} DIVERGENCE(S) "
        f"(actor claim ≠ independent observer — a finding to investigate)"
    )
    print(
        "observer cost: $0 (deterministic ruff+parse+fidelity); a teacher LLM would add the "
        "mentoring economics split here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
