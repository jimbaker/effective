"""Tuning the permit policy with `improve`: the classifier x improve junction.

The auto-mode allow-table (`effective.permission.PermitPolicy`/`allow_table`) is *data*, so it is
an `improve` candidate. The trick is the score: **offline replay** of a recorded `(op_key,
human_verdict)` corpus under a candidate policy — counterfactual policy evaluation at **zero live
risk and zero model calls** (the determinism dividend: every human approval decision is already a
recorded label). Objectives:

- ``false_allow`` (**min**, the hard safety gate): ops the policy auto-allowed that the
  human denied. A wrong auto-allow is the only expensive error; drive it to zero first.
- ``auto_rate`` (**max**): the fraction auto-settled, i.e. the reviewer friction we cut.

ASI = the ops the policy wrongly auto-allowed, rendered with the human's verdict — the GEPA
gradient the next policy fixes. The winner is a `PermitPolicy` whose digest a caller promotes
to the ledger (the only canonical event; `improve`'s reification rule).

Imports `effective` and `agent` only.
"""

from collections.abc import Sequence

from pydantic import BaseModel

from agent.bench_sweep import bench_sweep
from effective.api import Effect, step
from effective.domain import CallTool
from effective.improve import Measurement, Scored
from effective.keys import Key
from effective.pareto import Objective
from effective.permission import PermitPolicy

FALSE_ALLOW = "false_allow"
AUTO_RATE = "auto_rate"
PERMIT_OBJECTIVES = [Objective(FALSE_ALLOW, "min"), Objective(AUTO_RATE, "max")]


class Labeled(BaseModel):
    """One recorded decision from a deployment's approval queue: the op's key and the human's
    ground-truth verdict (``"allow"`` / ``"deny"``). The corpus of these is the classifier's
    labels."""

    op_key: str
    verdict: str  # "allow" | "deny"


def score_policy(corpus: Sequence[Labeled], policy: PermitPolicy) -> Measurement:
    """Replay ``corpus`` under ``policy`` (pure): count false-allows (auto-allowed but the
    human denied) and the auto-rate. The ASI is the false-allowed ops — the gradient."""
    false_allowed = [
        row for row in corpus if policy.permits(Key.parse(row.op_key)) and row.verdict == "deny"
    ]
    auto = sum(1 for row in corpus if policy.permits(Key.parse(row.op_key)))
    total = len(corpus) or 1
    asi = (
        ""
        if not false_allowed
        else "wrongly auto-allowed:\n"
        + "\n".join(f"{row.op_key} (human: deny)" for row in false_allowed)
    )
    return Measurement(
        measures={FALSE_ALLOW: float(len(false_allowed)), AUTO_RATE: auto / total}, asi=asi
    )


def make_policy_scorer(corpus: Sequence[Labeled]):
    """A ToolRunner for the ``score_policy`` op: offline replay of the recorded corpus under
    the candidate policy carried in the op args. Deterministic, so replay reuses the result."""

    def run(op: CallTool) -> Measurement:
        return score_policy(corpus, PermitPolicy.model_validate(op.args["policy"]))

    return run


def tune_permits(
    candidates: Sequence[PermitPolicy],
) -> Effect[list[Scored[PermitPolicy]]]:
    """A best-of-n sweep over candidate allow-tables, scored by offline replay (a
    ``bench_sweep`` over ``improve``). Returns the Pareto frontier of (false_allow, auto_rate)
    — the safety/friction menu. Use ``safest`` to pick the operator's default."""

    def evaluate(policy: PermitPolicy) -> Effect[Measurement]:
        m = yield from step(
            "permit:score",
            CallTool(
                name="score_policy",
                result_schema=Measurement,
                args={"policy": policy.model_dump()},
            ),
        )
        return m

    return bench_sweep(candidates, evaluate, objectives=PERMIT_OBJECTIVES)


def safest(frontier: Sequence[Scored[PermitPolicy]]) -> Scored[PermitPolicy] | None:
    """The operator's pick from the frontier: **zero false-allows** (the hard gate), then the
    highest auto-rate. ``None`` if no candidate is safe (every one auto-allows a denied op)."""
    safe = [s for s in frontier if s.measures[FALSE_ALLOW] == 0]
    return max(safe, key=lambda s: s.measures[AUTO_RATE]) if safe else None
