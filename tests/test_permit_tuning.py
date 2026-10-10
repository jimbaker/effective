"""Tuning the permit policy with improve (B-V0) — the classifier x improve junction.

- score_policy: offline replay of a recorded (op, human-verdict) corpus under a candidate
  policy counts false-allows and the auto-rate (pure, so replay-free and zero-cost).
- safest: the operator's pick from the frontier — zero false-allows, then max auto-rate.
- tune_permits: a live sweep (improve/bench_sweep over LocalCtx) finds the safe high-auto
  policy and drops the dominated ones — the whole tuning loop, no model calls.
"""

from agent.permit_tuning import (
    AUTO_RATE,
    FALSE_ALLOW,
    Labeled,
    make_policy_scorer,
    safest,
    score_policy,
    tune_permits,
)
from effective.cost import CostBudget, MeteredInterpreter
from effective.handlers.durable import DurableHandler
from effective.improve import Scored
from effective.permission import PermitPolicy

# reads and tests are always approved; apply_edit is sometimes denied (the risky op)
CORPUS = [
    Labeled(op_key="tool:read_file,0", verdict="allow"),
    Labeled(op_key="tool:read_file,1", verdict="allow"),
    Labeled(op_key="tool:apply_edit,0", verdict="allow"),
    Labeled(op_key="tool:apply_edit,1", verdict="deny"),
    Labeled(op_key="tool:run_tests,0", verdict="allow"),
]


# --- score_policy: offline replay, pure ------------------------------------


def test_score_policy_counts_false_allows_and_auto_rate():
    safe = PermitPolicy(allow=("tool:read_file", "tool:run_tests"))  # never touches the deny
    m = score_policy(CORPUS, safe)
    assert m.measures[FALSE_ALLOW] == 0.0
    assert m.measures[AUTO_RATE] == 3 / 5

    greedy = PermitPolicy(allow=("tool:",))  # auto-allows everything, incl. the denied edit
    g = score_policy(CORPUS, greedy)
    assert g.measures[FALSE_ALLOW] == 1.0
    assert g.measures[AUTO_RATE] == 1.0
    assert "tool:apply_edit,1" in g.asi  # the mis-allowed op is the ASI (the gradient)


# --- safest: the hard-gated pick over the frontier -------------------------


def test_safest_takes_zero_false_allow_then_max_auto():
    front = [
        Scored(PermitPolicy(allow=("tool:read_file",)), {FALSE_ALLOW: 0.0, AUTO_RATE: 0.4}),
        Scored(
            PermitPolicy(allow=("tool:read_file", "tool:run_tests")),
            {FALSE_ALLOW: 0.0, AUTO_RATE: 0.6},
        ),
        Scored(
            PermitPolicy(allow=("tool:",)), {FALSE_ALLOW: 1.0, AUTO_RATE: 1.0}
        ),  # higher auto, unsafe
    ]
    pick = safest(front)
    assert pick is not None
    assert pick.candidate.allow == ("tool:read_file", "tool:run_tests")  # safe + highest auto


def test_safest_is_none_when_every_candidate_is_unsafe():
    front = [Scored(PermitPolicy(allow=("tool:",)), {FALSE_ALLOW: 1.0, AUTO_RATE: 1.0})]
    assert safest(front) is None


# --- tune_permits: the live sweep over LocalCtx ----------------------------

CANDIDATES = [
    PermitPolicy(allow=()),  # allow nothing: safe but useless (auto 0)
    PermitPolicy(allow=("tool:read_file",)),  # safe, auto 0.4 -> dominated by the next
    PermitPolicy(allow=("tool:read_file", "tool:run_tests")),  # safe, auto 0.6
    PermitPolicy(allow=("tool:",)),  # greedy: auto 1.0 but false_allow 1 (the tradeoff point)
]


def _no_llm(op):
    raise AssertionError("a permit sweep makes no model calls")


def test_tune_permits_finds_the_safe_high_auto_policy():
    from effective.contexts import LocalCtx

    interp = MeteredInterpreter(
        llm=_no_llm, tools=make_policy_scorer(CORPUS), budget=CostBudget(1.0)
    )
    raw = DurableHandler(ctx=LocalCtx(), domain=interp).run(lambda: tune_permits(CANDIDATES))

    # the frontier round-trips through JSON as a list of Scored dicts
    front = {tuple(s["candidate"]["allow"]): s["measures"] for s in raw}
    # the two dominated safe policies drop out; the safe-best and the greedy tradeoff remain
    assert ("tool:read_file", "tool:run_tests") in front  # safe, auto 0.6
    assert ("tool:",) in front  # unsafe but higher auto -> a real Pareto point
    assert () not in front  # dominated
    assert ("tool:read_file",) not in front  # dominated by read_file+run_tests

    # the operator's pick: zero false-allow, max auto
    safe = {k: m for k, m in front.items() if m[FALSE_ALLOW] == 0}
    assert max(safe, key=lambda k: safe[k][AUTO_RATE]) == ("tool:read_file", "tool:run_tests")
