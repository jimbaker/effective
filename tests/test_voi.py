"""VOI probe deliverables: the informed grant prompt and the informed-vs-blind DoE that scores
prediction P1.

P1 (preregistered): informed grants beat a fixed policy over a mixed, headroom-stratified batch
**only when `probe ≪ grant`**; when the probe costs as much as the grant it informs, informed is
dominated. Scored here on a synthetic task family driven through the REAL fork driver
(`effective.fork`/`agent.voi`): the (quality, cost) come from actual `ForkTail.usage` and an
external grader. Infra-free.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from agent.voi import GrantPrompt, grant_full, probe_prompt, should_grant
from effective.api import step
from effective.budget import depth_grant_name
from effective.combinators import Answered, Deeper, descend
from effective.cost import Usage
from effective.domain import CallTool
from effective.handlers.recording import RecordingHandler, Suspended

UNIT = 0.001  # dollars per level (the surrogate's per-call cost)
THRESHOLD = 0.10  # grant iff a probe reveals >= this quality gain
GRANT_N = 4  # the blind grant size


def _judge(ctx, level):
    # the level's depth rides in the op args so the domain can produce a depth-tagged answer
    raw = yield from step(
        "judge",
        CallTool(name="judge", args={"depth": level.depth}, result_schema=str),
    )
    return Answered(raw) if level.final else Deeper(raw)


def _program():
    return (yield from descend("ctx", _judge, budget=1, run_id="r"))


class DepthDomain:
    """A metered surrogate: each level returns a depth-tagged answer at `UNIT` cost. Depth
    comes from the judge's op args, so the answer encodes how deep the drill went."""

    def run_metered(self, op: CallTool) -> tuple[str, Usage]:
        return f"d{op.args['depth']}", Usage(cost=UNIT)


@dataclass(frozen=True)
class Task:
    """A synthetic task = a quality-by-depth curve (the ground truth an external grader
    knows). `answer` strings are `d{depth}`, so the grader reads depth and returns quality."""

    name: str
    curve: Callable[[int], float]

    def grader(self, answer: str) -> float:
        return self.curve(int(answer[1:]))


SATURATED = Task("saturated", lambda d: 0.9)  # already good at d0 — no headroom
HEADROOM = Task("headroom", lambda d: min(1.0, 0.4 + 0.15 * d))  # improves with depth
BATCH = [SATURATED, HEADROOM]


def _parked_base() -> RecordingHandler:
    base = RecordingHandler(responses={"d:0;judge": "d0"})  # the recorded current best is d0
    parked = base.run(_program)
    assert isinstance(parked, Suspended)
    assert parked.awaiting == depth_grant_name("r", depth=1, generation=0)
    return base


# ---------------------------------------------------------------- the acceptance demo


def _prompt_for(task: Task, probe_levels: int = 1) -> GrantPrompt:
    trace = _parked_base().trace
    return probe_prompt(
        _program, trace, "r", 1, DepthDomain(), task.grader, probe_levels=probe_levels
    )


def test_informed_grant_prompt_shows_a_priced_graded_marginal():
    # The demo artifact: free current best vs a cheap one-level probe, each priced and graded.
    prompt = _prompt_for(HEADROOM)
    assert prompt.current_best.cost == 0.0  # free-and-exact
    assert prompt.current_best.quality == pytest.approx(0.4)  # d0
    assert prompt.probe.cost == pytest.approx(UNIT)  # exactly one level
    assert prompt.probe.quality == pytest.approx(0.55)  # d1
    assert prompt.marginal_quality == pytest.approx(0.15)
    assert should_grant(prompt, THRESHOLD) is True  # headroom -> grant


def test_informed_prompt_declines_a_saturated_task():
    prompt = _prompt_for(SATURATED)
    assert prompt.marginal_quality == pytest.approx(0.0)  # probing buys nothing
    assert should_grant(prompt, THRESHOLD) is False  # -> don't grant (avoid the waste)


# ---------------------------------------------------------------- the DoE (P1)


def _outcome_always_stop(task: Task) -> tuple[float, float]:
    current = _parked_base().trace[-1].result  # d0, free
    return task.grader(current), 0.0


def _outcome_always_grant(task: Task) -> tuple[float, float]:
    tail = grant_full(_program, _parked_base().trace, "r", 1, DepthDomain(), GRANT_N)
    return task.grader(tail.result), tail.usage.cost


def _outcome_informed(task: Task, probe_levels: int) -> tuple[float, float]:
    trace = _parked_base().trace
    prompt = probe_prompt(
        _program, trace, "r", 1, DepthDomain(), task.grader, probe_levels=probe_levels
    )
    if not should_grant(prompt, THRESHOLD):
        return prompt.current_best.quality, prompt.probe.cost  # stopped, but paid the probe
    tail = grant_full(_program, trace, "r", 1, DepthDomain(), GRANT_N)
    return task.grader(tail.result), prompt.probe.cost + tail.usage.cost  # re-run: probe + grant


def _totals(outcome: Callable[[Task], tuple[float, float]]) -> tuple[float, float]:
    quality = cost = 0.0
    for task in BATCH:
        q, c = outcome(task)
        quality += q
        cost += c
    return quality, cost


def test_P1_informed_dominates_blind_when_probe_is_cheap():
    stop_q, _stop_c = _totals(_outcome_always_stop)
    grant_q, grant_c = _totals(_outcome_always_grant)
    cheap_q, cheap_c = _totals(lambda t: _outcome_informed(t, probe_levels=1))

    # informed matches always-grant's quality (it captures the headroom task) ...
    assert cheap_q == pytest.approx(grant_q)
    # ... at strictly lower cost (it skipped the wasteful grant on the saturated task) ...
    assert cheap_c < grant_c
    # ... and beats always-stop on quality (stop is cheap, leaves headroom on the table).
    assert cheap_q > stop_q


def test_P1_informed_loses_when_the_probe_costs_as_much_as_the_grant():
    grant_q, grant_c = _totals(_outcome_always_grant)
    # a degenerate probe (as deep as the grant it informs) — the requirement #3 violation
    deg_q, deg_c = _totals(lambda t: _outcome_informed(t, probe_levels=GRANT_N))

    assert deg_q == pytest.approx(grant_q)  # same quality ...
    # ... but now MORE expensive: informed is dominated (the probe ≪ grant requirement fails)
    assert deg_c > grant_c
