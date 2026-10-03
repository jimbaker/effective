"""bench_sweep — improve generalized to a config sweep (design-space §3 consumer #2).

Proves the same objective loop works off the coding domain: score a grid of agent configs
on a (cost, quality) frontier, keep the non-dominated set, drop the dominated — all as a
recorded op stream that replays with no evaluation. This is the second consumer that earns
`improve` its slot.
"""

from collections.abc import Callable

from agent.bench_sweep import bench_sweep
from effective import Effect, RecordingHandler, ReplayHandler, Suspended, step
from effective.domain import CallTool
from effective.improve import Frontier, Measurement
from effective.pareto import Objective


def _run[T](handler, wf: Callable[[], Effect[T]]) -> T:
    """Run and narrow away the never-taken Suspended branch (the sweep doesn't await)."""
    out = handler.run(wf)
    assert not isinstance(out, Suspended)
    return out


# a grid: `slow` is strictly worse than `nano` on both axes -> it must be dropped
CONFIGS = ["nano", "mini", "big", "slow"]
MEASURES = {
    "nano": {"cost": 1.0, "quality": 0.6},
    "mini": {"cost": 3.0, "quality": 0.8},
    "big": {"cost": 10.0, "quality": 0.9},
    "slow": {"cost": 5.0, "quality": 0.5},  # dominated by nano (worse cost AND quality)
}
OBJECTIVES = [Objective("cost", "min"), Objective("quality", "max")]


def _evaluate(cfg):
    """A sealed evaluation op — canned by name in the recorder."""
    m = yield from step(
        "bench", CallTool(name="bench", result_schema=Measurement, args={"cfg": cfg})
    )
    return m


def _responses() -> dict:
    # seed (nano) scored directly; the other three in parallel via gather (bare op keys)
    return {
        "seed;bench": Measurement(measures=MEASURES["nano"], asi="cheapest"),
        "gather:0,0;cand:0,0;bench": Measurement(measures=MEASURES["mini"], asi="balanced"),
        "gather:0,1;cand:0,1;bench": Measurement(measures=MEASURES["big"], asi="best quality"),
        "gather:0,2;cand:0,2;bench": Measurement(measures=MEASURES["slow"], asi="dominated"),
    }


def test_sweep_returns_the_frontier_and_drops_the_dominated():
    h = RecordingHandler(_responses())
    front = _run(h, lambda: bench_sweep(CONFIGS, _evaluate, objectives=OBJECTIVES))
    on_frontier = {s.candidate for s in front}
    assert on_frontier == {"nano", "mini", "big"}  # slow is dominated -> excluded
    # the operator's single pick per preference (the convenience over the frontier)
    f = Frontier(front)
    best_quality = f.best("quality")
    best_cost = f.best("cost", "min")
    assert best_quality is not None
    assert best_quality.candidate == "big"
    assert best_cost is not None
    assert best_cost.candidate == "nano"


def test_single_objective_sweep_is_argmax():
    # the MOO/scalar unification: one objective collapses the frontier to the winner
    h = RecordingHandler(_responses())
    front = _run(
        h, lambda: bench_sweep(CONFIGS, _evaluate, objectives=[Objective("quality", "max")])
    )
    assert {s.candidate for s in front} == {"big"}  # just the argmax


def test_sweep_replays_without_evaluation():
    rec = RecordingHandler(_responses())
    live = _run(rec, lambda: bench_sweep(CONFIGS, _evaluate, objectives=OBJECTIVES))
    replayed = _run(
        ReplayHandler(rec.trace), lambda: bench_sweep(CONFIGS, _evaluate, objectives=OBJECTIVES)
    )
    assert {s.candidate for s in replayed} == {s.candidate for s in live}
