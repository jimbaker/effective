"""bench_sweep: `improve` as a config sweep.

A best-of-n sweep over agent configs, scoring each on a multi-objective
(cost, quality, latency, …) frontier and returning the non-dominated set. It proves the
same loop generalizes past coding (different domain, identical machinery) and that Pareto
is just the MOO case (one objective collapses to argmax).

No new op: each config's evaluation is a sealed scoring op (recorded, replay-free rescore —
`scoring.py`'s property extended to a whole sweep); selection is pure `effective.pareto`; the
frontier is the return value, never state. Best-of-n is `improve` with `rounds=1` and a
fixed proposal (the grid), so nothing reflects — a reflective sweep (propose the next
configs from the frontier's ASI) is the same call with a live `propose`.
"""

from collections.abc import Callable, Sequence

from effective.api import Effect
from effective.improve import Measurement, Reflection, Scored, improve
from effective.pareto import Objective

type EvalConfig[C] = Callable[[C], Effect[Measurement]]


def bench_sweep[C](
    configs: Sequence[C],
    evaluate: EvalConfig[C],
    *,
    objectives: Sequence[Objective],
) -> Effect[list[Scored[C]]]:
    """Score every config and return the Pareto frontier. ``evaluate(cfg)`` is the
    sealed evaluation op -> ``Measurement`` (the objective vector + its ASI). Requires at
    least one config; the first is the seed, the rest are scored in parallel via ``gather``."""
    if not configs:
        raise ValueError("bench_sweep needs at least one config")
    seed, rest = configs[0], list(configs[1:])

    def propose(_parents: Sequence[Scored[C]], _refl: Reflection) -> Effect[list[C]]:
        yield from ()  # best-of-n: the grid is fixed, so no reflective proposal
        return rest

    front = yield from improve(seed, propose, evaluate, objectives=objectives, rounds=1)
    return front
