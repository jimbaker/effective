"""improve: the objective loop as a combinator, GEPA-shaped.

One loop for *optimize anything*: propose candidates, score them, keep the Pareto
frontier, repeat. Single-objective and multi-objective are the **same loop**: Pareto
is the MOO variant, so arity is `len(objectives)` (with one objective the frontier is
the argmax set). The four correctness rules are structural, as with the RLM
combinators:

| rule                       | holds because                                                  |
|----------------------------|----------------------------------------------------------------|
| checkpoint per candidate   | every `score`/`propose` is a sealed op (recorded), so a crash  |
|                            | mid-round resumes without re-paying scored candidates;         |
|                            | parallel scores ride `gather`                                  |
| pure selection             | the frontier is `effective.pareto.frontier` over recorded      |
|                            | measures; replay re-derives it with zero model calls           |
| the frontier is the return | no per-candidate ledger event; the only canonical event a      |
| value, never state         | caller appends is the *promotion* of the winner                |
| budget is termination      | `rounds` bounds the search; `done` is a pure early stop        |

**ASI is the gradient (GEPA).** A `Scored` carries the objective vector and the
**actionable side-information**: the textual diagnostic the round threw off (a test
failure, a lint/ty error, a rubric note). `propose` receives the *frontier with its ASI*
and reflects on *why* before mutating: "text-optimization's analogue of the gradient." A
scalar reward discards exactly this, which is why `score` returns a `Measurement`, not a float.

The combinator adds no replay machinery: `gather` and the sealed `score` and `propose` ops
carry all durability.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel

from effective.api import Effect, gather, scoped
from effective.keys import Index, compose_key
from effective.pareto import Objective, frontier


class Measurement(BaseModel):
    """What `score` returns for one candidate: the objective vector plus the ASI — the
    textual diagnostic the evaluation produced (why it scored so). Recorded (a sealed op
    result), so the reflective signal survives replay exactly like any op result."""

    measures: dict[str, float]
    asi: str = ""  # actionable side-information: the reflective feedback (GEPA)


@dataclass(frozen=True)
class Scored[C]:
    """A candidate with its measured objectives and the ASI that explains them."""

    candidate: C
    measures: dict[str, float]
    asi: str = ""


@dataclass(frozen=True)
class Reflection:
    """The proposer's reflective context, in two parts with different lifetimes:

    - ``rubric`` — **pinned**: the standing objective / requirements / constraints (in this
      repo, the CLAUDE.md rules). It defines what "quality" *means*, so it is never
      compacted — the summarizer never even sees it. Losing it would silently redefine the
      objective mid-search.
    - ``digest`` — the running ASI history (prior diagnostics), which *is* compacted when it
      grows. The GEPA gradient, folded when it gets long.
    """

    rubric: str = ""
    digest: str = ""

    def text(self) -> str:
        return "\n\n".join(p for p in (self.rubric, self.digest) if p)


# Each callback names its ops plainly; `improve` runs it inside a `scoped(...)` (`seed`,
# `gen:{r}`, `cand:{r}:{i}`, `compact:{r}`), so a callback carries no tag of its own.
type Propose[C] = Callable[[Sequence["Scored[C]"], Reflection], Effect[Sequence[C]]]
type Score[C] = Callable[[C], Effect[Measurement]]
type Summarize = Callable[[str], Effect[str]]  # asi_history -> a shorter digest


def _measurement(m: Measurement | dict) -> Measurement:
    """Coerce a score result to `Measurement`. A direct `step` returns it typed (its
    `result_schema` coerces), but a `gather` branch returns JSON-reconstructed data (a dict)
    on the durable path — so a multi-child round trips its measures as a dict. Normalize both."""
    # lint: totality(coercion) — at a boundary. The `else` is not an arm, it is
    # `model_validate` on whatever the durable path reconstructed; naming it changes nothing.
    return m if isinstance(m, Measurement) else Measurement.model_validate(m)


def _front[C](pool: Sequence[Scored[C]], objectives: Sequence[Objective]) -> list[Scored[C]]:
    """The Pareto frontier of the pool, by measures — pure (replay re-derives it)."""
    points = [{**s.measures, "__i": i} for i, s in enumerate(pool)]
    keep = {p["__i"] for p in frontier(points, objectives)}
    return [pool[i] for i in sorted(keep)]


def improve[C](
    seed: C,
    propose: Propose[C],
    score: Score[C],
    *,
    objectives: Sequence[Objective],
    rounds: int,
    done: Callable[[Sequence[Scored[C]]], bool] = lambda _front: False,
    rubric: str = "",
    compact: Callable[[str], bool] | None = None,
    summarize: Summarize | None = None,
) -> Effect[list[Scored[C]]]:
    """Drive ``seed`` toward the objectives and return the Pareto frontier.

    ``propose`` sees the current frontier *with ASI* plus a ``Reflection`` and returns the
    next candidates (the GEPA reflection step); ``score`` measures each (parallel, via
    ``gather``). Selection and termination are pure over recorded measures. ``objectives``
    of length 1 makes the frontier the argmax singleton — same loop, no special case.
    Best-of-n is ``rounds=1``; self-refine is ``propose`` = "mutate the current front"; a
    staged MOO (get it working, then raise quality) is a ``done`` that gates on the hard
    objective before the soft one.

    **Compaction, rubric pinned.** When the ASI history grows, ``compact(history)`` (a pure
    trigger) fires a recorded ``summarize`` op that folds it into a shorter ``digest`` — but
    the ``rubric`` is *never* passed to ``summarize`` and always rides in the ``Reflection``
    verbatim, so the objective's definition survives every compaction epoch. Off by default
    (no ``compact``/``summarize`` → the reflection carries the full ASI history)."""
    seed_m = _measurement((yield from scoped(compose_key(t"seed"), lambda: score(seed))))
    pool: list[Scored[C]] = [Scored(seed, seed_m.measures, seed_m.asi)]
    digest = ""  # the compacted ASI history
    recent: list[str] = [seed_m.asi] if seed_m.asi else []  # ASI not yet folded into digest

    for r in range(rounds):
        parents = _front(pool, objectives)
        if done(parents):
            break
        # compaction seam (rubric NEVER summarized): fold the ASI history when it grows
        history = "\n".join(x for x in (digest, *recent) if x)
        if compact is not None and summarize is not None and compact(history):
            digest = yield from scoped(
                compose_key(t"compact:{Index(r)}"), lambda h=history: summarize(h)
            )
            recent = []
        reflection = Reflection(rubric=rubric, digest="\n".join(x for x in (digest, *recent) if x))
        children = list(
            (
                yield from scoped(
                    compose_key(t"gen:{Index(r)}"),
                    lambda ps=parents, rf=reflection: propose(ps, rf),
                )
            )
        )
        if not children:
            break
        if len(children) == 1:  # self-refine: score directly, no gather-branch prefix
            measured: list[Measurement] = [
                (
                    yield from scoped(
                        compose_key(t"cand:{Index(r)},{0}"), lambda c=children[0]: score(c)
                    )
                )
            ]
        else:  # a real fan-out: score the batch in parallel under structured concurrency
            measured = yield from gather(
                [
                    (
                        lambda c=c, i=i, r=r: scoped(
                            compose_key(t"cand:{Index(r)},{Index(i)}"), lambda c=c: score(c)
                        )
                    )
                    for i, c in enumerate(children)
                ]
            )
        ms = [_measurement(m) for m in measured]
        pool += [Scored(c, m.measures, m.asi) for c, m in zip(children, ms, strict=True)]
        recent += [m.asi for m in ms if m.asi]  # this round's diagnostics -> ASI history

    return _front(pool, objectives)


@dataclass(frozen=True)
class Frontier[C]:
    """A small convenience over ``improve``'s return: the frontier plus the single best by a
    named objective (the point an operator would ship when they must pick one)."""

    points: list[Scored[C]] = field(default_factory=list)

    def best(self, key: str, direction: str = "max") -> Scored[C] | None:
        if not self.points:
            return None
        pick = max if direction == "max" else min
        return pick(self.points, key=lambda s: s.measures.get(key, 0.0))
