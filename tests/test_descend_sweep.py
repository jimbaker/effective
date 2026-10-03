"""A sampled property sweep over `descend`: the drill's SPACE, not one path.

Role: **sampled property**. Its journey-side complement is
`tests/test_combinators.py`, which drives descend along chosen paths, and
`tests/test_conformance.py`, which drives one path on both engines. This asserts what must hold
over the space those pick points out of: budget, the depth an answer arrives at, how many refills
a grantor authorizes, and whether the drill sits inside a gather branch.

**The oracle is written before the generator**, and its center is `expected_levels`: the depths
and finality flags a case must produce, computed as arithmetic from the CASE. An invariant
re-derived from the run it judges is a mirror, and `expected_levels(case) == observed` is the one
comparison here that cannot be, since the case is written down before anything runs.

What the sampling reaches that a chosen path does not is the *interaction* of budget with grants:
a drill that exhausts, is granted more, and exhausts again, several times in one run.
"""

from dataclasses import dataclass, field
from functools import cache

import pytest

from effective.api import GatherBranch, gather, qualified_event_name, step
from effective.budget import Grant, depth_grant_name
from effective.combinators import Answered, Deeper, descend
from effective.domain import CallTool
from effective.graphview import bare_name, fold_cycles, from_keys
from effective.handlers.recording import RecordingHandler, Suspended
from effective.handlers.replay import ReplayHandler
from effective.keys import Key

# The pieces compose: a drill, its grant parks, the scopes it mints, and the fold that reads them
# back. `spine` rather than `unit` for that reason; the test role is finer than the marker.
pytestmark = pytest.mark.spine

RUN_ID = "drill-2026"
ASSESS = "assess"
"""The judge's one op. Named so the fold has something to count and the tape reads plainly."""

SIBLING = "sibling"
"""The op a gather-borne case runs in the branch BESIDE the drill, so `in_gather` is a real
gather (width 2) rather than a one-branch fan-out that would place keys differently."""


@dataclass(frozen=True)
class Case:
    """One descent, described before it runs.

    `answer_at` is the depth the judge would answer at *if nothing stopped it*; a budget or an
    exhausted grant can stop it sooner, which is the whole interaction this file exists to sample.
    `grants` is the sequence of `add_depth` values delivered at successive exhaustions — a case
    that runs out of the tuple gets 0, which is a human saying *answer with what you have*, and is
    what makes every case terminate."""

    budget: int
    answer_at: int
    grants: tuple[int, ...] = ()
    in_gather: bool = False


def expected_levels(case: Case) -> tuple[tuple[int, bool], ...]:
    """The `(depth, final)` pairs the judge must be asked at — arithmetic, from the case alone.

    Derived from `combinators.unfold`'s refill rule rather than observed from it: refill when the
    budget is out, `final` when the refill left nothing, judge once, and descend only if the judge
    asked to AND was not final. Written out so it CAN disagree with the code: calling `descend` to
    find out how often `descend` judges would be a mirror, and the whole oracle rests on this.

    The two ways to get it wrong are refilling *after* reading `final` rather than before (which
    makes the last level of an exhausted budget look non-final), and decrementing the budget on
    the level that answers (which is off by one at every boundary). `test_the_level_arithmetic_is
    _what_descend_does` pins both against a hand-derived table."""
    levels: list[tuple[int, bool]] = []
    remaining, depth, refills = case.budget, 0, 0
    while True:
        if remaining == 0:
            remaining = case.grants[refills] if refills < len(case.grants) else 0
            refills += 1
        final = remaining == 0
        levels.append((depth, final))
        if depth >= case.answer_at or final:
            return tuple(levels)
        remaining -= 1
        depth += 1


def expected_parks(case: Case) -> tuple[int, ...]:
    """The depths at which the drill exhausts and asks for more — one park each, in order.

    A refill happens exactly where `expected_levels` consumed one, so this reads the same walk
    and reports the other half of it."""
    parks: list[int] = []
    remaining, depth, refills = case.budget, 0, 0
    while True:
        if remaining == 0:
            parks.append(depth)
            remaining = case.grants[refills] if refills < len(case.grants) else 0
            refills += 1
        final = remaining == 0
        if depth >= case.answer_at or final:
            return tuple(parks)
        remaining -= 1
        depth += 1


def park_address(case: Case, depth: int) -> Key:
    """Where an emitter would send the grant for `depth` — composed by the same function a real
    emitter calls, so this compares IDENTITIES rather than wire text.

    The property it carries: a grant park sits at the DESCENT's own scope, never inside the
    level's, so the address gains a `gather:` frame when the drill runs in a branch and never a
    `d:{depth}` one however deep the drill has gone."""
    name = depth_grant_name(RUN_ID, depth=depth, generation=0).stored()
    if case.in_gather:
        return qualified_event_name(GatherBranch(0, 0), name=name)
    return qualified_event_name(name=name)


@dataclass(frozen=True)
class Outcome:
    """One driven descent: what was asked for, what came back, and the tape between them."""

    case: Case
    result: object
    trace: list
    tape: list[str]
    levels: tuple[tuple[int, bool], ...]
    """The `(depth, final)` pairs the judge was actually asked at, collected at the judge.

    An observation of the RUN — the invariants that judge it read the tape, so this is a second
    witness rather than a restatement of the first."""

    parks: tuple[Key, ...] = field(default=())
    """Each suspension's address, in wake order."""

    @property
    def program(self):
        """The tape folded onto the program. `d:` is `descend`'s own unrolling scope, so the
        caller declares it."""
        return fold_cycles(from_keys(RUN_ID, self.tape))

    @property
    def counts(self) -> dict[str, int]:
        return {node.key: node.count for node in self.program.nodes}


def _drill(case: Case, seen: list[tuple[int, bool]]):
    """The workflow under test: `descend`, optionally inside a gather branch."""

    def judge(ctx, level):
        seen.append((level.depth, level.final))
        yield from step(ASSESS, CallTool(name="assess", args={}, result_schema=str))
        if level.depth >= case.answer_at or level.final:
            return Answered(f"answered at {level.depth}")
        return Deeper(f"{ctx}/{level.depth}")

    def drill():
        return (yield from descend("ctx", judge, budget=case.budget, run_id=RUN_ID))

    if not case.in_gather:
        return drill

    def sibling():
        return (yield from step(SIBLING, CallTool(name="sibling", args={}, result_schema=str)))

    def in_gather():
        rows = yield from gather([drill, sibling])
        return rows[0]

    return in_gather


@cache
def drive(case: Case) -> Outcome:
    """Run `case`, delivering the grant it asked for at each park.

    Memoized on the case — a `Case` is frozen and every invariant only reads its `Outcome`, so
    the parametrization judges one run many ways instead of re-driving it per invariant.

    The grant delivered at the i-th park is `case.grants[i]`, and past the end of the tuple it is
    `Grant()` — add nothing, answer now. That is what bounds every case: a drill cannot be granted
    levels forever, so the sweep cannot hang on a generated case."""
    seen: list[tuple[int, bool]] = []
    handler = RecordingHandler(responses={ASSESS: "ok", SIBLING: "ok"})
    outcome = handler.run(_drill(case, seen))
    parks: list[Key] = []
    while isinstance(outcome, Suspended):
        grant = case.grants[len(parks)] if len(parks) < len(case.grants) else 0
        parks.append(outcome.awaiting)
        outcome = outcome.resume(Grant(add_depth=grant))
    return Outcome(
        case=case,
        result=outcome,
        trace=list(handler.trace),
        tape=[entry.key.stored() for entry in handler.trace],
        levels=tuple(seen),
        parks=tuple(parks),
    )


def levels_in(key: str) -> tuple[str, ...]:
    """Every `d:` coordinate a placed key carries, read through the production parser.

    The same move `_funnel.lanes_in` makes on the lane axis, and for the same reason: `d:1` is a
    prefix of `d:10`, so a substring test reports the tenth level as sitting inside the first."""
    from effective.keys.grammar import KeySyntaxError, parse

    try:
        terms = parse(key).terms
    except KeySyntaxError:
        return ()
    return tuple(
        term.coordinates[0].atoms[0].text for term in terms if term.tag == "d" and term.coordinates
    )


# --- the oracle ------------------------------------------------------------------------------
#
# Each invariant takes an `Outcome` and asserts one property of it against the case. They are
# exercised below against the scripted descent; the generator arrives next.


def check_levels_match_the_case(run: Outcome) -> None:
    """The judge is asked at exactly the depths the case implies, with exactly the finality flags
    it implies — the drill's whole control flow, stated in advance.

    Reddens when: the budget is decremented on the level that answers, the refill is consulted
    after `final` is read instead of before, or a granted refill is dropped."""
    assert run.levels == expected_levels(run.case)


def check_every_level_ran_in_its_own_scope(run: Outcome) -> None:
    """The tape carries one `d:{depth}` scope per level, contiguous from 0 — so a level's ops are
    namespaced by the level, and two levels of one drill cannot collide.

    Reddens when: the level scope is hoisted out of the loop, or minted from something other than
    the depth (both make two levels share a scope)."""
    depths = [levels_in(key) for key in run.tape]
    assert all(len(d) <= 1 for d in depths), run.tape
    scoped = [d[0] for d in depths if d]
    assert scoped == [str(depth) for depth, _ in run.levels]


def check_parks_are_addressed_at_the_descents_own_scope(run: Outcome) -> None:
    """Each grant park is addressed where an emitter would send it: at the descent's own scope,
    carrying the enclosing gather frame if there is one and NEVER the level's `d:` frame.

    This is `descend`'s stated contract — "the park sits at the descent's own scope, NOT inside
    the level's, so its name does not move as the drill deepens" — and the depth it names is a
    coordinate of the NAME, not of the placement.

    Reddens when: the refill is moved inside `scoped(compose_key(t"d:{depth}"))`, which is the
    one-line change that would make a call-site reader's guess wrong on both engines."""
    assert run.parks == tuple(park_address(run.case, depth) for depth in expected_parks(run.case))
    for address in run.parks:
        assert not levels_in(address.stored()), address


def check_the_answer_is_the_deepest_levels(run: Outcome) -> None:
    """The value that comes back is the one the LAST level produced — a drill returns its
    deepest answer, not the first verdict it saw.

    Reddens when: the trampoline returns on the first `Done` it constructs rather than iterating,
    or carries a stale value out of the loop."""
    deepest, _ = run.levels[-1]
    assert run.result == f"answered at {deepest}"


def check_replays(run: Outcome) -> None:
    """Re-execution re-binds every level — the grant parks included — to the identical result.

    Reddens when: a level's key is minted from something the second execution cannot re-derive.
    Measured with a counter in place of the depth — the first run looks correct (`d:0`, `d:1`, …)
    and the replay continues the counter, so nothing binds. Wrong-but-deterministic keys do NOT
    redden this, and cannot: replay re-executes the same code, so it agrees with itself. That is
    what makes this a distinct invariant rather than a second reading of the one above."""
    assert ReplayHandler(run.trace).run(_drill(run.case, [])) == run.result


def check_the_fold_collapses_the_drill(run: Outcome) -> None:
    """`d:` is an unrolling scope, so the whole drill projects onto ONE judge node whose count is
    the number of levels — however deep it went, and however many times it was granted more.

    The node key is the same with or without the gather, because branch is an axis the fold drops
    too: a drill in a branch and a drill at the top project onto one program point, which is what
    makes two runs of differing width comparable at all.

    Reddens when: the fold stops summing counts into the group it collapses (`graphview`'s
    `executions[folded[node.key]] += node.count`), and now also when `d:{depth}` stops declaring
    `Index` at its mint, which is where the fold reads what the coordinate means."""
    # Summed by the OP rather than by a folded label: the sweep generates widths, so whether a
    # `gather:` frame wraps the drill is a property of the case, not of the invariant.
    assessed = sum(
        node.count for node in run.program.nodes if bare_name(node.key) == f"step:{ASSESS}"
    )
    assert assessed == len(run.levels)


ORACLE = (
    check_levels_match_the_case,
    check_every_level_ran_in_its_own_scope,
    check_parks_are_addressed_at_the_descents_own_scope,
    check_the_answer_is_the_deepest_levels,
    check_replays,
    check_the_fold_collapses_the_drill,
)


SCRIPTED = Case(budget=2, answer_at=5, grants=(2,))
"""The case whose answers are established, and the gate on the oracle itself.

Chosen to exercise what the invariants inspect rather than to be small: the budget runs out at
depth 2 and a grant of 2 more carries it to depth 4, where the tuple is spent and the judge is
handed `final=True` and must answer. Five levels, two parks, one grant honored and one refused —
so every invariant here has something to be wrong about."""


SCRIPTED_IN_GATHER = Case(budget=2, answer_at=5, grants=(2,), in_gather=True)
"""The same descent inside a gather branch — the composition the durable suite pins on one path
(`test_durable_descend_leaf_parks_on_the_gather_qualified_grant`) and the reason `park_address`
takes the case rather than a depth alone. Scripted rather than left to the generator, because an
invariant with an arm nothing has exercised is an invariant that has not been calibrated."""


@pytest.mark.parametrize("case", [SCRIPTED, SCRIPTED_IN_GATHER], ids=["flat", "in-gather"])
@pytest.mark.parametrize("invariant", ORACLE, ids=lambda f: f.__name__)
def test_the_oracle_holds_on_the_case_we_already_know(invariant, case):
    """Every invariant against the scripted descent, at the top and inside a branch. An invariant
    that cannot pass here has no business judging a generated case, and one that passes here for
    the wrong reason is exposed by the mutation named in its docstring."""
    invariant(drive(case))


def test_the_scripted_case_is_the_shape_the_oracle_assumes():
    """Anti-vacuity for the gate above: it is only meaningful if the known case exercises what
    the invariants inspect. Named rather than counted, so a change says which property lapsed."""
    levels = expected_levels(SCRIPTED)
    parks = expected_parks(SCRIPTED)
    assert len(levels) > 1, "the drill must actually descend"
    assert len(parks) > 1, "the budget must run out more than once"
    assert levels[-1][1], "the last level must be FINAL — a judge forced to answer"
    assert not any(final for _, final in levels[:-1]), "and no earlier level may be"
    assert SCRIPTED.grants[0] > 0, "one grant must be honored, or the refill arm is untested"


@pytest.mark.parametrize(
    ("case", "levels", "parks"),
    [
        # budget alone: the judge answers before the budget bites, so nothing refills
        (Case(budget=5, answer_at=2), ((0, False), (1, False), (2, False)), ()),
        # the budget bites: depth 2 is where it runs out, and `final` forces the answer there
        (Case(budget=2, answer_at=9), ((0, False), (1, False), (2, True)), (2,)),
        # budget 1, one park, refused: the drill answers at depth 1
        (Case(budget=1, answer_at=9), ((0, False), (1, True)), (1,)),
        # a grant of 2 carries it two levels past the park, then the tuple is spent
        (
            Case(budget=1, answer_at=9, grants=(2,)),
            ((0, False), (1, False), (2, False), (3, True)),
            (1, 3),
        ),
        # answering at depth 0 is a real case: one level, no park
        (Case(budget=3, answer_at=0), ((0, False),), ()),
    ],
)
def test_the_level_arithmetic_is_what_descend_does(case, levels, parks):
    """The oracle's own arithmetic, pinned against a hand-derived table rather than a run.

    Separate because an oracle computed by the code it judges is a mirror: deriving these by hand
    keeps `expected_levels` a claim ABOUT `descend` rather than a restatement of it. Read the
    shape off the rows — a budget of B is exhausted at depth B, `final` is only ever true on the
    last row, and a grant of g buys exactly g more levels before the next park."""
    assert expected_levels(case) == levels
    assert expected_parks(case) == parks
    run = drive(case)
    assert run.levels == levels


# --- the generator ---------------------------------------------------------------------------


def case(seed: int) -> Case:
    """A generated descent. Deterministic in `seed` — the whole reproducibility story, since a
    failure reports the seed and nothing else is needed to re-run it.

    Every axis is drawn independently rather than by perturbing the scripted case, so the space
    includes shapes a chosen path does not reach: a grant of 0 in the MIDDLE of the tuple (a
    refusal that is not the tuple ending), a judge that answers at depth 0, and a drill granted
    more levels twice.

    `grants` is bounded and so is `answer_at`, which is what makes every case terminate: past the
    tuple every refill returns 0, and a level handed `final=True` must answer."""
    from random import Random

    rng = Random(seed)
    budget = rng.randint(1, 6)
    answer_at = rng.randint(0, 12)
    grants = tuple(rng.randint(0, 3) for _ in range(rng.randint(0, 3)))
    return Case(budget, answer_at, grants, in_gather=rng.random() < 0.5)


SEEDS = tuple(range(20))
"""A FIXED list, checked in, never random-per-run: a failure has to be reproducible by seed alone,
and per-seed determinism is machine state rather than a property of the code.

Twenty because every corner below is reached by twelve and this leaves margin — measured, not
guessed."""


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("invariant", ORACLE, ids=lambda f: f.__name__)
def test_the_oracle_holds_over_generated_cases(invariant, seed):
    """The sweep. Every invariant against every seeded case — the drill's space, not its path."""
    invariant(drive(case(seed)))


def test_the_seeds_reach_the_corners_the_scripted_cases_cannot():
    """Anti-vacuity for the sweep, and the reason its seed list is fixed rather than sized.

    Each corner is named, so a change to the generator that quietly stops producing one fails HERE
    rather than silently narrowing the sweep to a slower version of the two scripted cases."""
    cases = [case(seed) for seed in SEEDS]
    levels = [expected_levels(c) for c in cases]
    parks = [expected_parks(c) for c in cases]
    assert any(not p for p in parks), "a drill that answers before the budget ever bites"
    assert any(len(p) == 1 and not c.grants for c, p in zip(cases, parks, strict=True)), (
        "a budget exhausted with no grantor behind it — the forced-final arm"
    )
    assert any(len(p) >= 2 for p in parks), "granted more, then refused: two parks in one run"
    assert any(
        0 in c.grants[: len(p)] and len(p) > 1 for c, p in zip(cases, parks, strict=True)
    ), "a grant of ZERO in the middle of the tuple — a refusal that is not the tuple running out"
    assert any(len(level) == 1 for level in levels), "an answer at depth 0, with no descent at all"
    assert any(level[-1][0] >= 4 for level in levels), "a drill four levels deep or more"
    assert any(not level[-1][1] for level in levels), "an answer that was NOT forced by finality"
    assert any(c.in_gather for c in cases), "a drill inside a gather branch"
    assert any(not c.in_gather for c in cases), "and a drill at the top, under no branch at all"


def test_the_generator_is_deterministic():
    """Same seed, same case — twice. Without this the seed in a failure report means nothing."""
    assert case(7) == case(7)
    assert case(7) != case(8)


def test_every_invariant_declared_here_is_in_the_oracle():
    """`ORACLE` is what the sweep runs, so a check function left out of it is a test that exists
    and never executes. Reflective rather than counted: a count notices a dropped member only if
    nobody adjusts the number in the same edit, which is the one edit that drops one."""
    declared = {name for name in globals() if name.startswith("check_")}
    assert declared, "the reflection found no check functions at all"
    assert {invariant.__name__ for invariant in ORACLE} == declared
