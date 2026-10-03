"""A sampled property sweep over `_funnel`: the funnel's SPACE, not one path.

Role: **sampled property**. Its sibling `test_funnel_triage_example.py` is the *journey*
(one realistic path asserted end to end), and this is the other half: the same workflow driven
over many generated cases, judged by invariants rather than by a recorded answer.

**The oracle is written before the generator.** A sampled test whose assertions are weak
degenerates into "it ran and nothing crashed". So the invariants below are stated first and made
to pass against the ONE case whose answers are already known — `_funnel.SUBMISSIONS`, three talks,
one material flag, two rubrics — and only then handed cases nobody has checked. An invariant that
cannot be stated and passed against the known case is not ready to judge an unknown one.

Each invariant records the mutation that reddens it.

An assertion re-derived from the run it is judging is a MIRROR, not an oracle: `folded ==
fold(tape)` passes on any behaviour at all. Every predicate here is computed from the CASE — what
was submitted — and compared against what the run did with it.

**What the sampling reaches that one path cannot** is the reason to run it, and
`test_the_seeds_reach_the_corners_the_scripted_case_cannot` names each corner so a generator
change that stops producing one fails loudly. Widths run 1 through 10; twenty-three of the forty
seeds escalate more than one lane, seven simultaneously at the extreme, and five escalate none — a
run that never suspends at all. Every lane awaits `review:{run_id}`, separated only by its branch
frame, so several pending at once is the composition the sampling reaches.
"""

from dataclasses import dataclass
from functools import cache

import _funnel as funnel
import pytest
from _funnel import Submission, Triage
from _keymap import by_op, including

from effective.graphview import bare_name, fold_cycles, from_keys
from effective.handlers.recording import RecordingHandler, Suspended
from effective.handlers.replay import ReplayHandler
from effective.keys import Key

pytestmark = pytest.mark.journey

RUN_ID = "cfp-2026"
RULING = "uphold — reviewed"


@dataclass(frozen=True)
class Outcome:
    """One driven run: what went in, what came out, and the tape between them."""

    submissions: tuple[Submission, ...]
    result: Triage
    trace: list
    tape: list[str]
    deliveries: tuple[tuple[Key, str], ...]
    """The park address and the ruling handed to it, in wake order — one pair per suspension.

    Recorded at the driver, so it is an observation of the RUN rather than a re-reading of the
    tape the invariants also judge. Each ruling is DISTINCT, which is what makes the delivery
    traceable: one constant for every lane would say how many wakes happened and nothing about
    where any of them went."""

    @property
    def parks(self) -> int:
        return len(self.deliveries)

    @property
    def program(self):
        """The tape folded onto the program. `talk:` is this workflow's own unrolling scope, so
        the caller declares it."""
        return fold_cycles(from_keys(RUN_ID, self.tape), keymap=including("tests/_funnel.py"))

    @property
    def counts(self) -> dict[str, int]:
        return {node.key: node.count for node in self.program.nodes}


def drive(submissions: tuple[Submission, ...]) -> Outcome:
    """Run the funnel over `submissions`, resuming every park in turn.

    The resume loop is what lets a case park more than once: the scripted batch parks exactly
    once, and generated batches escalate up to five lanes in a single run.

    Every ruling carries its wake ORDINAL, so the rulings are distinguishable in the tape. That
    is what lets an invariant ask where a delivery landed rather than only how many there were:
    the recorder wakes branches by POSITION while `awaiting` names one by address, so a resume
    loop that hands every lane the same string cannot tell the two apart."""
    handler = RecordingHandler(responses=funnel.Answers(submissions))
    outcome = handler.run(lambda: funnel.triage(RUN_ID))
    deliveries: list[tuple[Key, str]] = []
    while isinstance(outcome, Suspended):
        ruling = f"{RULING} #{len(deliveries)}"
        deliveries.append((outcome.awaiting, ruling))
        outcome = outcome.resume(ruling)
    return Outcome(
        submissions=submissions,
        result=outcome,
        trace=list(handler.trace),
        tape=[entry.key.stored() for entry in handler.trace],
        deliveries=tuple(deliveries),
    )


@cache
def driven(submissions: tuple[Submission, ...]) -> Outcome:
    """`drive`, memoized on the batch. The sweep asks for the same run once per invariant, and
    a `Submission` is frozen, so the batch is its own cache key.

    Sound because every invariant only READS its `Outcome` — the driving is where the cost is,
    and judging one run eight ways costs nothing extra."""
    return drive(submissions)


def material(submissions: tuple[Submission, ...]) -> list[Submission]:
    """The talks the skeptic flags — computed from the CASE, never read back off the run."""
    return [s for s in submissions if s.audit.startswith("flag")]


CHUNK = 2
"""The chunk size that fixes how many LEAVES the fold gets.

The fold reads TWO of the several 2s in `_funnel`: the chunk slice in `Answers._chunks`, which
fixes the leaf count, and `recurse`'s `fanin=2`, which fixes the grouping. `_pairs`' `{"size": 2}`
CallTool argument is decorative — a responder answers by placement and cannot see op args, so
changing it to 3 changes nothing."""


def leaves_for(n: int) -> int:
    """How many `recurse` leaves a population of `n` produces. `_pairs` chunks first, and one
    leaf runs per chunk — so the tree-fold's input is the chunk count, not the population."""
    return -(-n // CHUNK)  # ceil


def merges_for(leaves: int, fanin: int = 2) -> int:
    """How many `combine` calls a balanced tree-fold makes over `leaves` results.

    Derived from `combinators.recurse`, not observed: it groups by `fanin` and combines each
    group of MORE THAN ONE, so a singleton group passes through uncombined — a real branch the
    scripted case barely touches.

    Written as arithmetic so it CAN disagree with the code: calling `recurse` to count how
    often `recurse` combines would be a mirror. The two ways to get it wrong are feeding it the
    population size rather than the chunk count, and forgetting that a singleton group passes
    through uncombined. The table below pins the second; the first shows up only on a case where
    the chunk count differs from the population, which is `check_tree_fold_shape` over the
    generated seeds."""
    merges = 0
    while leaves > 1:
        groups = [min(fanin, leaves - i) for i in range(0, leaves, fanin)]
        merges += sum(1 for size in groups if size > 1)
        leaves = len(groups)
    return merges


# --- the oracle ------------------------------------------------------------------------------
#
# Each invariant takes an `Outcome` and asserts one property of it against the case. They are
# exercised below against the scripted submissions; the generator arrives next.


def check_fanout_width(run: Outcome) -> None:
    """One lane per submission, and every lane ran its full sequence.

    Reddens when: the fan-out is built over a truncated batch."""
    n = len(run.submissions)
    assert run.counts["gather:*,*;talk:*;step:enrich"] == n
    assert run.counts["gather:*,*;talk:*;step:classify"] == n
    assert run.counts["gather:*,*;talk:*;step:audit"] == n


def check_route_totality(run: Outcome) -> None:
    """`route` dispatched every talk to exactly the rubric its classifier chose — the dispatch is
    TOTAL over what the classifier produced, and nothing was scored twice or not at all.

    Reddens when: the classifier answers one fixed label, or a lane reads a sibling's track."""
    # `bare_name` rather than a prefix on the node key: the question is which op ran, and the
    # frames the fold keeps sit in front of it.
    scored = {
        bare_name(node.key).removeprefix("step;score:"): node.count
        for node in run.program.nodes
        if bare_name(node.key).startswith("step;score:")
    }
    expected: dict[str, int] = {}
    for submission in run.submissions:
        expected[submission.track] = expected.get(submission.track, 0) + 1
    assert scored == expected
    assert sum(scored.values()) == len(run.submissions)


def check_closed_book(run: Outcome) -> None:
    """Every op inside a lane carries THAT lane's scope and no sibling's — isolation is
    structural, so a lane physically cannot read another talk's evidence.

    Reddens when: the lane scope is hoisted above the gather, or every lane is answered alike."""
    idents = {s.ident for s in run.submissions}
    examined = 0
    for key in run.tape:
        if not (lanes := funnel.lanes_in(key)):
            continue
        examined += 1
        assert len(lanes) == 1, f"{key} sits in {len(lanes)} lanes"
        assert lanes[0] in idents, key
    assert examined == 4 * len(run.submissions) + len(material(run.submissions))


def check_park_iff_material(run: Outcome) -> None:
    """The run suspends exactly once per flagged talk — including ZERO times when nothing is
    flagged, which is a whole branch of the escalation the scripted case never takes.

    Reddens when: the audit flags unconditionally, or the escalation is unconditional."""
    flagged = material(run.submissions)
    assert run.parks == len(flagged)
    assert by_op(run.program).get(f"event;review:{RUN_ID}", 0) == len(flagged)
    assert set(run.result.flagged) == {s.talk for s in flagged}


def check_each_ruling_reaches_the_lane_that_asked(run: Outcome) -> None:
    """A reviewer ruling lands in the lane whose skeptic escalated — the wake is ADDRESSED, not
    merely counted. `check_park_iff_material` above says how many wakes happened; this says
    where each one went, which is the property the serialized-wake rule actually claims.

    Reddens when: the branch `awaiting` names and the branch `resume` delivers into come apart —
    reverse either scan for the lowest parked slot in `RecordingHandler`
    (`GatherSuspended._refresh` for a later wake, the `lowest = next(...)` at construction for
    the first)."""
    flagged = {s.ident for s in material(run.submissions)}
    reached = set()
    for awaiting, ruling in run.deliveries:
        (asked,) = funnel.lanes_in(awaiting.stored())
        carriers = [entry for entry in run.trace if entry.result == ruling]
        assert len(carriers) == 1, f"{ruling!r} was recorded {len(carriers)} times, not once"
        (landed,) = funnel.lanes_in(carriers[0].key.stored())
        assert landed == asked, f"the ruling for lane {asked!r} was recorded in lane {landed!r}"
        reached.add(asked)
    assert reached == flagged, "the lanes that were ruled on are not the lanes that escalated"


def check_tree_fold_shape(run: Outcome) -> None:
    """The ranking is a balanced tree-fold, so its `merge` count is arithmetic in the population
    size — not whatever the run happened to do.

    Reddens when: the escalation or the fold structure changes — BUT NOT AT n=3. Measured: with
    three talks the population chunks to two leaves and the fold sees a single group of two, so
    there is no singleton to pass through; mutating `recurse`'s `if len(group) > 1` to `>= 1`
    leaves this green. The invariant is sound and the CASE is too small to exercise it; the
    generator supplies cases that do, and the arithmetic itself is pinned directly by
    `test_the_fold_arithmetic_is_what_recurse_does`."""
    leaves = leaves_for(len(run.submissions))

    # By the OP, not by a folded label: a one-chunk case mints no `gather:`/`rec:` frames at all,
    # so the label carries a different prefix at each generated width.

    ops = by_op(run.program)
    assert ops["step:shortlist"] == leaves  # one leaf per chunk
    assert ops.get("step:merge", 0) == merges_for(leaves)
    assert run.counts["rank;step:chunk"] == 1  # the decompose commits BEFORE the fan-out


def check_replays(run: Outcome) -> None:
    """Re-execution re-binds every op — the in-lane events included — to the identical result.
    Replay re-derives every key by running the program again, so a key that composes
    differently the second time cannot bind.

    Reddens when: any op's key is not re-derived identically on replay."""
    assert ReplayHandler(run.trace).run(lambda: funnel.triage(RUN_ID)) == run.result


def check_drift_is_the_baseline_diff(run: Outcome) -> None:
    """Drift reports exactly the comparable talks that moved: a talk with a baseline row whose
    tier changed. One with no row has not drifted — it is new, and absent from the diff rather
    than a lookup error.

    Reddens when: the membership test is dropped (any batch containing an unknown talk raises),
    or unchanged talks are reported, or the diff is emptied."""
    expected = {
        s.ident: (funnel.BASELINE[s.ident], s.tier)
        for s in run.submissions
        if s.ident in funnel.BASELINE and funnel.BASELINE[s.ident] != s.tier
    }
    assert run.result.drift == expected


def check_ranking_is_a_total_order(run: Outcome) -> None:
    """Every submission appears exactly once in the ranked program, strongest tier first.

    Reddens when: the tree-fold drops a singleton group, or a merge loses a branch's result."""
    ranked = run.result.program.split(" > ")
    assert sorted(ranked) == sorted(s.ident for s in run.submissions)
    tiers = [funnel.TIERS.index(s.tier) for s in run.submissions if s.ident in ranked]
    by_rank = {s.ident: funnel.TIERS.index(s.tier) for s in run.submissions}
    assert [by_rank[i] for i in ranked] == sorted(by_rank[i] for i in ranked), ranked
    assert tiers  # anti-vacuity: there was something to order


ORACLE = (
    check_fanout_width,
    check_route_totality,
    check_closed_book,
    check_park_iff_material,
    check_each_ruling_reaches_the_lane_that_asked,
    check_tree_fold_shape,
    check_drift_is_the_baseline_diff,
    check_replays,
    check_ranking_is_a_total_order,
)


@pytest.mark.parametrize("invariant", ORACLE, ids=lambda f: f.__name__)
def test_the_oracle_holds_on_the_case_we_already_know(invariant):
    """Every invariant, against the scripted submissions — the one case whose answers are
    established. This is the gate on the oracle itself: an invariant that cannot pass here has no
    business judging a generated case, and one that passes here for the wrong reason is exposed
    by the mutation named in its docstring."""
    invariant(driven(funnel.SUBMISSIONS))


@pytest.mark.parametrize(
    ("n", "leaves", "merges"),
    [(1, 1, 0), (2, 1, 0), (3, 2, 1), (4, 2, 1), (5, 3, 2), (6, 3, 2), (7, 4, 3), (9, 5, 4)],
)
def test_the_fold_arithmetic_is_what_recurse_does(n, leaves, merges):
    """The oracle's own arithmetic, pinned against a hand-derived table rather than a run.

    Two reasons this is separate. The invariant that USES it cannot discriminate at n=3 (see
    `check_tree_fold_shape`), so the arithmetic would otherwise be unpinned until the generator
    lands. And an oracle computed by the same code it judges is a mirror — deriving these by hand
    keeps `merges_for` a claim about `recurse` rather than a restatement of it.

    Read the shape off the numbers: leaves is ceil(n/2) because `_pairs` chunks first, and merges
    is one per group of MORE than one, so n=5 (three leaves -> groups of 2 and 1) is the smallest
    case where a singleton passes through uncombined."""
    assert leaves_for(n) == leaves
    assert merges_for(leaves_for(n)) == merges


# --- the generator ---------------------------------------------------------------------------

NAMES = (
    "aspen", "birch", "cedar", "dogwood", "elm", "fir", "ginkgo", "hazel", "ironwood", "juniper",
)  # fmt: skip
"""A fixed vocabulary, so an ident is always a well-formed atom and a failing seed names talks a
reader can recognise. Alphabetical, which also fixes batch order."""


def case(seed: int) -> tuple[Submission, ...]:
    """A generated batch of submissions. Deterministic in `seed` — the whole reproducibility
    story, since a failure reports the seed and nothing else is needed to re-run it.

    Built from the STRUCTURE outward rather than by perturbing the scripted case: the width, each
    talk's track and tier, and whether the skeptic flags it are all drawn independently, so the
    space includes the corners the scripted path cannot reach — one lane, no escalation at all,
    every lane escalating, and populations whose chunking leaves a singleton group."""
    from random import Random

    rng = Random(seed)
    n = rng.randint(1, len(NAMES))
    return tuple(
        Submission(
            ident=name,
            title=f"a talk about {name}",
            evidence=f"evidence for {name}",
            track=rng.choice(("systems", "story")),
            tier=rng.choice(funnel.TIERS),
            audit="flag: material" if rng.random() < 0.4 else "pass",
        )
        for name in NAMES[:n]
    )


SEEDS = tuple(range(40))
"""A FIXED list, checked in, never random-per-run: a failure has to be reproducible by seed alone,
and per-seed determinism is machine state rather than a property of the code.

The window is forty because the corner gate below demands it: the first batch whose lanes ALL
escalate and which is wider than one is seed 32. Twenty seeds reached that corner only through
the two n==1 cases, which is the same corner as `a gather with ONE branch` wearing another name.
The next such seeds are 108, 149 and 188, so nothing between 33 and 107 buys it back."""


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("invariant", ORACLE, ids=lambda f: f.__name__)
def test_the_oracle_holds_over_generated_cases(invariant, seed):
    """The sweep. Every invariant against every seeded case — the funnel's space, not its path."""
    invariant(driven(case(seed)))


def test_the_seeds_reach_the_corners_the_scripted_case_cannot():
    """Anti-vacuity for the sweep, and the reason its seed list is fixed rather than sized.

    Twenty seeds are worth running only if they actually reach the shapes the scripted case
    misses. Each corner is named, so a change to the generator that quietly stops producing one
    fails HERE rather than silently narrowing the sweep to a slower version of the journey test."""
    cases = [case(seed) for seed in SEEDS]
    assert any(len(c) == 1 for c in cases), "a gather with ONE branch"
    assert any(not material(c) for c in cases), "a run that never parks"
    assert any(len(material(c)) > 1 for c in cases), "several lanes parked at once"
    assert any(len(material(c)) == len(c) and len(c) > 1 for c in cases), (
        "every lane parked, in a batch WIDER than one — an n==1 case satisfies "
        "'all parked' and 'a gather with ONE branch' at once, so without the width "
        "condition this corner is the one above it under another name"
    )
    assert any(leaves_for(len(c)) % 2 == 1 and leaves_for(len(c)) > 1 for c in cases), (
        "a chunking that leaves a SINGLETON group for the tree-fold to pass through"
    )
    assert any(len({s.track for s in c}) == 1 for c in cases), "one rubric for the whole batch"
    assert any(any(s.ident not in funnel.BASELINE for s in c) for c in cases), (
        "a talk with NO baseline row — the drift diff must treat it as new, not subscript it"
    )


def test_every_invariant_declared_here_is_in_the_oracle():
    """`ORACLE` is what the sweep runs, so a check function left out of it is a test that exists
    and never executes — silence of exactly the kind the sweep was built to remove.

    Reflective rather than counted. A count notices a dropped member only if nobody adjusts the
    number in the same edit, which is the one edit that drops one."""
    declared = {name for name in globals() if name.startswith("check_")}
    assert declared, "the reflection found no check functions at all"
    assert {invariant.__name__ for invariant in ORACLE} == declared


def test_the_generator_is_deterministic():
    """Same seed, same case — twice. Without this the seed in a failure report means nothing."""
    assert case(7) == case(7)
    assert case(7) != case(8)


def test_the_scripted_case_is_the_shape_the_oracle_assumes():
    """Anti-vacuity for the gate above: it is only meaningful if the known case actually
    exercises what the invariants inspect. Named rather than counted, so a change says which
    property lapsed."""
    submissions = funnel.SUBMISSIONS
    assert len(submissions) >= 3, "the fan-out must be wide enough to fold"
    assert len({s.track for s in submissions}) >= 2, "the route must actually split"
    assert 0 < len(material(submissions)) < len(submissions), "some park, some do not"
    assert merges_for(leaves_for(len(submissions))) >= 1, "the tree-fold must combine once"
