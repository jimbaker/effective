"""The states, and one verdict enum per state — the dependent sum's index and its fibres.

**Why a verdict enum PER STATE rather than one shared enum.** The transition's domain is
Sigma_s Verdict(s), not the product State x Verdict. A product demands an arm for
`(TEST, ReviewVerdict.APPROVED)`, which no judge can produce: 25 real cells against 175, with
150 that exist only to be refused. Giving each state its own enum makes those 150 unspellable
at the call site instead of refused at run time — `route_test` simply does not accept a
`ReviewVerdict`.

**Why enums and not frozen dataclasses.** Every other closed union here is dataclasses, so this
is the exception, for a measured reason: `ty` narrows a `StrEnum`
match to `Never`, so `assert_never` reddens on both a dropped arm and an added one, naming the
arm in the diagnostic. `StrEnum` also round-trips through a JSON checkpoint as its value, which
a dataclass would not without a serializer.

**`Exhausted` is deliberately not an arm of anything here.** It is minted by the interpreter at
the final level, never by a judge, so a judge's `Effect[V]` — where `V` is one of these enums —
cannot spell it. That is the whole mechanism: a judge that never says "exhausted" would relocate
the livelock rather than kill it, so the vocabulary makes saying it impossible.
`combinators.unfold` does the same thing with `final = remaining == 0`, and refuses a node that
descends past an exhausted budget rather than trusting it not to.

**`DONE` is not a state.** The incumbent (`tests/_coding.py`) already says why — *"not a phase;
nothing runs in it"* — and typing it as an outcome rather than a state keeps
`Mapping[State, StateSpec]` TOTAL. A `State` with no worker, no judge and no route would make
that map partial, and a partial map keyed by an enum is the nominal-where-structural shape this
repo keeps finding. Reaching the end is `transition.Finish`.

`transition.py` holds the edges; this module holds only the vocabulary.
"""

from enum import StrEnum


class State(StrEnum):
    """The seven states that RUN something. Every one has a `StateSpec`; see the module
    docstring for why the terminal is an outcome instead of an eighth member."""

    PLAN = "plan"
    EXPLORE = "explore"
    REPL = "repl"
    TEST = "test"
    DRAFT = "draft"
    FINALIZE = "finalize"
    REVIEW = "review"


class PlanVerdict(StrEnum):
    """The human's word on a decomposition, gathered at the one park that precedes any world
    change. Not the model's — PLAN's gate is `await_event`, so this enum records an answer that
    came from outside the machine."""

    APPROVED = "approved"
    REVISE = "revise"
    REJECTED = "rejected"


class ExploreVerdict(StrEnum):
    MORE_EXPLORATION = "more-exploration"
    READY_FOR_REPL = "ready-for-repl"


class ReplVerdict(StrEnum):
    """Generated code cannot compute this one itself: Monty has no `match`/`enum` (blocked by
    class inheritance, measured), so REPL's verdict is always judged rather than derived."""

    NEED_MORE_EVIDENCE = "need-more-evidence"
    HYPOTHESIS_REJECTED = "hypothesis-rejected"
    HYPOTHESIS_SUPPORTED = "hypothesis-supported"


class TestVerdict(StrEnum):
    """`TEST` means WRITE a test, so RED is the pass condition and a green suite proves nothing
    about the test just written."""

    # Not a pytest fixture class — `Test*` is pytest's collection prefix, and without this the
    # suite warns on every run. A dunder is not turned into an enum member, so this is free.
    __test__ = False

    RED = "red"
    GREEN_ALREADY = "green-already"
    RED_ELSEWHERE = "red-elsewhere"
    BROKEN_ENV = "broken-env"


class DraftVerdict(StrEnum):
    GREEN = "green"
    STILL_RED = "still-red"
    REGRESSED = "regressed"


class FinalizeVerdict(StrEnum):
    """A genuine product of a mechanical axis (does the suite still pass?) and a judged one (is
    the structure paid off?) — and the impossible cell is named rather than defaulted: when the
    suite is red, `BROKE_IT` is the answer and no opinion about structure is worth having yet."""

    STILL_GREEN_TIDY = "still-green-tidy"
    STILL_GREEN_DEBT_REMAINS = "still-green-debt-remains"
    BROKE_IT = "broke-it"


class ReviewVerdict(StrEnum):
    """The proposal's best move: *"review found a problem"* stops being one fuzzy back edge, and
    the KIND of problem picks the edge — which is what makes the process measurable, since after
    N runs you can ask what fraction of reviews returned to EXPLORE rather than DRAFT."""

    APPROVED = "approved"
    MISSING_UNDERSTANDING = "missing-understanding"
    INVALID_ASSUMPTION = "invalid-assumption"
    NEED_RUNTIME_EVIDENCE = "need-runtime-evidence"
    TEST_GAP = "test-gap"
    IMPLEMENTATION_DEFECT = "implementation-defect"
    STRUCTURAL_DEBT = "structural-debt"


type Verdict = (
    PlanVerdict
    | ExploreVerdict
    | ReplVerdict
    | TestVerdict
    | DraftVerdict
    | FinalizeVerdict
    | ReviewVerdict
)
"""Every verdict any judge can return. `Exhausted` is deliberately OUTSIDE this union."""


VERDICTS: dict[State, type[Verdict]] = {
    State.PLAN: PlanVerdict,
    State.EXPLORE: ExploreVerdict,
    State.REPL: ReplVerdict,
    State.TEST: TestVerdict,
    State.DRAFT: DraftVerdict,
    State.FINALIZE: FinalizeVerdict,
    State.REVIEW: ReviewVerdict,
}
"""The dependent sum's fibres, as data — so a test can ENUMERATE the domain (and its complement,
the 150 excluded cells) rather than hand-listing either. One mint, two consumers: `transition`
reads it to check a verdict belongs to its state, and the graph gate reads it to build the edge
relation by calling `transition` rather than by re-spelling the table."""

__all__ = [n for n in dir() if not n.startswith("_")]
