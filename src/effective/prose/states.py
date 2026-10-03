"""The states of a de-essaying pass, and one verdict enum per state.

The second embodiment of `effective.machine`, and it exists to be DRIVEN rather than to be read:
its whole purpose is that a human answers every judgment through a terminal surface, so each
verdict is on the tape and the pass is auditable afterwards.

**Five states are the coding machine's, relabelled; three are not.** `SELECT`, `READ`, `DRAFT`,
`VERIFY` and `REVIEW` are `PLAN`/`EXPLORE`/`DRAFT`/`TEST`/`REVIEW`. `CLASSIFY`, `RELOCATE` and
`RUBRIC` have no counterpart, and they are what makes this a different machine rather than the
same one over different files: de-essaying is mostly a REHOMING problem. Only one of the rubric's
seven destinations deletes; the other six move prose somewhere it belongs, and a machine with no
relocation state would have to pretend they were all deletions.

**`RELOCATE` runs before `DRAFT`, and the ordering is a constraint rather than a preference.** You
cannot compress what you have not rehomed: the history has to reach a commit message and the
deferred option has to reach the wiki *before* anyone decides how short the docstring can be.
Drafting first produces a lossy result that still passes every mechanical check.

Per-state enums, not one shared enum, for the reason `coding.states` gives: the transition's domain
is the dependent sum, and giving each state its own fibre makes the excluded cells unspellable at
the call site rather than refused at run time.

This module is the vocabulary; `transition.py` holds the edges.
"""

from enum import StrEnum


class State(StrEnum):
    """The eight states that RUN something. Reaching the end is `transition.Finish`, not a state —
    the same rule `coding.states` states, and for the same reason: a `State` with no `StateSpec`
    makes the spec map partial."""

    SELECT = "select"
    READ = "read"
    CLASSIFY = "classify"
    RELOCATE = "relocate"
    DRAFT = "draft"
    VERIFY = "verify"
    REVIEW = "review"
    RUBRIC = "rubric"


class SelectVerdict(StrEnum):
    """Which region to work, or the worklist is empty.

    A REGION rather than a file: `keys.py` is 1,593 lines and 10,484 prose words, which is more
    than one pass can hold, and regions also give a driver several short runs instead of one long
    one. `doc_bloat`'s outlier sites are the natural regions."""

    CHOSE = "chose"
    EXHAUSTED_WORKLIST = "exhausted-worklist"


class ReadVerdict(StrEnum):
    """One arm, deliberately. Inventorying prose units cannot fail — a region with no prose is
    still inventoried, and it is `CLASSIFY` that finds nothing to do with it."""

    INVENTORIED = "inventoried"


class ClassifyVerdict(StrEnum):
    """Does every unit in the region have one of the rubric's seven destinations?

    Per-REGION rather than per-unit, because a rubric gap is not local: one unclassifiable unit
    is a fact about the rubric and sends the whole region to `RUBRIC`. Choosing regions small
    enough is what stops that from stalling a large file on one hard paragraph."""

    CLASSIFIED = "classified"
    NO_DESTINATION = "no-destination"


class RelocateVerdict(StrEnum):
    """Did the prose ARRIVE where the classification sent it?

    `DESTINATION_MISSING` exists because all five mechanical gates scan the SOURCE file, so a run
    could rehome nothing, delete everything, and pass every one of them. This edge checks the
    destination, so arrival does not rest on discipline."""

    REHOMED = "rehomed"
    DESTINATION_MISSING = "destination-missing"


class DraftVerdict(StrEnum):
    """One arm. A draft that could not be written is not a verdict, it is a state that never
    returned — and the budget, not a verdict, is what bounds that."""

    REWRITTEN = "rewritten"


class VerifyVerdict(StrEnum):
    """The five mechanical gates, folded into four outcomes.

    `NOT_PROSE_ONLY` is separate from `MECHANICAL` because they send you to different work: a line
    over the limit is a rewording, while a changed AST skeleton means the edit moved code and has
    to be undone. `BROKEN_ENV` covers a red gate with no prose defect anywhere (`just check`
    reddens when the tmpfs Postgres container vanishes), where prescribing "reword and retry" is
    the failure `TestVerdict.BROKEN_ENV` exists to prevent."""

    CLEAN = "clean"
    MECHANICAL = "mechanical"
    NOT_PROSE_ONLY = "not-prose-only"
    BROKEN_ENV = "broken-env"


class ReviewVerdict(StrEnum):
    """The human's word, and the KIND of problem picks the edge.

    `RUBRIC_GREW` is the deep back edge and the one this design exists for: a review finding that
    adds a *destination criterion* is not a rewording, because where things go has changed, so it
    lands on `CLASSIFY` rather than on `DRAFT`.

    `LEAVE_IT_ALONE` is the refusal. Without it `REVIEW`'s only exits are approval and loops, and
    routing a human "no" to `Finish` would make *we improved this file* and *a reviewer said
    stop* indistinguishable on the tape, which `coding.transition.route_plan` also refuses."""

    APPROVED = "approved"
    REVISE = "revise"
    RUBRIC_GREW = "rubric-grew"
    LEAVE_IT_ALONE = "leave-it-alone"


class RubricVerdict(StrEnum):
    """Did the rubric actually grow?

    `REFUSED` exists because a single `AMENDED` arm assumes the rubric
    always grows on demand, and it does not: sometimes the honest answer is that a unit has no
    destination *and* no new rule is warranted, which is a question for a human rather than another
    lap."""

    AMENDED = "amended"
    REFUSED = "refused"


type Verdict = (
    SelectVerdict
    | ReadVerdict
    | ClassifyVerdict
    | RelocateVerdict
    | DraftVerdict
    | VerifyVerdict
    | ReviewVerdict
    | RubricVerdict
)
"""Every verdict any judge here can return. `Exhausted` is deliberately OUTSIDE this union, so a
judge cannot spell it and the livelock stays the interpreter's to kill."""


VERDICTS: dict[State, type[Verdict]] = {
    State.SELECT: SelectVerdict,
    State.READ: ReadVerdict,
    State.CLASSIFY: ClassifyVerdict,
    State.RELOCATE: RelocateVerdict,
    State.DRAFT: DraftVerdict,
    State.VERIFY: VerifyVerdict,
    State.REVIEW: ReviewVerdict,
    State.RUBRIC: RubricVerdict,
}
"""The dependent sum's fibres as data, so a test enumerates the domain rather than hand-listing
it. One mint, two consumers: `transition` checks a verdict belongs to its state, and a driver
decodes an answer typed at a terminal into the right enum."""

__all__ = [n for n in dir() if not n.startswith("_")]
