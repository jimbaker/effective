"""A triage funnel on the RLM combinators.

A shared fixture, like `_conformance.py` and `_composition.py`. Two test modules import it:
`test_funnel_triage_example.py` drives the scripted batch along one path, `test_funnel_sweep.py`
drives generated batches and judges them against invariants. It is registered in
`lint.WORKFLOW_ROLE_SRCS`, so the determinism-boundary lint checks it like any workflow.

**What it is for is DEPTH.** `_composition.py` covers every ordered PAIR of combinator holes,
and this runs five combinators in one workflow along two arms neither a pair nor a single nest
can reach: `hoisted ∘ gather ∘ scoped ∘ route` for the lanes, with a park inside one of them
(`gather:0,1;talk:birch;event;review:cfp-2026`), and a separate `scoped ∘ recurse` for the
barrier. Four frames deep at the most, joined by a `gather` whose width is decided at runtime.
The domain is a conference CFP pass; the shape is a common triage topology, and the phases
below are its beats.

**profile** — the fan-out width is data: N comes back from the inbox, not from this file.

**lanes** — one sequential sub-pipeline per talk, all under ONE `gather`. Closed-book is
structural rather than conventional: a lane's closure holds only its own talk, and the handler
places its keys under `talk:{ident}`, so a lane cannot name a sibling's op even by accident.

**route** — a recorded classifier picks each talk's rubric (dict-of-generators dispatch). The
taken path is in the tape, so replay re-dispatches identically rather than re-deciding.

**the skeptic** — an adversarial audit per lane. A MATERIAL flag escalates IN THE LANE: the
await parks the whole run on that branch's fully-qualified event, which
`qualified_event_name` composes as `gather:0,{i};talk:{ident};review:{run}`, while the sibling
lanes' work holds. The park sits inside a branch inside a scope, which is the composition this
fixture exists to keep exercised.

**rank** — the population barrier as `recurse`'s balanced tree-fold, never one flat reduce.

**drift** — the diff against the last adjudicated pass, pure Python between yields.

**the tape** — the run's graph, read off the trace. Folded back onto the program it came from,
its node counts ARE the funnel: one op per lane per talk, the route's split across rubrics, one
park per escalation.

**Nothing here spells a key.** Every answer the run needs is decided by `Answers`, which reads
the PLACEMENT off the key it is asked for, so a new talk or a new frame grammar rewrites no key.
`scripted_run` still holds three expected VALUES — the ranked program, the flagged talk, the
drift — which a fourth submission would change.

`hoisted` activates the rubric skill once, ABOVE the fan-out: activation is value-independent, so
paying it per branch would be pure waste.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from _keymap import including

from effective.api import (
    Effect,
    GatherBranch,
    ask_llm,
    await_event,
    call_tool,
    gather,
    qualified_event_name,
    scoped,
    step,
)
from effective.combinators import hoisted, recurse, route
from effective.domain import CallTool
from effective.graphview import bare_name, fold_cycles, from_keys
from effective.handlers.recording import RecordingHandler, Suspended
from effective.keys import Index, compose_key
from effective.keys.grammar import KeySyntaxError, ParsedKey, parse
from effective.skills import Pin

# --- the drift reference: last pass's adjudicated tiers -----------------------

BASELINE = {"aspen": "accept", "birch": "accept", "cedar": "waitlist"}

TIERS = ("accept", "waitlist", "reject")
"""Strongest first — the order the ranking sorts by."""


@dataclass(frozen=True)
class Submission:
    """A CFP submission and the judgements a scripted run makes about it.

    One row per talk, keyed by NAME. That is what lets `Answers` below decide what `enrich`
    returns in THIS lane without any key being written down."""

    ident: str
    title: str
    evidence: str
    track: str
    tier: str
    audit: str  # a verdict starting `flag:` is MATERIAL and escalates in the lane

    @property
    def talk(self) -> str:
        return f"{self.ident} — {self.title}"


SUBMISSIONS = (
    Submission(
        "aspen",
        "profiling a GC pause down to a missing write barrier",
        "reproducible benchmark; profiles attached; prior talks well rated",
        track="systems",
        tier="accept",
        audit="pass",
    ),
    Submission(
        "birch",
        "how our team shipped the migration in one weekend",
        "no benchmark; strong narrative arc; single-team anecdote",
        track="story",
        tier="waitlist",
        audit="flag: single-team anecdote generalized to a method claim — material",
    ),
    Submission(
        "cedar",
        "free-threading in production: numbers from three services",
        "production numbers across three services on 3.14t",
        track="story",
        tier="accept",
        audit="pass",
    ),
)


@dataclass(frozen=True)
class Scored:
    """One lane's joined row: the talk, its routed tier, the skeptic's verdict,
    and — for a material flag — the in-lane reviewer ruling that resolved it."""

    talk: str
    tier: str
    audit: str
    review: str = ""


@dataclass(frozen=True)
class Triage:
    """The funnel's result: the ranked program plus everything the barrier saw."""

    program: str
    tiers: dict[str, str]
    flagged: tuple[str, ...]
    review: str
    drift: dict[str, tuple[str, str]]


def _ident(talk: str) -> str:
    return talk.split(" — ")[0]


# --- one talk's closed-book lane: enrich, route, audit, maybe escalate --------


def _classify(chunk: tuple[str, str]) -> Effect[str]:
    talk, evidence = chunk
    return (yield from ask_llm("classify", f"Track for {_ident(talk)} given: {evidence}", str))


def _fanout(batch: list[str], pins, run_id: str) -> Effect[list[Scored]]:
    pin = pins["rubric"]

    def scorer(label: str):
        def handler(chunk: tuple[str, str]) -> Effect[str]:
            talk, evidence = chunk
            return ask_llm(
                f"score:{label}",
                f"Tier {_ident(talk)} on the {label} rubric ({pin.body}): {evidence}",
                str,
            )

        return handler

    rubrics = {label: scorer(label) for label in ("systems", "story")}

    def lane(talk: str, run_id: str) -> Effect[Scored]:
        """enrich -> route -> audit, sequentially, seeing only this talk.

        A material flag escalates HERE rather than at the join: the await parks the whole run
        on this lane's qualified event while the sibling lanes' work holds.

        Every name below is PLAIN — the lane runs inside `scoped(t"talk:{ident}")` and the
        handler places the keys, so what a reader sees here is what this lane does."""
        evidence = yield from ask_llm("enrich", f"Collect evidence on {talk!r}.", str)
        tier = yield from route((talk, evidence), _classify, rubrics)
        audit = yield from ask_llm(
            "audit",
            f"SKEPTIC: try to break tier {tier!r} for {_ident(talk)}. "
            "Flag ONLY if the finding would change the tier.",
            str,
        )
        review = ""
        if audit.startswith("flag"):
            review = yield from await_event(f"review:{run_id}", str)
        return Scored(talk, tier, audit, review)

    # the fork/join band: one branch per discovered talk, joined in batch order.
    # The lane is scoped by the talk's NAME, not its index: `talk:birch` says which lane in
    # the tape, where `talk:1` only restated the `gather:0,1` coordinate beside it.
    # `Index` all the same, and that pair is the reason it takes a `str`: the three lanes run
    # IDENTICAL code, so the coordinate counts repetitions of one position however it is spelled.
    # A fold drops it; the name is there for whoever reads the tape.
    rows = yield from gather(
        [
            (
                lambda t=talk: scoped(
                    compose_key(t"talk:{Index(_ident(t))}"), lambda: lane(t, run_id)
                )
            )
            for talk in batch
        ]
    )
    return rows


# --- the population barrier as a balanced tree-fold ---------------------------


def _pairs(population: tuple[str, ...]) -> Effect[list[tuple[str, ...]]]:
    """The chunking is a recorded op, and it commits BEFORE the fan-out — so a crash mid-rank
    replays with the same branch structure instead of re-chunking a changed population."""
    op = CallTool(
        name="chunk", args={"population": list(population), "size": 2}, result_schema=list
    )
    return (yield from step("chunk", op))


def _shortlist(chunk: tuple[str, ...]) -> Effect[str]:
    return (yield from ask_llm("shortlist", f"Order by strength: {', '.join(chunk)}", str))


def _merge(group) -> Effect[str]:
    return (yield from ask_llm("merge", f"Merge rankings: {' | '.join(group)}", str))


# --- the workflow: the whole funnel, one generator -----------------------------


def triage(run_id: str) -> Effect[Triage]:
    # the width is data: N comes back from the inbox, not from this file
    batch = yield from call_tool("inbox", {"tab": "unprocessed"}, list[str])
    # one disclosure, above the fan-out: activation is value-independent
    rows = yield from hoisted(("rubric",), lambda pins: _fanout(batch, pins, run_id))
    # the escalation happened inside the lanes: a material flag parked the run mid-fan-out
    # and a reviewer ruling resumed it — collect what the join carried out
    flagged = tuple(r.talk for r in rows if r.audit.startswith("flag"))
    review = next((r.review for r in rows if r.review), "")
    tiers = {_ident(r.talk): r.tier for r in rows}
    # rank the whole population — recurse's tree-fold, never one flat reduce
    population = tuple(f"{ident}={tier}" for ident, tier in tiers.items())
    program = yield from scoped(
        compose_key(t"rank"),
        lambda: recurse(population, _pairs, _shortlist, _merge, fanin=2),
    )
    # drift vs the last adjudicated pass — pure Python between yields
    # A talk with no previous pass has not DRIFTED — it is new, and absent from this diff.
    # The membership test is what keeps `BASELINE[i]` total: `BASELINE.get(i) != t` alone
    # admits an ident the body then subscripts, so a talk with no baseline row raises.
    # The inequality decides whether a comparable talk moved.
    drift = {i: (BASELINE[i], t) for i, t in tiers.items() if i in BASELINE and BASELINE[i] != t}
    return Triage(program, tiers, flagged, review, drift)


# --- the scripted run: the answers, and the spine that drives them -------------

_FRAME_TAGS = ("gather", "talk", "rank", "rec", "fold")
"""The frames this run's combinators mint. A frame is placed BY the handler, so a fixture
that wants to know where it is reads them back rather than writing them down."""


def _placed(key: str) -> tuple[dict[str, str], str]:
    """Split a placed key into `{frame tag: its first coordinate}` and the author's own op name.

    `gather:0,1;talk:birch;enrich` -> `({"gather": "0", "talk": "birch"}, "enrich")`.

    The lanes all run the same code and name their ops plainly, so a
    fixture answering `enrich` has to know WHICH lane is asking — and the placement is where
    that fact lives. `grammar.parse` is the production reader, so this tracks the grammar
    rather than restating it, and it is total: text the parser refuses comes back as its own
    name, which the caller then fails to find rather than mis-answering."""
    try:
        terms = parse(key).terms
    except KeySyntaxError:
        return {}, key
    frames = {
        term.tag: term.coordinates[0].atoms[0].text
        for term in terms
        if term.tag in _FRAME_TAGS and term.coordinates
    }
    body = next((i for i, term in enumerate(terms) if term.tag not in _FRAME_TAGS), len(terms))
    return frames, ParsedKey(terms[body:]).render() if body < len(terms) else key


def lanes_in(key: str) -> tuple[str, ...]:
    """Every lane a placed key sits in — its `talk:` coordinates, read through the parser.

    A key outside the fan-out, and text the parser refuses, both come back empty, so a caller
    asks one question — *which lane is this?* — rather than matching a prefix. Matching is what
    a lane check must not do: `talk:elm` is a prefix of `talk:elmwood`, so a substring test
    reports elmwood's lane as reading elm's evidence.

    A tuple rather than a single coordinate because the interesting failure is arity: a lane
    key carries exactly one `talk:`, and both a hoisted scope and a nested one are visible here
    and invisible to a membership test."""
    try:
        terms = parse(key).terms
    except KeySyntaxError:
        return ()
    return tuple(
        term.coordinates[0].atoms[0].text
        for term in terms
        if term.tag == "talk" and term.coordinates
    )


class Answers(Mapping[str, object]):
    """The canned side of the run, answered by PLACEMENT rather than by a spelled key.

    `RecordingHandler` consults `responses` as a plain Mapping — the qualified key first, then
    the bare one — so a Mapping that COMPUTES needs no handler change and no new seam.

    Note what is NOT here: `review:{run_id}`. An unanswered await is what parks the run; the
    ruling arrives through `resume`."""

    def __init__(self, submissions: tuple[Submission, ...]) -> None:
        self._submissions = submissions
        self._by_ident = {s.ident: s for s in submissions}
        self._rank = {s.ident: (TIERS.index(s.tier), i) for i, s in enumerate(submissions)}

    def __iter__(self):
        """The names answerable WITHOUT a lane frame — so iterating and then indexing agrees,
        and `dict(...)` round-trips.

        A lane's ops (`enrich`, `classify`, `audit`, `score:{track}`) are deliberately absent:
        each needs a `talk:` frame to say which lane is asking, so there is no bare key that
        answers them and enumerating one would hand a caller a `KeyError`."""
        return iter(("tool:inbox", "skill:rubric,activate", "chunk", "merge"))

    def __len__(self) -> int:
        return 4

    def _population(self) -> tuple[str, ...]:
        return tuple(f"{s.ident}={s.tier}" for s in self._submissions)

    def _chunks(self) -> list[tuple[str, ...]]:
        population = self._population()
        return [tuple(population[i : i + 2]) for i in range(0, len(population), 2)]

    def _ordered(self, rows: tuple[str, ...]) -> str:
        """Strongest tier first, batch order breaking ties — what the tree-fold converges to."""
        idents = [row.split("=")[0] for row in rows]
        return " > ".join(sorted(idents, key=lambda ident: self._rank[ident]))

    def _in_lane(self, submission: Submission, name: str) -> object:
        """What a lane's op returns. The rubric arm is conditional on the TRACK: only the
        rubric `classify` chose is ever asked, so answering `score:{other}` would be
        answering an op this run cannot reach."""
        match name:
            case "enrich":
                return submission.evidence
            case "classify":
                return submission.track
            case "audit":
                return submission.audit
            case _ if name == f"score:{submission.track}":
                return submission.tier
        raise KeyError(name)

    def __getitem__(self, key: str) -> object:
        frames, name = _placed(str(key))
        match name:
            case "tool:inbox":
                return [s.talk for s in self._submissions]
            case "skill:rubric,activate":
                return Pin(name="rubric", content_hash="9f2c", body="score evidence")
            case "chunk":
                return self._chunks()
            case "shortlist":
                return self._ordered(self._chunks()[int(frames["rec"])])
            case "merge":
                return self._ordered(self._population())
        if (submission := self._by_ident.get(frames.get("talk", ""))) is not None:
            return self._in_lane(submission, name)
        raise KeyError(key)


def scripted_run() -> Triage:
    """Drive the scripted batch: park in a lane, resume, assert the run and its shape.

    Every answer here is known in advance, so the sampled sweep calibrates its invariants
    against this run before judging generated ones. `test_funnel_triage_example.py` calls it."""
    handler = RecordingHandler(responses=Answers(SUBMISSIONS))
    parked = handler.run(lambda: triage("cfp-2026"))
    # The material flag parked the run INSIDE birch's lane — the whole run waits
    # on that lane's fully-qualified event while the sibling lanes' work holds.
    # The name is COMPOSED, not spelled: the lane wrote `await_event("review:cfp-2026")`
    # and knows nothing of the two frames above it. An emitter — a reviewer's UI, a script —
    # walks the same path to reach the same name, and both sides are `Key`s, so this
    # compares IDENTITIES rather than wire text.
    assert isinstance(parked, Suspended)
    assert parked.awaiting == qualified_event_name(
        GatherBranch(0, 1), compose_key(t"talk:{Index('birch')}"), name="review:cfp-2026"
    )
    result = parked.resume("uphold — the anecdote reads as an anecdote")
    assert isinstance(result, Triage)
    assert result.program == "aspen > cedar > birch"
    assert result.flagged == ("birch — how our team shipped the migration in one weekend",)
    assert result.drift == {"birch": ("accept", "waitlist"), "cedar": ("waitlist", "accept")}

    # The tape IS the graph, so what is asserted is its SHAPE rather than its text: fold the
    # tape back onto the program and the node counts are the funnel — one op per lane per talk,
    # the route's split across two rubrics, one park.
    tape = [entry.key.stored() for entry in handler.trace]
    program = fold_cycles(from_keys("cfp-2026", tape), keymap=including("tests/_funnel.py"))
    counts = {node.key: node.count for node in program.nodes}
    assert counts["gather:*,*;talk:*;step:enrich"] == len(SUBMISSIONS)
    assert counts["gather:*,*;talk:*;step:audit"] == len(SUBMISSIONS)
    assert sum(
        n.count for n in program.nodes if bare_name(n.key).startswith("step;score:")
    ) == len(SUBMISSIONS)
    assert counts["gather:*,*;talk:*;event;review:cfp-2026"] == sum(
        s.audit.startswith("flag") for s in SUBMISSIONS
    )
    assert program.executions == len(tape)  # fewer nodes, never fewer facts
    return result
