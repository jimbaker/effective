"""The recursive-language-model combinator sugar: recurse / route / descend /
hoisted, plus the STRUCTURAL scope the sugar places its callbacks under.

All on RecordingHandler (infra-free): canned responses pin the *op-key
structure* (a combinator that assembled a wrong key would miss its canned
response and fail loudly), and the parent trace pins the sequencing rules
(decompose before the fan-out; the fold's tree shape; activation above the
gather). Durable crash/suspend coverage for the substrate underneath
(``gather``, ``run_code``, ``await_event``) is the conformance/pgt suites' job;
the sugar adds no replay machinery of its own.
"""

import pytest
from pydantic import TypeAdapter

from effective.api import GatherBranch, await_event, qualified_event_name, scoped, step
from effective.budget import Grant, depth_grant_name
from effective.code import CodeOutcome, run_code
from effective.combinators import (
    AcrossTasks,
    Answered,
    Branch,
    Deeper,
    DescendedPastBudget,
    Level,
    descend,
    fix,
    grant_cascade,
    granted_levels,
    hoisted,
    human_grant,
    recurse,
    refill_levels,
    route,
    tree_search,
    unfold,
)
from effective.compose import code_act
from effective.domain import CallTool
from effective.handlers.recording import RecordingHandler, Suspended
from effective.keys import Index, Segment, compose_key
from effective.ops import AwaitEvent, Step
from effective.react import AssistantTurn, ToolRequest, ToolResult, Trajectory, run_agent
from effective.skills import Pin

# ------------------------------------------------------------- run_code + code_act


def test_run_code_names_its_own_key_and_a_scope_places_it():
    """`run_code` spells `code:{name}:...` and nothing else; wrapping it in `scoped(...)`
    is what puts it under a namespace. Same bytes as the old `scope=` parameter produced —
    the decision moved, the key did not."""

    def wf():
        return (
            yield from scoped(
                compose_key(t"rec:{0}"), lambda: run_code("audit", "1 + 1", schema=int)
            )
        )

    handler = RecordingHandler(
        responses={"rec:0;code:seg,0,audit": CodeOutcome(status="complete", output=2)}
    )
    assert handler.run(wf) == 2
    assert [e.key.stored() for e in handler.trace] == ["rec:0;step;code:seg,0,audit"]


def test_run_code_rejects_a_slashed_name():
    """`/` is the path delimiter no key atom may hold; `run_code` refuses it before composing,
    so the refusal names the call."""

    def bad_name():
        return (yield from run_code("a/b", "1", schema=int))

    with pytest.raises(ValueError, match="the key grammar's path delimiter"):
        RecordingHandler().run(bad_name)


def scripted(*turns: AssistantTurn):
    """A `decide` that answers each turn from `turns` in order, yielding no op."""
    script = iter(turns)

    def decide(messages, level):
        yield from ()
        return next(script)

    return decide


ANSWER = AssistantTurn(thought="done", answer="done")


def test_code_act_runs_code_named_act_inside_the_turns_frame():
    run = AssistantTurn(thought="add", tool=ToolRequest(name="run_code", args={"code": "1 + 1"}))

    def wf():
        return (
            yield from scoped(
                compose_key(t"sub:{Segment('a')}"),
                lambda: run_agent("go", decide=scripted(run, ANSWER), act=code_act()),
            )
        )

    handler = RecordingHandler(
        responses={"sub:a;d:0;code:seg,0,act": CodeOutcome(status="complete", output={"n": 2})}
    )
    trajectory = handler.run(wf)
    assert isinstance(trajectory, Trajectory)
    assert trajectory.steps[0].observation == '{"n": 2}'


def test_code_act_falls_through_to_plain_tool_dispatch():
    """The fall-through names `tool:{request.name}` — no scope INSIDE the tag. That infix
    splice was the aliasing hazard: `request.name` is model-controlled, so a scope spliced
    after `tool:` let a tool named `sub/search` forge a coordinate PATH."""
    search = AssistantTurn(thought="look", tool=ToolRequest(name="search", args={"q": "x"}))

    def wf():
        return (
            yield from scoped(
                compose_key(t"sub:{Segment('a')}"),
                lambda: run_agent("go", decide=scripted(search, ANSWER), act=code_act()),
            )
        )

    handler = RecordingHandler(responses={"sub:a;d:0;tool:search": ToolResult(content="hit")})
    trajectory = handler.run(wf)
    assert isinstance(trajectory, Trajectory)
    assert trajectory.steps[0].observation == "hit"


def test_a_code_act_request_without_a_code_string_is_an_observation():
    """A call the loop cannot make is the next observation, as `typed_act`'s bad arguments are.
    Raised, it would end the task: only a refusal crosses the turn's scope."""
    no_code = AssistantTurn(thought="run", tool=ToolRequest(name="run_code", args={"lang": "py"}))

    def wf():
        return (yield from run_agent("go", decide=scripted(no_code, ANSWER), act=code_act()))

    trajectory = RecordingHandler().run(wf)
    assert isinstance(trajectory, Trajectory)
    assert (trajectory.steps[0].observation or "").startswith("[bad arguments for run_code] ")
    assert trajectory.answer == "done"


# --------------------------------------------------------------------- recurse


def _op(name: str, **args):
    """One canned observational step (schema=str keeps the canning trivial)."""
    return step(name, CallTool(name="op", args=args, result_schema=str))


# Every callback names its op PLAINLY — `recurse` places each one under `rec:{i}` /
# `fold:{level},{k}`. The canned keys below are the qualified names, so a combinator that
# assembled a wrong path would miss its response and fail loudly (the same proof as before,
# now over a path the handler composed rather than one the callback spliced).
def _decompose(ctx):
    raw = yield from _op("decompose", ctx=ctx)
    return raw.split(",")


def _leaf(chunk):
    return (yield from _op("leaf", chunk=chunk))


def _combine(group):
    return (yield from _op("merge", parts=list(group)))


def test_recurse_tree_folds_with_singleton_passthrough():
    """5 leaves at fanin 2: three fold levels, the odd leaf carried untouched —
    and the canned keys prove every scope the sugar assembled."""
    handler = RecordingHandler(
        responses={
            "decompose": "a,b,c,d,e",
            "gather:0,0;rec:0;leaf": "A",
            "gather:0,1;rec:1;leaf": "B",
            "gather:0,2;rec:2;leaf": "C",
            "gather:0,3;rec:3;leaf": "D",
            "gather:0,4;rec:4;leaf": "E",
            "gather:1,0;fold:0,0;merge": "AB",
            "gather:1,1;fold:0,1;merge": "CD",
            "gather:2,0;fold:1,0;merge": "ABCD",
            "gather:3,0;fold:2,0;merge": "ABCDE",
        }
    )

    def wf():
        return (yield from recurse("ctx", _decompose, _leaf, _combine, fanin=2))

    assert handler.run(wf) == "ABCDE"
    # Rule 1a sequencing in the parent trace: decompose commits BEFORE the fan-out; then the
    # fold is a tree of gathers, never one reduce. 1b: each gather is pure structure (no node
    # entry) and its leaves surface path-prefixed `gather:{g},{i};` — so the four gathers get
    # distinct positional namespaces g=0..3. (Pre-B3 this trace showed `gather:5, gather:2,
    # gather:1, gather:1` — fold levels 1 and 2 both `gather:1`, the very arity collision B3
    # fixes; they are now `gather:2` and `gather:3`.)
    assert [e.key.stored() for e in handler.trace] == [
        "step:decompose",
        "gather:0,0;rec:0;step:leaf",  # g=0: the leaf map over 5 chunks
        "gather:0,1;rec:1;step:leaf",
        "gather:0,2;rec:2;step:leaf",
        "gather:0,3;rec:3;step:leaf",
        "gather:0,4;rec:4;step:leaf",
        # g=1: fold level 0 — [A,B] [C,D] merge; [E] passes through
        "gather:1,0;fold:0,0;step:merge",
        "gather:1,1;fold:0,1;step:merge",
        "gather:2,0;fold:1,0;step:merge",  # g=2: fold level 1 — [AB,CD] merge; [E] passes through
        "gather:3,0;fold:2,0;step:merge",  # g=3: fold level 2 — [ABCD,E] merge
    ]


def test_an_enclosing_scope_encloses_the_whole_recurse():
    """Wrapping the call namespaces everything inside it — including the gather coordinates,
    which restart at 0 within the new frame. That ordering is the honest one: the frames
    appear in nesting order, where the old author-spliced scope landed INSIDE each leaf name
    and so read as `gather:…:outer;rec:0;leaf`."""
    handler = RecordingHandler(
        responses={
            "outer;decompose": "a,b",
            "outer;gather:0,0;rec:0;leaf": "A",
            "outer;gather:0,1;rec:1;leaf": "B",
            "outer;gather:1,0;fold:0,0;merge": "AB",
        }
    )

    def wf():
        return (
            yield from scoped(
                compose_key(t"outer"),
                lambda: recurse("ctx", _decompose, _leaf, _combine),
            )
        )

    assert handler.run(wf) == "AB"


def test_recurse_rejects_empty_decomposition_and_degenerate_fanin():
    def empty(ctx):
        return []
        yield  # pragma: no cover — makes this a generator

    def wf():
        return (yield from recurse("ctx", empty, _leaf, _combine))

    with pytest.raises(ValueError, match="no chunks"):
        RecordingHandler().run(wf)

    def flat():
        return (yield from recurse("ctx", _decompose, _leaf, _combine, fanin=1))

    with pytest.raises(ValueError, match="fanin must be >= 2"):
        RecordingHandler().run(flat)


# ----------------------------------------------------------------------- route


def _classifier(chunk):
    return (yield from _op("classify", chunk=chunk))


_HANDLERS = {
    "flat": lambda c: _op("answer", chunk=c),
    "code": lambda c: _op("script", chunk=c),
}


def test_route_dispatches_on_the_recorded_label():
    """`route` adds no frame of its own — it dispatches, and any namespace is the caller's."""
    handler = RecordingHandler(responses={"q:0;classify": "code", "q:0;script": "ran"})

    def wf():
        return (
            yield from scoped(
                compose_key(t"q:{0}"), lambda: route("chunk", _classifier, _HANDLERS)
            )
        )

    assert handler.run(wf) == "ran"
    assert [e.key.stored() for e in handler.trace] == ["q:0;step:classify", "q:0;step:script"]


def test_route_unknown_label_is_loud():
    handler = RecordingHandler(responses={"classify": "mystery"})

    def wf():
        return (yield from route("chunk", _classifier, _HANDLERS))

    with pytest.raises(LookupError, match="not a handler"):
        handler.run(wf)


# --------------------------------------------------------------------- descend


def _judge(ctx, level):
    # A plain name: `descend` runs each level inside `scoped(compose_key(t"d:{depth}"))`.
    raw = yield from _op("judge", ctx=ctx, final=level.final, model=level.model)
    if raw.startswith("deeper:"):
        return Deeper(raw.removeprefix("deeper:"))
    return Answered(raw)


def test_descend_stops_on_the_confidence_gate():
    handler = RecordingHandler(responses={"d:0;judge": "deeper:x1", "d:1;judge": "the answer"})

    def wf():
        return (yield from descend("ctx", _judge, budget=5))

    assert handler.run(wf) == "the answer"


def test_descend_budget_exhaustion_parks_then_grant_extends():
    handler = RecordingHandler(
        responses={"d:0;judge": "deeper:x1", "d:1;judge": "deeper:x2", "d:2;judge": "best"}
    )

    def wf():
        return (yield from descend("ctx", _judge, budget=1, run_id="r"))

    parked = handler.run(wf)
    assert isinstance(parked, Suspended)
    assert (
        parked.awaiting.stored() == f"{depth_grant_name('r', depth=1, generation=0).stored()}"
    )  # the cost boundary IS a permission boundary
    parked = parked.resume(Grant(add_depth=1))
    assert isinstance(parked, Suspended)
    assert parked.awaiting.stored() == f"{depth_grant_name('r', depth=2, generation=0).stored()}"
    # a 0-level grant demands the best answer now (the final nudge)
    assert parked.resume(Grant(add_depth=0)) == "best"


def test_descend_full_trace_pins_the_durable_key_sequence():
    """The equivalence pin for the recursive rewrite (and permanent hardening):
    the exact op-key sequence across every op-emitting path of descend — clean
    drill, exhaustion park, grant extension, re-exhaustion park, 0-grant final
    — inside an enclosing scope, so frame composition is pinned too. descend itself mints
    only the grant EVENT names plus the judge invocation order/args (the judge names its ops
    plainly and the per-level scope places them); both are exactly what this trace captures.

    Note where the grant park sits: `q;event;depth-grant:r,0,depth=1`, at the DESCENT's scope
    and not the level's — a grant is asked once per exhaustion, so its name must not move as
    the drill deepens. The `event;` tag sits directly on the awaited name."""
    handler = RecordingHandler(
        responses={
            "q;d:0;judge": "deeper:x1",
            "q;d:1;judge": "deeper:x2",
            "q;d:2;judge": "deeper:x3",
            "q;d:3;judge": "best",
        }
    )

    def wf():
        return (
            yield from scoped(
                compose_key(t"q"),
                lambda: descend("ctx", _judge, budget=1, run_id="r"),
            )
        )

    parked = handler.run(wf)
    assert isinstance(parked, Suspended)
    assert parked.awaiting.stored() == f"q;{depth_grant_name('r', depth=1, generation=0).stored()}"
    parked = parked.resume(Grant(add_depth=2))
    assert isinstance(parked, Suspended)
    assert parked.awaiting.stored() == f"q;{depth_grant_name('r', depth=3, generation=0).stored()}"
    assert parked.resume(Grant(add_depth=0)) == "best"
    assert [e.key.stored() for e in handler.trace] == [
        "q;d:0;step:judge",
        f"q;event;{depth_grant_name('r', depth=1, generation=0).stored()}",
        "q;d:1;step:judge",
        "q;d:2;step:judge",
        f"q;event;{depth_grant_name('r', depth=3, generation=0).stored()}",
        "q;d:3;step:judge",
    ]


def test_a_grant_rejects_a_negative_refill():
    """The cost boundary must not evaporate on a malformed HITL payload: `add_depth=-1` — a
    typo away from `1` — made `remaining` skip the `== 0` gate forever, granting UNBOUNDED
    descent. This is the one place a human's raw JSON reaches combinator state."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Grant(add_depth=-1)
    assert Grant(add_depth=0).add_depth == 0  # 0 stays legal: "answer now"


def test_descend_judge_may_not_descend_past_a_final_level():
    handler = RecordingHandler(responses={"d:0;judge": "deeper:x1", "d:1;judge": "deeper:x2"})

    def wf():
        return (yield from descend("ctx", _judge, budget=1))

    with pytest.raises(DescendedPastBudget, match="descended past an exhausted budget"):
        handler.run(wf)


def test_descend_grantor_cascade_human_parks_then_grant_extends():
    """The Budget/Grant grantor seam: the same park/resume as the built-in
    the built-in arm, but the answerer is a ``grant_cascade([human_grant(...)])``
    returning a ``Grant``. The park name matches descend's contract
    (``depth_grant_name(run_id, depth)``, placed by whatever frames enclose it); ``add_depth``
    refills the level budget and
    ``add_depth=0`` is the answer-now nudge."""
    handler = RecordingHandler(
        responses={"d:0;judge": "deeper:x1", "d:1;judge": "deeper:x2", "d:2;judge": "best"}
    )
    grantor = grant_cascade([human_grant("r")])

    def wf():
        return (yield from descend("ctx", _judge, budget=1, grantor=grantor))

    parked = handler.run(wf)
    assert isinstance(parked, Suspended)
    assert parked.awaiting == depth_grant_name(
        "r", depth=1, generation=0
    )  # the built-in arm's own name
    parked = parked.resume(Grant(add_depth=1))
    assert isinstance(parked, Suspended)
    assert parked.awaiting.stored() == f"{depth_grant_name('r', depth=2, generation=0).stored()}"
    assert parked.resume(Grant(add_depth=0)) == "best"  # add nothing = answer now


def test_descend_run_id_and_grantor_are_mutually_exclusive():
    handler = RecordingHandler(responses={})

    def wf():
        return (
            yield from descend(
                "ctx", _judge, budget=1, run_id="r", grantor=grant_cascade([human_grant("g")])
            )
        )

    with pytest.raises(ValueError, match="run_id OR grantor"):
        handler.run(wf)


def test_granted_levels_classifies_both_park_payloads():
    # The one classification point for the two schemas — pure, no handler needed.
    assert granted_levels(Grant(add_depth=3)) == 3
    assert granted_levels(Grant(add_depth=2)) == 2
    assert granted_levels(Grant()) == 0  # add nothing = answer now


def test_granted_levels_stop_forces_zero():
    # B3: `stop` is honored regardless of `add_depth` — the dead field, now enforced.
    assert granted_levels(Grant(add_depth=2, stop=True)) == 0
    assert granted_levels(Grant(add_depth=0, stop=True)) == 0


def test_descend_grant_stop_forces_final_answer_now():
    """End to end: `Grant(stop=True)` forces the final level regardless of `add_depth`, so
    the descent answers now instead of consuming the levels and re-parking deeper."""

    def judge(ctx, level):
        raw = yield from _op("judge", ctx=ctx, final=level.final)
        return Answered(raw) if level.final else Deeper(f"{ctx}.")

    handler = RecordingHandler(responses={"d:0;judge": "-", "d:1;judge": "answered-at-d1"})

    def wf():
        return (
            yield from descend("c", judge, budget=1, grantor=grant_cascade([human_grant("r")]))
        )

    parked = handler.run(wf)
    assert isinstance(parked, Suspended)
    assert parked.awaiting.stored() == f"{depth_grant_name('r', depth=1, generation=0).stored()}"
    # one forced-final judge call at d1 answers; no re-park deeper.
    assert parked.resume(Grant(add_depth=2, stop=True)) == "answered-at-d1"


def test_grant_cascade_falls_to_default_when_all_tiers_escalate():
    """First decisive tier wins; a tier returning ``None`` escalates; all-escalate
    falls to the default ``Grant()`` (add nothing = answer now) — the ``Grant``-typed
    analog of the permission cascade's fail-closed default."""

    def escalating_tier(depth):
        yield from ()  # deterministic, yields nothing
        return None

    grantor = grant_cascade([escalating_tier])
    gen = grantor(0)
    with pytest.raises(StopIteration) as done:
        next(gen)  # no tier yields or decides → returns the default immediately
    assert done.value.value == Grant()  # add_depth 0, stop False


def test_descend_promote_maps_depth_to_the_judged_model():
    handler = RecordingHandler(responses={"d:0;judge": "deeper:x1", "d:1;judge": "done"})

    def wf():
        return (
            yield from descend(
                "ctx", _judge, budget=3, promote=lambda d: "opus" if d == 0 else "haiku"
            )
        )

    assert handler.run(wf) == "done"
    judged = []
    for entry in handler.trace:
        assert isinstance(entry.op, Step)
        assert isinstance(entry.op.op, CallTool)
        judged.append(entry.op.op.args["model"])
    assert judged == ["opus", "haiku"]


def test_recurse_with_parking_leaves_serializes_wakes_through_the_scopes():
    """The retracted caveat, pinned (V1): recurse leaves may park durably; two
    parked leaves resume lowest-first, each on its fully-qualified event —
    the gather prefix composed with the combinator's own `rec:{i}` frame. The leaf awaits a
    PLAIN `ruling`; both frames are applied by the handler."""

    def parking_leaf(chunk):
        if chunk == "a":
            return (yield from _op("leaf", chunk=chunk))
        ruling = yield from await_event("ruling", str)
        return f"{chunk}+{ruling}"

    handler = RecordingHandler(
        responses={
            "decompose": "a,b,c",
            "gather:0,0;rec:0;leaf": "A",
            "gather:1,0;fold:0,0;merge": "AB",
            "gather:2,0;fold:1,0;merge": "ABC",
        }
    )

    def wf():
        return (yield from recurse("ctx", _decompose, parking_leaf, _combine, fanin=2))

    parked = handler.run(wf)
    assert isinstance(parked, Suspended)
    assert parked.awaiting.stored() == "gather:0,1;rec:1;ruling"
    parked = parked.resume("ok1")
    assert isinstance(parked, Suspended)
    assert parked.awaiting.stored() == "gather:0,2;rec:2;ruling"
    assert parked.resume("ok2") == "ABC"


def test_qualified_event_name_pins_the_qualification_grammar():
    """The helper must match the pinned conformance literals exactly: the literals are the
    contract; the helper is its importable anchor.

    It takes the PATH, outermost frame first, and both frame kinds compose in one call, so no
    caller splices a scope onto a name by hand (`"rec:1;grant:…"`) and makes the delimiter
    decision itself."""
    assert (
        qualified_event_name(
            GatherBranch(0, 1),
            compose_key(t"rec:{1}"),
            name=f"{depth_grant_name('r1', depth=1, generation=0).stored()}",
        ).stored()
        == f"gather:0,1;rec:1;{depth_grant_name('r1', depth=1, generation=0).stored()}"
    )
    # nested gathers — the nested_gather_park_wf conformance literal
    assert (
        qualified_event_name(GatherBranch(0, 0), GatherBranch(0, 0), name="ev:r1").stored()
        == "gather:0,0;gather:0,0;ev:r1"
    )
    # a bare name is its own qualified form
    assert qualified_event_name(name="ev:r1").stored() == "ev:r1"


def test_descend_leaf_in_recurse_parks_on_the_gather_qualified_grant():
    """Item 1b's recorder twin (durable: the conformance suite): a budget-1
    descend leaf under recurse parks on the gather-qualified grant —
    `gather:{g},{i};` (handler-side) composed with the combinator's own
    `rec:{i}` frame — and BOTH are invisible at the descend call site, which names nothing."""

    def drilling_leaf(chunk):
        if chunk == "a":
            return (yield from _op("leaf", chunk=chunk))
        return (yield from descend(chunk, _judge, budget=1, run_id="r"))

    handler = RecordingHandler(
        responses={
            "decompose": "a,b",
            "gather:0,0;rec:0;leaf": "A",
            "gather:0,1;rec:1;d:0;judge": "deeper:x1",
            "gather:0,1;rec:1;d:1;judge": "B",
            "gather:1,0;fold:0,0;merge": "AB",
        }
    )

    def wf():
        return (yield from recurse("ctx", _decompose, drilling_leaf, _combine, fanin=2))

    parked = handler.run(wf)
    assert isinstance(parked, Suspended)
    assert (
        parked.awaiting.stored()
        == f"gather:0,1;rec:1;{depth_grant_name('r', depth=1, generation=0).stored()}"
    )
    assert parked.resume(Grant(add_depth=0)) == "AB"


# -------------------------------------------------------------------- unfold + fix


def _splitter(ctx, level: Level):
    """Branches once at the root, and answers at every level below it."""
    if level.depth == 0:
        yield from _op("split", ctx=ctx)
        return Branch([f"{ctx}.a", f"{ctx}.b"], lambda values: _op("join", values=list(values)))
    return Answered((yield from _op("leaf", ctx=ctx)))


def _always_branch(ctx, level: Level):
    yield from _op("split", ctx=ctx)
    return Branch([ctx], lambda values: _op("join", values=list(values)))


def _always_deeper(ctx, level: Level):
    yield from _op("judge", ctx=ctx)
    return Deeper(ctx)


def test_a_descend_judge_unfolds_under_the_names_descend_gives_it():
    """A judge is a node that never branches, so the two drivers must agree byte for byte."""
    responses = {"d:0;judge": "deeper:x1", "d:1;judge": "deeper:x2", "d:2;judge": "the answer"}
    descended, unfolded = (
        RecordingHandler(responses=responses),
        RecordingHandler(responses=responses),
    )

    assert descended.run(lambda: descend("ctx", _judge, budget=5)) == "the answer"
    assert unfolded.run(lambda: unfold("ctx", _judge, budget=5)) == "the answer"
    assert [e.key.stored() for e in unfolded.trace] == [e.key.stored() for e in descended.trace]


def test_a_branch_gathers_its_children_one_level_deeper_then_joins():
    handler = RecordingHandler(responses={"split": "s", "leaf": "v", "join": "joined"})

    assert handler.run(lambda: unfold("root", _splitter, budget=3)) == "joined"
    assert [e.key.stored() for e in handler.trace] == [
        "d:0;step:split",
        "gather:0,0;rec:0;d:1;step:leaf",
        "gather:0,1;rec:1;d:1;step:leaf",
        "step:join",
    ]


def test_a_node_that_descends_at_its_final_level_is_refused():
    handler = RecordingHandler(responses={"judge": "j"})

    with pytest.raises(DescendedPastBudget, match="at depth 1"):
        handler.run(lambda: unfold("ctx", _always_deeper, budget=1))


def test_a_child_that_branches_at_its_final_level_is_refused_inside_its_gather():
    """The final level is a child's, so the refusal leaves through the gather as a group.

    The only pin on `unfold`'s final-level guard for `Branch`: without it a node that always
    branches recurses until memory runs out."""
    handler = RecordingHandler(responses={"split": "s", "join": "joined"})

    with pytest.raises(ExceptionGroup) as refused:
        handler.run(lambda: unfold("ctx", _always_branch, budget=1))
    assert refused.group_contains(DescendedPastBudget, match="at depth 1")


def test_a_node_sees_its_depth_and_whether_it_is_final_and_no_model():
    seen: list[Level] = []

    def node(ctx, level: Level):
        seen.append(level)
        yield from _op("judge", ctx=ctx)
        return Answered(ctx) if level.final else Deeper(ctx)

    RecordingHandler(responses={"judge": "j"}).run(lambda: unfold("ctx", node, budget=2))
    assert seen == [Level(0, "", False), Level(1, "", False), Level(2, "", True)]


def test_a_zero_budget_is_one_final_level():
    seen: list[Level] = []

    def node(ctx, level: Level):
        seen.append(level)
        return Answered((yield from _op("judge", ctx=ctx)))

    RecordingHandler(responses={"judge": "j"}).run(lambda: unfold("ctx", node, budget=0))
    assert seen == [Level(0, "", True)]


def test_unfold_refuses_a_negative_budget():
    with pytest.raises(ValueError, match="budget must be >= 0"):
        RecordingHandler().run(lambda: unfold("ctx", _splitter, budget=-1))


def test_tree_search_runs_each_round_under_its_own_scope():
    seen: list[list[str]] = []

    def node_for(state: list[str]):
        seen.append(list(state))

        def node(ctx, level: Level):
            return Answered((yield from _op("look", ctx=ctx)))

        return node

    handler = RecordingHandler(responses={"look": "seen"})
    result = handler.run(
        lambda: tree_search(
            "root",
            node_for,
            lambda state, value: [*state, value],
            initial=[],
            iterations=2,
            depth=1,
        )
    )
    assert result == ["seen", "seen"]
    assert seen == [[], ["seen"]]
    assert [e.key.stored() for e in handler.trace] == [
        "search:0;d:0;step:look",
        "search:1;d:0;step:look",
    ]


def test_tree_search_needs_a_round():
    with pytest.raises(ValueError, match="iterations must be >= 1"):
        RecordingHandler().run(
            lambda: tree_search(
                "root", lambda _s: _judge, lambda s, v: s, initial=0, iterations=0, depth=1
            )
        )


def _z(f):
    """Z = λf. (λx. f (λv. x x v)) (λx. f (λv. x x v)), spelled literally."""

    def half(x):
        return f(lambda *v: x(x)(*v))

    return half(half)


def _countdown(recur):
    def body(n: int):
        here = yield from scoped(compose_key(t"d:{Index(n)}"), lambda: _op("leaf", n=n))
        return here if n == 0 else here + (yield from recur(n - 1))

    return body


def test_fix_closes_an_open_recursion_as_the_literal_z_does():
    closed, literal = (
        RecordingHandler(responses={"leaf": "x"}),
        RecordingHandler(responses={"leaf": "x"}),
    )

    assert closed.run(lambda: fix(_countdown)(3)) == "xxxx"
    assert literal.run(lambda: _z(_countdown)(3)) == "xxxx"
    assert [e.key.stored() for e in closed.trace] == [e.key.stored() for e in literal.trace]
    assert [e.key.stored() for e in closed.trace] == [
        "d:3;step:leaf",
        "d:2;step:leaf",
        "d:1;step:leaf",
        "d:0;step:leaf",
    ]


def _drive(parked, grants):
    """Resume a suspended run with each grant in turn, recording what it awaited."""
    awaited = []
    for grant in grants:
        assert isinstance(parked, Suspended), parked
        awaited.append(parked.awaiting.stored())
        parked = parked.resume(grant)
    return parked, awaited


@pytest.mark.parametrize(
    "answerer",
    [{"run_id": "r"}, {"grantor": grant_cascade([human_grant("r")])}],
    ids=["run_id", "grantor"],
)
def test_unfold_parks_for_levels_where_descend_does(answerer):
    """The exhaustion park, a grant, a second park and a 0-level grant, inside an enclosing scope:
    `unfold` and `descend` must await the same names and write the same trace, byte for byte."""
    responses = {
        "q;d:0;judge": "deeper:x1",
        "q;d:1;judge": "deeper:x2",
        "q;d:2;judge": "deeper:x3",
        "q;d:3;judge": "best",
    }
    grants = [Grant(add_depth=2), Grant(add_depth=0)]
    runs = {}
    for driver in (descend, unfold):
        handler = RecordingHandler(responses=responses)
        parked = handler.run(
            lambda d=driver: scoped(
                compose_key(t"q"), lambda: d("ctx", _judge, budget=1, **answerer)
            )
        )
        result, awaited = _drive(parked, grants)
        runs[driver.__name__] = (result, awaited, [e.key.stored() for e in handler.trace])

    assert runs["unfold"] == runs["descend"]
    assert runs["unfold"][1] == [
        f"q;{depth_grant_name('r', depth=1, generation=0).stored()}",
        f"q;{depth_grant_name('r', depth=3, generation=0).stored()}",
    ]


def test_a_stop_grant_makes_the_level_final_under_unfold():
    handler = RecordingHandler(responses={"d:0;judge": "deeper:x1", "d:1;judge": "answered"})

    parked = handler.run(lambda: unfold("c", _judge, budget=1, run_id="r"))
    result, _ = _drive(parked, [Grant(add_depth=2, stop=True)])
    assert result == "answered"


def test_each_child_of_a_branch_parks_for_its_own_levels():
    """The budget counts levels along a path, so two children exhausting together park twice,
    each under its own branch's frames."""

    def node(ctx, level: Level):
        yield from _op("look", ctx=ctx)
        if level.depth == 0:
            return Branch(["a", "b"], lambda values: _op("join", values=list(values)))
        return Answered(f"{ctx}@{level.depth}") if level.final else Deeper(ctx)

    handler = RecordingHandler(responses={"look": "seen", "join": "joined"})
    parked = handler.run(lambda: unfold("root", node, budget=1, run_id="r"))
    result, awaited = _drive(parked, [Grant(add_depth=0), Grant(add_depth=0)])

    grant = depth_grant_name("r", depth=1, generation=0).stored()
    assert result == "joined"
    assert awaited == [
        qualified_event_name(
            GatherBranch(0, i), compose_key(t"rec:{Index(i)}"), name=grant
        ).stored()
        for i in range(2)
    ]


def test_promote_names_each_levels_model():
    seen: list[Level] = []

    def node(ctx, level: Level):
        seen.append(level)
        yield from _op("judge", ctx=ctx)
        return Answered(ctx) if level.final else Deeper(ctx)

    RecordingHandler(responses={"judge": "j"}).run(
        lambda: unfold("ctx", node, budget=1, promote=lambda depth: f"m{depth}")
    )
    assert seen == [Level(0, "m0", False), Level(1, "m1", True)]


def test_unfold_run_id_and_grantor_are_mutually_exclusive():
    with pytest.raises(ValueError, match="run_id OR grantor"):
        RecordingHandler().run(
            lambda: unfold("ctx", _judge, budget=1, run_id="r", grantor=grant_cascade([]))
        )


# --------------------------------------------------------------------- hoisted


def test_hoisted_activates_once_above_the_fanout():
    pin = Pin(name="slicing", content_hash="abc", body="how to slice")
    handler = RecordingHandler(
        responses={
            "skill:slicing,activate": pin,
            "decompose": "a,b",
            "gather:0,0;rec:0;leaf": "A",
            "gather:0,1;rec:1;leaf": "B",
            "gather:1,0;fold:0,0;merge": "AB",
        }
    )

    def wf():
        def body(pins):
            assert pins["slicing"].body == "how to slice"
            return recurse("ctx", _decompose, _leaf, _combine)

        # rule 4: duplicate names still activate once, above the gather
        return (yield from hoisted(("slicing", "slicing"), body))

    assert handler.run(wf) == "AB"
    keys = [e.key.stored() for e in handler.trace]
    assert keys[0] == "step;skill:slicing,activate"  # hoisted: before decompose + gathers
    assert keys.count("step;skill:slicing,activate") == 1


# --- one park name must mean one schema -------------------------------------------------


def test_the_two_grant_arms_park_on_one_name_with_one_schema():
    """`human_grant` and `descend`'s built-in arm park on byte-identical names, so they await
    one schema. Two schemas would let an emitter following one contract be silently misread by
    the other: `{"extra_levels": 3}` validated to `Grant()`, granting ZERO levels with nothing
    in the log."""
    park = next(refill_levels(2, run_id="r1", grantor=None))
    # Narrowed, not assumed: `refill` yields a `WorkflowOp`, and the two assertions below read
    # `AwaitEvent`'s own fields. Stating the arm is what the test meant all along.
    assert isinstance(park, AwaitEvent)
    assert park.name.stored() == f"{depth_grant_name('r1', depth=2, generation=0).stored()}"
    assert park.schema is Grant, "the built-in arm must await what `human_grant` awaits"


def test_a_foreign_field_is_refused_rather_than_granting_zero():
    """The silent zero: a wrong payload would produce a VALID all-defaults grant. Pydantic's
    default drops unknown fields; `extra="forbid"` turns a foreign field name, such as the one a
    stale emitter sends, into a refusal at the boundary that `ge=0` already promises is loud."""
    import pytest
    from pydantic import ValidationError

    from effective.budget import Grant

    with pytest.raises(ValidationError, match="extra_levels"):
        Grant.model_validate({"extra_levels": 3})

    assert granted_levels(Grant.model_validate({"add_depth": 3})) == 3  # the right field works


def test_all_three_grant_parks_compose_through_the_registry():
    """`human_grant`, `descend`'s built-in arm and `respawn`'s generation refill all compose
    this name through one registry, so none can drift from the others' payload schema."""
    from effective.budget import Budget, chain_grant_name, depth_grant_name
    from effective.combinators import _refill_generations, human_grant

    human_park = next(human_grant("r1")(2))
    assert isinstance(human_park, AwaitEvent)
    assert human_park.name == depth_grant_name("r1", depth=2, generation=0)

    built_in_park = next(refill_levels(2, run_id="r1", grantor=None))
    assert isinstance(built_in_park, AwaitEvent)
    assert built_in_park.name == depth_grant_name("r1", depth=2, generation=0)

    chain = _refill_generations(Budget(run_id="r1", on_exhaust="park"), 3)
    chain_park = next(chain)
    assert isinstance(chain_park, AwaitEvent)
    assert chain_park.name == chain_grant_name("r1", 3)


def test_each_grant_axis_owns_its_own_namespace():
    """The axis is the TAG, not a letter inside a shared name. That is what makes these
    identities visible to the key registry, which is what enforces one tag / one arity — a
    letter inside `{base}:d{n}` was invisible to it, and a typo would have minted a fourth
    namespace nobody emits to."""
    from effective.budget import CHAIN_GRANT, DEPTH_GRANT, chain_grant_name, depth_grant_name

    assert depth_grant_name("r1", depth=1, generation=0) != chain_grant_name("r1", 1)
    assert depth_grant_name("r1", depth=1, generation=0).stored().startswith(f"{DEPTH_GRANT}:")
    assert chain_grant_name("r1", 1).stored().startswith(f"{CHAIN_GRANT}:")
    # A run id carrying the delimiter would forge a level; `Segment` refuses it at composition.
    with pytest.raises(ValueError, match="delimiter"):
        depth_grant_name("r1:9", depth=1, generation=0)


def test_a_refill_does_not_cross_tasks():
    """Reddens if an `unfold` across tasks accepts a refill: a granted level grants no spawn depth,
    so the handler would refuse the spawns the grant allowed."""
    crossing = AcrossTasks("node", TypeAdapter(str), TypeAdapter(str))

    def wf():
        return (yield from unfold("ctx", _judge, budget=1, run_id="r", crossing=crossing))

    with pytest.raises(ValueError, match="cannot cross tasks"):
        RecordingHandler(responses={}).run(wf)
