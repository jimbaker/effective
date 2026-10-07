"""A ReAct agent loop expressed as an `effective` workflow (prototype spec).

These tests are the executable form of one recommendation: house a ReAct loop
*inside* an effective workflow generator, so every
model turn is a checkpointed ``AskLLM`` op and every tool action a ``CallTool``
op. The loop's branching is pure code between yields — which means the existing
record / replay handlers make the *entire agent trajectory* durable and
replayable, i.e. test-without-LLM for free.

Red-green: written before `effective.react` exists (it was `agent.react` then).
"""

import pytest

from effective import (
    RecordingHandler,
    ReplayHandler,
    ReplayMismatch,
    Suspended,
    ask_llm,
    call_tool,
)
from effective.api import Effect
from effective.domain import AskLLM, CallTool
from effective.interrupts import Phase, Quiet, Redirect, Signal
from effective.react import (
    AssistantTurn,
    ToolRequest,
    ToolResult,
    Trajectory,
    TrajectorySummary,
    _assistant,
    run_agent,
    summary_message,
    tool_key,
)
from effective.tape import TurnKind, transcript_violations, turn_kind

QUESTION = "when was acme founded?"


def responses_two_step() -> dict:
    """search -> lookup -> finish, each answer keyed under the `d:{i}` frame of its turn."""
    return {
        "d:0;react:turn": AssistantTurn(
            thought="search first", tool=ToolRequest(name="search", args={"q": "acme"})
        ),
        "d:0;tool:search": ToolResult(content="found id=7"),
        "d:1;react:turn": AssistantTurn(
            thought="now look it up", tool=ToolRequest(name="lookup", args={"id": 7})
        ),
        "d:1;tool:lookup": ToolResult(content="Acme Corp, founded 1999"),
        "d:2;react:turn": AssistantTurn(thought="done", answer="Acme Corp was founded in 1999."),
    }


TWO_STEP_TRACE = [
    "d:0;step;react:turn",
    "d:0;step;tool:search",
    "d:1;step;react:turn",
    "d:1;step;tool:lookup",
    "d:2;step;react:turn",
]


def responses_never_finish() -> dict:
    """The model keeps acting and never answers — exercises the iter ceiling."""
    return {
        "d:0;react:turn": AssistantTurn(
            thought="t0", tool=ToolRequest(name="search", args={"q": "x"})
        ),
        "d:0;tool:search": ToolResult(content="r0"),
        "d:1;react:turn": AssistantTurn(
            thought="t1", tool=ToolRequest(name="lookup", args={"id": 1})
        ),
        "d:1;tool:lookup": ToolResult(content="r1"),
        "react:final": AssistantTurn(thought="wrap up", answer="best effort from the context"),
    }


# --- multi-step run, zero I/O ----------------------------------------------


def test_multi_step_agent_runs_with_zero_io():
    h = RecordingHandler(responses_two_step())
    traj = h.run(lambda: run_agent(QUESTION))

    assert isinstance(traj, Trajectory)
    assert traj.answer == "Acme Corp was founded in 1999."
    assert traj.stop_reason == "finish"
    # two tool actions + the finishing turn
    assert len(traj.steps) == 3
    assert [s.tool.name for s in traj.steps if s.tool is not None] == ["search", "lookup"]
    assert traj.steps[0].observation == "found id=7"
    # every model turn and every tool action is a recorded op, in order
    assert [e.key.stored() for e in h.trace] == TWO_STEP_TRACE


# --- durability: the whole trajectory replays without the model ------------


def test_agent_trajectory_replays_without_the_model():
    rec = RecordingHandler(responses_two_step())
    live = rec.run(lambda: run_agent(QUESTION))

    # ReplayHandler gets *no* canned responses — only the recorded op trace.
    replayed = ReplayHandler(rec.trace).run(lambda: run_agent(QUESTION))

    assert replayed == live
    assert replayed.answer == "Acme Corp was founded in 1999."


def test_replay_detects_a_changed_agent_control_flow():
    rec = RecordingHandler(responses_two_step())
    rec.run(lambda: run_agent(QUESTION))

    def altered():
        # op 0 matches the first turn's name, then we call a *different* tool than recorded
        yield from ask_llm("react:turn", [{"role": "user", "content": QUESTION}], AssistantTurn)
        yield from call_tool("other", {}, ToolResult)  # recorded op 1 was tool:search
        return Trajectory(answer="x", steps=[], stop_reason="finish")

    with pytest.raises(ReplayMismatch):
        ReplayHandler(rec.trace).run(altered)


# --- the iteration ceiling forces a final answer ----------------------------


def test_max_iters_forces_a_synthesized_final_answer():
    h = RecordingHandler(responses_never_finish())
    traj = h.run(lambda: run_agent(QUESTION, max_iters=2))
    assert not isinstance(traj, Suspended)

    assert traj.stop_reason == "max_iters"
    assert traj.answer == "best effort from the context"
    assert [e.key.stored() for e in h.trace] == [
        "d:0;step;react:turn",
        "d:0;step;tool:search",
        "d:1;step;react:turn",
        "d:1;step;tool:lookup",
        "d:2;step;react:final",
    ]


# --- the compaction seam (ADR-less SVS: prove it before promoting) ----------


def responses_with_compaction() -> dict:
    """Same two-step run, plus a canned summary for the compaction turn that fires
    once the transcript grows past the trigger."""
    return {
        **responses_two_step(),
        "d:2;react:compact": TrajectorySummary(
            facts=["acme has id=7"], tried=["searched, then looked up"], open_questions=[]
        ),
    }


def test_compact_none_is_identical_to_no_compaction():
    """The seam's core guarantee: default off changes nothing — same result, same
    op stream as a run that never knew the parameter existed."""
    off = RecordingHandler(responses_two_step())
    a = off.run(lambda: run_agent(QUESTION))
    none = RecordingHandler(responses_two_step())
    b = none.run(lambda: run_agent(QUESTION, compact=None))
    assert a == b
    assert (
        [e.key.stored() for e in off.trace]
        == [e.key.stored() for e in none.trace]
        == TWO_STEP_TRACE
    )


def test_compaction_inserts_one_recorded_summary_op():
    """When the trigger fires, exactly one ``react:compact`` Step appears in the op
    stream; the reasoning trajectory (steps) is untouched — compaction is context
    management, not a reasoning step."""
    h = RecordingHandler(responses_with_compaction())
    traj = h.run(lambda: run_agent(QUESTION, compact=lambda m: len(m) > 4))
    assert not isinstance(traj, Suspended)

    assert traj.answer == "Acme Corp was founded in 1999."
    assert traj.stop_reason == "finish"
    assert len(traj.steps) == 3  # search, lookup, finish — compaction adds no Step
    assert [e.key.stored() for e in h.trace] == [
        "d:0;step;react:turn",
        "d:0;step;tool:search",
        "d:1;step;react:turn",
        "d:1;step;tool:lookup",
        "d:2;step;react:compact",  # the reification — a recorded summarize op
        "d:2;step;react:turn",
    ]


def test_compaction_run_replays_without_the_model():
    """The recorded summary makes the compacted run replay-exact: ReplayHandler gets
    only the trace (no canned responses, no model)."""
    rec = RecordingHandler(responses_with_compaction())
    live = rec.run(lambda: run_agent(QUESTION, compact=lambda m: len(m) > 4))

    replayed = ReplayHandler(rec.trace).run(
        lambda: run_agent(QUESTION, compact=lambda m: len(m) > 4)
    )
    assert replayed == live
    assert replayed.answer == "Acme Corp was founded in 1999."


# --- the turn frame ------------------------------------------------------------------------------


def test_one_tool_called_on_two_turns_composes_TWO_keys():
    """One tool called on two turns lands two keys, told apart by the `d:{i}` frame of each turn.

    A reader deriving identity from the placed key (`graphview`, the key registry, the paths sweep)
    keeps both calls."""
    responses = {
        "d:0;react:turn": AssistantTurn(
            thought="once", tool=ToolRequest(name="search", args={"q": "a"})
        ),
        "d:0;tool:search": ToolResult(content="first"),
        "d:1;react:turn": AssistantTurn(
            thought="twice", tool=ToolRequest(name="search", args={"q": "b"})
        ),
        "d:1;tool:search": ToolResult(content="second"),
        "d:2;react:turn": AssistantTurn(thought="done", answer="two searches"),
    }
    h = RecordingHandler(responses)
    traj = h.run(lambda: run_agent(QUESTION))
    assert isinstance(traj, Trajectory)

    keys = [e.key.stored() for e in h.trace if "tool:" in e.key.stored()]
    assert keys == ["d:0;step;tool:search", "d:1;step;tool:search"]
    assert len(set(keys)) == 2, "one tool called twice still composes one key"
    assert [s.observation for s in traj.steps[:2]] == ["first", "second"]


def test_the_loops_tool_key_is_the_direct_call_key():
    """A loop's action and a direct `call_tool` share `tool:{name}`: the turn is the frame's."""
    from effective.api import direct_tool_key

    assert direct_tool_key("run_suite").stored() == "tool:run_suite"
    assert tool_key("run_suite") == direct_tool_key("run_suite")


def test_the_transcript_carries_the_ACTION_not_only_the_thought():
    """ReAct is Thought/Action/Observation, and the middle term is load-bearing.

    Rendering only `thought` leaves a model re-reading its own monologue beside an unattributed
    `Tool result:`: it cannot tell WHICH tool produced it, or that it has already run one. The
    observed failure is the obvious one: it calls the same tool again, forever.

    **Measured on a live provider**, gpt-5-nano over the code-agent fixture, six runs
    per arm: thought-only passed **1/6**, thought+action passed **6/6** (Fisher exact two-sided
    p = 0.015). The control's transcript showed `list_dir` called six times against a correct and
    present observation.

    Pinned here because nothing else in the suite asserts what the loop actually sends back."""
    acting = _assistant(
        AssistantTurn(thought="look around", tool=ToolRequest(name="list_dir", args={}))
    )
    assert "list_dir" in acting["content"], acting
    assert "look around" in acting["content"], acting

    # The ARGUMENTS too: two calls to one tool differing only in their args are otherwise
    # indistinguishable in the transcript, which is the same repeat-forever failure one level in.
    read = _assistant(
        AssistantTurn(
            thought="read it", tool=ToolRequest(name="read_file", args={"path": "mod.py"})
        )
    )
    assert "mod.py" in read["content"], read

    # A finishing turn is unchanged — there is no action to report.
    done = _assistant(AssistantTurn(thought="reasoning", answer="the answer"))
    assert done["content"] == "the answer"


# --- the transitions, checked at every point on a recorded tape --------------


def _user_msg(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant_msg(text: str) -> dict:
    return {"role": "assistant", "content": text + "\nAction: list_dir({})"}


class _K:
    """The smallest thing the walk asks of a key: `.stored()`. A synthetic tape needs no more."""

    def __init__(self, text: str) -> None:
        self._t = text

    def stored(self) -> str:
        return self._t


class _Entry:
    """A hand-built `TraceEntry` stand-in — the walk duck-types `key`/`op`/`result`.

    Hand-built rather than driven, because these tapes state shapes the live loop cannot currently
    produce (see the compaction tests below); driving them would mean bending the loop to make a
    point about the walk."""

    def __init__(self, key: str, op, result=None) -> None:
        self.key, self.op, self.result = _K(key), op, result


def test_every_turn_can_see_the_previous_action_and_its_observation():
    """End-to-end over the tape, with no model and no I/O: the shape that catches a dropped
    Action.

    The earlier tests here assert the RESULT and the op-key SEQUENCE, both of which stay correct
    while the loop is unusable: `list_dir` six times in a row is a valid key sequence. Only the
    content of each transition shows that defect."""
    rec = RecordingHandler(responses_two_step())
    rec.run(lambda: run_agent(QUESTION))

    acted = [e for e in rec.trace if isinstance(getattr(e.op, "op", e.op), CallTool)]
    assert len(acted) == 2, [e.key.stored() for e in rec.trace]  # anti-vacuity: it really acted
    # `require_clean` carries the second half of that anti-vacuity claim: three ReAct turns
    # were SCORED, so an empty result means the property held rather than never having looked.
    transcript_violations(rec.trace).require_clean(at_least=3)


def test_the_walk_survives_the_round_trip_through_replay():
    """The same check over a REPLAYED tape. Replay re-serves recorded results rather than calling
    a model, so the transcript is re-derived by the loop's own code — if replay reconstructed the
    messages differently from the live run, this is where it would show."""
    rec = RecordingHandler(responses_two_step())
    live = rec.run(lambda: run_agent(QUESTION))

    replayed = ReplayHandler(rec.trace)
    assert replayed.run(lambda: run_agent(QUESTION)) == live
    transcript_violations(rec.trace).require_clean(at_least=3)


def test_the_walk_scores_only_ReAct_turns_and_not_the_other_askllms():
    """A real tape carries `AskLLM`s that are NOT turns, and scoring them invents violations.

    Two kinds, both on a live coding tape: an emitter (`write:3` — a metered call that produces
    file contents, with its own prompt and no sight of the prior tool call) and the summarizer
    (`react:compact`). The second is why this is not a substring test: its key CONTAINS `react:`.

    Measured on a live sidecar: unscoped, 5 of 24 runs clean; scoped, 24 of 24. The check that
    counts is the committed function pointed at a tape with an emitter in it."""
    assert turn_kind("d:0;step;react:turn") is TurnKind.REACT
    assert turn_kind("step;react:final") is TurnKind.REACT
    assert (
        turn_kind("d:1;state:draft;d:2;step;react:turn") is TurnKind.REACT
    )  # frames do not matter
    assert turn_kind("d:3;step;react:compact") is TurnKind.COMPACTION  # the summariser, not a turn
    assert turn_kind("step;write:3") is TurnKind.OTHER
    assert turn_kind("d:1;step;tool:read_file") is TurnKind.OTHER


def test_an_emitter_between_a_tool_and_the_next_turn_is_not_a_violation():
    """One emitter `AskLLM` between the tool call and the next ReAct turn, which is what a live
    coding-agent run emits. Unscoped, the check reports two false violations against a correct
    loop."""

    tape = [
        _Entry(
            "d:0;step;react:turn",
            AskLLM(messages=[_user_msg("go")], response_schema=dict),
            AssistantTurn(thought="looked", tool=ToolRequest(name="list_dir", args={})),
        ),
        _Entry(
            "d:0;step;tool:list_dir",
            CallTool(name="list_dir", args={}, result_schema=ToolResult),
            ToolResult(content="mod.py"),
        ),
        # the emitter: its prompt is its own, and it has never seen `list_dir`
        _Entry(
            "step;write:0",
            AskLLM(messages=[_user_msg("write it")], response_schema=dict),
        ),
        _Entry(
            "d:1;step;react:turn",
            AskLLM(
                messages=[
                    _user_msg("go"),
                    _assistant_msg("looked"),
                    _user_msg("Tool result: mod.py"),
                ],
                response_schema=dict,
            ),
        ),
    ]

    transcript_violations(tape).require_clean(at_least=2)


# --- the walk across a compaction ------------------------------------------------------------


def test_the_walk_is_clean_across_a_compaction():
    """A compacting run satisfies the property, and the walk SCORES every turn in it.

    Compaction is the one thing on this tape that can legitimately break "turn `i` sees turn
    `i-1`": `KEEP_TAIL` rewrites the transcript. The walk therefore treats `react:compact` as a
    boundary rather than as a turn — and `require_clean` states that the three real turns were
    still scored, so this is not clean-by-not-looking."""
    rec = RecordingHandler(responses_with_compaction())
    rec.run(lambda: run_agent(QUESTION, compact=lambda m: len(m) > 4))

    assert [turn_kind(e.key.stored()) for e in rec.trace].count(TurnKind.COMPACTION) == 1
    transcript_violations(rec.trace).require_clean(at_least=3)


def test_a_compaction_that_drops_the_summary_is_a_violation():
    """The boundary REPLACES the obligation rather than waiving it.

    Exempting the turn after a compaction and stopping there would let a compaction that spliced
    nothing pass clean — the model would have lost the prefix and gained nothing, which is the
    failure compaction exists to avoid. So the next scored turn must be able to see the summary,
    and this tape is that turn without it."""
    summary = TrajectorySummary(facts=["acme has id=7"], tried=[], open_questions=[])
    tape = [
        _Entry("d:0;step;react:turn", AskLLM(messages=[_user_msg("go")], response_schema=dict)),
        _Entry(
            "d:1;step;react:compact",
            AskLLM(messages=[_user_msg("summarize")], response_schema=dict),
            summary,
        ),
        # the splice never happened: the prefix is gone and the summary is not there either
        _Entry("d:1;step;react:turn", AskLLM(messages=[_user_msg("go")], response_schema=dict)),
    ]
    verdict = transcript_violations(tape)
    assert verdict.witnessed == 2
    assert [v.why for v in verdict.violations] == [
        "cannot see the summary that replaced the compacted prefix"
    ]


def test_the_boundary_exempts_a_prefix_the_splice_really_dropped():
    """A turn whose prior action the splice removed is NOT a violation — that is the false
    positive the boundary exists to prevent.

    **This tape is synthetic on purpose, and the reason is worth recording.** On the plain loop the
    exemption never fires: a turn contributes two messages (assistant, tool), so `KEEP_TAIL = 4` is
    exactly the last two turns and the immediately-prior action always survives the splice —
    measured for 1..5 prior turns. It takes three extra interleaved messages to push it out, which
    today means interrupts. So the exemption is DEFENSIVE rather than currently load-bearing on
    the plain loop, and this test says which of the two it is instead of claiming the stronger
    thing."""
    summary = TrajectorySummary(facts=["acme has id=7"], tried=[], open_questions=[])
    spliced = str(summary_message(summary)["content"])
    tape = [
        _Entry(
            "d:0;step;react:turn",
            AskLLM(messages=[_user_msg("go")], response_schema=dict),
            AssistantTurn(
                thought="search first", tool=ToolRequest(name="search", args={"q": "acme"})
            ),
        ),
        _Entry(
            "d:0;step;tool:search",
            CallTool(name="search", args={"q": "acme"}, result_schema=ToolResult),
            ToolResult(content="found id=7"),
        ),
        _Entry(
            "d:1;step;react:compact",
            AskLLM(messages=[_user_msg("summarize")], response_schema=dict),
            summary,
        ),
        # the splice dropped both the action and its observation; only the summary carries them
        _Entry(
            "d:1;step;react:turn",
            AskLLM(messages=[_user_msg("go"), _user_msg(spliced)], response_schema=dict),
        ),
    ]
    transcript_violations(tape).require_clean(at_least=2)


def test_without_the_boundary_that_same_tape_would_report_two_false_violations():
    """The boundary's cost, stated as a number rather than asserted as a principle.

    Same tape as above with the compaction relabelled as an ordinary turn: the walk now scores it,
    the obligation from `search` survives into it, and two violations appear against a loop that
    did nothing wrong. This is the test that fails if someone deletes the `COMPACTION` arm."""
    summary = TrajectorySummary(facts=["acme has id=7"], tried=[], open_questions=[])
    spliced = str(summary_message(summary)["content"])
    tape = [
        _Entry(
            "d:0;step;react:turn",
            AskLLM(messages=[_user_msg("go")], response_schema=dict),
            AssistantTurn(
                thought="search first", tool=ToolRequest(name="search", args={"q": "acme"})
            ),
        ),
        _Entry(
            "d:0;step;tool:search",
            CallTool(name="search", args={"q": "acme"}, result_schema=ToolResult),
            ToolResult(content="found id=7"),
        ),
        _Entry(
            "d:9;step;react:turn",
            AskLLM(messages=[_user_msg("summarize")], response_schema=dict),
            summary,
        ),
        _Entry(
            "d:1;step;react:turn",
            AskLLM(messages=[_user_msg("go"), _user_msg(spliced)], response_schema=dict),
        ),
    ]
    verdict = transcript_violations(tape)
    assert verdict.witnessed == 3
    # TWO violations, one per obligation — asserted by what each names rather than by its exact
    # prose. This test pinned both sentences verbatim and went red when the action check started
    # asking the renderer for its line instead of rebuilding the name and the values; the property
    # it exists to state (delete the `COMPACTION` arm and a correct loop reports two violations)
    # never moved. Pinning the wording pins today's phrasing, which is the instance, not the class.
    assert len(verdict.violations) == 2, [v.why for v in verdict.violations]
    assert "action" in verdict.violations[0].why
    assert "observation" in verdict.violations[1].why


def test_loop_injected_args_are_not_the_models_action():
    """The action a turn must be able to see is the one the MODEL chose, not the one the loop
    executed — and on a real tape those differ.

    A coding agent splices its workflow-local file `tree` into every op's args
    (`args={**request.args, "tree": tree}`), so the executed `CallTool` carries an entire source
    tree the model never chose and could not possibly be shown. A walk that read the executed op
    reported four violations against a live, SUCCEEDING run — confident, wrong, and invisible to
    the canned fixture, where the model's args and the executed args happen to be equal.

    This tape makes them differ, which is the only way to state the difference as a test."""
    tape = [
        _Entry(
            "d:0;step;react:turn",
            AskLLM(messages=[_user_msg("go")], response_schema=dict),
            # the model asked for a bare `list_dir()` — no arguments at all
            AssistantTurn(thought="look around", tool=ToolRequest(name="list_dir", args={})),
        ),
        _Entry(
            "d:0;step;tool:list_dir",
            # ...but the loop executed it with its whole workflow-local tree spliced in
            CallTool(
                name="list_dir",
                args={"tree": {"mod.py": "def add(a, b): return a - b"}},
                result_schema=ToolResult,
            ),
            ToolResult(content="mod.py"),
        ),
        _Entry(
            "d:1;step;react:turn",
            AskLLM(
                messages=[
                    _user_msg("go"),
                    {"role": "assistant", "content": "look around\nAction: list_dir({})"},
                    {"role": "tool", "content": "mod.py"},
                ],
                response_schema=dict,
            ),
        ),
    ]
    transcript_violations(tape).require_clean(at_least=2)


# --- the walk agrees with the renderer, or it does not check what it claims -------------------
#
# Both pins below close together, on one change: `_turn_violations` should build the expected line
# with `_assistant` and search for THAT, instead of reconstructing name-plus-values by hand. They
# point in opposite directions on purpose — one says the check is too weak, the other says the same
# check is too strong — because a hand-rolled substring match fails both ways at once, and fixing
# only the direction you noticed leaves the other live.


def test_a_turn_that_only_MENTIONS_the_tool_has_not_seen_the_action():
    """Too weak. The property's claim is that turn `i` can see what turn `i-1` DID.

    What it tests is that the name appears in some message and each argument's `str()` appears in
    that same message. A goal like "use read_file on mod.py" satisfies both while the transcript
    contains no action line at all — the model is being asked to infer causation from adjacency,
    which is the exact failure the walk was written to catch, passing the walk.

    The name is model-controlled on the chat path (`_WireTool.name` is free text), so a
    single-character tool name makes the check unfalsifiable outright: `"o" in content` is true of
    nearly every transcript. That is the general form of this pin; the goal-mention case is the one
    that arises without an adversary."""
    tape = [
        _Entry(
            "d:0;step;react:turn",
            AskLLM(messages=[_user_msg("use read_file on mod.py")], response_schema=dict),
            AssistantTurn(
                thought="read it", tool=ToolRequest(name="read_file", args={"path": "mod.py"})
            ),
        ),
        _Entry(
            "d:0;step;tool:read_file",
            CallTool(name="read_file", args={"path": "mod.py"}, result_schema=ToolResult),
            ToolResult(content="contents"),
        ),
        # The next turn never saw an `Action:` line — only the goal it started from, plus the
        # observation. The goal happens to name the tool and the path.
        _Entry(
            "d:1;step;react:turn",
            AskLLM(
                messages=[
                    _user_msg("use read_file on mod.py"),
                    _user_msg("Tool result: contents"),
                ],
                response_schema=dict,
            ),
        ),
    ]

    assert transcript_violations(tape).violations, (
        "a turn that never saw the action was scored clean"
    )


def test_a_correct_turn_with_a_non_ascii_argument_is_clean():
    """Too strong, on the same line of code — and this one reddens `just tape-check` against a
    working agent.

    `_assistant` renders arguments with `json.dumps`, which escapes non-ASCII (`café` becomes
    `caf\\u00e9`). The walk then asks whether `str(value)` — the unescaped `café` — appears in that
    text, and it does not. Every other JSON-vs-`str` divergence is here too (`True` vs `true`,
    `None` vs `null`, nested containers), but those need a non-string argument; this one fires with
    the shipped all-string tool catalog, on a goal like "handle café encoding".

    The module states the correct rule two functions down — *ASK the renderer, never re-implement
    it* — and applies it to the compaction summary only. This is the same rule reaching the
    function that was violating it."""
    acted = AssistantTurn(thought="search", tool=ToolRequest(name="probe", args={"q": "café"}))
    tape = [
        _Entry(
            "d:0;step;react:turn",
            AskLLM(messages=[_user_msg("handle café encoding")], response_schema=dict),
            acted,
        ),
        _Entry(
            "d:0;step;tool:probe",
            CallTool(name="probe", args={"q": "café"}, result_schema=ToolResult),
            ToolResult(content="found"),
        ),
        _Entry(
            "d:1;step;react:turn",
            AskLLM(
                messages=[
                    _user_msg("handle café encoding"),
                    _assistant(acted),  # the REAL rendering, byte for byte
                    _user_msg("Tool result: found"),
                ],
                response_schema=dict,
            ),
        ),
    ]

    transcript_violations(tape).require_clean(at_least=2)


def test_a_post_turn_interrupt_leaves_no_action_in_the_transcript():
    """An interrupt after the model chose but before the tool ran must not leave the action behind.

    Were `_assistant(turn)` appended BEFORE the post-turn poll, an interrupt that won would record
    `[interrupted]` while the transcript kept `Action: write_file({...})`: a call that never
    happened, which every later turn re-reads as history.

    **The tape walk structurally cannot catch this.** Its invariant is that a turn can SEE the
    action before it; here the transcript shows MORE than happened, not less. A property about
    sufficiency says nothing about excess, so the guard has to live at the only place that knows
    the action was dropped.

    Both halves asserted: the thought survives (the model should see what it was thinking when it
    was cut off), and the action does not."""
    responses = {
        "d:0;react:turn": AssistantTurn(
            thought="rewrite it", tool=ToolRequest(name="write_file", args={"path": "mod.py"})
        ),
        "d:1;react:turn": AssistantTurn(thought="stopped", answer="halted"),
    }

    def interrupt(i: int, when: Phase) -> Effect[Signal]:
        yield from ()
        return Redirect("user said stop") if (i, when) == (0, "post") else Quiet()

    rec = RecordingHandler(responses)
    traj = rec.run(lambda: run_agent("go", interrupt=interrupt, max_iters=2))

    asks = [getattr(e.op, "op", e.op) for e in rec.trace]
    asks = [op for op in asks if isinstance(op, AskLLM)]
    assert len(asks) == 2, [type(a).__name__ for a in asks]  # anti-vacuity: a turn really followed
    later = [str(m.get("content", "")) for m in asks[-1].messages]

    assert any("rewrite it" in c for c in later), later
    assert not any("write_file" in c for c in later), later
    assert isinstance(traj, Trajectory)
    assert traj.steps[0].observation == "[interrupted] user said stop"
