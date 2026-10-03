"""`run_agent`: a ReAct loop as a `descend` judge, one turn per level.

Every model turn is an `AskLLM` step named `react:turn` and every action a `CallTool` step named
`tool:{name}`, inside the level's `d:{i}` frame, so a recorded trajectory replays with no model.
Parsing a turn and choosing between an answer and an action is pure code between yields.

Three policy seams are parameters:

- `decide` takes the transcript and the `Level` and yields one `AssistantTurn`; a turn with a tool
  acts, a turn without one answers. At the final level the transcript ends in a nudge to answer.
- `act` runs a `ToolRequest`. The default is `call_tool`; a HITL `act` yields `await_event`, so the
  loop suspends durably.
- `compact` is a pure trigger over the transcript; when it fires the loop records a
  `TrajectorySummary` and splices it over the transcript's prefix.

Cost and cache accounting stay a handler concern over `AskLLM`.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache, partial
from string.templatelib import Template
from typing import Any, Literal, assert_never

from pydantic import BaseModel, TypeAdapter, ValidationError

from effective import Effect, Refused, ask_llm, await_event, step
from effective.api import direct_tool_key
from effective.cancel import Cancelled, OpCancelled
from effective.channels import directives, render
from effective.combinators import Answered, Deeper, Grantor, Level, descend
from effective.domain import ASK_TOOL, CallTool, ToolRefused
from effective.govern import routable
from effective.interrupts import Escape, Interrupt, Quiet, Redirect, Signal
from effective.keys import Key, Name, compose_key
from effective.ops import CompositionRefused


class ToolRequest(BaseModel):
    """One action the model wants to take."""

    name: str
    args: dict[str, Any] = {}


class AssistantTurn(BaseModel):
    """One model turn. A ``tool`` means *act*; no tool means *finish*.

    This is the ``AskLLM`` response schema: a real DomainInterpreter coerces the
    provider response into it (a typical loop reads ``tool_calls`` off the
    message); the RecordingHandler returns a canned one.
    """

    thought: str = ""
    tool: ToolRequest | None = None
    answer: str | None = None


class ToolResult(BaseModel):
    """The observation returned by a tool — fed back into the next prompt."""

    content: str


class Step(BaseModel):
    """One recorded iteration of the loop (the trajectory is a list of these)."""

    thought: str
    tool: ToolRequest | None = None
    observation: str | None = None


type StopReason = Literal["finish", "max_iters", "escaped"]


class Trajectory(BaseModel):
    answer: str
    steps: list[Step]
    stop_reason: StopReason


class TrajectorySummary(BaseModel):
    """The recorded schema of a compaction turn — a self-continuation summary that
    replaces the compacted transcript prefix. This is the parse contract (the data
    axis); a handler-side channel template + ``FormGate`` (a ``Gated``
    extractor guardrail, e.g. "every `tried` entry names a tool op in
    the transcript") is the guardrail follow-up, not needed for the recorded seam."""

    facts: list[str] = []  # established by tool evidence
    tried: list[str] = []  # approaches attempted, and how each failed
    open_questions: list[str] = []


class UncarriedDirective(CompositionRefused):
    """A template declared something the transcript has nowhere to put.

    A loop prompt's CACHE boundary: the cached prefix is the framing composed once above the loop,
    where a provider sees the same bytes every turn, and this template is the volatile tail. An
    observation's ROLE: an observation is one tool message whatever its template says. Dropping
    either would leave the author believing the declaration was placed, so each refusal names
    where it belongs instead."""


def _user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _messages(prompt: str | Template) -> list[dict[str, Any]]:
    """The context a run starts from.

    A `str` is one user message, as it always was. A `Template` goes through the channel
    processor, so its roles, its cache boundary and its data fences are that processor's
    decisions rather than this loop's, and the loop stays the control axis it is.

    Rendering here is workflow-side, and it gets no registry, so a `Skill` node is a located
    `SkillResolutionError` instead of a body read from whatever snapshot this worker happens to
    hold. A disclosure that must survive a park carries a pin, threaded through the template a
    caller composes.

    **`render` does no I/O of its own and is not thereby deterministic**: it calls `__format__` on
    the values it is handed, so a value carrying state renders differently on a replay and nothing
    compares the two. A review measured that on this path as well as on an observation's."""
    match prompt:
        case Template():
            if any(declared.cache is not None for declared in directives(prompt)):
                raise UncarriedDirective(
                    "a loop prompt declared a cache flag; compose the cached framing once above "
                    "the loop and leave this template the volatile tail"
                )
            rendered = render(prompt, output=str).messages
            return [{"role": message.role, "content": message.content} for message in rendered]
        case str():
            return [_user(prompt)]
        case unreachable:
            assert_never(unreachable)


def _observed(observation: str | Template) -> str:
    """One observation's text.

    An observation is a single tool message, so a template declaring a role is refused rather than
    flattened into it. The template is ASKED rather than its rendering inspected, because a
    rendering cannot tell an explicit `role=user` from a hole that declared nothing, and a review
    measured that one going through.

    **A hole's value must render the same bytes every time.** This runs workflow-side on a replay
    as well, where the recorded RESULT is re-served and rendered again, so a formatter carrying
    state changes the transcript and nothing compares the two renderings."""
    match observation:
        case Template() as template:
            if any(declared.role is not None for declared in directives(template)):
                raise UncarriedDirective(
                    "an observation is one tool message and carries no role; drop the directive"
                )
        case str():
            pass
        case unreachable:
            assert_never(unreachable)
    return "".join(part["content"] for part in _messages(observation))


def action_line(tool: ToolRequest) -> str:
    """The ONE rendering of a chosen action, so a reader can ask for it instead of rebuilding it.

    Public because `effective.tape` has to check that a turn could SEE the action before it, and
    the only honest way to check that is to ask what was written rather than to reconstruct it.
    The walk did reconstruct it — `name in some_message` and then `str(value) in that_message` —
    and was wrong in both directions at once: a goal that merely mentioned the tool satisfied it
    (an unanchored substring, and a one-character tool name makes it unfalsifiable), while a
    correct run with a non-ASCII argument FAILED it, because this side writes `json.dumps` and
    escapes `café` to `caf\u00e9` where the checker looked for the unescaped text.

    The module states the rule for the compaction summary — *ask the renderer, never re-implement
    it* — and this is that rule reaching the function that was breaking it. `sort_keys=True` is
    load-bearing for the same reason: two renderings of one action must be one string."""
    return f"Action: {tool.name}({json.dumps(tool.args, sort_keys=True)})"


def _assistant(turn: AssistantTurn) -> dict[str, Any]:
    """The assistant's turn as the model will re-read it — thought AND action."""
    if turn.tool is None:
        return {"role": "assistant", "content": turn.answer or turn.thought}
    return {"role": "assistant", "content": f"{turn.thought}\n{action_line(turn.tool)}"}


def _observation(result: ToolResult) -> dict[str, Any]:
    return {"role": "tool", "content": result.content}


_NUDGE = _user(
    "You have reached the step limit. Give your best final answer now, grounded "
    "in the tool results above. Do not call any tools."
)

KEEP_TAIL = 4  # verbatim recent turns kept after a compaction


def _transcript(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _summary_request(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The messages for a compaction turn: plain, because there is nothing here to decide.

    The transcript is already this loop's own rendering of its own turns, so it carries no hole a
    processor would fence and no role a template would declare. A summarization that grows either
    becomes a template like any other prompt."""
    return [
        _user(
            "Summarize this agent transcript for your own continuation. Preserve "
            "every tool result you would need to not repeat work.\n\n" + _transcript(messages)
        )
    ]


def summary_message(summary: TrajectorySummary) -> dict[str, Any]:
    """The summary as the model will re-read it, after the compacted prefix is gone.

    PUBLIC because it now has a second consumer: `effective.tape` checks that the turn after a
    compaction can actually see this message. That check must ASK for the rendering rather than
    spell it — a walk that re-implemented this would drift from it and then report confident false
    violations, which is worse than no check at all (the failure mode measured on the span
    sidecar)."""
    return _user(
        "[summary of earlier turns]\n"
        f"facts: {summary.facts}\n"
        f"tried: {summary.tried}\n"
        f"open_questions: {summary.open_questions}"
    )


# --- the policy seams ----------------------------------------------------------

type Decide = Callable[[list[dict[str, Any]], Level], Effect[AssistantTurn]]
type Act = Callable[[ToolRequest, Level], Effect[ToolResult]]
type Compact = Callable[[list[dict[str, Any]]], bool]  # PURE trigger over workflow state


@dataclass(frozen=True, slots=True)
class Proceed:
    """The action runs."""


@dataclass(frozen=True, slots=True)
class Refuse:
    """The action does not run, and `reason` is the observation the model reads."""

    reason: str


@dataclass(frozen=True, slots=True)
class Ask:
    """The user decides: `question` reaches them through the `ask` tool, and the run parks."""

    question: str


type Verdict = Proceed | Refuse | Ask
type Guard = Callable[[ToolRequest, list[dict[str, Any]], Level], Effect[Verdict]]
"""Judges an action before it runs, over the transcript that chose it. It may yield ops."""


class Allowed(BaseModel):
    """The user's answer to an `Ask`, delivered to the turn's `guard:ask` park."""

    allow: bool


type ReactStep = Literal["turn", "compact", "final"]


def react_key(kind: ReactStep) -> Key:
    """The name of a loop's own model call. The `d:{i}` frame around it says which turn."""
    return compose_key(t"react:{Name(kind)}")


def refusal(tool: str, diagnostic: str) -> ToolResult:
    """A refusal as the model reads it, from the ONE place that renders one.

    The diagnostic is a data hole, because a tool composes it from what it was handed: a path the
    model chose, a filename the project carries. A review measured a newline in a filename forging
    an `Assistant:` line through this path, which never touches `Tool.observe`.

    **`tool` is the name this loop KEYED, never the one a refusal echoed back.** `tool_key` fences
    the name an action is dispatched under, and `ToolRefused.tool` is handler-returned data that
    nothing ties to it: a refusal whose `tool` carried a newline would forge a transcript line
    through an unfenced field."""
    return ToolResult(content=_observed(t"[refused] {tool}: {diagnostic:data}"))


def default_decide(messages: list[dict[str, Any]], level: Level) -> Effect[AssistantTurn]:
    """One reasoning step: `react:turn`, or `react:final` at the final level.

    It takes no scope: a nested agent is `scoped(...)` around its `run_agent` call."""
    key = react_key("final" if level.final else "turn")
    decided = yield from ask_llm(key.stored(), messages, AssistantTurn)
    return decided


class UnusableToolName(CompositionRefused):
    """The model named a tool whose name cannot BE a key — so the action has no identity.

    A `CompositionRefused` (`effective.ops`), so a ReAct loop running as a spawned child fails of
    it on the attempt that raised it and its parent hears `Failed`, where a crash would spend every
    attempt first.

    **The guarantee was working; the refusal had nowhere to go.** `compose_key` is *supposed* to
    refuse a name it cannot make an atom of — that is the whole point of composing a
    model-controlled value instead of f-stringing it. What was missing is that it raised
    `KeySyntaxError`, a bare `ValueError` the loop's `except Refused` arm does not catch, so a
    tool named ``7zip`` took the run down instead of becoming an observation to route around."""


def tool_key(name: str) -> Key:
    """The op key for a model-chosen tool, `tool:{name}`, which the loop's `d:{i}` frame places.

    `name` is model-controlled, and the atom rule refuses more than a delimiter: a leading digit
    (`7zip`), a space (`rm -rf`), punctuation outside `[A-Za-z0-9_.@-]`. The refusal becomes
    `UnusableToolName`, which the loop routes around as an observation."""
    try:
        return direct_tool_key(name)
    except ValueError as refused:  # `Segment`'s delimiter fence, or the atom rule below it
        raise UnusableToolName(
            f"the tool name {name!r} cannot be composed into an op key, so this action has no "
            f"durable identity and cannot be checkpointed: {refused}"
        ) from refused


def default_act(request: ToolRequest, level: Level) -> Effect[ToolResult]:
    """An action named `tool:{name}` and dispatched by the bare name.

    A scope arrives as a leading term, never a splice into this one: `request.name` is
    model-controlled, and `tool_key` refuses a name carrying a delimiter, so no tool name can
    compose `sub;tool:read_file`.

    **A refusal is a result here as it is for a typed tool.** A deployment asked for something it
    cannot do answers `ToolRefused`; declaring `ToolResult` alone would leave the refusal unable to
    load from its own checkpoint, taking the run down. A model naming a tool the deployment does
    not serve reaches this path."""
    schema: Any = ToolRefused | ToolResult
    result: ToolRefused | ToolResult = yield from step(
        tool_key(request.name).stored(),
        CallTool(name=request.name, result_schema=schema, args=request.args),
    )
    match result:
        case ToolRefused(diagnostic=diagnostic):
            return refusal(request.name, diagnostic)
        case ToolResult():
            return result
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class Tool[A, R]:
    """One tool at the act seam, with its name, its ARGUMENT shape and its RESULT type bound
    together in one place.

    **Two consumers want different things from one call.** `default_act` names every action's
    result `ToolResult` (one field, `content: str`) because that is what the transcript needs.
    The WORKFLOW needs whatever the tool actually
    produced: `coding.runners._write_file` returns the resulting tree precisely so the workflow
    can learn about a handler-side edit through a recorded op result, which is the only kind of
    state a replay can reconstruct. One untyped `CallTool` cannot serve both, and weakening
    either end fails: a tree squeezed into `content` would be spliced into every subsequent
    prompt, and a `write_file` that returned a message would break replay-derivability.

    So the two readings are separated rather than reconciled: `result` is what the OP carries and
    the workflow folds, `observe` is what the TRANSCRIPT sees. Both are declared once, beside the
    tool's name, instead of being restated at each construction site as a bare string no checker
    binds to a definition.

    **`args` is validated.** A field written and never read is a recurring defect, so binding the
    argument shape has to buy something. It buys the call-side half
    of a crash: a model naming `write_file` with no `path` reaches `KeyError` inside the runner,
    which is a handler-side exception and therefore kills the task rather than becoming a turn's
    observation. Checked here, it is an observation the loop routes around. The tool-side half, a
    tool that ran and refused, is a different mechanism.

    **No `key` property, and not because it was forgotten.** `step_key` wraps every `Step` in its
    own arm, so a `call_tool`'s letter is `step;tool:{name}`; and the substrate decouples the
    dispatch name from the durable key on purpose (`activate_skill` keys under `skill:{name}
    ,activate` while its `CallTool` name is `skill-disclose`). A name-derived key would match
    neither. `docs/repertoire-axis.md` §6b holds that question and declares it unowned."""

    name: str
    args: type[A]
    result: type[R]
    observe: Callable[[R], str | Template]


@dataclass(slots=True)
class ToolLog:
    """What the typed tools RETURNED, in order — before `observe` rendered them for the prompt.

    The placement is the mechanism. The loop keeps only `Step.observation: str | None`, so a typed
    result is destroyed at the loop boundary and a reader handed a finished `Trajectory` could only
    parse a rendered string back into a tree — the immediate-flattening antipattern, arriving
    through the back door. The fold therefore happens where the typed value still exists, which is
    inside `act`.

    **Replay-safe by construction, not by discipline.** Every entry here came from a `step(...)`
    result, so a resume re-serves it from the checkpoint and a replay re-derives the same log with
    no tool running. That is exactly the rule `Evidence.tree` states, and the reason the machine's
    workspace may not be a shared object a handler-side tool mutates.

    Per WORKER INVOCATION, never per factory. A `Worker` closure is re-entered on every visit, so
    a log built once and closed over would carry visit 0's edits into visit 3's evidence."""

    entries: list[tuple[Tool[Any, Any], Any]] = field(default_factory=list)

    def last[T](self, schema: type[T]) -> T | None:
        """The most recent result a tool DECLARED as `schema` — by the declaration, never by
        sniffing the value.

        Sniffing would be the nominal-where-structural move one grain down: two tools can return
        the same runtime shape for different reasons, and a fold that guessed from `isinstance`
        would pick whichever ran last rather than whichever the caller meant.

        **A `ToolRefused` is skipped, and the declaration is exactly why it has to be.** A refused
        `write_file` is still an entry whose TOOL declares `dict[str, str]`, so matching on the
        declaration alone would hand a refusal back as the workspace — and the trampoline commits
        whatever `Evidence.tree` holds. The entry stays in the log so `names` still says the tool
        ran; what it did not produce is a value to fold."""
        for tool, value in reversed(self.entries):
            # lint: totality(filter): a refused entry is skipped; the fold takes the last value.
            if tool.result == schema and not isinstance(value, ToolRefused):
                return value
        return None

    @property
    def names(self) -> tuple[str, ...]:
        """Which tools ran, in order — for a reader that wants to say what a state did."""
        return tuple(tool.name for tool, _ in self.entries)


def bad_arguments(tool: str, malformed: ValidationError) -> ToolResult:
    """The observation for a call whose arguments do not fit its tool; the next turn reads it.

    The validator's message quotes the input it rejected, which the model wrote, so it arrives as
    data. `tool` is the declared name the call was looked up by, as in `refusal`."""
    message = str(malformed)
    return ToolResult(content=_observed(t"[bad arguments for {tool}] {message:data}"))


@cache
def _args_adapter(schema: type) -> TypeAdapter[Any]:
    return TypeAdapter(schema)


def typed_act(
    tools: Mapping[str, Tool[Any, Any]],
    log: ToolLog,
    *,
    fallback: Act | None = None,
    bind: Callable[[], Mapping[str, Any]] | None = None,
) -> Act:
    """An `act` that yields each tool's OWN result schema and records what came back.

    The op is the same one `default_act` yields — `tool:{name}`, same key, same `CallTool` — with
    `result_schema` taken from the declaration instead of pinned to `ToolResult`. So the recorded
    op stream is byte-identical for any tool that already declared `ToolResult`, and a recorded
    trajectory replays unchanged.

    **An undeclared name falls through to `fallback`**, which keeps the loop's route-around for
    model-invented names exactly as it was: `UnusableToolName` still becomes an observation, an
    unserved name still reaches the domain and refuses there. Constraining what a model MAY name
    is a decode-side question and belongs to whoever builds the `Decide`, not here.

    **`bind` supplies the arguments the workflow owns**, merged over the model's after they are
    validated, so a model cannot supply or override them. It is called per action: a workspace
    bound this way is the one the recorded results so far describe, and a replay derives the same
    one."""
    dispatch = fallback if fallback is not None else default_act

    def act(request: ToolRequest, level: Level) -> Effect[ToolResult]:
        tool = tools.get(request.name)
        if tool is None:
            return (yield from dispatch(request, level))
        try:
            _args_adapter(tool.args).validate_python(request.args)
        except ValidationError as malformed:
            # The loop's own stance, applied one step earlier: a call it cannot make becomes an
            # observation to route around rather than a `KeyError` inside the runner, which is
            # handler-side and would take the task down.
            return bad_arguments(tool.name, malformed)
        # `ToolRefused | R`, composed HERE rather than declared on the `Tool`, for two reasons.
        # A tool's declaration should say what it produces, not restate that anything can fail;
        # and `ty` refuses a `types.UnionType` where `call_tool`'s `type[T]` is expected, so a
        # union in a caller's hands does not type-check while one in the substrate's does.
        # Additive on the wire: a successful call records the same bytes it always did.
        schema: Any = ToolRefused | tool.result
        value = yield from step(
            # Composed, not f-stringed, and `tool_key` carries the reason. `tool.name` rather
            # than `request.name`: they are equal by the lookup above, and taking it from the
            # DECLARATION means the key and the dispatch cannot name two different tools.
            tool_key(tool.name).stored(),
            CallTool(
                name=tool.name,
                result_schema=schema,
                args=request.args if bind is None else {**request.args, **bind()},
            ),
        )
        log.entries.append((tool, value))
        if isinstance(value, ToolRefused):
            # The loop's existing stance, arriving as a value instead of an exception: a refusal
            # becomes the next observation, so the turn can try a different edit. Rendered here
            # rather than by `observe`, which is typed on what the tool PRODUCES.
            return refusal(tool.name, value.diagnostic)
        return ToolResult(content=_observed(tool.observe(value)))

    return act


@dataclass(frozen=True, slots=True)
class _Loop:
    """What one turn hands the next: the transcript the model reads and the steps so far."""

    messages: list[dict[str, Any]]
    steps: list[Step]


@dataclass(frozen=True, slots=True)
class Ran:
    """A finished run: its trajectory, and the transcript a later run continues from."""

    trajectory: Trajectory
    messages: list[dict[str, Any]]


ESCAPED = "[escaped]"
"""The marker a transcript carries where the user escaped a turn."""


def _escaped(messages: list[dict[str, Any]], steps: list[Step], thought: str) -> Answered[Ran]:
    """The turn the user abandoned. What already ran stays in the transcript, and the thought that
    chose an action the loop did not run is recorded without the action."""
    step_ = Step(thought=thought, observation=ESCAPED)
    trajectory = Trajectory(answer="", steps=[*steps, step_], stop_reason="escaped")
    return Answered(Ran(trajectory, [*messages, _user(ESCAPED)]))


type _Ended = Answered[Ran] | Deeper[_Loop]


def _before_decide(
    signal: Signal, messages: list[dict[str, Any]], steps: list[Step]
) -> _Ended | None:
    """A pending redirect wins before the model is called; an escape ends the run."""
    match signal:
        case Quiet():
            return None
        case Redirect(text):
            return Deeper(
                _Loop(
                    [*messages, _user(f"[interrupt] {text}")],
                    [*steps, Step(thought="", observation=f"[interrupted] {text}")],
                )
            )
        case Escape():
            return _escaped(messages, steps, "")
        case unreachable:
            assert_never(unreachable)


def _after_decide(
    signal: Signal, messages: list[dict[str, Any]], steps: list[Step], turn: AssistantTurn
) -> _Ended | None:
    """A signal that landed during the decide drops its action.

    The thought goes into the transcript without `action_line`: the action did not run, and a
    later turn reads the transcript as history."""
    thought = {"role": "assistant", "content": turn.thought}
    match signal:
        case Quiet():
            return None
        case Redirect(text):
            return Deeper(
                _Loop(
                    [*messages, thought, _user(f"[interrupt] {text}")],
                    [*steps, Step(thought=turn.thought, observation=f"[interrupted] {text}")],
                )
            )
        case Escape():
            return _escaped([*messages, thought], steps, turn.thought)
        case unreachable:
            assert_never(unreachable)


def _after_act(signal: Signal, messages: list[dict[str, Any]], steps: list[Step]) -> _Ended:
    """The observation stands whatever arrived while the action ran."""
    match signal:
        case Quiet():
            return Deeper(_Loop(messages, steps))
        case Redirect(text):
            return Deeper(_Loop([*messages, _user(f"[interrupt] {text}")], steps))
        case Escape():
            trajectory = Trajectory(answer="", steps=steps, stop_reason="escaped")
            return Answered(Ran(trajectory, [*messages, _user(ESCAPED)]))
        case unreachable:
            assert_never(unreachable)


def _decided(
    decide: Decide, compact: Compact | None, messages: list[dict[str, Any]], level: Level
) -> Effect[tuple[list[dict[str, Any]], AssistantTurn] | Cancelled]:
    """The decide, over the transcript compacted first when `compact` fires, with the transcript
    it read. Either model call cancelled while it ran is a `Cancelled`."""
    try:
        if compact is not None and compact(messages):
            messages = yield from _compacted(messages)
        return messages, (yield from decide(messages, level))
    except OpCancelled as cancelled:
        return Cancelled(partial=cancelled.partial)


def _final(
    decide: Decide, messages: list[dict[str, Any]], steps: list[Step], level: Level
) -> Effect[_Ended]:
    """The last level's turn, which must answer: its decide reads the nudge."""
    match (yield from _decided(decide, None, [*messages, _NUDGE], level)):
        case Cancelled() as cancelled:
            return _escaped(messages, steps, cancelled.partial)
        case (_, final):
            answer = final.answer or final.thought
            closing = [*messages, {"role": "assistant", "content": answer}]
            return _answered(closing, steps, final.thought, answer, "max_iters")
        case unreachable:
            assert_never(unreachable)


def _compacted(messages: list[dict[str, Any]]) -> Effect[list[dict[str, Any]]]:
    """The transcript with its prefix replaced by a recorded summary, keeping the task and the last
    `KEEP_TAIL` messages."""
    summary = yield from ask_llm(
        react_key("compact").stored(), _summary_request(messages), TrajectorySummary
    )
    return [messages[0], summary_message(summary), *messages[1:][-KEEP_TAIL:]]


def _guarded(
    guard: Guard, request: ToolRequest, messages: list[dict[str, Any]], level: Level
) -> Effect[ToolResult | None]:
    """`None` when the action may run, else the refusal that stands in for it."""
    verdict: Verdict = yield from guard(request, messages, level)
    match verdict:
        case Proceed():
            return None
        case Refuse(reason=reason):
            return refusal(request.name, reason)
        case Ask(question=question):
            asked = CallTool(name=ASK_TOOL, args={"question": question}, result_schema=ToolResult)
            yield from step(compose_key(t"guard:asked").stored(), asked)
            answer = yield from await_event(compose_key(t"guard:ask"), Allowed)
            return None if answer.allow else refusal(request.name, "the user declined")
        case unreachable:
            assert_never(unreachable)


def _acted(act: Act, request: ToolRequest, level: Level) -> Effect[ToolResult | Cancelled]:
    """The action's result, with a refusal as an observation the model can route around."""
    try:
        return (yield from act(request, level))
    except OpCancelled as cancelled:
        return Cancelled(partial=cancelled.partial)
    except Refused as refusal:
        return ToolResult(content=f"[denied] {routable(refusal).reason}")
    except UnusableToolName as unusable:
        # its base `CompositionRefused` also names programming errors, which propagate
        return ToolResult(content=f"[unusable tool name] {unusable}")


def _act(
    act: Act,
    guard: Guard | None,
    interrupt: Interrupt | None,
    turn: AssistantTurn,
    request: ToolRequest,
    messages: list[dict[str, Any]],
    steps: list[Step],
    level: Level,
) -> Effect[_Ended]:
    """Run the chosen action unless the guard refuses it, then poll; an action cancelled while it
    ran ends the run."""
    refused = None if guard is None else (yield from _guarded(guard, request, messages, level))
    outcome: ToolResult | Cancelled = (
        refused if refused is not None else (yield from _acted(act, request, level))
    )
    match outcome:
        case Cancelled() as cancelled:  # the output so far is what the model sees ran
            result, signal = ToolResult(content=f"{cancelled.partial}\n{ESCAPED}"), Escape()
        case ToolResult() as result:
            signal = Quiet() if interrupt is None else (yield from interrupt(level.depth, "act"))
        case unreachable:
            assert_never(unreachable)
    messages = [*messages, _observation(result)]
    steps = [*steps, Step(thought=turn.thought, tool=request, observation=result.content)]
    return _after_act(signal, messages, steps)


def _answered(
    messages: list[dict[str, Any]], steps: list[Step], thought: str, answer: str, why: StopReason
) -> Answered[Ran]:
    trajectory = Trajectory(answer=answer, steps=[*steps, Step(thought=thought)], stop_reason=why)
    return Answered(Ran(trajectory, messages))


def _turn(
    decide: Decide,
    act: Act,
    guard: Guard | None,
    interrupt: Interrupt | None,
    compact: Compact | None,
    loop: _Loop,
    level: Level,
) -> Effect[_Ended]:
    """One turn: an answer or an escape ends the loop; an action or a redirect goes a level deeper.

    The final level polls no interrupt and compacts nothing; its decide reads the nudge."""
    messages, steps = loop.messages, loop.steps
    if level.final:
        return (yield from _final(decide, messages, steps, level))

    if interrupt is not None:
        signal = yield from interrupt(level.depth, "pre")
        if (ended := _before_decide(signal, messages, steps)) is not None:
            return ended

    match (yield from _decided(decide, compact, messages, level)):
        case Cancelled() as cancelled:
            return _escaped(messages, steps, cancelled.partial)
        case (compacted, turn):
            messages = compacted
        case unreachable:
            assert_never(unreachable)

    if interrupt is not None:
        signal = yield from interrupt(level.depth, "post")
        if (ended := _after_decide(signal, messages, steps, turn)) is not None:
            return ended

    messages = [*messages, _assistant(turn)]
    if turn.tool is None:
        answer = turn.answer if turn.answer is not None else turn.thought
        return _answered(messages, steps, turn.thought, answer, "finish")

    return (yield from _act(act, guard, interrupt, turn, turn.tool, messages, steps, level))


def run_turns(
    messages: list[dict[str, Any]],
    max_iters: int = 6,
    *,
    decide: Decide | None = None,
    act: Act | None = None,
    guard: Guard | None = None,
    interrupt: Interrupt | None = None,
    compact: Compact | None = None,
    grantor: Grantor | None = None,
) -> Effect[Ran]:
    """`run_agent` from a transcript, handing the transcript back with the trajectory."""
    turn = partial(
        _turn,
        decide if decide is not None else default_decide,
        act if act is not None else default_act,
        guard,
        interrupt,
        compact,
    )
    return (yield from descend(_Loop(messages, []), turn, budget=max_iters, grantor=grantor))


def run_agent(
    prompt: str | Template,
    max_iters: int = 6,
    *,
    decide: Decide | None = None,
    act: Act | None = None,
    guard: Guard | None = None,
    interrupt: Interrupt | None = None,
    compact: Compact | None = None,
    grantor: Grantor | None = None,
) -> Effect[Trajectory]:
    """Drive the turns to an answer: `max_iters` turns, then one final turn that must answer.

    Each turn is a `descend` level under `d:{i}`; a `grantor` adds turns before the final one. A
    nested agent is `scoped(compose_key(t"sub:{name}"), lambda: run_agent(...))`. A turn is a
    scoped body, so a refusal raised in it reaches this call's caller and any other exception ends
    the run.

    `interrupt` is polled before the decide, after it, and after the action, each a recorded op
    when the poll subscribes to that phase (`effective.interrupts`). A redirect becomes the next
    observation, and one that lands before the action runs drops it. An escape ends the run with
    `stop_reason="escaped"`, dropping an action the model chose but the loop had not run.

    `guard` judges each action before it runs, over the transcript that chose it: a `Refuse` is
    the observation in place of the action, and an `Ask` puts its question to the user and parks
    on `guard:ask` for their `Allowed`.

    A `Template` prompt renders through `effective.channels`, so the run starts from the roles
    that template declares; a `str` is one user message, which is what every caller passed until
    the data axis grew a processor to hand this to.

    `compact` fires on the transcript before a decide and records one `react:compact` step whose
    summary replaces the transcript's prefix, keeping the task and the last `KEEP_TAIL` messages.
    The splice is loop logic because the transcript is workflow state between yields, where no
    handler layer reaches."""
    ran = yield from run_turns(
        _messages(prompt),
        max_iters,
        decide=decide,
        act=act,
        guard=guard,
        interrupt=interrupt,
        compact=compact,
        grantor=grantor,
    )
    return ran.trajectory
