"""`run_agent`: a ReAct loop as a `descend` judge, one turn per level.

Every model turn is an `AskLLM` step named `react:turn` and every action a `CallTool` step named
`tool:{name}`, inside the level's `d:{i}` frame, so a recorded trajectory replays with no model.
Parsing a turn and choosing between an answer and an action is pure code between yields.

Three policy seams are parameters:

- `decide` takes a `TurnRequest` (the transcript, the `Level` and a typed run's answer schema)
  and yields one `AssistantTurn`; a turn with a tool acts, a turn without one answers. At the
  final level the transcript ends in a nudge to answer.
- `act` runs a `ToolRequest`. The default is `call_tool`; a HITL `act` yields `await_event`, so the
  loop suspends durably.
- `compact` is a pure trigger over the transcript; when it fires the loop records a
  `TrajectorySummary` and splices it over the transcript's prefix.

Cost and cache accounting stay a handler concern over `AskLLM`.
"""

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache, partial, reduce
from inspect import Parameter, signature
from operator import or_
from string.templatelib import Template
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    TypeAliasType,
    Union,
    assert_never,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError

from effective import Effect, Refused, ask_llm, await_event, step
from effective.api import direct_tool_key
from effective.cancel import OpCancelled
from effective.channels import Prompt, Repair, answer_model, answering, directives, render
from effective.combinators import Answered, Deeper, Grantor, Level, descend
from effective.domain import ASK_TOOL, AskLLM, CallTool, ToolRefused
from effective.govern import ChildRefused, routable
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
    answer: JsonValue | None = None


class ToolResult(BaseModel):
    """The observation returned by a tool — fed back into the next prompt."""

    content: str


class Step(BaseModel):
    """One recorded iteration of the loop (the trajectory is a list of these). `repair` is why a
    typed run refused the answer this turn gave."""

    thought: str
    tool: ToolRequest | None = None
    observation: str | None = None
    repair: str | None = None


type StopReason = Literal["finish", "max_iters"]


class Trajectory[T = str](BaseModel):
    """A run that answered."""

    kind: Literal["answered"] = "answered"
    answer: T
    steps: list[Step]
    stop_reason: StopReason


type EscapeOrigin = Literal["escape", "decide", "action", "compaction"]
"""What ended a run without an answer, and what its `partial` holds.

| origin       | what happened                               | `partial`                    |
|--------------|---------------------------------------------|------------------------------|
| `escape`     | a poll reported the user's Esc              | the dropped action's thought |
| `decide`     | a decide was cancelled while it ran         | the model's output so far    |
| `action`     | an action was cancelled while it ran        | the tool's output so far     |
| `compaction` | a compaction was cancelled while it ran     | the summary so far           |
"""


class Escaped(BaseModel):
    """A run that ended without an answer. `name` is the name the op or poll that reported it was
    raised under: an op's authored name, a joined child's done event, or `None` for a poll that
    records nothing. It addresses the report, and the checkpoint stays readable from the tape by
    position."""

    kind: Literal["escaped"] = "escaped"
    origin: EscapeOrigin
    name: str | None
    partial: str
    steps: list[Step]


class Exhausted(BaseModel):
    """A typed run whose answers were refused until its repairs ran out: `repair` is the last
    refusal and `repairs` the budget the run had."""

    kind: Literal["exhausted"] = "exhausted"
    repair: Repair
    repairs: int
    steps: list[Step]


type Ending[M] = Trajectory[M] | Escaped | Exhausted


_ENDING: TypeAdapter[Ending[Any]] = TypeAdapter(
    Annotated[Trajectory[Any] | Escaped | Exhausted, Field(discriminator="kind")]
)


@cache
def _typed_ending(answer: type[Any]) -> TypeAdapter[Any]:
    return TypeAdapter(
        # ty: ignore[invalid-type-form]  -- the run's answer model is a value known at run time
        Annotated[Trajectory[answer] | Escaped | Exhausted, Field(discriminator="kind")]
    )


@overload
def load_ending(raw: Any) -> Ending[Any]: ...
@overload
def load_ending[M](raw: Any, *, answer: type[M]) -> Ending[M]: ...
def load_ending(raw: Any, *, answer: type[Any] | None = None) -> Ending[Any]:
    """A run's ending as a handler hands it back: the value itself, or its stored form. A typed
    run's answer validates as `answer`; without it, an answer is any JSON value."""
    adapter = _ENDING if answer is None else _typed_ending(answer)
    return adapter.validate_python(raw)


LOOP_ESCAPE = "react:escape"
"""The name an escape carries out of `unwrap` when its poll recorded nothing."""


class RepairsExhausted(ChildRefused):
    """A typed run's answers were refused until its repairs ran out. A refusal, so it climbs a
    spawn tree; `reason` is the last answer's refusal and `repairs` the budget."""

    def __init__(self, reason: str, repairs: int) -> None:
        super().__init__(reason, repairs)
        self.reason = reason
        self.repairs = repairs

    def __str__(self) -> str:
        return f"the answer was refused after {self.repairs} repairs: {self.reason}"


def unwrap[T](ending: Trajectory[T] | Escaped | Exhausted) -> T:
    """The answer; `OpCancelled` with the escape's partial text; or `RepairsExhausted` with the
    last refusal: the counterpart of `join_answer` for a run in this task."""
    match ending:
        case Trajectory(answer=answer):
            return answer
        case Escaped(name=name, partial=partial):
            raise OpCancelled(name if name is not None else LOOP_ESCAPE, partial)
        case Exhausted(repair=Repair(reason=reason), repairs=repairs):
            raise RepairsExhausted(reason, repairs)
        case unreachable:
            assert_never(unreachable)


class TrajectorySummary(BaseModel):
    """The recorded schema of a compaction turn — a self-continuation summary that
    replaces the compacted transcript prefix. This is the parse contract (the data
    axis); a handler-side channel template + ``FormGate`` (a ``Gated``
    extractor guardrail, e.g. "every `tried` entry names a tool op in
    the transcript") is not needed for the recorded seam."""

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


def answer_text(answer: JsonValue) -> str:
    """An answer as transcript text: a string as itself, any other JSON value compact."""
    # lint: totality(coercion): a JSON value is text or is encoded as text.
    match answer:
        case str():
            return answer
        case _:
            return json.dumps(answer, separators=(",", ":"), sort_keys=True)


def _said(turn: AssistantTurn) -> str:
    """The answer a turn gave, or its thought when it gave none."""
    return turn.thought if turn.answer is None or turn.answer == "" else answer_text(turn.answer)


def _assistant(turn: AssistantTurn) -> dict[str, Any]:
    """The assistant's turn as the model will re-read it — thought AND action."""
    if turn.tool is None:
        return {"role": "assistant", "content": _said(turn)}
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


@dataclass(frozen=True, slots=True)
class TurnRequest:
    """What a decide is asked for one turn: the transcript, the level, and the JSON schema of the
    answer the run expects, or `None` for a run that answers in text."""

    messages: list[dict[str, Any]]
    level: Level
    answer_schema: dict[str, Any] | None = None


type Decide = Callable[[TurnRequest], Effect[AssistantTurn]]


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


GUARD_ASKED = compose_key(t"guard:asked")
"""The name of the step that puts a guard's question to the user."""


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


def escape_observation(ending: Escaped) -> ToolResult:
    """An escaped run as the model that delegated it reads it. The partial text is a data hole,
    since a model or a tool wrote it."""
    return ToolResult(content=_observed(t"[escaped: {ending.origin}] {ending.partial:data}"))


def exhausted_observation(ending: Exhausted) -> ToolResult:
    """An exhausted run as the model that delegated it reads it. The reason is a data hole, since
    a validator quoting the model's answer wrote it."""
    return ToolResult(content=_observed(t"[exhausted: repairs] {ending.repair.reason:data}"))


def default_decide(request: TurnRequest) -> Effect[AssistantTurn]:
    """One reasoning step: `react:turn`, or `react:final` at the final level.

    It takes no scope: a nested agent is `scoped(...)` around its `run_agent` call."""
    key = react_key("final" if request.level.final else "turn")
    asked = AskLLM(
        messages=request.messages,
        response_schema=AssistantTurn,
        answer_schema=request.answer_schema,
    )
    decided = yield from step(key, asked)
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
    tool_key(request.name)  # an unusable name is refused here, as `UnusableToolName`
    result: ToolRefused | ToolResult = yield from step(
        request.name,
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

    **No `key` property.** `step_key` wraps every `Step` in its own arm, so a `call_tool`'s letter
    is `step;tool:{name}`, and the dispatch name and the durable key are separate on purpose
    (`activate_skill` keys under `skill:{name},activate` while its `CallTool` name is
    `skill-disclose`). A key derived from the name would match neither."""

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

    def last_from(self, tool: Tool[Any, Any]) -> Any | None:
        """The most recent result `tool` itself returned, skipping a refusal as `last` does."""
        for logged, value in reversed(self.entries):
            # lint: totality(filter): a refused entry is skipped; the fold takes the last value.
            if logged is tool and not isinstance(value, ToolRefused):
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
        # `tool.name` rather than `request.name`: they are equal by the lookup above, and taking
        # it from the DECLARATION means the key and the dispatch cannot name two different tools.
        tool_key(tool.name)  # an unusable name is refused here, as `UnusableToolName`
        value = yield from step(
            tool.name,
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


class DuplicateToolName(CompositionRefused):
    """Two declared tools share a dispatch name, so a call by that name could reach either."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name

    def __str__(self) -> str:
        return f"two tools are declared under the name {self.name!r}, and a call reaches one"


def declared(tools: Iterable[Tool[Any, Any]]) -> dict[str, Tool[Any, Any]]:
    """The dispatch table for `tools`, each under its own name."""
    table: dict[str, Tool[Any, Any]] = {}
    for tool in tools:
        if tool.name in table:
            raise DuplicateToolName(tool.name)
        table[tool.name] = tool
    return table


@dataclass(slots=True)
class Toolbox:
    """What one run, or one visit of a machine state, dispatches actions through: the declared
    tools, the log of what they returned, the arguments the workflow binds into their calls, and
    the `fallback` an undeclared name goes to. A declaration says what a name means; which names a
    model may call is the `Decide`'s business."""

    tools: Mapping[str, Tool[Any, Any]]
    log: ToolLog = field(default_factory=ToolLog)
    bind: Callable[[], Mapping[str, Any]] | None = None
    fallback: Act | None = None

    def act(self) -> Act:
        return typed_act(self.tools, self.log, fallback=self.fallback, bind=self.bind)


type EvidenceFault = Literal[
    "unannotated",
    "unpassable",
    "unsupported",
    "unsupported-result",
    "unproduced",
    "ambiguous",
    "undeclared-tool",
    "other-result",
]

_EVIDENCE_FAULTS: dict[EvidenceFault, str] = {
    "unannotated": "has no annotation",
    "unpassable": "cannot be passed by name; take it as a positional-or-keyword parameter",
    "unsupported-result": "meets a declared tool whose result type evidence cannot bind",
    "unsupported": "is a form evidence cannot bind, such as a generic alias",
    "unproduced": "is the result of no declared tool",
    "ambiguous": "is the result of more than one declared tool; name one with Annotated[T, tool]",
    "undeclared-tool": "names a tool the run does not declare",
    "other-result": "names a tool whose result is another type",
}


class EvidenceUnbound(CompositionRefused):
    """An answer contract's parameter the run's declared tools cannot supply exactly once.
    `declared` holds each declared tool's name and its result type, rendered."""

    def __init__(
        self,
        contract: str,
        parameter: str,
        fault: EvidenceFault,
        annotation: str,
        declared: tuple[tuple[str, str], ...],
    ) -> None:
        super().__init__(contract, parameter, fault)
        self.contract = contract
        self.parameter = parameter
        self.fault: EvidenceFault = fault
        self.annotation = annotation
        self.declared = declared

    def __str__(self) -> str:
        return (
            f"{self.contract}'s parameter {self.parameter}: {self.annotation} "
            f"{_EVIDENCE_FAULTS[self.fault]}; "
            f"the run declares {', '.join(f'{n} -> {r}' for n, r in self.declared) or 'none'}"
        )


class _Unsupported(Exception):
    """An annotation form evidence cannot bind."""


def _plain(annotation: Any) -> Any:
    """`annotation` with every type alias replaced by what it names, at any depth."""
    if isinstance(annotation, TypeAliasType):
        return _plain(annotation.__value__)
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    if isinstance(origin, TypeAliasType):
        raise _Unsupported(annotation)
    args = tuple(_plain(arg) for arg in get_args(annotation))
    if origin in (Union, UnionType):
        return reduce(or_, args)
    try:
        return origin[args]
    except TypeError as unsupported:
        raise _Unsupported(annotation) from unsupported


@dataclass(frozen=True, slots=True)
class _Evidence:
    """A contract parameter, the declared tool whose last result it takes, and what it receives
    before that tool has answered: its `default`, else `None` when it is `optional`."""

    parameter: str
    annotation: Any
    tool: Tool[Any, Any]
    optional: bool
    default: Any = Parameter.empty


def _bound(
    contract: Callable[..., Prompt[Any]], tools: Mapping[str, Tool[Any, Any]]
) -> tuple[_Evidence, ...]:
    """Each parameter of `contract`, bound to the one declaration in `tools` whose result type it
    names. An `Annotated[T, tool]` parameter names its tool."""
    name = getattr(contract, "__qualname__", repr(contract))
    hints = get_type_hints(contract, include_extras=True)
    declared = tuple((tool.name, repr(tool.result)) for tool in tools.values())
    bound: list[_Evidence] = []
    for parameter, spec in signature(contract).parameters.items():
        if spec.kind not in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
            raise EvidenceUnbound(name, parameter, "unpassable", str(spec), declared)
        if (annotation := hints.get(parameter)) is None:
            raise EvidenceUnbound(name, parameter, "unannotated", "", declared)
        evidence = _bind(name, parameter, annotation, tools, declared)
        bound.append(replace(evidence, default=spec.default))
    return tuple(bound)


def _named(annotation: Any) -> tuple[Any, list[Tool[Any, Any]]]:
    """The type an `Annotated` parameter wraps, and the tools its metadata names."""
    if get_origin(annotation) is not Annotated:
        return annotation, []
    base, *metadata = get_args(annotation)
    named: list[Tool[Any, Any]] = []
    for marker in metadata:
        match marker:
            case Tool():
                named.append(marker)
            case _:
                pass
    return base, named


def _optional(wanted: Any) -> tuple[Any, bool]:
    """`wanted` without `None`, and whether it held `None`."""
    if get_origin(wanted) in (Union, UnionType) and type(None) in get_args(wanted):
        return reduce(or_, (arg for arg in get_args(wanted) if arg is not type(None))), True
    return wanted, False


def _result(
    contract: str, parameter: str, tool: Tool[Any, Any], declared: tuple[tuple[str, str], ...]
) -> Any:
    """`tool`'s result type with its aliases replaced, or the refusal of a form evidence cannot
    bind, met while binding `parameter`."""
    try:
        return _plain(tool.result)
    except _Unsupported:
        raise EvidenceUnbound(
            contract, parameter, "unsupported-result", repr(tool.result), declared
        ) from None


def _bind(
    contract: str,
    parameter: str,
    annotation: Any,
    tools: Mapping[str, Tool[Any, Any]],
    declared: tuple[tuple[str, str], ...],
) -> _Evidence:
    base, named = _named(annotation)
    try:
        wanted, optional = _optional(_plain(base))
    except _Unsupported:
        raise EvidenceUnbound(contract, parameter, "unsupported", repr(base), declared) from None

    def refused(fault: EvidenceFault) -> EvidenceUnbound:
        return EvidenceUnbound(contract, parameter, fault, repr(base), declared)

    def result(tool: Tool[Any, Any]) -> Any:
        return _result(contract, parameter, tool, declared)

    match named:
        case [tool]:
            if not any(tool is held for held in tools.values()):
                raise refused("undeclared-tool")
            if result(tool) != wanted:
                raise refused("other-result")
            return _Evidence(parameter, wanted, tool, optional)
        case []:
            pass
        case _:
            raise refused("ambiguous")
    match [tool for tool in tools.values() if result(tool) == wanted]:
        case [tool]:
            return _Evidence(parameter, wanted, tool, optional)
        case []:
            raise refused("unproduced")
        case _:
            raise refused("ambiguous")


TOOL_AND_ANSWER = "a turn calls a tool or answers, never both"
FINAL_TOOL = "the final turn answers and calls no tool"
NOT_JSON = "the answer is not a JSON object"


@dataclass(frozen=True, slots=True)
class _Answering:
    """A typed run's answer contract: the function, the model it declares, that model's JSON
    schema, the evidence its parameters bind, the log they read, and the repair budget.

    The contract and its gates must be deterministic over their arguments: an answer resolves on
    a replay from the recorded turn and the recorded tool results alone."""

    contract: Callable[..., Prompt[Any]]
    model: type[BaseModel]
    schema: dict[str, Any]
    evidence: tuple[_Evidence, ...]
    log: ToolLog
    repairs: int

    def values(self) -> dict[str, Any] | Repair:
        """The contract's arguments from the log, or the repair naming evidence not yet
        produced."""
        values: dict[str, Any] = {}
        for evidence in self.evidence:
            if (value := self.log.last_from(evidence.tool)) is None:
                if evidence.default is not Parameter.empty:
                    value = evidence.default
                elif not evidence.optional:
                    wanted = repr(evidence.annotation)
                    return Repair(
                        _observed(t"the answer needs {wanted}, which no tool has returned")
                    )
            values[evidence.parameter] = value
        return values

    def prompt(self, values: dict[str, Any]) -> Prompt[BaseModel]:
        return answering(self.contract, self.contract(**values), self.model)

    def resolved(self, turn: AssistantTurn) -> BaseModel | Repair:
        """The model `turn`'s answer resolves to, or the repair to ask for."""
        if turn.tool is not None:
            return Repair(TOOL_AND_ANSWER)
        raw = turn.answer
        # lint: totality(coercion): the fallback transport writes the answer's JSON in a string.
        match raw:
            case str():
                try:
                    raw = json.loads(raw)
                except ValueError:
                    return Repair(NOT_JSON)
            case _:
                pass
        match self.values():
            case Repair() as repair:
                return repair
            case dict() as values:
                return self.prompt(values).resolve(raw)
            case unreachable:
                assert_never(unreachable)

    def tail(self, repair: Repair | None) -> list[dict[str, Any]]:
        """The volatile message a decide reads last: the answer's fields, the rendered contract
        once its evidence binds, and the last refusal on a repair turn."""
        properties = self.schema.get("properties", {})
        fields = "\n".join(
            _observed(t"{name}: {json.dumps(spec, sort_keys=True)}")
            for name, spec in properties.items()
        )
        parts = [_observed(t"The answer has these fields:\n{fields}")]
        match self.values():
            case Repair():
                rendered = []
            case dict() as values:
                rendered = self.prompt(values).messages
            case unreachable:
                assert_never(unreachable)
        if rendered:
            if any(message.cache for message in rendered):
                raise UncarriedDirective(
                    "an answer contract declared a cache flag, and the loop places its "
                    "rendering in the volatile tail"
                )
            if any(message.role != "user" for message in rendered):
                raise UncarriedDirective(
                    "an answer contract declared a role, and the loop places its rendering in "
                    "one user message"
                )
            parts.extend(message.content for message in rendered)
        if repair is not None:
            parts.append(_observed(t"The last answer was refused: {repair.reason:data}"))
        return [_user("\n".join(parts))]


def _answering(
    contract: Callable[..., Prompt[Any]] | None, repairs: int, toolbox: Toolbox | None
) -> _Answering | None:
    if contract is None:
        return None
    model = answer_model(contract)
    tools = toolbox.tools if toolbox is not None else {}
    log = toolbox.log if toolbox is not None else ToolLog()
    evidence = _bound(contract, tools)
    return _Answering(contract, model, model.model_json_schema(), evidence, log, repairs)


@dataclass(frozen=True, slots=True)
class _Loop:
    """What one turn hands the next: the transcript the model reads, the steps so far, and the
    refusal a redirect carried past the repair turn it interrupted."""

    messages: list[dict[str, Any]]
    steps: list[Step]
    repair: Repair | None = None


@dataclass(frozen=True, slots=True)
class Ran:
    """A finished run: how it ended, and the transcript a later run continues from."""

    trajectory: Ending[Any]
    messages: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _Cut:
    """An op cancelled while it ran: what it was, its authored name, and its output so far."""

    origin: EscapeOrigin
    name: str
    partial: str


ESCAPED = "[escaped]"
"""The marker a transcript carries where the user escaped a turn."""


def _escaped(
    messages: list[dict[str, Any]],
    steps: list[Step],
    thought: str,
    origin: EscapeOrigin,
    name: str | None,
) -> Answered[Ran]:
    """The turn the run abandoned. What already ran stays in the transcript, and the thought that
    chose an action the loop did not run is recorded without the action."""
    step_ = Step(thought=thought, observation=ESCAPED)
    ending = Escaped(origin=origin, name=name, partial=thought, steps=[*steps, step_])
    return Answered(Ran(ending, [*messages, _user(ESCAPED)]))


def _cut(messages: list[dict[str, Any]], steps: list[Step], cut: _Cut) -> Answered[Ran]:
    return _escaped(messages, steps, cut.partial, cut.origin, cut.name)


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
        case Escape(name=name):
            return _escaped(messages, steps, "", "escape", name)
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
        case Escape(name=name):
            return _escaped([*messages, thought], steps, turn.thought, "escape", name)
        case unreachable:
            assert_never(unreachable)


def _after_act(signal: Signal, messages: list[dict[str, Any]], steps: list[Step]) -> _Ended:
    """The observation stands whatever arrived while the action ran."""
    match signal:
        case Quiet():
            return Deeper(_Loop(messages, steps))
        case Redirect(text):
            return Deeper(_Loop([*messages, _user(f"[interrupt] {text}")], steps))
        case Escape(name=name):
            ending = Escaped(origin="escape", name=name, partial="", steps=steps)
            return Answered(Ran(ending, [*messages, _user(ESCAPED)]))
        case unreachable:
            assert_never(unreachable)


def _decided(
    decide: Decide,
    compact: Compact | None,
    messages: list[dict[str, Any]],
    level: Level,
    answering: _Answering | None = None,
    repair: Repair | None = None,
) -> Effect[tuple[list[dict[str, Any]], AssistantTurn] | _Cut]:
    """The decide, over the transcript compacted first when `compact` fires, with the transcript
    it read. A typed run's decide reads the answer's tail after it, which the transcript does not
    keep. Either model call cancelled while it ran is a `_Cut`."""
    if compact is not None and compact(messages):
        try:
            messages = yield from _compacted(messages)
        except OpCancelled as cancelled:
            return _Cut("compaction", cancelled.name, cancelled.partial)
    if answering is None:
        request = TurnRequest(messages, level)
    else:
        request = TurnRequest([*messages, *answering.tail(repair)], level, answering.schema)
    try:
        return messages, (yield from decide(request))
    except OpCancelled as cancelled:
        return _Cut("decide", cancelled.name, cancelled.partial)


type _Again = Callable[[_Loop, Level, Repair], Effect[_Ended]]


def _final(
    decide: Decide,
    answering: _Answering | None,
    again: _Again,
    messages: list[dict[str, Any]],
    steps: list[Step],
    level: Level,
    repair: Repair | None,
) -> Effect[_Ended]:
    """The last level's turn, which must answer: its decide reads the nudge. A typed run's final
    turn calls no tool, and its answer may spend the repairs left."""
    match (yield from _decided(decide, None, [*messages, _NUDGE], level, answering, repair)):
        case _Cut() as cut:
            return _cut(messages, steps, cut)
        case (_, final):
            pass
        case unreachable:
            assert_never(unreachable)
    closing = [*messages, {"role": "assistant", "content": _said(final)}]
    if answering is None:
        return _answered(closing, steps, final.thought, _said(final), "max_iters")
    resolved = Repair(FINAL_TOOL) if final.tool is not None else answering.resolved(final)
    return (yield from _settled(again, answering, closing, steps, final, level, resolved))


def _settled(
    again: _Again,
    answering: _Answering,
    messages: list[dict[str, Any]],
    steps: list[Step],
    turn: AssistantTurn,
    level: Level,
    resolved: BaseModel | Repair,
) -> Effect[_Ended]:
    """A typed answer: the trajectory when it resolved, else another turn at this level while
    repairs are left, else the run exhausted."""
    why: StopReason = "max_iters" if level.final else "finish"
    match resolved:
        case Repair(reason=reason) as repair:
            steps = [*steps, Step(thought=turn.thought, repair=reason)]
            if sum(step.repair is not None for step in steps) > answering.repairs:
                ending = Exhausted(repair=repair, repairs=answering.repairs, steps=steps)
                return Answered(Ran(ending, messages))
            return (yield from again(_Loop(messages, steps), level, repair))
        case BaseModel() as answer:
            steps = [*steps, Step(thought=turn.thought)]
            trajectory = Trajectory[Any](answer=answer, steps=steps, stop_reason=why)
            return Answered(Ran(trajectory, messages))
        case unreachable:
            assert_never(unreachable)


def _compacted(messages: list[dict[str, Any]]) -> Effect[list[dict[str, Any]]]:
    """The transcript with its prefix replaced by a recorded summary, keeping the task and the last
    `KEEP_TAIL` messages."""
    summary = yield from ask_llm(
        react_key("compact"), _summary_request(messages), TrajectorySummary
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
            yield from step(GUARD_ASKED, asked)
            answer = yield from await_event(compose_key(t"guard:ask"), Allowed)
            return None if answer.allow else refusal(request.name, "the user declined")
        case unreachable:
            assert_never(unreachable)


def _acted(act: Act, request: ToolRequest, level: Level) -> Effect[ToolResult | _Cut]:
    """The action's result, with a refusal as an observation the model can route around."""
    try:
        return (yield from act(request, level))
    except OpCancelled as cancelled:
        return _Cut("action", cancelled.name, cancelled.partial)
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
    outcome: ToolResult | _Cut = (
        refused if refused is not None else (yield from _acted(act, request, level))
    )
    match outcome:
        case _Cut(partial=partial) as cut:  # the output so far is what the model sees ran
            result = ToolResult(content=f"{partial}\n{ESCAPED}")
            messages = [*messages, _observation(result)]
            steps = [*steps, Step(thought=turn.thought, tool=request, observation=result.content)]
            ending = Escaped(origin=cut.origin, name=cut.name, partial=partial, steps=steps)
            return Answered(Ran(ending, [*messages, _user(ESCAPED)]))
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
    answering: _Answering | None,
    loop: _Loop,
    level: Level,
    repair: Repair | None = None,
) -> Effect[_Ended]:
    """One turn: an answer or an escape ends the loop; an action or a redirect goes a level deeper.

    The final level polls no interrupt and compacts nothing; its decide reads the nudge. A typed
    run's refused answer is decided again at the same level, with the refusal in its tail; that
    decide compacts nothing, since compaction belongs to a level. A redirect on a repair turn
    carries the refusal to the next level's tail."""
    messages, steps = loop.messages, loop.steps
    again: _Again = partial(_turn, decide, act, guard, interrupt, compact, answering)
    compacting = compact if repair is None else None
    repair = repair if repair is not None else loop.repair
    if level.final:
        return (yield from _final(decide, answering, again, messages, steps, level, repair))

    if interrupt is not None:
        signal = yield from interrupt(level.depth, "pre")
        if (ended := _before_decide(signal, messages, steps)) is not None:
            return _carrying(ended, repair)

    match (yield from _decided(decide, compacting, messages, level, answering, repair)):
        case _Cut() as cut:
            return _cut(messages, steps, cut)
        case (compacted, turn):
            messages = compacted
        case unreachable:
            assert_never(unreachable)

    if interrupt is not None:
        signal = yield from interrupt(level.depth, "post")
        if (ended := _after_decide(signal, messages, steps, turn)) is not None:
            return _carrying(ended, repair)

    if answering is not None and (turn.tool is None or turn.answer is not None):
        # An action the loop refuses with the answer beside it never runs, so the transcript keeps
        # what the turn said and leaves the action out.
        messages = [*messages, {"role": "assistant", "content": _said(turn)}]
        resolved = answering.resolved(turn)
        return (yield from _settled(again, answering, messages, steps, turn, level, resolved))
    messages = [*messages, _assistant(turn)]
    if turn.tool is None:
        answer = answer_text(turn.answer) if turn.answer is not None else turn.thought
        return _answered(messages, steps, turn.thought, answer, "finish")

    return (yield from _act(act, guard, interrupt, turn, turn.tool, messages, steps, level))


def _carrying(ended: _Ended, repair: Repair | None) -> _Ended:
    """`ended`, a redirect carrying `repair` into the next level."""
    match ended:
        case Deeper(context=loop) if repair is not None:
            return Deeper(replace(loop, repair=repair))
        case Answered() | Deeper():
            return ended
        case unreachable:
            assert_never(unreachable)


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
    return (
        yield from _ran(messages, max_iters, decide, act, guard, interrupt, compact, grantor, None)
    )


def _ran(
    messages: list[dict[str, Any]],
    max_iters: int,
    decide: Decide | None,
    act: Act | None,
    guard: Guard | None,
    interrupt: Interrupt | None,
    compact: Compact | None,
    grantor: Grantor | None,
    answering: _Answering | None,
) -> Effect[Ran]:
    turn = partial(
        _turn,
        decide if decide is not None else default_decide,
        act if act is not None else default_act,
        guard,
        interrupt,
        compact,
        answering,
    )
    return (yield from descend(_Loop(messages, []), turn, budget=max_iters, grantor=grantor))


REPAIRS = 1
"""The answers a typed run may have refused and ask again for, unless a caller says otherwise."""


@overload
def run_agent(
    prompt: str | Template,
    max_iters: int = 6,
    *,
    answer: None = None,
    decide: Decide | None = None,
    tools: Sequence[Tool[Any, Any]] = (),
    act: Act | None = None,
    guard: Guard | None = None,
    interrupt: Interrupt | None = None,
    compact: Compact | None = None,
    grantor: Grantor | None = None,
) -> Effect[Trajectory | Escaped]: ...
@overload
def run_agent[M: BaseModel](
    prompt: str | Template,
    max_iters: int = 6,
    *,
    answer: Callable[..., Prompt[M]],
    repairs: int = REPAIRS,
    decide: Decide | None = None,
    tools: Sequence[Tool[Any, Any]] = (),
    act: Act | None = None,
    guard: Guard | None = None,
    interrupt: Interrupt | None = None,
    compact: Compact | None = None,
    grantor: Grantor | None = None,
) -> Effect[Ending[M]]: ...


def run_agent(
    prompt: str | Template,
    max_iters: int = 6,
    *,
    answer: Callable[..., Prompt[Any]] | None = None,
    repairs: int | None = None,
    decide: Decide | None = None,
    tools: Sequence[Tool[Any, Any]] = (),
    act: Act | None = None,
    guard: Guard | None = None,
    interrupt: Interrupt | None = None,
    compact: Compact | None = None,
    grantor: Grantor | None = None,
) -> Effect[Ending[Any]]:
    """Drive the turns to an answer: `max_iters` turns, then one final turn that must answer.

    Each turn is a `descend` level under `d:{i}`; a `grantor` adds turns before the final one. A
    nested agent is `scoped(compose_key(t"sub:{name}"), lambda: run_agent(...))`. A turn is a
    scoped body, so a refusal raised in it reaches this call's caller and any other exception ends
    the run.

    `interrupt` is polled before the decide, after it, and after the action, each a recorded op
    when the poll subscribes to that phase (`effective.interrupts`). A redirect becomes the next
    observation, and one that lands before the action runs drops it. An escape ends the run in
    `Escaped`, dropping an action the model chose but the loop had not run, and so does an op
    cancelled while it ran. `unwrap` hands back the answer or raises the escape as a cancel.

    `guard` judges each action before it runs, over the transcript that chose it: a `Refuse` is
    the observation in place of the action, and an `Ask` puts its question to the user and parks
    on `guard:ask` for their `Allowed`.

    A `Template` prompt renders through `effective.channels`, so the run starts from the roles
    that template declares; a `str` is one user message, which is what every caller passed until
    the data axis grew a processor to hand this to.

    `compact` fires on the transcript before a decide and records one `react:compact` step whose
    summary replaces the transcript's prefix, keeping the task and the last `KEEP_TAIL` messages.
    The splice is loop logic because the transcript is workflow state between yields, where no
    handler layer reaches.

    `tools` declares what each name an action calls means, its arguments and its result type,
    and `act` dispatches every action itself, so a run takes one or the other. An undeclared name
    still goes to `default_act`. Declarations are not an authorization boundary.

    `answer` types the run's answer: a function returning `Prompt[M]` whose parameters bind, by
    result type, the last result of the one declared tool producing each (`Annotated[T, tool]`
    names one). Every decide receives `M`'s schema and a tail naming its fields, the
    rendered contract once its evidence binds, and the last refusal on a repair turn. A refused
    answer is decided again at the same level, `repairs` times over the run, and then the run
    ends `Exhausted`, which `unwrap` raises as `RepairsExhausted`. A turn calling a tool spends
    no repair, and the final turn calls none. Evidence from an earlier occurrence of a tool, or
    produced inside a subagent, is out of reach."""
    if tools and act is not None:
        raise ValueError("`tools` and `act` each dispatch the run's actions; pass one")
    toolbox = Toolbox(declared(tools)) if tools else None
    return (
        yield from _run(
            toolbox,
            prompt,
            max_iters,
            answer,
            repairs,
            decide,
            act,
            guard,
            interrupt,
            compact,
            grantor,
        )
    )


@overload
def run_in(
    toolbox: Toolbox | None,
    prompt: str | Template,
    max_iters: int = 6,
    *,
    decide: Decide | None = None,
    act: Act | None = None,
    guard: Guard | None = None,
    interrupt: Interrupt | None = None,
    compact: Compact | None = None,
    grantor: Grantor | None = None,
) -> Effect[Trajectory | Escaped]: ...
@overload
def run_in[M: BaseModel](
    toolbox: Toolbox | None,
    prompt: str | Template,
    max_iters: int = 6,
    *,
    answer: Callable[..., Prompt[M]],
    repairs: int = REPAIRS,
    decide: Decide | None = None,
    act: Act | None = None,
    guard: Guard | None = None,
    interrupt: Interrupt | None = None,
    compact: Compact | None = None,
    grantor: Grantor | None = None,
) -> Effect[Ending[M]]: ...
@overload
def run_in(
    toolbox: Toolbox | None,
    prompt: str | Template,
    max_iters: int = 6,
    *,
    answer: Callable[..., Prompt[Any]] | None,
    repairs: int | None = None,
    decide: Decide | None = None,
    act: Act | None = None,
    guard: Guard | None = None,
    interrupt: Interrupt | None = None,
    compact: Compact | None = None,
    grantor: Grantor | None = None,
) -> Effect[Ending[Any]]: ...
def run_in(
    toolbox: Toolbox | None,
    prompt: str | Template,
    max_iters: int = 6,
    *,
    answer: Callable[..., Prompt[Any]] | None = None,
    repairs: int | None = None,
    decide: Decide | None = None,
    act: Act | None = None,
    guard: Guard | None = None,
    interrupt: Interrupt | None = None,
    compact: Compact | None = None,
    grantor: Grantor | None = None,
) -> Effect[Ending[Any]]:
    """`run_agent` dispatching through `toolbox`, whose fallback takes the undeclared names, or
    through `act` without one. The caller holds the toolbox, so it reads the log the run wrote."""
    return (
        yield from _run(
            toolbox,
            prompt,
            max_iters,
            answer,
            repairs,
            decide,
            act,
            guard,
            interrupt,
            compact,
            grantor,
        )
    )


def _run(
    toolbox: Toolbox | None,
    prompt: str | Template,
    max_iters: int,
    answer: Callable[..., Prompt[Any]] | None,
    repairs: int | None,
    decide: Decide | None,
    act: Act | None,
    guard: Guard | None,
    interrupt: Interrupt | None,
    compact: Compact | None,
    grantor: Grantor | None,
) -> Effect[Ending[Any]]:
    if answer is None and repairs is not None:
        raise ValueError("`repairs` bounds a typed run's answers, and no `answer` was given")
    answering = _answering(answer, REPAIRS if repairs is None else repairs, toolbox)
    ran = yield from _ran(
        _messages(prompt),
        max_iters,
        decide,
        toolbox.act() if toolbox is not None else act,
        guard,
        interrupt,
        compact,
        grantor,
        answering,
    )
    return ran.trajectory
