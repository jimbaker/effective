"""Sandboxed code execution as an effect: the ``run_code`` combinator.

RLM's defining affordance, where the model (or a pinned skill script) supplies code
that slices and recombines data without paying tokens per intermediate step,
lands as *existing* op kinds: no new ``WorkflowOp`` and no handler changes (the
``skills.py`` reserved-tool idiom):

- each **segment** is a checkpointed ``Step`` carrying the reserved
  ``code-execute`` ``CallTool``, which the deployment's domain interpreter
  implements with a code engine (``effective.monty.MontyEngine``: deployment
  infrastructure, exactly like the model caller);
- each **action** (a world-mutating host call the code makes) surfaces as its
  **own** ``CallTool`` op in the parent stream, recorded (exactly-once across
  a crash), routed through the permission cascade and deniable, before the code
  resumes. A denial re-enters the sandbox as ``PermissionError``, so code can
  route around it (denial-as-routing-signal, the ``run_agent`` rule).

Durability is **replay-by-re-execution**: the sandbox has no ambient authority,
so its only effects are host calls. A segment re-executes the code from the
start, re-binding every prior host-call result by call index (functions from the
threaded ``fn_log``, actions from the workflow's own checkpointed results), and
runs live only past that frontier. Committed segments replay as checkpoint
lookups with no re-execution at all. All state is threaded explicitly through op
results, with no handler pin-state and no captured continuation: the same rule
as skill pins.

Contracts:

- **One canonical form.** Every value crossing the sandbox boundary (a live
  host-call result, a re-bound one, a threaded action result) crosses in
  ``canonical()`` (checkpoint JSON) form: nan/inf → null, int dict keys → str,
  tuples → lists. Live and re-bound values are therefore byte-identical by
  construction, a set-bearing value included: ``canonical`` routes through the
  shared ``handlers/base.canonical_form``, which sorts sets, so set hash order
  cannot diverge across a fresh-worker resume. ``inputs`` must already BE JSON
  values, checked loudly at entry, because a ``set`` input's iteration order
  leaks the host process's hash seed into the sandbox and no normalization at
  entry can fix that (entry re-runs on every re-execution).
- **Growth shape.** The cumulative ``fn_log`` rides every segment's args and
  result: O(segments x log) bytes through the checkpoint store. Fine for glue
  code; a ``run_code`` making hundreds of large host calls wants restructuring
  into fewer, bigger functions.
- An exception raised *by* the code propagates as the segment step's failure
  (deterministic: a retry re-raises identically, re-paying live host calls).
"""

from collections.abc import Mapping, Sequence
from functools import cache
from math import isfinite
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, TypeAdapter
from pydantic_core import to_jsonable_python

from effective.api import Effect, scoped, step
from effective.domain import CallTool
from effective.govern import routable
from effective.handlers.base import canonical_form
from effective.keys import (
    Index,
    Key,
    Name,
    Tag,
    authored_key,
    carries_structure,
    compose_key,
)
from effective.ops import Minted
from effective.permission import Refused


class PinnedScript(Protocol):
    """A code source plus the KEY of whatever pinned it — structurally
    ``effective.skills.Script`` (``pin.script("x.py")``); the combinator stays decoupled
    from the skills module, the same rule as ``channels.SkillPin``.

    A `Key` rather than a provenance string, because `run_code` does not RECORD it — it
    runs the segments inside ``scoped(key)``, so the pin lands in every segment's key as a
    frame. Anything that can name a scope can pin code; nothing here knows about skills."""

    @property
    def source(self) -> str: ...
    @property
    def key(self) -> Key: ...


CODE = Tag("code")
"""The run_code namespace. A `Tag` constant rather than an inline literal so the registry
scanner resolves it and both VARIANTS below register under one namespace."""


def _segment_key(name: str, structured: bool, seg: int) -> Key:
    """The checkpoint key of one code SEGMENT: `code:seg,{seg},{name}` or `code;seg:{seg};{name}`.

    Two variants because a wrapper supplies the tag, so an author's bare name can only be a
    COORDINATE while a structured one is spliced as its own terms (`keys.carries_structure`, the
    same split `handlers.base.step_key` makes).

    **The discriminator leads and the identity trails**, in both. The order is forced: a splice
    absorbs every remaining term, so `registry._bind` refuses one that is not LAST: `code;{name};
    seg:{n}` does not decode at all. Every splice in this tree is terminal for the same reason.

    `code:` is a tagged union because the name, which may carry delimiters, is never interior. The
    four variants are pairwise disjoint (`seg` vs `action` at coordinate 0, arity 3 vs 0 at term
    0)."""
    if structured:
        return compose_key(t"{CODE};seg:{Index(seg)};{authored_key(name):domain=address}")
    return compose_key(t"{CODE}:seg,{Index(seg)},{Name(name)}")


def _action_key(name: str, structured: bool, j: int, tool: str) -> Key:
    """The checkpoint key of the `j`-th ACTION a code run requests. Companion to `_segment_key`."""
    if structured:
        return compose_key(
            t"{CODE};action:{Index(j)};tool:{Name(tool)};{authored_key(name):domain=address}"
        )
    return compose_key(t"{CODE}:action,{Index(j)},{Name(name)};tool:{Name(tool)}")


EXECUTE_TOOL = "code-execute"
"""The reserved CallTool name for one code segment — the handler's domain
interpreter implements it by delegating to a code engine (the registry is
deployment infrastructure, exactly like ``skill-disclose``)."""

# A run that keeps requesting actions is runaway fan-out, not computation; each
# action is a durable op, so bound the loop legibly rather than let it creep.
MAX_ACTIONS = 64


def canonical(value: Any) -> Any:
    """The checkpoint form of ``value``: pydantic JSON-mode semantics (nan/inf →
    null, int dict keys → str, tuples → lists, datetimes → ISO strings). Values
    cross the sandbox boundary ONLY in this form, live and re-bound alike; any
    other form diverges from its own checkpoint round-trip on replay.

    The **deterministic normal form** is the shared ``canonical_form`` (it sorts
    sets, the one hazard ``to_jsonable_python`` leaves seed-dependent); the outer
    ``to_jsonable_python`` supplies the checkpoint-form leaf normalization on top
    (int keys → str, nan/inf → null). One canonicalizer with ``content_digest``, so
    a set-bearing value cannot re-serialize two ways across a fresh-worker resume."""
    return to_jsonable_python(canonical_form(value), inf_nan_mode="null")


def _require_json(value: Any, path: str) -> None:
    """Reject a non-JSON input loudly, located. Normalizing instead would be
    unsound: ``run_code``'s entry re-runs on every re-execution, so a ``set``
    normalized here still leaks the resuming process's hash order."""
    match value:
        case str() | int() | bool() | None:
            return
        case float() if not isfinite(value):
            raise TypeError(f"run_code input {path} is non-finite ({value!r}); not a JSON value")
        case float():
            return
        case list():
            for i, v in enumerate(value):
                _require_json(v, f"{path}[{i}]")
        case dict():
            for k, v in value.items():
                if not isinstance(k, str):
                    raise TypeError(f"run_code input {path} has a non-str key {k!r}")
                _require_json(v, f"{path}[{k!r}]")
        case _:
            raise TypeError(
                f"run_code input {path} is {type(value).__name__!r}, not a JSON value — "
                "convert explicitly (e.g. sorted(a_set), list(a_tuple), dt.isoformat())"
            )


class ActionCall(BaseModel):
    """A world-mutating host call the sandbox requested — the segment boundary."""

    name: str
    args: list[Any] = []
    kwargs: dict[str, Any] = {}


class CodeOutcome(BaseModel):
    """One segment's result: the run completed, or it paused at an action.

    ``fn_log`` is the **cumulative** host-function call log, in call order: the segment's
    checkpoint records it, and the next segment re-binds from it. That threaded state makes
    re-execution deterministic without any handler pin-state. Each entry is
    ``{"name", "args" (a digest), "value"}``: the digest makes a re-bind that lands on the
    wrong call site a **loud** ``CodeEngineError``.

    There is no provenance field. A pinned run is scoped, so the pin is a frame in every
    segment's key and the improvised-vs-pinned Pareto signal is a question about keys; a
    field here would be read from the args and echoed back unchanged, recorded weight that no
    computation reads."""

    status: Literal["complete", "action"]
    output: Any = None
    action: ActionCall | None = None
    fn_log: list[Any] = []


@cache
def _adapter(schema: type) -> TypeAdapter[Any]:
    return TypeAdapter(schema)


def _validate[T](schema: type[T], raw: Any) -> T:
    # No None bypass: `schema=int` means int. A code run whose final value can be
    # None must say so in its schema.
    if schema is object:
        return cast(T, raw)
    return _adapter(schema).validate_python(raw)


def _frame_and_source(code: str | PinnedScript) -> tuple[Key | None, str]:
    """A script's pin-frame and its text — ONE decision over `str | PinnedScript`.

    These were two conditional expressions asking the same question of the same value, on
    consecutive lines. That is how a pair drifts: nothing makes them answer together, and the
    shape is the one the `compose_key` pilot found four times over in a single function.
    Returning the pair makes the coupling structural."""
    # lint: totality(total) — `str` returns directly. `PinnedScript` is a non-runtime-checkable
    # protocol and cannot be named as a class pattern, so the wildcard is the declared second arm.
    match code:
        case str():
            return None, code
        case _:
            return code.key, code.source


def run_code[T](
    name: str,
    code: str | PinnedScript,
    *,
    schema: type[T],
    inputs: Mapping[str, Any] | None = None,
    functions: Sequence[str] = (),
    actions: Mapping[str, type[Any]] | None = None,
) -> Effect[T]:
    """Run sandboxed code as a durable sub-workflow, namespaced by ``name``.

    ``code`` is either a raw string — **improvised** (model-written this turn) —
    or a pinned skill ``Script`` (``pin.script("x.py")``), whose source came off
    the activation checkpoint. **Provenance is a FRAME:** a pinned run's segments are
    composed inside ``scoped(script.key)``, so the pack rides every segment key and the
    ledger separates pinned from improvised code structurally. That is the promotion /
    Pareto signal (improvised-code rate is a cost to watch shrinking), answered by selecting
    on keys rather than by scanning payloads.

    ``functions`` names the observational host calls the code may make (bodies
    live in the deployment's engine registry); their results seal into each
    segment's recorded ``fn_log``. ``actions`` maps world-mutating host-call
    names to their result schemas; each such call surfaces as its own
    ``code:action,{j},{name};tool:{tool}`` op — cascade-gated, and **exactly-once per
    committed checkpoint**: a crash after the tool fires but before its
    checkpoint commits re-runs it (standard step-thunk semantics — the crash
    window shrinks from RLM's whole-exec to the one call, it does not vanish).
    Each action asks the handler for an ``idempotency_key`` (`Minted`), minted from the
    task and the action's placement, gather frames included; an action tool whose
    receiver dedupes on it performs the action once across a crash.
    The final expression's value is validated to ``schema``.

    ``name`` must be unique per ``run_code`` call within the workflow (it keys
    the checkpoints, like every step name) and must not contain ``/``, the path
    delimiter no key atom may hold. No name can forge a scope boundary that no
    ``scoped(...)`` opened: every key below leads with the ``code`` tag, so a name
    carrying ``;`` splices after ``code;seg:{j};`` and never stands where a frame
    stands. ``scoped(rec:0, run_code("x"))`` records ``rec:0;step;code:seg,0,x``,
    while a run named ``rec:0;x`` records ``step;code;seg:0;rec:0;x``.

    **Namespacing is not this function's job.** It names its checkpoints
    ``code:seg,{j},{name}`` / ``code:action,{j},{name};tool:{tool}`` and nothing
    else; to place a whole ``run_code`` under a namespace, wrap the call —
    ``scoped(compose_key(t"rec:{i}"), lambda: run_code(...))``. The handler applies a
    namespace, in the same position, for both scopes and the structural
    ``gather:{g},{i};`` prefix, so no call site decides a delimiter.
    (The durable handler places a duplicate name as ``name#2``: a safety net, not a
    license, since occurrences bind by yield *order*, so an edit that inserts an
    earlier same-named call shifts every later binding.)
    """
    if "/" in name:
        raise ValueError(
            f"run_code name {name!r} contains '/', the key grammar's path delimiter; "
            "wrap the call in `scoped(...)` to namespace it"
        )
    # **A pinned run is SCOPED.** The pin's key becomes a frame, so every checkpoint below
    # carries it and "was this pinned?" is a question about the key. Applied HERE rather than
    # left to the caller, so the property is structural and automatic: an author cannot pin a
    # script and forget to say so.
    frame, source = _frame_and_source(code)
    action_schemas = dict(actions or {})
    for key, value in dict(inputs or {}).items():
        _require_json(value, f"inputs[{key!r}]")
    structured = carries_structure(name)

    def segments() -> Effect[T]:
        fn_log: list[Any] = []
        action_results: list[dict[str, Any]] = []
        seg = 0
        while True:
            outcome = yield from step(
                _segment_key(name, structured, seg).stored(),
                CallTool(
                    name=EXECUTE_TOOL,
                    args={
                        "code": source,
                        "inputs": dict(inputs or {}),
                        "functions": list(functions),
                        "actions": sorted(action_schemas),
                        "fn_log": fn_log,
                        "action_results": action_results,
                    },
                    result_schema=CodeOutcome,
                ),
            )
            fn_log = list(outcome.fn_log)
            if outcome.status == "complete":
                return _validate(schema, outcome.output)
            request = outcome.action
            if request is None:  # an engine bug, not a user error — fail located
                paused = _segment_key(name, structured, seg).stored()
                raise ValueError(f"{paused} paused without an action request")
            j = len(action_results)
            if j >= MAX_ACTIONS:
                raise RuntimeError(
                    f"{CODE}:{name} requested more than {MAX_ACTIONS} actions; "
                    "runaway action loop (raise MAX_ACTIONS deliberately if real)"
                )
            action_key = _action_key(name, structured, j, request.name).stored()
            try:
                # result_schema=object: the step returns the RAW checkpoint form and
                # we thread exactly that, never a validated object re-dumped, which
                # is not a fixpoint (a set field re-dumps in the resuming process's
                # hash order, so two processes diverge).
                # The declared schema still gates: malformed tool output fails HERE.
                raw = yield from step(
                    action_key,
                    CallTool(
                        name=request.name,
                        args={"args": request.args, "kwargs": request.kwargs},
                        result_schema=object,
                    ),
                    idempotency_key=Minted(),
                )
                _validate(action_schemas[request.name], raw)
                action_results = [*action_results, {"return_value": canonical(raw)}]
            except Refused as refusal:
                denial = routable(refusal)
                # The cascade blocked this action. Thread the denial *into* the
                # sandbox: the next segment resumes the call as a PermissionError
                # the code can catch — route around, or let it fail the run.
                action_results = [
                    *action_results,
                    {"exception": {"type": "PermissionError", "message": denial.reason}},
                ]
            seg += 1

    if frame is None:
        return (yield from segments())
    return (yield from scoped(frame, segments))
