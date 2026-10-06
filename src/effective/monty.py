"""MontyEngine: the default code engine behind ``code-execute``.

Deployment infrastructure, exactly like the model caller and the tool registry:
``effective.code`` never imports this module. A deployment builds
``MontyEngine(functions=...)`` and registers ``execute_tool(engine)`` under
``EXECUTE_TOOL`` in its domain interpreter (the ``agents`` slot of
``effective.interpreters.tools.make_tool_runner`` fits: it hands over the whole op).

Monty (pydantic-monty, pinned; alpha, API in flux) is capability-based: no
``import``/``eval``/filesystem/network, and every ambient-looking call, even
``datetime.now()``, surfaces as a ``FunctionSnapshot`` the host resolves. So
sandboxed code satisfies the determinism boundary **by construction**:
deterministic computation between recorded host calls.

The engine executes ONE segment per call: it (re-)runs the code from the start,
re-binding recorded host-call results by call index — functions from
``fn_log``, actions from ``action_results`` — then runs live past that
frontier, until the next unresolved *action* call (pause: the workflow yields
it as a real op) or completion. Every host call is resumed with the
**checkpoint form** of its result (``to_jsonable_python``), on the live path
and the re-bind path alike, so the sandbox sees identical values on first run
and on re-execution.

Undeclared-name policy (all deterministic, so re-execution agrees):

- a *declared function* miss in the registry is a deployment bug →
  ``CodeEngineError`` (the step fails loudly);
- an undeclared **OS-function** call (``open``, ``os.environ``, …) is resumed
  ``not_handled`` → the sandbox raises ``PermissionError`` (undeclared world
  access *is* a permission denial — catchable as control flow);
- any other undeclared call → ``NameError`` inside the sandbox;
- a bare undefined *name* (``NameLookupSnapshot``) → ``CodeEngineError`` (plain
  code bug; fail located rather than guess a value).
"""

from collections.abc import Callable, Mapping
from hashlib import blake2b
from typing import Any

import pydantic_monty as monty
from pydantic_core import to_json

from effective.code import ActionCall, CodeOutcome, canonical
from effective.domain import CallTool


class CodeEngineError(Exception):
    """The engine cannot (or will not) execute this program shape — a
    deployment/authoring error, distinct from an exception the code raised."""


# The exception types a threaded action outcome may re-raise *inside* the
# sandbox. Deliberately small: denial is control flow, not a vector for
# arbitrary host exception types crossing the boundary.
_INJECTABLE: dict[str, type[BaseException]] = {
    "PermissionError": PermissionError,
    "ValueError": ValueError,
    "RuntimeError": RuntimeError,
    "TimeoutError": TimeoutError,
}


def _resume_value(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Decode a threaded action outcome into Monty's ExternalResult envelope."""
    if "exception" in entry:
        info = entry["exception"]
        exc_type = _INJECTABLE.get(info.get("type", ""), RuntimeError)
        return {"exception": exc_type(info.get("message", ""))}
    return {"return_value": entry["return_value"]}


def _action_call(state: Any) -> ActionCall:
    """The paused call, in canonical (checkpoint) form — what the workflow
    yields as an op."""
    return ActionCall(
        name=state.function_name,
        args=[canonical(a) for a in state.args],
        kwargs=canonical(dict(state.kwargs or {})),
    )


def _args_digest(state: Any) -> str:
    """A short digest of a host call's (args, kwargs) in canonical form: the
    call-site fingerprint. Re-binding checks it, so a recorded value landing on
    the WRONG call (a nondeterministic sandbox, a serde divergence, a code edit
    mid-flight) is a loud ``CodeEngineError``, never a silent misbinding."""
    payload = to_json([canonical(list(state.args)), canonical(dict(state.kwargs or {}))])
    return blake2b(payload, digest_size=8).hexdigest()


class MontyEngine:
    """One segment executor over a host-function registry.

    ``functions`` holds the bodies for the *declared* observational calls; a
    workflow's ``functions=("llm_query", ...)`` declaration is the contract,
    this registry is the implementation. Registering an ambient-looking name
    (e.g. ``"datetime.now"``) makes it a recorded effect — clock-as-effect.
    ``limits`` applies Monty's ``ResourceLimits`` to every segment.
    """

    def __init__(
        self,
        functions: Mapping[str, Callable[..., Any]] | None = None,
        *,
        limits: Any | None = None,
    ) -> None:
        self.functions: dict[str, Callable[..., Any]] = dict(functions or {})
        self.limits = limits

    def execute(self, args: Mapping[str, Any]) -> CodeOutcome:
        code: str = args["code"]
        inputs: dict[str, Any] = dict(args.get("inputs") or {})
        declared = frozenset(args.get("functions") or ())
        actions = frozenset(args.get("actions") or ())
        action_results: list[dict[str, Any]] = list(args.get("action_results") or ())

        program = monty.Monty(code, inputs=sorted(inputs))
        state: Any = program.start(inputs=inputs, limits=self.limits)
        log: list[Any] = list(args.get("fn_log") or ())  # re-bound prefix + live appends
        fn_idx = 0
        act_idx = 0
        while True:
            # one match over sandbox events; the FunctionSnapshot arms are the
            # undeclared-name policy ladder from the module docstring, in order
            match state:
                case monty.MontyComplete():
                    return CodeOutcome(
                        status="complete",
                        output=canonical(state.output),
                        fn_log=log,
                    )
                case monty.FunctionSnapshot(function_name=fname) if fname in actions:
                    if act_idx < len(action_results):  # re-bind a resolved action
                        state = state.resume(_resume_value(action_results[act_idx]))
                        act_idx += 1
                    else:  # pause: the workflow owns this call
                        return CodeOutcome(status="action", action=_action_call(state), fn_log=log)
                case monty.FunctionSnapshot(function_name=fname) if fname in declared:
                    state, fn_idx = self._resume_function(state, log, fn_idx)
                case monty.FunctionSnapshot(is_os_function=True):
                    # undeclared world access = permission denial
                    state = state.resume_not_handled()
                case monty.FunctionSnapshot(function_name=fname):
                    state = state.resume(
                        {"exception": NameError(f"name {fname!r} is not defined")}
                    )
                case monty.NameLookupSnapshot(variable_name=name):
                    raise CodeEngineError(f"undefined name {name!r} in sandboxed code")
                case _:  # FutureSnapshot etc. — not supported in M0
                    raise CodeEngineError(f"unsupported sandbox event: {type(state).__name__}")

    def _resume_function(self, state: Any, log: list[Any], fn_idx: int) -> tuple[Any, int]:
        """One declared-function call: re-bind from the log when the index is
        still inside the recorded prefix (replay-by-re-execution), else run the
        registry body live and append its canonical-form result. The re-bind
        verifies the recorded ``(name, args-digest)`` against the actual call
        site — a mismatch is loud, never a silent misbinding."""
        fname: str = state.function_name
        digest = _args_digest(state)
        if fn_idx < len(log):
            entry = log[fn_idx]
            if entry.get("name") != fname or entry.get("args") != digest:
                raise CodeEngineError(
                    f"re-bind mismatch at host call #{fn_idx}: recorded "
                    f"{entry.get('name')!r}@{entry.get('args')}, re-executed "
                    f"{fname!r}@{digest} — nondeterministic sandbox path, serde "
                    "divergence, or the code changed mid-flight"
                )
            return state.resume({"return_value": entry["value"]}), fn_idx + 1
        fn = self.functions.get(fname)
        if fn is None:
            raise CodeEngineError(f"declared function {fname!r} missing from the engine registry")
        value = canonical(fn(*state.args, **dict(state.kwargs or {})))
        log.append({"name": fname, "args": digest, "value": value})
        return state.resume({"return_value": value}), fn_idx + 1


def execute_tool(engine: MontyEngine) -> Callable[[CallTool[Any]], CodeOutcome]:
    """The handler-side ``code-execute`` implementation: register it under
    ``EXECUTE_TOOL`` where the deployment dispatches tools that receive the
    whole op (they need the structured args, not a stringified result)."""

    def run(op: CallTool[Any]) -> CodeOutcome:
        return engine.execute(op.args)

    return run
