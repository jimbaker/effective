"""Cost / KV-cache as a handler layer over the model calls, `AskLLM` and `Judge`.

Token cost, cache-hit ratio, and budget enforcement do not belong to a workflow's
logic: they are observations and rewrites over the model-interaction stream,
the textbook cross-cutting concern. In this substrate that stream is the model-call
ops, so each concern is a layer over their interpretation, invisible to the workflow.

`Usage` is the accumulator folded over the stream, associative on its token counts and
order-dependent on its float `cost` and `latency_s`. `MeteredInterpreter` is a
`DomainInterpreter` decorator: the production seam that runs a model call for real also
yields the provider `usage`, which accrues into a running meter; a `CallTool` passes through
untouched. A `CostBudget` makes the ceiling enforceable: once it is spent, the next model call
is refused with `BudgetExceeded`.

DB-free, provider-agnostic: `from_response` duck-types the response (no litellm
import), so the core never pulls a provider SDK.
"""

import threading
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, assert_never, runtime_checkable

from effective.domain import AskLLM, AsksModel, CallTool, DomainOp, Judge

if TYPE_CHECKING:
    # `cache` imports this module, so the annotation is the only place `Cache` appears here.
    from effective.cache import Cache
from effective.layers import (
    DomainLayer,
    Interpreter,
    compose_domain,
    domain_layer,
    drive_through,
)
from effective.ops import FORMAT_PARAM, UnreadFormat

type Dollars = float


# --- the recorded format ---------------------------------------------------------------
# A model call's `Step` checkpoint is `{result, usage}`, and a reader knows it from the run's
# format rather than by sniffing for those keys, which a workflow result may also hold. The
# format is marked at spawn in the task's immutable params under `FORMAT_PARAM`, so a task
# never holds rows of two formats.


class RecordedFormat(StrEnum):
    """The version of a run's recorded format, marked at spawn in its params.

    | version | a step's key names    | a model call's step records | an authority answer |
    |---------|-----------------------|-----------------------------|---------------------|
    | `V1`    | its op's kind         | `{result, usage}`           | a ruling            |
    """

    V1 = "v1"

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> RecordedFormat:
        """The format a task records under, read from its immutable spawn params. Params that mark
        none, or a format this release does not read, are `UnreadFormat`."""
        match params.get(FORMAT_PARAM):
            case None:
                raise UnreadFormat(
                    f"a run whose params mark no {FORMAT_PARAM!r}, which this release does not "
                    "read; finish it under the release that spawned it"
                )
            case raw:
                try:
                    return cls(raw)
                except ValueError as unread:
                    raise UnreadFormat(
                        f"a run spawned under recorded format {raw!r}, which this release does "
                        "not read; finish it under the release that spawned it"
                    ) from unread


# The format a new run records under.
NEWEST_FORMAT = list(RecordedFormat)[-1]


def marked(params: Mapping[str, Any]) -> dict[str, Any]:
    """`params` marked with the format a new run records under, whatever mark they carry."""
    return {**params, FORMAT_PARAM: NEWEST_FORMAT.value}


def refuse_an_unread_format(params: Mapping[str, Any]) -> None:
    """Refuse a run whose `params` mark no format this release reads, as `UnreadFormat`."""
    RecordedFormat.from_params(params)


@dataclass(frozen=True)
class Usage:
    """Token counts and cost for one AskLLM, or a sum across many.

    `+` is associative over the integer token counts and NOT over `cost` and `latency_s`, which
    are floats: `(a + b) + c` and `a + (b + c)` differ in the last bits. Fold in a fixed order
    wherever the total is compared, and read a summed float as a report rather than as a key."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost: Dollars = 0.0
    latency_s: float = 0.0  # wall time of the call(s); the local analog of cost

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cost=self.cost + other.cost,
            latency_s=self.latency_s + other.latency_s,
        )

    @property
    def tokens_per_second(self) -> float:
        """Generation throughput (completion tokens / wall time); 0 if untimed."""
        if self.latency_s <= 0:
            return 0.0
        return round(self.completion_tokens / self.latency_s, 2)

    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of input tokens served from cache (0.0 on an empty meter)."""
        if self.prompt_tokens <= 0:
            return 0.0
        return round(self.cache_read_input_tokens / self.prompt_tokens, 4)

    def as_attributes(self) -> dict[str, Any]:
        """The usage as span attributes, named by the OpenTelemetry GenAI conventions.

        The cost layer owns the meaning of the meter, so the attribute mapping lives here.
        `input_tokens` counts cached tokens too, as `prompt_tokens` does and the conventions
        ask."""
        return {
            "gen_ai.usage.input_tokens": self.prompt_tokens,
            "gen_ai.usage.output_tokens": self.completion_tokens,
            "gen_ai.usage.cache_read.input_tokens": self.cache_read_input_tokens,
            "gen_ai.usage.cache_write.input_tokens": self.cache_creation_input_tokens,
            "effective.cost.usd": self.cost,
            "effective.cache_hit_ratio": self.cache_hit_ratio,
            "effective.latency_s": self.latency_s,
        }

    @classmethod
    def from_response(cls, response: Any) -> Usage:
        """Read a provider response (duck-typed; litellm-shaped) into a Usage."""
        usage = getattr(response, "usage", None) or {}
        hidden = getattr(response, "_hidden_params", None) or {}
        return cls(
            prompt_tokens=int(_read(usage, "prompt_tokens")),
            completion_tokens=int(_read(usage, "completion_tokens")),
            cache_read_input_tokens=int(_read(usage, "cache_read_input_tokens")),
            cache_creation_input_tokens=int(_read(usage, "cache_creation_input_tokens")),
            cost=float(_read(hidden, "response_cost")),
        )


def _read(source: Any, key: str) -> Any:
    """Field access over a mapping or an attribute-bearing object; 0 if absent."""
    if isinstance(source, dict):
        return source.get(key) or 0
    return getattr(source, key, 0) or 0


@dataclass
class CostBudget:
    """A running spend tracker with a hard ceiling.

    A *telemetry* accumulator; the measured trip that enforces spend sits above the checkpoint,
    in `DurableHandler`. The lock closes the lost-update race: `spent = spent + usage` is a
    read-modify-write, so two concurrent `gather` branches accruing at once could drop one, and
    a lock cures the *total*. A lock cannot make a shared enforcement gate confluent, which
    needs the per-branch fold (`Budget.lean`'s `shared_lock_insufficient`)."""

    limit: Dollars
    spent: Usage = Usage()
    _lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)

    def add(self, usage: Usage) -> None:
        with self._lock:
            self.spent = self.spent + usage

    def exceeded(self) -> bool:
        return self.spent.cost >= self.limit


class BudgetExceeded(RuntimeError):
    """Raised when an AskLLM is refused because the cost ceiling is reached."""

    def __init__(self, spent: Dollars, limit: Dollars) -> None:
        super().__init__(f"cost budget exceeded: spent ${spent:.4f} >= limit ${limit:.4f}")
        self.spent = spent
        self.limit = limit


type LLMCall = Callable[[AskLLM[Any]], tuple[Any, Usage]]
type ToolRunner = Callable[[CallTool[Any]], Any]
type JudgeCall = Callable[[Judge[Any]], tuple[Any, Usage]]


def _no_judge(op: Judge[Any]) -> tuple[Any, Usage]:
    raise LookupError(
        f"a Judge op reached an interpreter with no judge caller ({sorted(op.questions)}); "
        "pass judge= to MeteredInterpreter"
    )


class _Calls:
    """The production base interpreter: runs an op for real. ``llm`` and ``judge``
    return ``(result, Usage)``; ``tools`` returns the tool result directly. Metering is the
    layer's job."""

    def __init__(self, llm: LLMCall, tools: ToolRunner, judge: JudgeCall = _no_judge) -> None:
        self.llm = llm
        self.tools = tools
        self.judge = judge

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case AskLLM():
                return self.llm(op)  # (result, usage)
            case Judge():
                return self.judge(op)  # (result, usage)
            case CallTool():
                return self.tools(op)
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead

    def run_metered(self, op: DomainOp[Any]) -> tuple[Any, Usage]:
        """Like `run`, but always returns `(result, usage)`: the durable handler's
        usage-in-checkpoint seam. A `CallTool` reports zero usage."""
        match op:
            case AskLLM():
                return self.llm(op)  # already (result, usage)
            case Judge():
                return self.judge(op)
            case CallTool():
                return self.tools(op), Usage()
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead


@dataclass(frozen=True)
class _Enveloped:
    """A `MeteredDomain` read the way `metered` expects its base: a model call answers
    `(result, usage)` and a tool call its bare result."""

    domain: MeteredDomain

    def run(self, op: DomainOp[Any]) -> Any:
        result, usage = self.domain.run_metered(op)
        match op:
            case AsksModel():
                return result, usage
            case CallTool():
                return result
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead

    def run_metered(self, op: DomainOp[Any]) -> tuple[Any, Usage]:
        return self.domain.run_metered(op)


def metered(
    accrue: Callable[[Usage], None], budget: CostBudget | None = None
) -> Callable[[DomainOp[Any]], Generator[DomainOp[Any], Any, Any]]:
    """The cost concern as a `@domain_layer`.

    Refuses an over-budget model call before forwarding; on the way back, folds the
    `(result, usage)` from the base into the meter and unwraps to the bare result
    the workflow sees. A `CallTool` passes through untouched. `accrue` and `budget` carry
    the cross-op state in the closure.
    """

    @domain_layer
    def run(op: DomainOp[Any]) -> Generator[DomainOp[Any], Any, Any]:
        match op:
            case AsksModel():
                if budget is not None and budget.exceeded():
                    raise BudgetExceeded(budget.spent.cost, budget.limit)
                result, usage = yield op
                accrue(usage)
                if budget is not None:
                    budget.add(usage)
                return result
            case CallTool():
                return (yield op)
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead

    return run


class MeteredInterpreter:
    """A `DomainInterpreter` that folds model-call cost into a running meter.

    A `metered` `@domain_layer` over a `_Calls` base, assembled with `compose_domain`: it
    satisfies the structural ``run(op) -> Any`` contract `DurableHandler` expects, exposes the
    running ``meter``, and refuses the next model call once a ``budget`` ceiling is reached.
    """

    def __init__(
        self,
        llm: LLMCall,
        tools: ToolRunner,
        budget: CostBudget | None = None,
        domain_layers: Sequence[DomainLayer[Any]] = (),
        judge: JudgeCall = _no_judge,
        cache: Cache | None = None,
    ) -> None:
        self.meter = Usage()
        self._budget = budget
        # Guards the meter RMW (`meter = meter + usage`) so concurrent gather branches
        # accruing at once (via `run_metered` or the `metered` layer) don't drop an update
        # (the telemetry counterpart of the CostBudget race).
        self._lock = threading.Lock()
        # A cache answers below every layer, where each op answers `(result, usage)`: `traced`
        # above it sees each hit, and `metered` accrues a hit's zero usage.
        calls = _Calls(llm, tools, judge)
        base = _Enveloped(cache.over(calls)) if cache is not None else calls
        # metered is outermost so it unwraps `(result, usage)`; extra layers (e.g. a
        # telemetry `traced`) compose *under* it and see the tuple before the unwrap.
        self._stack: Interpreter = compose_domain(
            [metered(self._accrue, budget), *domain_layers], base=base
        )
        # The same layers WITHOUT the enforcing/unwrapping `metered` head — traced still
        # fires, but the `(result, usage)` tuple surfaces intact. This is the durable
        # handler's `run_metered` path: usage rides above the checkpoint so
        # the handler owns the replay-derived meter; `metered` stays a telemetry observer.
        self._base_stack: Interpreter = compose_domain(list(domain_layers), base=base)

    def _accrue(self, usage: Usage) -> None:
        with self._lock:
            self.meter = self.meter + usage

    def run(self, op: DomainOp[Any]) -> Any:
        return self._stack.run(op)

    def run_metered(self, op: DomainOp[Any]) -> tuple[Any, Usage]:
        """Return `(result, usage)` without `metered`'s unwrap: the usage-in-checkpoint seam
        the durable handler drives. Folds into the telemetry `meter` and charges the `budget`
        ceiling as `metered` does; the measured trip is the handler's. `traced` composes under
        here and fires once."""
        match op:
            case AsksModel():
                if self._budget is not None and self._budget.exceeded():
                    raise BudgetExceeded(self._budget.spent.cost, self._budget.limit)
                result, usage = self._base_stack.run(op)
                self._accrue(usage)  # locked: the same meter RMW the layer's accrue uses
                if self._budget is not None:
                    self._budget.add(usage)
                return result, usage
            case CallTool():
                return self._base_stack.run(op), Usage()
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead


@runtime_checkable
class MeteredDomain(Protocol):
    """Anything with `run_metered(op) -> (result, usage)` — the metered domain-seam base,
    structural so callers needn't name a concrete class (mirrors `layers.Interpreter`).

    `runtime_checkable` because `metered_call` asks this question AT RUNTIME, once per op, to
    decide whether the measured trip applies. `isinstance(domain, MeteredDomain)` is a `getattr`
    sniff with the role named.

    The usual `runtime_checkable` caveat holds and is fine here: it checks member PRESENCE, not
    signature. A domain with a wrong-shaped `run_metered` still passes; `ty` catches the shape,
    statically, at the call site."""

    def run_metered(self, op: DomainOp[Any]) -> tuple[Any, Usage]: ...


# A `serve` element: a domain-seam middleware (`@domain_layer`) wrapping ONE metered call,
# the call-axis sibling of a `Policy`. A service *does* (retry/cache/meter/trace) and never
# parks; it threads the `(result, usage)` it forwards. Structurally a `DomainLayer`, so the
# alias IS the named role: it types `serve`'s parameter and appears in its errors.
type Service = DomainLayer[Any]


@dataclass
class _ServedDomain:
    services: tuple[Service, ...]
    base: MeteredDomain

    def run_metered(self, op: DomainOp[Any]) -> tuple[Any, Usage]:
        # First come, first served: leftmost service is outermost. drive_through pumps the
        # stack down to the base call and threads `(result, usage)` back up intact (a service
        # that only forwards/retries is value-agnostic); a downstream raise re-enters each
        # service's `yield` via `.throw`, which is what makes `retry_domain` work.
        out = drive_through(self.services, op, self.base.run_metered)
        if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], Usage)):
            # Shape-guard the pipe: a service that unwraps to a bare value (e.g. a
            # `compose_domain`-style `metered` returning just the result) would let `result, usage`
            # tuple-unpack a 2-char string into `"o", "k"` — a silent injectivity corruption. (It
            # cannot catch a service whose *result* is itself an (Any, Usage) pair — inherent to a
            # shape check on a non-injective encoding.)
            raise TypeError(
                f"a serve service must thread the metered (result, usage) pair; the stack "
                f"returned {out!r}. Unwrapping to a bare value corrupts the pipe."
            )
        return out

    def run(self, op: DomainOp[Any]) -> Any:
        # A serve(...) stack is METERED-ONLY (run_metered, no plain run). DurableHandler calls
        # domain.run(op) for a CallTool; a serve stack cannot serve one. Fail LEGIBLY here,
        # before a mid-checkpoint AttributeError.
        # Defining `.run` means a serve stack satisfies `layers.Interpreter` structurally while
        # it always raises, so a code path is chosen by `run_metered`, as `metered_call` does,
        # and never by the presence of `.run`.
        raise TypeError(
            "a serve(...) stack is metered-only: it backs the measured path (run_metered) and "
            "cannot run a CallTool. Put serve inside the metered domain, not around the whole "
            "handler."
        )


def serve(*services: Service, base: MeteredDomain) -> MeteredDomain:
    """Compose call-execution Services around a metered domain: the `govern` sibling for the
    domain seam. One trampoline (`drive_through`), and it never parks. Preserves `run_metered`,
    which `compose_domain` does not: a `.run`-only wrapper would hide the metered arm and
    silently disable the budget trip (`metered_call` probes `run_metered` by `getattr`).

    Order IS the semantics (first come, first served): `serve(retry, cache, meter)` retries the
    whole cache+call, and a cache hit skips the meter; put `meter` inside `cache` for free hits,
    outside to bill saved cost.

    `serve(base=d)` with no services returns `d` UNTOUCHED: the zero-service form inherits the
    base's contract *unguarded* (no shape guard, no `.run` fence). Guarding a bad bare base is
    the consumer's job (`measured_drive` unpacks `(result, usage)`).

    Rejects an `@op_layer` service at composition time, for consumers outside the `ty` gate: a
    park-capable op layer belongs to `govern`."""
    for svc in services:
        if getattr(svc, "__effective_layer__", "domain") == "op":
            name = getattr(svc, "__qualname__", None) or getattr(svc, "__name__", repr(svc))
            raise TypeError(
                f"serve composes DOMAIN-seam services, but {name!r} is an @op_layer (it may "
                f"park/suspend). Did you mean `retry_domain` (the domain twin of `retry`)? "
                f"The rule: park ⇒ govern; transform-a-call ⇒ serve."
            )
    return _ServedDomain(tuple(services), base) if services else base


__all__ = [
    "BudgetExceeded",
    "CostBudget",
    "Dollars",
    "LLMCall",
    "MeteredDomain",
    "MeteredInterpreter",
    "Service",
    "ToolRunner",
    "Usage",
    "metered",
    "serve",
]
