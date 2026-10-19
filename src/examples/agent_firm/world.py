"""The scripted world: what the firm's ops reach, as functions, metered."""

import threading
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from effective import MeteredInterpreter, Usage
from effective.domain import AskLLM, CallTool, Judge

from .cases import CASES
from .erp.port import OPS, Erp
from .erp.scripted import ScriptedOdoo
from .knowledge import K0, K1, Knowledge


def scripted() -> Erp:
    """The ERP port on a scripted Odoo holding every case's books."""
    return ScriptedOdoo.seeded(CASES.values()).erp()


@dataclass
class World:
    """What the firm's ops reach, as functions, metered: the ERP port behind the `erp` tool, the
    snapshots of its knowledge behind the `knowledge` tool, the firm's model `llm`, an outside
    `critic` that answers every ask whose schema is in `critiques`, a judge, the `claimant` tool,
    which answers with the idempotency key minted for its call, and any other tool by name.
    `price` is the dollar cost of one call of each kind, so a run's spend is a function of what it
    asked.

    A tool honors its idempotency key, as a service taking an `Idempotency-Key` header does: the
    first answer under a key is the answer to every later call under it. Gather branches call
    from their own threads, so one lock serializes every tool call, the ERP's included, which is
    what makes the first answer the one kept; a tool that waits on another branch would deadlock.
    The models run outside the lock, so a scripted model that keeps state guards it itself."""

    erp: Erp = field(default_factory=scripted)
    llm: Callable[[AskLLM[Any]], Any] = lambda op: {}
    critic: Callable[[AskLLM[Any]], tuple[Any, Usage]] | None = None
    critiques: frozenset[type] = frozenset()
    judge: Callable[[Judge[Any]], Mapping[str, Any]] = lambda op: {}
    tools: Mapping[str, Callable[[Mapping[str, Any]], Any]] = field(default_factory=dict)
    price: Mapping[str, float] = field(
        default_factory=lambda: {"llm": 0.01, "judge": 0.002, "tool": 0.0}
    )
    knowledge: Mapping[str, Knowledge] = field(
        default_factory=lambda: {snapshot.digest: snapshot for snapshot in (K0, K1)}
    )
    """The snapshots of the firm's knowledge, by digest."""
    calls: Counter[str] = field(default_factory=Counter)
    """Each call by kind, as the world received it: a repeat answered from `answered` counts."""
    asked: Counter[str] = field(default_factory=Counter)
    """Each model ask, the firm's and the critic's, by the schema it answers."""
    answered: dict[str, Any] = field(default_factory=dict)
    """Each keyed tool call's answer, by its idempotency key."""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _model(self, op: AskLLM[Any]) -> tuple[Any, Usage]:
        with self._lock:
            self.asked[op_name(op)] += 1
        if self.critic is not None and op.response_schema in self.critiques:
            with self._lock:
                self.calls["critic"] += 1
            return self.critic(op)
        with self._lock:
            self.calls["llm"] += 1
        return self.llm(op), Usage(cost=self.price["llm"])

    def _tool(self, op: CallTool[Any]) -> Any:
        with self._lock:
            self.calls[op.name] += 1
            match op.name, op.args.get("idempotency_key"):
                case "knowledge", None:
                    return self.knowledge[op.args["digest"]].model_dump(mode="json")
                case "claimant", str(key):
                    return key
                case "erp", str(key):
                    performed = self.erp.perform(OPS.validate_python(op.args), key)
                    return performed.model_dump(mode="json")
                case name, None:
                    return self.tools[name](op.args)
                case name, str(key):
                    if key not in self.answered:
                        self.answered[key] = self.tools[name](op.args)
                    return self.answered[key]
                case unreadable:
                    raise ValueError(f"an unreadable tool call: {unreadable}")

    def _judged(self, op: Judge[Any]) -> tuple[Any, Usage]:
        with self._lock:
            self.calls["judge"] += 1
        return self.judge(op), Usage(cost=self.price["judge"])

    def domain(self) -> MeteredInterpreter:
        return MeteredInterpreter(llm=self._model, tools=self._tool, judge=self._judged)


def op_name(op: AskLLM[Any]) -> str:
    """The schema an ask answers, which is how a scripted model tells two asks apart."""
    return getattr(op.response_schema, "__name__", str(op.response_schema))
