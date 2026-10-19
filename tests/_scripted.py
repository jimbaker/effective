"""A scripted model that answers one step more than once: each answer is handed out once, in order.

`RecordingHandler` looks a canned answer up by the step's authored name, qualified by its scope
and never by its occurrence, so a plain table hands a repeated step the same answer each time.
`Sequenced` holds a list per name and consumes it; `Scripted` is the same idea as a domain, for a
`DurableHandler` that numbers each repeat (`react:turn#2`) in its keys. Each says what it has not
handed out, so a test asserts every scripted answer was asked for.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from effective.domain import DomainOp


@dataclass
class Sequenced(Mapping[str, Any]):
    """A canned-response table whose names each hold answers, consumed in order.

    A name holds an answer while its list has one left, so the step after the last answer finds
    nothing canned and the handler refuses it."""

    answers: dict[str, list[Any]]

    def __getitem__(self, name: str) -> Any:
        if not self.answers.get(name):
            raise KeyError(name)
        return self.answers[name].pop(0)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and bool(self.answers.get(name))

    def __iter__(self) -> Iterator[str]:
        return (name for name, left in self.answers.items() if left)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    @property
    def unasked(self) -> dict[str, list[Any]]:
        """The answers no step asked for."""
        return {name: left for name, left in self.answers.items() if left}


@dataclass
class Scripted:
    """A domain that answers each op it runs with the next scripted answer, and records the op."""

    answers: list[Any]
    ran: list[DomainOp[Any]] = field(default_factory=list)

    def run(self, op: DomainOp[Any]) -> Any:
        assert self.answers, f"no scripted answer left for {op!r}"
        self.ran.append(op)
        return self.answers.pop(0)
