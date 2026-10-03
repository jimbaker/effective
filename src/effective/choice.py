"""What a race decides, and how each of its branches ended.

A race reads its branches' completions in batches. A batch is every branch whose result is
available when the race wakes, ranked by branch index, and `decide` is the one rule every
interpreter applies to it:

| after the batch                         | the choice                                        |
|-----------------------------------------|---------------------------------------------------|
| `s >= k`                                | winners: the earlier successes, then the lowest   |
|                                         | indices of this batch's, until there are `k`      |
| the deadline has arrived                | timeout                                           |
| `s + u < k`                             | impossible                                        |
| otherwise                               | undecided; the race waits for the next batch      |

`s` counts successes so far and `u` the branches still running, and the rows are tried in that
order: enough winners answer a race that also reached its deadline. A success counts toward `s`
only if it landed strictly before the deadline, so a branch that returns at the instant itself is
a loss. The choice is saved before any loser is told to stop, and whatever the store holds is the
choice.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, assert_never

from effective.ops import Unretryable


@dataclass(frozen=True)
class Won[T]:
    """A winner: its branch index and the value it returned."""

    index: int
    value: T


@dataclass(frozen=True)
class Unchosen[T]:
    """A branch that returned a value the choice did not take."""

    index: int
    value: T


@dataclass(frozen=True)
class Refusal:
    """A branch that ended in a refusal, which counts as a loss."""

    index: int
    reason: str


@dataclass(frozen=True)
class Stopped:
    """A branch that did not end on its own: a loser stopped at an op admission once the choice
    was saved, or a branch of a quorum of none, which never starts."""

    index: int


@dataclass(frozen=True)
class Raised:
    """A loser whose op raised a programming error after the choice was saved."""

    index: int
    error: str


type Ending[T] = Won[T] | Unchosen[T] | Refusal | Stopped | Raised


@dataclass(frozen=True)
class Chosen[T]:
    """A race that found its `k` winners, listed in branch-index order."""

    winners: tuple[Won[T], ...]
    endings: tuple[Ending[T], ...]


@dataclass(frozen=True)
class TimedOut[T]:
    """A race whose deadline arrived before `k` branches had succeeded."""

    endings: tuple[Ending[T], ...]


@dataclass(frozen=True)
class Impossible[T]:
    """A race whose successes and unresolved branches together fall short of `k`."""

    endings: tuple[Ending[T], ...]


type Answer[T] = Chosen[T] | TimedOut[T] | Impossible[T]


@dataclass(frozen=True)
class Choice:
    """The choice as it is stored: its kind, the winners' indices, and the batch that decided it.

    The batch is kept to explain the choice. Replay serves the choice and never the batch."""

    kind: Literal["winners", "timeout", "impossible"]
    winners: tuple[int, ...]
    batch: tuple[int, ...]

    def stored(self) -> dict[str, Any]:
        return {"kind": self.kind, "winners": list(self.winners), "batch": list(self.batch)}

    @classmethod
    def from_stored(cls, raw: dict[str, Any]) -> Choice:
        match raw:
            case {
                "kind": "winners" | "timeout" | "impossible" as kind,
                "winners": list(w),
                "batch": list(b),
            }:
                return cls(kind, tuple(w), tuple(b))
        raise ValueError(f"not a stored choice: {raw!r}")

    def loses(self, index: int) -> bool:
        """Whether branch `index` is to stop at its next admission the record cannot serve."""
        return index not in self.winners


def decide(
    want: int,
    earlier: Iterable[int],
    succeeded: Iterable[int],
    failed: Iterable[int],
    running: int,
    expired: bool = False,
) -> Choice | None:
    """The choice after one batch, or `None` while the race must wait for another.

    `earlier` are the successes of batches already read; `succeeded` and `failed` partition this
    batch; `running` counts the branches in no batch yet. `expired` says the deadline has arrived,
    and a race without one never passes it."""
    earlier, batch_wins = sorted(earlier), sorted(succeeded)
    batch = tuple(sorted((*batch_wins, *failed)))
    if len(earlier) + len(batch_wins) >= want:
        winners = tuple(sorted((*earlier, *batch_wins[: want - len(earlier)])))
        return Choice("winners", winners, batch)
    if expired:
        return Choice("timeout", (), batch)
    if len(earlier) + len(batch_wins) + running < want:
        return Choice("impossible", (), batch)
    return None


class EndingLost(Unretryable):
    """A stored ending names a value its branch did not replay to.

    A loser's value can rest on an op its record does not hold, such as a refusal it caught
    after the choice: the retry stops the loser at that op, and nothing on record carries the
    value. A stored copy would come back as JSON, a different type than the one returned, so
    the race fails here instead."""


def stored_endings(
    endings: Iterable[Ending[Any]], inputs: Sequence[str | None] | None = None
) -> list[dict[str, Any]]:
    """The endings as a checkpoint holds them: each branch's kind, with a refusal's reason or an
    error's text. A value is not stored: a branch replays to its own. With `inputs`, a value
    ending also carries the digest of what its branch was handed, which a retry must reproduce."""
    out: list[dict[str, Any]] = []
    for i, ending in enumerate(endings):
        witness = {} if inputs is None else {"inputs_sha256": inputs[i]}
        match ending:
            case Won():
                out.append({"ending": "won"} | witness)
            case Unchosen():
                out.append({"ending": "unchosen"} | witness)
            case Refusal(reason=reason):
                out.append({"ending": "refusal", "reason": reason})
            case Stopped():
                out.append({"ending": "stopped"})
            case Raised(error=error):
                out.append({"ending": "raised", "error": error})
            case unreachable:
                assert_never(unreachable)
    return out


def endings_from_stored(
    stored: list[dict[str, Any]], values: dict[int, Any]
) -> tuple[Ending[Any], ...]:
    """Endings rebuilt from a checkpoint, each value taken from the branch that replayed to it."""
    out: list[Ending[Any]] = []
    for index, raw in enumerate(stored):
        match raw:
            case {"ending": "won" | "unchosen"} if index not in values:
                raise EndingLost(
                    f"branch {index} ended with a value, and this attempt stopped it before it "
                    "returned one: its value rests on an op its record does not hold"
                )
            case {"ending": "won"}:
                out.append(Won(index, values[index]))
            case {"ending": "unchosen"}:
                out.append(Unchosen(index, values[index]))
            case {"ending": "refusal", "reason": str(reason)}:
                out.append(Refusal(index, reason))
            case {"ending": "stopped"}:
                out.append(Stopped(index))
            case {"ending": "raised", "error": str(error)}:
                out.append(Raised(index, error))
            case _:
                raise ValueError(f"not a stored ending: {raw!r}")
    return tuple(out)


def answer[T](choice: Choice, endings: tuple[Ending[T], ...]) -> Answer[T]:
    """The value the race returns: its winners in branch-index order, or its impossibility."""
    match choice.kind:
        case "winners":
            winners = tuple(e for e in endings if isinstance(e, Won))
            return Chosen(winners, endings)
        case "timeout":
            return TimedOut(endings)
        case "impossible":
            return Impossible(endings)
        case unreachable:
            assert_never(unreachable)
