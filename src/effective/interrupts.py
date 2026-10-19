"""The interrupt channel a loop polls: steer a turn, or escape it.

A poll is a recorded `CallTool` on `INTERRUPT_TOOL`, keyed `step;tool;interrupt:{phase}` inside the
turn's `d:{i}` frame, so whichever of a keypress and an op wins the race is a checkpoint, and a
replay re-serves it. The domain answers the poll from the channel a face writes into.

| signal     | what the face sent                 | what the loop does                           |
|------------|------------------------------------|----------------------------------------------|
| `Quiet`    | nothing                            | continues                                    |
| `Redirect` | a line of steering, delivered once | the text is the next observation; turn spent |
| `Escape`   | Esc, held until the next user line | drops the pending action and ends the turn   |

`Interrupted` is the wire shape a domain answers with; `signal_of` is where it becomes a value.
"""

from collections.abc import Callable, Set
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from effective.api import Effect, step
from effective.domain import INTERRUPT_TOOL, CallTool
from effective.keys import Key, Name, compose_key

type Phase = Literal["pre", "post", "act"]
"""Where a turn polls: before the decide, after it, and after the action returns."""

BETWEEN_TURNS: frozenset[Phase] = frozenset({"pre", "post"})
EVERY_PHASE: frozenset[Phase] = frozenset({"pre", "post", "act"})


@dataclass(frozen=True, slots=True)
class Quiet:
    """Nothing is pending."""


@dataclass(frozen=True, slots=True)
class Redirect:
    """Steering the face sent while the turn ran."""

    text: str


@dataclass(frozen=True, slots=True)
class Escape:
    """The user abandoned this turn. `name` is the poll's authored name, `None` for a poll
    that records nothing."""

    name: str | None = None


type Signal = Quiet | Redirect | Escape

type Interrupt = Callable[[int, Phase], Effect[Signal]]
"""A poll, given the turn's depth and the phase."""


class Interrupted(BaseModel):
    """What a domain answers a poll with. `escape` outranks a `redirect` sent in the same poll."""

    redirect: str | None = None
    escape: bool = False


def signal_of(answer: Interrupted, name: str | None = None) -> Signal:
    """The signal a poll named `name` answered."""
    if answer.escape:
        return Escape(name)
    return Quiet() if answer.redirect is None else Redirect(answer.redirect)


def interrupt_key(phase: Phase) -> Key:
    """The name of the step that polls for an interrupt at `phase`."""
    return compose_key(t"interrupt:{Name(phase)}")


def tool_interrupt(phases: Set[Phase] = BETWEEN_TURNS) -> Interrupt:
    """A non-blocking poll, recorded as a `CallTool` per turn and phase in `phases`.

    A phase outside `phases` answers `Quiet` and yields nothing, so a loop that polls at every
    phase records only the ones this poll subscribes to.

    The step is named `interrupt:{phase}`, a minter's key, so no author's tool named `interrupt`
    shares it; the domain dispatches on `INTERRUPT_TOOL` and reads the turn and the phase from
    the args."""

    def poll(turn: int, phase: Phase) -> Effect[Signal]:
        if phase not in phases:
            return Quiet()
        name = interrupt_key(phase)
        answer = yield from step(
            name,
            CallTool(
                name=INTERRUPT_TOOL, args={"turn": turn, "phase": phase}, result_schema=Interrupted
            ),
        )
        return signal_of(answer, name.stored())

    return poll
