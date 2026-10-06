"""ReAct as a generator: an agent loop that performs no I/O, driven by a script.

Run it with `uv run python examples/react_toy.py`. The README's state machine section shows
`react`; `hooks_as_layers.py` is the same idea in Effective.
"""

from collections.abc import Callable, Generator
from dataclasses import dataclass


@dataclass(frozen=True)
class Reason:
    """Ask the model to choose an action, given the goal and the last observation."""

    goal: str
    observation: str | None


@dataclass(frozen=True)
class Act:
    """Ask the world to carry out an action, and observe what comes back."""

    action: str


type Request = Reason | Act


def react(goal: str) -> Generator[Request, str]:
    observation = None
    while True:
        action = yield Reason(goal, observation)
        observation = yield Act(action)


def scripted(request: Request) -> str:
    match request:
        case Reason(observation=None):
            return "text the painter a door code for 13:00"
        case Reason():
            return "the painter has a code"
        case Act():
            return "sent a door code for 13:00 to 14:00"


def drive(loop: Generator[Request, str], answer: Callable[[Request], str], steps: int) -> None:
    """The only code that calls `send`: each answer becomes the value of the paused `yield`."""
    request = next(loop)
    for _ in range(steps):
        reply = answer(request)
        print(f"{request} -> {reply!r}")
        request = loop.send(reply)


if __name__ == "__main__":
    drive(react("let the painter in at 13:00"), scripted, steps=3)
