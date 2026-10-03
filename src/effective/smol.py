"""`smol`: the two loops of `examples/smol_agent.py`, durable and interruptible.

The outer loop waits for a user line and the inner loop runs turns until the model answers, over
one append-only transcript. Each user turn is one `respawn` generation, so a replay walks only the
turn in progress, and the transcript is the carry. The wait for a line is a park named
`user:{n}`, answered by whatever face the user types into, and the turn's own ops and parks run
under `turn:{n}`, so a park inside one turn never shares a name with the same park in another.

| the user sends          | the loop                                                        |
|-------------------------|-----------------------------------------------------------------|
| a line                  | runs turns over the transcript until the model answers          |
| Esc, during a turn      | ends the turn, marking the transcript `[escaped]`               |
| an empty line, `/quit`  | ends the conversation with the transcript                       |

Esc reaches the loop two ways. A poll between steps answers `Escape` (`effective.interrupts`), and
an op running when the Esc arrives completes as `Cancelled` (`effective.cancel`). Either way the
run keeps what already ran and drops what the model chose but the loop had not run.
"""

from typing import Any

from pydantic import BaseModel

from effective.api import Effect, await_event, scoped
from effective.combinators import Again, Chain, Done, Turn, respawn
from effective.interrupts import Interrupt
from effective.keys import Index, Run, compose_key
from effective.react import Act, Decide, Guard, run_turns

QUIT = frozenset({"", "/quit"})
"""The lines, stripped, that end the conversation."""


class Conversation(BaseModel):
    """The transcript, which is everything a generation hands the next."""

    messages: list[dict[str, Any]] = []


class UserLine(BaseModel):
    """A line the user typed, delivered as the `user:{n}` park's answer."""

    text: str


def smol(
    chain: Chain[Conversation],
    *,
    decide: Decide | None = None,
    act: Act | None = None,
    guard: Guard | None = None,
    interrupt: Interrupt | None = None,
    max_iters: int = 6,
) -> Effect[Conversation]:
    """Converse until the user quits, one generation of `chain` per user turn.

    Scoped under the conversation's run, so two conversations in one engine park on different
    names. Nothing may run before this call that must run once: a generation re-enters the
    program from the top."""

    def user_turn(
        conversation: Conversation, turn: Turn
    ) -> Effect[Again[Conversation] | Done[Conversation]]:
        line = yield from await_event(compose_key(t"user:{Index(turn.generation)}"), UserLine)
        if line.text.strip() in QUIT:
            return Done(conversation)
        messages = [*conversation.messages, {"role": "user", "content": line.text}]
        ran = yield from scoped(
            compose_key(t"turn:{Index(turn.generation)}"),
            lambda: run_turns(
                messages, max_iters, decide=decide, act=act, guard=guard, interrupt=interrupt
            ),
        )
        carried = Conversation(messages=ran.messages)
        return Done(carried) if turn.final else Again(carried)

    return (
        yield from scoped(
            compose_key(t"smol:{Run(chain.run_id)}"), lambda: respawn(user_turn, chain)
        )
    )
