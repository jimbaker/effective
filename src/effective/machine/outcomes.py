"""The outcome vocabulary a machine walks — generic over the states it walks between.

Nothing here names a state: `S` is whatever
`StrEnum` an embodiment declares, and the three terminal shapes plus the reason for parking are
the same whether the machine is editing code, revising a document, or driving a process nobody
has written yet. The routers that decide WHICH outcome a verdict produces stay with the
embodiment, because an edge table is a design rather than a mechanism.
"""

from dataclasses import dataclass
from enum import StrEnum


class ParkReason(StrEnum):
    """Why the machine stopped for a human, as the outcome row records it.

    | reason       | who stopped it              | the work it leaves                     |
    |--------------|-----------------------------|----------------------------------------|
    | `rejected`   | a human, at the plan        | the tree as it stood before the plan   |
    | `exhausted`  | the visit budget            | the tree the last visit left           |
    | `broken-env` | the environment             | a tree the predicate has yet to judge  |
    | `refused`    | a governed boundary         | the tree before the refused visit      |

    A transition returns the first three. The walk mints `refused` itself, when a visit raises a
    refusal, and `RunRefused` carries that run to its caller."""

    EXHAUSTED = "exhausted"
    REJECTED = "rejected"
    BROKEN_ENV = "broken-env"
    """The environment failed to run the predicate, so the run waits for an operator.

    A re-draft answers bad work, and this is an infrastructure fault: the prose machine's VERIFY
    reads gates whose container can vanish between runs, which is a named CLAUDE.md hazard."""
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class Advance[S: StrEnum]:
    """Run `to` next. The self-edge (`Advance(State.TEST)` out of TEST) is a legitimate arm, not
    a missing one — `GREEN_ALREADY` means the test proves nothing, so write a better one."""

    to: S


@dataclass(frozen=True, slots=True)
class Finish:
    """The work is done. Reached from EXACTLY ONE cell — `(REVIEW, APPROVED)` — which the graph
    gate pins, because a second route to `Finish` is how a machine ships unreviewed work."""


@dataclass(frozen=True, slots=True)
class Park[S: StrEnum]:
    """Stop and wait for a human. Reachable from every state under `Exhausted`, which is what
    makes exhaustion a bounded park rather than a silent loop or a discarded run."""

    at: S
    why: ParkReason


type Outcome[S: StrEnum] = Advance[S] | Finish | Park[S]


class VerdictOutOfDomain(ValueError):
    """A judge for one state returned another state's verdict — one of the 150 cells the
    dependent sum excludes. Refused LOUDLY and immediately: the incumbent's defect was exactly
    that an unrecognised outcome silently took an edge, and the edge it took appended to the
    canonical record on every visit."""


@dataclass(frozen=True, slots=True)
class Exhausted[S: StrEnum]:
    """The visits ran out. **Minted by the interpreter, never by a judge** (see the module
    docstring), in `state` at `level`, the final visit, which a grantor may move past the
    budget."""

    state: S
    level: int
