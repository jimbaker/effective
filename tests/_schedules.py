"""A schedule, forced through the op-layer seam.

Nothing in `src/` chooses which branch of a `gather` advances (`docs/effective-design.md` §3.6),
and the seam that could is already there: an `OpLayer` blocking before its `yield op` decides, on
both handlers, because a gather branch's ops run through the branch handler's own layer stack.
This turns that seam into an instrument, so a test can assert over two named schedules rather than
over whichever interleaving the run happened to take.

**Release on COMPLETION, never on admission.** A layer that let its op through and then recorded
the turn has ordered the *entries* and left the commits to race. Neither engine preserves
admission order: blocking one branch below the admission gate lets a sibling take the store
first, measured on both as admitted `[a0,b0,a1,b1]` against stored `[b0,a0,a1,b1]`. An
admission-release mutant does redden more readily on Postgres, whose `PostgresLedger` pools its
connections, than on SQLite, whose single write lock often hands the commits back in the order
they arrived. That is a difference in how easily the mistake is caught, never a guarantee either
engine offers.

A `Turnstile` deadlocks by design when the run cannot supply its schedule, which is the correct
failure: a sequential gather cannot run branch 1 first. The wait is bounded and the timeout
reports the labels that did commit.

A turnstile's own rules, each pinned in `test_schedules.py`:

| rule                                      | what it prevents                                    |
|-------------------------------------------|-----------------------------------------------------|
| a label names one op                      | two branches sharing a turn and running together    |
| a turn ends with its op: commit, raise    | a sibling waiting out its timeout behind an op that |
| or park                                   | will never commit                                   |
| the layer sits above what it orders       | an order among ops a gate below has already decided |
| `check`: the order used up, two siblings  | a schedule that ordered nothing a sibling could     |
|                                           | reorder                                             |

One turnstile serves one run to its end: a replay, or a resume after a park, re-admits the ops it
already released."""

import threading
from collections.abc import Callable, Collection, Generator, Sequence
from itertools import combinations
from typing import Any

from effective.keys.frame import branch_frames
from effective.layers import current_placement, op_layer
from effective.ops import AppendLedgerRow, Step, WorkflowOp

type Label = Callable[[WorkflowOp], str | None]
"""What an op is called for ordering purposes, or `None` to let it through unordered."""


def ledger_path(op: WorkflowOp) -> str | None:
    """Order a ledger append by its row's `path` field."""
    return op.row.get("path") if isinstance(op, AppendLedgerRow) else None


def step_name(op: WorkflowOp) -> str | None:
    """Order a step by the name its author gave it."""
    return str(op.name) if isinstance(op, Step) else None


def branch_of(placement: str) -> tuple[str, ...]:
    """The frames of the innermost branch `placement` runs in, empty on the main line.

    A gather branch or a race branch: a turnstile orders siblings, and a race is where the order
    decides the answer."""
    return branch_frames(placement)


def nests(p: tuple[str, ...], q: tuple[str, ...]) -> bool:
    """Does one branch run inside the other, so program order already orders their ops?"""
    return p[: len(q)] == q or q[: len(p)] == p


class Turnstile:
    """Commits the ops it recognizes in a named order, whichever branch reaches them first."""

    def __init__(self, order: Sequence[str], label: Label, timeout: float = 20.0) -> None:
        if repeated := sorted({name for name in order if list(order).count(name) > 1}):
            raise ValueError(f"{repeated[0]!r} appears twice: a label names one op")
        self.order = list(order)
        self.label = label
        self.timeout = timeout
        self.ended: list[str] = []
        self.outcomes: list[tuple[str, str]] = []
        self.placements: dict[str, str | None] = {}
        self.threads: set[int] = set()
        self._arrived: set[str] = set()
        self._turn = threading.Condition()

    def layer(self) -> Any:
        @op_layer
        def release(op: WorkflowOp) -> Generator[WorkflowOp, Any, Any]:
            name = self.label(op)
            if name not in self.order:
                return (yield op)

            def its_turn() -> bool:
                filled = len(self.ended)
                return filled < len(self.order) and self.order[filled] == name

            with self._turn:
                assert name not in self._arrived, f"{name!r} names a second op"
                self._arrived.add(name)
                reached = self._turn.wait_for(its_turn, timeout=self.timeout)
            assert reached, f"{name!r} waited out the gather; ended {self.ended}"
            placement = current_placement()
            outcome = "closed"
            try:
                result = yield op
                outcome = "committed"
                return result
            except GeneratorExit:
                raise
            except BaseException as raised:
                outcome = type(raised).__name__
                raise
            finally:
                with self._turn:
                    self.threads.add(threading.get_ident())
                    self.placements[name] = None if placement is None else placement.stored()
                    self.ended.append(name)
                    self.outcomes.append((name, outcome))
                    self._turn.notify_all()

        return release

    def kept_its_schedule(self) -> bool:
        return self.ended == self.order

    def check(self, stopped: Collection[str] = ()) -> None:
        """Assert the run took its order, over placements spanning two sibling branches.

        | the order                      | verdict                    |
        |--------------------------------|----------------------------|
        | taken whole                    | passes                     |
        | missing an op `stopped` names  | passes                     |
        | missing an op it does not name | fails                      |
        | taken out of turn              | fails, whatever `stopped`  |

        A race stops its losers, so the ops after its choice never arrive and a cancelling shape
        cannot use its order up. `stopped` is read off the run's own record, never counted by the
        caller."""
        missing = [name for name in self.order if name not in self.ended]
        undeclared = [name for name in missing if name not in stopped]
        assert not undeclared, (
            f"the run would not take {self.order}: {self.ended}; nothing allows {undeclared}"
        )
        arrived = [name for name in self.order if name not in missing]
        assert self.ended == arrived, f"the run took {self.ended} out of the order {self.order}"
        placed = {name: at for name, at in self.placements.items() if at is not None}
        unplaced = set(arrived) - placed.keys()
        assert not unplaced, f"no placement for {unplaced}"
        branches = {branch_of(at) for at in placed.values()}
        assert any(not nests(p, q) for p, q in combinations(branches, 2)), (
            f"the order spans no two sibling branches: {sorted(branches)}"
        )
