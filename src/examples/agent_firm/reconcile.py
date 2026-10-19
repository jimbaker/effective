"""`reconcile`: hold every item to named invariants, and explain each item that breaks one.

The explanations run concurrently, one per exception, so a reconciliation of many items pays for
the slowest explanation rather than their sum. An item is matched when every invariant holds.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial

from effective.api import Effect, gather


@dataclass(frozen=True)
class Unmatched[S, X]:
    item: S
    broken: tuple[str, ...]
    explanation: X


@dataclass(frozen=True)
class Reconciled[S, X]:
    matched: tuple[S, ...]
    exceptions: tuple[Unmatched[S, X], ...]


def reconcile[S, X](
    items: Sequence[S],
    invariants: Mapping[str, Callable[[S], bool]],
    explain: Callable[[S, tuple[str, ...]], Effect[X]],
) -> Effect[Reconciled[S, X]]:
    broken = [tuple(name for name, holds in invariants.items() if not holds(i)) for i in items]
    unmatched = [(item, names) for item, names in zip(items, broken, strict=True) if names]
    explanations = yield from gather([partial(explain, item, names) for item, names in unmatched])
    return Reconciled(
        matched=tuple(item for item, names in zip(items, broken, strict=True) if not names),
        exceptions=tuple(
            Unmatched(item, names, explanation)
            for (item, names), explanation in zip(unmatched, explanations, strict=True)
        ),
    )
