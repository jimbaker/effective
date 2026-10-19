"""The factor account of one resolution: a projection over what the run was admitted with, the
rows it wrote to the ledger, and its tape, joined by the run.

| factor     | read from                                                              |
|------------|------------------------------------------------------------------------|
| assignment | the admission's manifest: the arm, the round, whether a critic runs    |
| H          | the ledger's ask and ruling rows: how many parks, and who ruled each   |
| A and C    | the tape's model steps, and their recorded usage under the v1 contract |
| K          | the snapshot the manifest pins                                         |

Each factor comes from the bookkeeper that holds it, so the account copies none of them into
another. Y is not here: the books are scored apart from the firm (`score`).
"""

from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from effective import Contract
from effective.graphview import bare_name
from effective.handlers.base import step_key
from effective.keys import Key

from .ar import Round
from .knowledge import Pinned
from .manifest import Manifest
from .resolve import ASKED, RULED

MODEL_STEPS = frozenset(step_key(name).stored() for name in ("explain", "engage"))
"""The steps at which the firm asks a model: its proposer, and its critic."""


class Account(BaseModel):
    run_id: str
    arm: str
    case: str
    """The admitted case's digest."""
    round: Round
    knowledge: Pinned
    critic: bool
    asks: int
    rulers: tuple[str | None, ...]
    model_calls: int
    spend: Decimal | None
    """Dollars the committed model calls recorded; `None` for a run under the v0 contract, whose
    tape records no usage."""


def account(
    params: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], tape: Mapping[Key, Any]
) -> Account:
    """The account of the run admitted with `params`, from its ledger `rows` and its `tape`."""
    manifest = Manifest.admitted(params)
    calls = [state for key, state in tape.items() if bare_name(key.stored()) in MODEL_STEPS]
    metered = Contract.from_params(params) is Contract.V1
    spend = (
        sum((Decimal(str(state["usage"]["cost"])) for state in calls), Decimal(0))
        if metered
        else None
    )
    return Account(
        run_id=params["run_id"],
        arm=manifest.arm,
        case=manifest.case_digest,
        round=manifest.round,
        knowledge=manifest.knowledge,
        critic=manifest.critic,
        asks=sum(row["kind"] == ASKED for row in rows),
        rulers=tuple(row["ruler"] for row in rows if row["kind"] == RULED),
        model_calls=len(calls),
        spend=spend,
    )


type Cell = tuple[bool, bool]
"""Whether a run held the treated knowledge, and whether a critic ran."""


def interaction(
    outcomes: Iterable[tuple[Manifest, Decimal]], control: Pinned, treated: Pinned
) -> Decimal:
    """I = Y(K₁,A₁) - Y(K₁,A₀) - Y(K₀,A₁) + Y(K₀,A₀) over one outcome per cell, each run's cell
    read from its manifest: K₀ where it pins `control` and K₁ where it pins `treated`, A₁ where a
    critic runs. A run pinning neither, and a cell with no outcome or with two, are refused."""
    cells: dict[Cell, Decimal] = {}
    for manifest, outcome in outcomes:
        if manifest.knowledge not in (control, treated):
            raise ValueError(f"a run pins {manifest.knowledge}, neither arm's snapshot")
        cell = (manifest.knowledge == treated, manifest.critic)
        if cell in cells:
            raise ValueError(f"two outcomes in the cell {cell}")
        cells[cell] = outcome
    if missing := {(k, a) for k in (False, True) for a in (False, True)} - cells.keys():
        raise ValueError(f"no outcome in the cells {sorted(missing)}")
    return cells[True, True] - cells[True, False] - cells[False, True] + cells[False, False]
