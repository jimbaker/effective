"""The `sla-cap` case twice: once for a control the firm holds, once for one it lacks.

    uv run python -m examples.agent_firm.gaps

| control                          | the run                                                     |
|----------------------------------|-------------------------------------------------------------|
| an approver holds the role       | the reviewer's approval parks again; the controller's posts |
| acceptance apart from completion | a $3,000 critique citing the contract posts once approved   |

The contract caps the credit at $1,000, and the second run completes with nothing checking its
credit against the cap. The score after each run is computed outside the firm, from the books.
"""

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from effective import Usage
from effective.keys import Segment

from .__main__ import CONTROLLER, CRITIQUES, MAIN, admits, run, scripted_firm
from .cases import SLA_CAP
from .engage import CritiqueWire
from .score import scored
from .world import scripted

REVIEWER = Segment("reviewer")
"""A ruler who does not hold the approving role."""

UNCAPPED = CRITIQUES["sla-cap"].model_copy(
    update={"credit": Decimal(3000), "reason": "the outage credit the customer took"}
)
"""A critique citing the held contract, and recommending the whole deduction."""


def told(args: Mapping[str, Any]) -> dict[str, Any]:
    return {"ok": True}


def settled(critique: CritiqueWire, rulers: Sequence[Segment]) -> None:
    """Resolve `sla-cap` with `critique`, answering each approval as the next of `rulers`."""
    world = replace(
        scripted_firm(SLA_CAP, scripted(), told), critic=lambda op: (critique, Usage())
    )
    with tempfile.TemporaryDirectory() as directory:
        result = run(admits(SLA_CAP, MAIN), world, MAIN.books, Path(directory) / "firm.db", rulers)
    print("result:   ", result)
    disposition = scored(SLA_CAP, world.erp)
    print("scored:   ", disposition.credit, "credited of", disposition.gold, "supported")


def main() -> None:
    settled(CRITIQUES["sla-cap"], (REVIEWER, CONTROLLER))
    settled(UNCAPPED, (CONTROLLER,))


if __name__ == "__main__":
    main()
