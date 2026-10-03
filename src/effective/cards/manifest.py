"""``CardSpec`` → an inspectable manifest.

The manifest is the agent-customization-safety surface: a compact, diffable
description of what a card *exposes* — its fields, whether it carries a chart,
and the typed actions (with their risk tier) it can emit. An AI-proposed card
change can be reviewed against the manifest delta rather than the rendered HTML.
This is the view-axis analogue of the channel manifest.
"""

from typing import Any

from effective.cards.spec import CardSpec


def manifest(spec: CardSpec) -> dict[str, Any]:
    return {
        "card_id": spec.card_id,
        "title": spec.title,
        "badge": spec.badge.tone if spec.badge is not None else None,
        "fields": [cell.label for cell in spec.metrics],
        "has_chart": spec.chart is not None,
        "actions": [{"cmd": a.cmd, "target": a.target, "risk": a.risk} for a in spec.actions],
        "slots": [s.slot_id for s in spec.extras],
    }
