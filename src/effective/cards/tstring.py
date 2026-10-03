"""``card(t"…")``: the PEP 750 authoring face for a ``CardSpec``.

The exact sibling of ``effective.channels.render``: a function on a ``Template``
that walks the interpolations as *typed holes* and produces a typed result — here
a ``CardSpec`` instead of a ``Prompt[S]``. The static markup spans are the readable
scaffold (as prose is in ``render``); the holes carry the meaning, dispatched by
type. The result is exactly the ``CardSpec`` the declarative builder produces, so
the t-string face adds onto the IR and shares its manifest.

``CardId``/``Title``/``Question``/``Summary`` are the text holes; the rest of the
vocabulary (``Badge``/``Metric``/``Interval``/``VegaChart``/``Action``) is the IR's.
Like ``render``, every hole must be typed: a bare value is a located error.
"""

from dataclasses import dataclass
from string.templatelib import Template

from effective.cards.spec import Action, Badge, CardSpec, Cell, Interval, Metric, VegaChart


@dataclass(frozen=True)
class CardId:
    value: str


@dataclass(frozen=True)
class Title:
    value: str


@dataclass(frozen=True)
class Question:
    value: str


@dataclass(frozen=True)
class Summary:
    value: str


class CardTemplateError(Exception):
    """A card template hole was unknown, or a required hole was missing."""


def card(template: Template) -> CardSpec:
    """Walk a card ``Template`` into a ``CardSpec``, dispatching holes by type."""
    texts: dict[type, str] = {}  # CardId/Title/Question/Summary -> their value
    badge: Badge | None = None
    chart: VegaChart | None = None
    metrics: list[Cell] = []
    actions: list[Action] = []

    for interp in template.interpolations:
        match interp.value:
            case CardId() | Title() | Question() | Summary() as text_hole:
                texts[type(text_hole)] = text_hole.value
            case Badge() as b:
                badge = b
            case VegaChart() as c:
                chart = c
            case Metric() | Interval() as m:
                metrics.append(m)
            case Action() as a:
                actions.append(a)
            case other:
                raise CardTemplateError(
                    f"unexpected card hole {interp.expression!r}: {type(other).__name__} "
                    f"(every hole must be a typed card channel)"
                )

    card_id = texts.get(CardId)
    title = texts.get(Title)
    if card_id is None:
        raise CardTemplateError("card template needs a {CardId(...)} hole")
    if title is None:
        raise CardTemplateError("card template needs a {Title(...)} hole")

    return CardSpec(
        card_id=card_id,
        title=title,
        badge=badge,
        question=texts.get(Question),
        summary=texts.get(Summary),
        metrics=tuple(metrics),
        chart=chart,
        actions=tuple(actions),
    )
