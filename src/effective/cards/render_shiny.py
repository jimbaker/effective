"""``CardSpec`` → a reactive ``htmltools.Tag`` tree, the Shiny lowering.

The same ``CardSpec`` that ``render_html`` emits as a *string*, lowered into an
htmltools tag tree so Shiny owns the diff/patch: a changed metric becomes a
single-node patch, and chart nodes survive re-render.

``htmltools`` is Shiny's small standalone *tag* library. It is the one framework
import allowed in ``effective/cards/``, and only here; the spec and the other
projectors stay framework-free.

| piece  | lowers to                                                        |
|--------|------------------------------------------------------------------|
| chart  | nothing here: the app renders it as a widget Shiny owns          |
| action | a live Shiny action button, its id from ``action_id``            |
| slot   | an ``output_ui`` div the author fills with a normal Shiny output |
"""

from htmltools import Tag, tags

from effective.cards.spec import Action, Badge, CardSpec, Cell, Slot


def _badge(b: Badge) -> Tag:
    return tags.span(b.text, class_=f"ev-badge ev-badge--{b.tone}")


def _metric(c: Cell) -> Tag:
    return tags.div(
        tags.dt(c.label),
        tags.dd(c.render_value()),
        class_=f"ev-metric ev-metric--{c.kind}",
    )


def action_id(card_id: str, action: Action) -> str:
    """The deterministic Shiny input id for an action. Includes
    ``card_id`` so two cards with the same cmd/target don't collide; the app shell's
    one dispatcher reconstructs it to map a click back to a ledger event."""
    return f"act__{card_id}__{action.cmd}__{action.target}"


def _action(a: Action, card_id: str) -> Tag:
    # a real Shiny action button — the `action-button` class IS the input binding,
    # so no `import shiny` is needed (htmltools + the known class is enough).
    return tags.button(
        a.label,
        id=action_id(card_id, a),
        type="button",
        class_=f"action-button ev-action ev-action--{a.risk}",
        **{"data-cmd": a.cmd, "data-target": a.target},
    )


def _slot(s: Slot) -> Tag:
    # the output_ui binding: Shiny fills this div by id with the author's normal
    # @render.ui / @render_widget. No `import shiny` — the class is the binding hook.
    return tags.div(id=s.slot_id, class_="shiny-html-output ev-slot")


def render_shiny(spec: CardSpec) -> Tag:
    """Render ``spec`` to an htmltools tag tree."""
    header: list[Tag] = [tags.h3(spec.title, class_="ev-card__title")]
    if spec.badge is not None:
        header.append(_badge(spec.badge))

    children: list[Tag] = [tags.header(*header, class_="ev-card__header")]
    if spec.question:
        children.append(tags.p(spec.question, class_="ev-card__question"))
    if spec.metrics:
        children.append(tags.dl(*[_metric(c) for c in spec.metrics], class_="ev-metrics"))
    # NB: spec.chart is NOT rendered here. On the Shiny surface charts are app-level
    # @render_altair widgets Shiny owns, with no data-vega-spec and no observer.
    # The static render_html lowering keeps the self-contained chart div.
    if spec.summary:
        children.append(tags.p(spec.summary, class_="ev-card__summary"))
    if spec.actions:
        actions = [_action(a, spec.card_id) for a in spec.actions]
        children.append(tags.div(*actions, class_="ev-actions"))
    children.extend(_slot(s) for s in spec.extras)

    return tags.article(*children, class_="ev-card", **{"data-card-id": spec.card_id})
