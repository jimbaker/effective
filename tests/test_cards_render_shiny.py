"""render_shiny: CardSpec -> an htmltools.Tag tree.

The Shiny lowering: the same CardSpec render_html emits as a string, as a tag
tree Shiny can diff. Infra-free (htmltools only, no Shiny server).
"""

from htmltools import Tag

from effective.cards import Action, Badge, CardSpec, Interval, Metric, VegaChart, render_shiny


def test_render_shiny_emits_a_tag_tree_with_the_card_structure() -> None:
    spec = CardSpec(
        card_id="r1",
        title="Garden Shed",
        badge=Badge("approved", tone="success"),
        metrics=(Metric("Rank", 2, fmt="rank"), Interval("CI", 90, 150, fmt="integer")),
        chart=VegaChart({"mark": "area"}, title="savings"),
        actions=(Action("review", "Review", target="r1", risk="medium"),),
    )
    tag = render_shiny(spec)
    assert isinstance(tag, Tag)
    html = str(tag)
    assert 'data-card-id="r1"' in html
    assert "Garden Shed" in html
    assert "ev-badge--success" in html
    assert "#2" in html
    assert "90–150" in html  # noqa: RUF001 (en-dash range)
    assert "ev-action--medium" in html
    # a live Shiny action button (step 3): the deterministic id + the binding class
    assert 'id="act__r1__review__r1"' in html
    assert "action-button" in html
    # charts are NOT in the tag tree on the Shiny surface — they're app @render_altair
    # widgets (the static render_html lowering keeps the self-contained chart div)
    assert "data-vega-spec=" not in html


def test_slot_lowers_to_a_shiny_output_region() -> None:
    from effective.cards import Slot

    spec = CardSpec(card_id="r1", title="T", extras=(Slot("usage_table", fallback="open in app"),))
    html = str(render_shiny(spec))
    assert 'id="usage_table"' in html
    assert "shiny-html-output" in html  # the output_ui binding the author fills


def test_action_id_is_deterministic_and_includes_card_id() -> None:
    from effective.cards import action_id

    a = Action("review_schedule", "Review", target="r1")
    assert action_id("r1", a) == "act__r1__review_schedule__r1"


def test_render_shiny_parity_classes_with_render_html() -> None:
    from effective.cards import render_html

    spec = CardSpec(card_id="x", title="T", metrics=(Metric("Rank", 1, fmt="rank"),))
    shiny_html, stdlib_html = str(render_shiny(spec)), render_html(spec)
    for cls in ("ev-card", "ev-card__title", "ev-metrics", "ev-metric"):
        assert cls in shiny_html
        assert cls in stdlib_html
