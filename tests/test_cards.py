"""Card IR, the `CardSpec` view axis: formatting, renderers, manifest.

Infra-free (covered by `just test-core`). The fixture is a synthetic
thermostat-schedule card built from plain values, with no domain import.
"""

from _html import classes, elements, json_attr, one

from effective.cards import (
    Action,
    Badge,
    CardSpec,
    Interval,
    Metric,
    VegaChart,
    format_value,
    manifest,
    render_html,
    render_markdown,
)

# Rendered ranges use a typographic en-dash (see Interval.render_value); RUF001
# flags the literal as hyphen-confusable, so pin the expected text once.
RANGE = "90–150"  # noqa: RUF001


def _card() -> CardSpec:
    return CardSpec(
        card_id="room-3",
        title="Living Room Thermostat",
        badge=Badge("schedule change", tone="warn"),
        question="Should this room be pre-heated?",
        summary="High expected savings, low comfort risk.",
        metrics=(
            Metric("Expected savings", 120, fmt="integer", unit="kWh/mo"),
            Interval("95% CI", 90, 150, fmt="integer", unit="kWh/mo"),
            Metric("Rank", 7, fmt="rank"),
        ),
        chart=VegaChart({"mark": "area", "data": {"values": []}}, title="Cumulative savings"),
        actions=(Action("review", "Review schedule", target="room-3", risk="medium"),),
    )


def test_format_value_styles() -> None:
    assert format_value(1234, "integer") == "1,234"
    assert format_value(42.18, "currency") == "$42.18"
    assert format_value(0.935, "percent") == "93.5%"
    assert format_value(7, "rank") == "#7"
    assert format_value("n/a", "plain") == "n/a"
    # non-numeric values degrade gracefully under numeric formats
    assert format_value("n/a", "currency") == "n/a"


def test_metric_and_interval_render_value() -> None:
    assert Metric("Savings", 120, fmt="integer", unit="kWh/mo").render_value() == "120 kWh/mo"
    assert Interval("CI", 90, 150, fmt="integer").render_value() == RANGE


def test_render_html_contains_structure() -> None:
    """Asserted on RESULTS — the parsed attribute and text — so a renderer may spell an escape
    however it likes and this still means what it says (`tests/_html.py`)."""
    html = render_html(_card())
    assert one(html, "article").attrs["data-card-id"] == "room-3"
    assert one(html, "h3").text == "Living Room Thermostat"
    assert "ev-badge--warn" in classes(html)
    assert "120 kWh/mo" in one(html, "article").text
    assert f"{RANGE} kWh/mo" in one(html, "article").text
    assert "#7" in one(html, "article").text
    # typed action: a button carrying cmd + target, not a magic id
    button = one(html, "button")
    assert button.attrs["data-cmd"] == "review"
    assert button.attrs["data-target"] == "room-3"
    assert "ev-action--medium" in classes(html)
    # chart is a self-contained data-spec div (no Shiny id-wiring)
    assert "ev-chart" in classes(html)
    # What `VEGA_BOOTSTRAP`'s `JSON.parse(el.dataset.vegaSpec)` gets in the browser.
    assert json_attr(html, "div", "data-vega-spec")["mark"] == "area"


def test_render_html_escapes_untrusted_text() -> None:
    """The property is that the title is TEXT, not that some entity appears in the bytes.

    A `not in` on the raw markup passes for a renderer that drops the title entirely, and a
    `"&lt;script&gt;" in` pins one spelling of the escape. Parsing asks the question a browser
    would: is there a script element, and is the title still the string that was handed in?"""
    spec = CardSpec(card_id="x", title="<script>alert(1)</script>")
    html = render_html(spec)
    assert elements(html, "script") == []
    assert one(html, "h3").text == "<script>alert(1)</script>"


def test_render_markdown_is_faithful() -> None:
    md = render_markdown(_card())
    assert "### Living Room Thermostat" in md
    assert "_schedule change_" in md
    assert "| Expected savings | 120 kWh/mo |" in md
    assert f"| 95% CI | {RANGE} kWh/mo |" in md
    assert "Cumulative savings — Vega-Lite chart" in md
    assert "**Review schedule** (`review` → room-3)" in md


def test_manifest_is_the_inspectable_surface() -> None:
    m = manifest(_card())
    assert m["card_id"] == "room-3"
    assert m["badge"] == "warn"
    assert m["fields"] == ["Expected savings", "95% CI", "Rank"]
    assert m["has_chart"] is True
    assert m["actions"] == [{"cmd": "review", "target": "room-3", "risk": "medium"}]


def test_slot_lowers_to_its_fallback_in_static_html() -> None:
    from effective.cards import Slot

    spec = CardSpec(card_id="r1", title="T", extras=(Slot("m", fallback="snapshot here"),))
    html = render_html(spec)
    assert "snapshot here" in html  # static target renders the fallback
    assert "shiny-html-output" not in html  # not a live Shiny binding


def test_empty_card_renders_without_optional_sections() -> None:
    spec = CardSpec(card_id="empty", title="Empty")
    html = render_html(spec)
    assert "ev-metrics" not in html
    assert "ev-actions" not in html
    assert "ev-chart" not in html
    assert manifest(spec)["has_chart"] is False
    assert render_markdown(spec).startswith("### Empty")


def test_format_value_stringifies_a_str_SUBCLASS_rather_than_returning_it() -> None:
    """The cell a 133-case differential missed, because every string in it was a plain `str`.

    `str | int | float` includes subclasses, and a subclass with its own `__str__` is the case
    where "return the value" and "return `str(value)`" differ in BOTH text and type. Naming the
    value axis as a `match` makes `case str(): return value` look obviously right; it is not, and
    only a subclass shows it."""

    class Loud(str):
        def __str__(self) -> str:
            return f"<{super().__str__()}>"

    assert format_value(Loud("x"), "plain") == "<x>"
    assert type(format_value(Loud("x"), "plain")) is str, "the subclass must not survive"
    assert format_value(Loud("x"), "integer") == "<x>", "every non-rank format stringifies"
    assert format_value(Loud("x"), "rank") == "#<x>", "and rank already interpolated it"
