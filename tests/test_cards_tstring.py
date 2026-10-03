"""The tdom card face: card(t"...") -> CardSpec, rendered via tdom.

The headline assertion is the convergence: card(t"...") builds the *same* CardSpec a hand
constructor would, so the spec IS the manifest. card() is the sibling of
channels.render: a PEP 750 Template processor that dispatches interpolations by type.
"""

import pytest

from effective.cards import (
    Action,
    Badge,
    CardId,
    CardSpec,
    Interval,
    Metric,
    Question,
    Summary,
    Title,
    VegaChart,
    card,
)
from effective.cards.tstring import CardTemplateError


def test_card_builds_the_same_spec_as_a_hand_built_one() -> None:
    chart = {"mark": "area"}
    name = "Living Room"
    built = card(t"""
      <card>
        {CardId("room-3")} {Title(name)} {Badge("needs review", tone="info")}
        {Question("Should this room be pre-heated?")}
        <metrics>
          {Metric("Expected savings", 120, fmt="integer", unit="kWh/mo")}
          {Interval("95% CI", 90, 150, fmt="integer", unit="kWh/mo")}
          {Metric("Rank", 1, fmt="rank")}
        </metrics>
        {VegaChart(chart, title="Cumulative savings")}
        {Summary("Upstairs, rank 1 by expected savings")}
        <actions>
          {Action("review_schedule", "Review schedule", target="room-3", risk="medium")}
        </actions>
      </card>
    """)
    expected = CardSpec(
        card_id="room-3",
        title="Living Room",
        badge=Badge("needs review", tone="info"),
        question="Should this room be pre-heated?",
        summary="Upstairs, rank 1 by expected savings",
        metrics=(
            Metric("Expected savings", 120, fmt="integer", unit="kWh/mo"),
            Interval("95% CI", 90, 150, fmt="integer", unit="kWh/mo"),
            Metric("Rank", 1, fmt="rank"),
        ),
        chart=VegaChart(chart, title="Cumulative savings"),
        actions=(Action("review_schedule", "Review schedule", target="room-3", risk="medium"),),
    )
    assert built == expected  # A1's spec IS A2's manifest


def test_card_requires_id_and_title() -> None:
    with pytest.raises(CardTemplateError):
        card(t"<card>{Title('x')}</card>")  # missing CardId
    with pytest.raises(CardTemplateError):
        card(t"<card>{CardId('x')}</card>")  # missing Title


def test_card_rejects_an_untyped_hole() -> None:
    # like channels.render, every hole must be a typed channel — a bare value is an error
    with pytest.raises(CardTemplateError):
        card(t"<card>{CardId('x')}{Title('y')}{object()}</card>")
