# ADR-0010: The card is a typed `CardSpec` IR rendered to many targets; the t-string face emits the same IR

- **Date:** 2026-06-21
- **Status:** Accepted. Built in `src/effective/cards/`: the IR ([`src/effective/cards/spec.py`](../../src/effective/cards/spec.py)),
  `render_html`, `render_markdown`, `manifest`, the PEP 750 face `card(t"...")`
  ([`src/effective/cards/tstring.py`](../../src/effective/cards/tstring.py)), and a third target, `render_shiny` ([ADR-0011](0011-shiny-native-card-render-target.md)).
  `effective.runview` reuses `Action` as the write-back vocabulary for answering a park.
  A projector that builds a `CardSpec` from a domain read-model belongs to the host application.
- **Relates to:** [ADR-0001](0001-channel-processor.md) (the `render` → `Prompt[S]` processor whose shape `card()` mirrors),
  [ADR-0007](0007-channel-discovery-typed-manifest.md) (the inspectable-manifest principle, here applied to the view), [ADR-0011](0011-shiny-native-card-render-target.md) (the Shiny
  render target).

## Context

An operator dashboard needs cards: a title, a badge, a few metrics, a chart, and actions. A layer
that renders everything through one framework would bind the dashboard to that framework. Two
facts shrink the bet:

- **Charts are already declarative.** A Vega-Lite spec is a `dict` (altair's `.to_dict()`), a
  portable artifact any surface can embed. The open surface is the card chrome around it.
- **A declarative card spec and a t-string card template reduce to one object.** A face built
  later on `t"..."` can emit the same spec as direct construction, so building the object first
  is the cheap, reversible move.

Great DX is adjudicated by the reader, human and agent. A card the next agent can edit in one
place, whose actions carry typed meaning and whose structure is an inspectable manifest, beats one
wired by string ids to a server the reader must also find. The view axis also completes the
t-string thesis: `render()` on the data axis and `card()` on the view axis are the same processor
shape.

## Decision

**A card is a typed, domain-neutral `CardSpec` (the IR), populated by a projector and rendered to
many targets by pure functions. The PEP 750 `card(t"...")` face emits the same `CardSpec`.**

| piece | where | what it is |
|---|---|---|
| IR | `effective.cards.spec` | `CardSpec` and a small vocabulary: `Metric`, `Interval`, `Badge`, `Action`, `VegaChart`, `Slot`. Frozen dataclasses, so a card is a value that diffs cleanly |
| chart | `VegaChart.spec` | a Vega-Lite `dict`; the IR imports neither altair nor shiny |
| action | `Action` | a typed `cmd`, `label`, `target` and `risk`: the seam a ledger event binds to, in place of a magic string |
| HTML | `render_html(spec) -> str` | a self-contained fragment built with tdom's `html(t"...")`; the chart is a `data-vega-spec` div that a one-time `VEGA_BOOTSTRAP` hydrates, so one fragment serves a page, a Shiny `ui.HTML` and an MCP-App resource |
| Markdown | `render_markdown(spec) -> str` | the fallback that always works |
| manifest | `manifest(spec) -> dict` | the inspectable surface: field labels, badge tone, whether a chart is present, slot ids, and each action's command, target and risk. An agent's proposed card change is reviewed as a manifest delta |
| t-string face | `card(t"...")` | walks a `Template`'s interpolations as typed holes (`CardId`, `Title`, `Question`, `Summary`, and the IR's own types) into a `CardSpec`; a bare value is a located error |

**The projector is the join, kept out of the IR.** Building a `CardSpec` joins a domain
read-model with the Effective ledger (status, decisions, cost): the two bookkeepers. The renderers
stay pure and infra-free; the projector lives with the domain, never in `effective.cards`.

## Consequences

- **Reversible.** The IR is the one durable commitment; renderers and authoring faces swap behind
  it.
- **Foreclosed:** a card as a live framework fragment. The IR is what portability to MCP and
  agent-safety review need anyway.
- **Cost:** one package and its tests ([`tests/test_cards.py`](../../tests/test_cards.py), [`tests/test_cards_tstring.py`](../../tests/test_cards_tstring.py),
  [`tests/test_cards_render_shiny.py`](../../tests/test_cards_render_shiny.py)). The IR and the string renderers depend only on the standard
  library and tdom.
