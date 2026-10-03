"""The ``CardSpec`` IR and its typed wrapper vocabulary.

Every piece is a frozen value type — a card is *data*, so it renders to many
targets, diffs cleanly, and yields a manifest for free. ``Action`` carries a
typed ``cmd``/``target``/``risk`` (the seam a ledger event binds to), not a
magic string; that is the view-axis analogue of a channel's write-back.
"""

from dataclasses import dataclass
from typing import Any, Literal, assert_never

type Fmt = Literal["plain", "integer", "currency", "percent", "rank"]
type Tone = Literal["neutral", "info", "success", "warn", "danger"]
type Risk = Literal["low", "medium", "high", "critical"]


def format_value(value: float | int | str, fmt: Fmt = "plain") -> str:
    """Render a metric value as display text. Domain-neutral; the unit (if any)
    is appended by the caller from ``Metric.unit`` so this stays a pure style."""
    # TWO axes, each named: the value's type here, the `fmt` in `_format_number`. The value axis
    # is a table rather than a guard, because a guard rejects and both arms here return accepted
    # output over the declared `float | int | str`.
    match value:
        case str():
            # `str(value)`, not `value`: a `str` SUBCLASS with its own `__str__` is inside the
            # declared `float | int | str`, and returning it unchanged hands back both a different
            # text and a different type. A battery of plain `str` values does not exercise it.
            return f"#{value}" if fmt == "rank" else str(value)
        case int() | float():
            return _format_number(value, fmt)
        case unreachable:
            assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead


def _format_number(value: float, fmt: Fmt) -> str:
    """The numeric half of `format_value` — reached only once the value axis is decided.

    `"plain"` is NAMED rather than left to the wildcard, so the `Fmt` literals are a closed table
    and the wildcard means only what it says: a runtime string outside the type."""
    match fmt:
        case "integer":
            return f"{value:,.0f}"
        case "currency":
            return f"${value:,.2f}"
        case "percent":
            return f"{value:.1%}"
        case "rank":
            return f"#{value}"
        case "plain":
            return str(value)
        case _:
            # NOT `assert_never`: `Fmt` is a `Literal` alias, so an out-of-type string reaches
            # here at runtime from any untyped caller, and every literal above is named.
            return str(value)


@dataclass(frozen=True)
class Metric:
    """A single labelled value. ``unit`` is a free suffix (e.g. ``"kWh/mo"``)."""

    label: str
    value: float | int | str
    fmt: Fmt = "plain"
    unit: str | None = None
    kind: Literal["metric"] = "metric"

    def render_value(self) -> str:
        base = format_value(self.value, self.fmt)
        return f"{base} {self.unit}" if self.unit else base


@dataclass(frozen=True)
class Interval:
    """A labelled low-high range (e.g. a 95% CI)."""

    label: str
    lo: float
    hi: float
    fmt: Fmt = "plain"
    unit: str | None = None
    kind: Literal["interval"] = "interval"

    def render_value(self) -> str:
        lo = format_value(self.lo, self.fmt)
        hi = format_value(self.hi, self.fmt)
        # En-dash is the correct typographic range separator in rendered output;
        # RUF001 flags it as hyphen-confusable, which is a non-issue here.
        base = f"{lo}–{hi}"  # noqa: RUF001
        return f"{base} {self.unit}" if self.unit else base


type Cell = Metric | Interval


@dataclass(frozen=True)
class Badge:
    text: str
    tone: Tone = "neutral"


@dataclass(frozen=True)
class Action:
    """A typed command on an entity — the seam a ledger event binds to.

    ``cmd`` is the command name (the workflow op / event), ``target`` the entity
    id it applies to, ``risk`` the policy tier (cf. the dashboard-change ladder).
    """

    cmd: str
    label: str
    target: str
    risk: Risk = "low"


@dataclass(frozen=True)
class VegaChart:
    """A Vega-Lite spec (``alt.Chart(...).to_dict()``) plus an optional title.

    Stored as a plain ``dict`` so this package never imports altair; the domain
    builds the chart and passes the spec in."""

    spec: dict[str, Any]
    title: str | None = None


@dataclass(frozen=True)
class Slot:
    """A typed escape hatch: a named region the host fills. The IR carries the
    *hole* and the host supplies the widget, so a card can host something the cell vocabulary
    doesn't model (a live table, a map, a custom widget) and stay renderable on every
    target. On Shiny it lowers to an ``output_ui`` the author fills with a normal
    Shiny output; on static/MCP it renders ``fallback`` (a snapshot or placeholder)."""

    slot_id: str
    fallback: str | None = None
    kind: Literal["slot"] = "slot"


@dataclass(frozen=True)
class CardSpec:
    """One card: a readable front (title/badge/metrics) over an analytic chart,
    with typed actions. Populated by a *projector* that joins a domain
    read-model with the Effective ledger; the renderers stay pure."""

    card_id: str
    title: str
    badge: Badge | None = None
    question: str | None = None
    summary: str | None = None
    metrics: tuple[Cell, ...] = ()
    chart: VegaChart | None = None
    actions: tuple[Action, ...] = ()
    extras: tuple[Slot, ...] = ()  # bespoke host-filled regions (the escape hatch)
