"""The SQL boundary as a PEP 750 ``Template`` processor: the ``sqlite3`` half of the seam.

psycopg ≥ 3.3 takes a ``Template`` directly, so the Postgres/Absurd side of this repo already
writes its SQL the structural way::

    conn.execute(t"SELECT state FROM absurd.{c_tbl:i} WHERE task_id = {task_id}::uuid")

The stdlib ``sqlite3`` driver has no such support, and its native form is a string literal
beside a separate ``parameters`` tuple: the structure and its data split across two arguments a
reader has to zip up by eye. This module is the missing processor, so the two engines' SQL reads
the *same* at both boundaries::

    conn.execute(*bind(t"SELECT state FROM checkpoints WHERE task_id={task_id} AND name={name}"))

`bind` is a **processor**, not a formatter: it never flattens a value into the SQL text. Each
interpolation becomes a ``?`` placeholder plus one entry in ``Query.parameters``, in template
order, so the delimiter/data distinction the driver needs is a property of the *structure* rather
than of the author's discipline (immediate flattening is the Bobby Tables antipattern).
The f-string lives one layer down, in the render backend, where the structural decisions are
already made.

**What only the ``Template`` makes possible.** An f-string hands a processor a finished string;
a ``Template`` hands it position, arity, and each hole's source ``expression``. `bind` spends all
three on hazards a `%s`-and-a-tuple convention can only ask you to remember:

- a hole **inside a quoted SQL string literal** (``t"… LIKE '{prefix}%'"``) is refused. Rendered
  as text that would be a literal ``?`` *inside* the quotes and a parameter count that does not
  match: the classic silent break. The processor knows the hole's position in the statics, so
  it can say so and name the fix.
- a hole inside a ``"quoted identifier"`` or a comment — same knowledge, same refusal.
- a manual ``?`` in the static SQL is refused: mixing hand-written placeholders with
  interpolations silently permutes the parameter order.
- the format-spec slot is **owned by this DSL** (see `compose_key`, which reserves it the same
  way). ``{table:i}`` means *safely-quoted identifier*, spelled exactly
  as psycopg spells it so the two engines' SQL stays one vocabulary. Every other spec and every
  conversion is refused rather than silently dropped.
- a container value (``{ids}`` for an ``IN`` list) is refused with the fix named, instead of the
  driver's ``InterfaceError: unsupported type`` a stack frame later.

Every refusal names the offending **expression as the author wrote it** — that is the hole's
``expression``, which is exactly what an f-string throws away.

Scope: DB-API ``qmark`` paramstyle (``sqlite3``). Nothing here imports a driver — it is a pure
``Template`` processor over text, so it stays inside "keep the core DB-free". If ``sqlite3`` ever
accepts a ``Template`` itself, deleting `bind` is a mechanical unwrap of one call.
"""

import enum
import re
from string.templatelib import Interpolation, Template
from typing import Any, NamedTuple, assert_never

IDENTIFIER_SPEC = "i"
"""The format spec that marks a hole as an **identifier** rather than a value — psycopg's own
spelling (`{table:i}`), kept verbatim so a reader moving between the Absurd and SQLite readers
sees one grammar rather than two dialects of it."""

PLACEHOLDER = "?"
"""The qmark paramstyle's placeholder (PEP 249). Named so the processor, the manual-placeholder
refusal, and the tests agree by construction rather than by scattered literals."""


class SqlTemplateError(ValueError):
    """A SQL template the processor refuses to compose.

    A `ValueError` because that is what a bad *argument* to a processor is (`compose_key` raises
    the same), and its own class so a caller — a lint, a test, a fallback path — can catch exactly
    this rather than every `ValueError` the driver family raises."""


class Query(NamedTuple):
    """A composed statement: the SQL text and its bound parameters, kept apart.

    A `NamedTuple` for one concrete reason — it splats. ``conn.execute(*bind(t"…"))`` is the one
    canonical boundary form, and the splat is the visible seam between the two things DB-API keeps
    separate: ``sql`` is structure, ``parameters`` is data. Named fields keep it inspectable in a
    test or a log without positional guessing."""

    sql: str
    parameters: tuple[Any, ...]


class _Region(enum.Enum):
    """Where in the SQL text the scanner currently stands.

    States as data (the substrate's idiom, cf. `Phase`/`Inspection`): the scanner's whole job is to
    answer "may a hole appear *here*?", and only `OUTSIDE` may."""

    OUTSIDE = enum.auto()
    STRING = enum.auto()  # '…' — a value literal
    IDENTIFIER = enum.auto()  # "…" — a quoted identifier
    LINE_COMMENT = enum.auto()  # -- … to end of line
    BLOCK_COMMENT = enum.auto()  # /* … */


_REGION_FIX = {
    _Region.STRING: (
        "inside a quoted SQL string literal, where a placeholder is literal text: the driver "
        "would bind nothing and the parameter count would not match. Bind the whole value "
        "instead — t\"… LIKE {prefix + '%'}\", not t\"… LIKE '{prefix}%'\""
    ),
    _Region.IDENTIFIER: (
        'inside a "quoted identifier". Interpolate the identifier itself with the identifier spec '
        f'— t"… FROM {{table:{IDENTIFIER_SPEC}}}", which quotes it for you — not inside the quotes'
    ),
    _Region.LINE_COMMENT: "inside a `--` comment, where it would be discarded by the parser",
    _Region.BLOCK_COMMENT: "inside a `/* */` comment, where it would be discarded by the parser",
}


_OPENS = {
    "'": _Region.STRING,
    '"': _Region.IDENTIFIER,
    "--": _Region.LINE_COMMENT,
    "/*": _Region.BLOCK_COMMENT,
}
_CLOSES = {
    _Region.STRING: "'",
    _Region.IDENTIFIER: '"',
    _Region.LINE_COMMENT: "\n",
    _Region.BLOCK_COMMENT: "*/",
}
_TOKEN = re.compile(r"--|/\*|\*/|['\"\n?]")
"""Every byte that can move the scanner between regions, plus the placeholder it must notice.
Everything else in the SQL is inert to this scan — which is the point: the processor reads the
author's SQL only for *where a hole may stand*, and never rewrites it."""


def _advance(region: _Region, text: str) -> tuple[_Region, bool]:
    """Scan one static and return `(region at its end, saw a manual placeholder)`.

    Only an `OUTSIDE` placeholder counts: a `?` inside `'a?b'` is data. A doubled quote (`'it''s'`,
    SQLite's own escape) needs no special case — the second quote re-opens the literal the first
    one closed, so the parity works out on its own."""
    saw_placeholder = False
    for match in _TOKEN.finditer(text):
        token = match.group()
        if region is not _Region.OUTSIDE:
            if token == _CLOSES[region]:
                region = _Region.OUTSIDE
        elif token == PLACEHOLDER:
            saw_placeholder = True
        elif (opened := _OPENS.get(token)) is not None:
            region = opened
    return region, saw_placeholder


def quote_identifier(name: object, expression: str) -> str:
    """Render one `{table:i}` hole: a double-quoted SQL identifier, embedded quotes doubled.

    The identifier is the one place a value legitimately becomes *structure*, so it is the one
    place an injection could re-enter — hence quoting here, once, in the processor, rather than at
    each call site. `expression` is the hole's source text, carried in so a refusal can name what
    the author wrote."""
    if not isinstance(name, str):
        raise SqlTemplateError(
            f"{{{expression}:{IDENTIFIER_SPEC}}} is a {type(name).__name__}, not a str — an "
            f"identifier is a name. Drop the :{IDENTIFIER_SPEC} spec to bind it as a value."
        )
    if not name:
        raise SqlTemplateError(f"{{{expression}:{IDENTIFIER_SPEC}}} is empty — not an identifier.")
    if "\x00" in name:
        raise SqlTemplateError(
            f"{{{expression}:{IDENTIFIER_SPEC}}} contains a NUL, which cannot be quoted."
        )
    return '"' + name.replace('"', '""') + '"'


def bind(template: Template) -> Query:
    """Compose a `Template` into `(sql, parameters)` for a DB-API `qmark` driver (`sqlite3`).

    The one canonical boundary form is the splat::

        row = conn.execute(*bind(t"SELECT state FROM checkpoints WHERE name={name}")).fetchone()

    Every interpolation is a bound parameter — ``?`` in the text, the value in `parameters`, in
    template order — except a hole carrying the ``:i`` spec, which is a safely-quoted identifier.
    Statics pass through verbatim; they are the author's SQL and this processor never rewrites
    them. See the module docstring for what each refusal buys."""
    text: list[str] = []
    parameters: list[Any] = []
    region = _Region.OUTSIDE
    for item in template:
        match item:
            case str() as static:
                region, saw_placeholder = _advance(region, static)
                if saw_placeholder:
                    raise SqlTemplateError(
                        f"the SQL carries a manual {PLACEHOLDER!r} placeholder "
                        f"({static.strip()!r}) — mixing hand-written placeholders with "
                        f"interpolations permutes the "
                        f"parameter order silently. Write the value as an interpolation instead."
                    )
                text.append(static)
            case Interpolation() as hole:
                if region is not _Region.OUTSIDE:
                    raise SqlTemplateError(
                        f"interpolation {{{hole.expression}}} is {_REGION_FIX[region]}."
                    )
                if hole.conversion is not None:
                    raise SqlTemplateError(
                        f"interpolation {{{hole.expression}!{hole.conversion}}} carries a "
                        f"conversion. A parameter is passed to the driver as a Python value, so "
                        f"converting it to text here would bind a different value than the "
                        f"template says. Convert it before composing."
                    )
                # `''` is an absent spec while `None` is an absent conversion — the sentinel
                # asymmetry `compose_key` documents; testing both the same way is the bug.
                if (spec := hole.format_spec) == IDENTIFIER_SPEC:
                    text.append(quote_identifier(hole.value, hole.expression))
                elif spec:
                    raise SqlTemplateError(
                        f"interpolation {{{hole.expression}:{spec}}} carries a format spec this "
                        f"DSL does not define. The slot belongs to the processor: "
                        f"{IDENTIFIER_SPEC!r} means a quoted identifier, and a bare hole is a "
                        f"bound parameter. Render the value before composing."
                    )
                elif isinstance(hole.value, list | tuple | set | dict):
                    raise SqlTemplateError(
                        f"interpolation {{{hole.expression}}} is a "
                        f"{type(hole.value).__name__} — a driver binds one scalar per "
                        f"placeholder. For an IN list, compose one hole per element (the arity "
                        f"has to reach the SQL); for a JSON column, serialize it first."
                    )
                else:
                    text.append(PLACEHOLDER)
                    parameters.append(hole.value)
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
    return Query("".join(text), tuple(parameters))


__all__ = [
    "IDENTIFIER_SPEC",
    "PLACEHOLDER",
    "Query",
    "SqlTemplateError",
    "bind",
    "quote_identifier",
]
