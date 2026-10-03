"""The composer against the grammar's INDUCTION: a product of base and inductive cases.

**Why this file exists rather than more examples.** `compose_key` and `parse` are two
implementations of one language, so they can drift: a composer whose tag check fences only `:`
and `;` mints `Foo_Bar:1`, `a,b:1` and `a#2:1`, text its own parser refuses. A hand-written list
of such spellings does not generalize, because the next drift will be at a position nobody
listed.

**The method: take a reasonable PRODUCT of the base case(s) and the inductive
case(s), and show the paths through them work.** The grammar is four categories, each with a base
and an inductive constructor:

    key        := term (';' term)*  ('#' INTEGER)?     base: one term      | inductive: ';'
    term       := tag (':' coord (',' coord)*)?        base: bare tag      | inductive: arity 1..n
    coordinate := atom ('/' atom)*                     base: one atom      | inductive: '/' path
    atom       := INTEGER | UUID | DIGEST | NAME       four base cases, no induction

**And the enabling fact is PEP 750: a `Template` IS the AST**. It is an ordinary runtime
value — statics interleaved with `Interpolation`s — so a test can *construct* compositions instead
of writing them out, and enumerate AST shapes the way a compiler test enumerates terms. An
f-string cannot be generated this way: by the time you hold one it is already flattened, so there
is nothing left to take a product over. This file is the concrete payoff of the data axis being
reified.

Two directions, because the two defects are duals:

- **positive** (`test_every_path_through_the_grammar`): every path through the constructors
  composes, parses, round-trips, and yields the structure it was built from. This is what proves
  the induction is wired up.
- **negative** (`test_no_position_admits_a_value_outside_its_language`): at every POSITION, a
  value outside that position's language is refused. Drift shows up here, as a composer that
  accepts what it should reject.

Role: adversarial. The negative half is an attack surface; a pass means only that it failed.
"""

from collections.abc import Callable
from itertools import product
from string.templatelib import Interpolation, Template
from typing import Any

import pytest

from effective.keys import Key, Segment, Tag, compose_key
from effective.keys.grammar import KeySyntaxError, parse

pytestmark = pytest.mark.adversarial

# --- the base cases -------------------------------------------------------------------------
#
# One representative per ATOM kind. These are the leaves of the induction, and the four kinds are
# the whole of it — an atom has no inductive case, which is exactly what makes arity countable
# from the bytes with no registry (design §1a).
ATOMS = {
    "integer": "12",
    "uuid": "0197f3aa-1c2d-7e88-9a0b-2f3c4d5e6f70",
    "digest": "sha256-9f2c1a",
    "name": "a-b",
}

COORDINATE_SHAPE = ["single", "path"]  # base: one atom | inductive: atom '/' atom
BINDING = ["positional", "named"]  # required | optional-with-default
ARITY = [0, 1, 2]  # base: a bare tag (a qualifier) | inductive
TERM_COUNT = [1, 2]  # base: one term | inductive: ';'
OCCURRENCE = [None, 2]  # the suffix is a decoration on the whole key


def _illegal(shape: str, binding: str, arity: int) -> str | None:
    """Combinations the LANGUAGE excludes, each with the rule that excludes it.

    Named rather than silently skipped: a product whose exclusions are unexplained is a product
    someone will quietly widen. Both of these were *discovered by running the product* rather than
    known in advance — the second one especially, which is a `compose_key` rule I had not read.
    """
    if arity == 0 and (shape == "path" or binding == "named"):
        return "a bare tag is a qualifier — it has no coordinates to shape or name"
    if shape == "path" and binding == "named":
        return "a defaulted coordinate is ONE atom, not a path — a default is a single value"
    return None


def _compose(atom: str, shape: str, binding: str, arity: int, terms: int) -> Key:
    """Build the `Template` — the AST — for one cell, then compose it.

    Assembled as alternating statics and `Interpolation`s, which is precisely what the compiler
    hands `compose_key` for a literal `t"…"`. Same input, generated.
    """
    statics: list[str] = []
    holes: list[Interpolation] = []
    pending = ""
    for term in range(terms):
        pending += ("" if term == 0 else ";") + f"ns{term}"
        for position in range(arity):
            pending += ":" if position == 0 else ","
            if binding == "named":
                pending += f"k{term}{position}="
            statics.append(pending)
            pending = ""
            holes.append(
                Interpolation(
                    Segment(atom),
                    f"a{term}{position}",
                    None,
                    "default=0" if binding == "named" else "",
                )
            )
            if shape == "path":
                pending = "/"
                statics.append(pending)
                pending = ""
                holes.append(Interpolation(Segment(atom), f"b{term}{position}", None, ""))
    statics.append(pending)

    parts: list[object] = []
    for static, hole in zip(statics, [*holes, None], strict=True):
        parts.append(static)
        if hole is not None:
            parts.append(hole)
    return compose_key(Template(*parts))  # ty: ignore[invalid-argument-type]


CELLS = [
    (atom, shape, binding, arity, terms, occurrence)
    for atom, shape, binding, arity, terms, occurrence in product(
        ATOMS, COORDINATE_SHAPE, BINDING, ARITY, TERM_COUNT, OCCURRENCE
    )
    if _illegal(shape, binding, arity) is None
]


@pytest.mark.parametrize(("atom", "shape", "binding", "arity", "terms", "occurrence"), CELLS)
def test_every_path_through_the_grammar(atom, shape, binding, arity, terms, occurrence):
    """Compose it, parse it, render it back, and check the STRUCTURE is what was built.

    The last check is what separates this from a smoke test: `parse` returning *something* proves
    nothing, so the cell asserts the term count and the occurrence it was constructed with. A
    composer that dropped a term or swallowed the suffix would still round-trip its own mistake.
    """
    key = _compose(ATOMS[atom], shape, binding, arity, terms)
    if occurrence is not None:
        key = key.occurrence(occurrence)
    text = key.stored()

    parsed = parse(text)  # 1. the composer's output is in the language
    assert parsed.render() == text  # 2. and the bytes survive a round-trip
    assert len(parsed.terms) == terms  # 3. and the structure is the one we built
    assert parsed.occurrence == occurrence
    for term in parsed.terms:
        assert len(term.coordinates) == arity


def test_the_product_covers_both_ends_of_every_axis():
    """Anti-vacuity for the product itself: a filter bug could quietly empty an axis.

    Without this, tightening `_illegal` (or mistyping a constant) shrinks the matrix silently and
    every remaining cell still passes, an instrument that cannot fail. This guards the
    parameterization rather than the assertions.
    """
    assert len(CELLS) == 112, len(CELLS)
    for axis, values in (
        (0, set(ATOMS)),
        (1, set(COORDINATE_SHAPE)),
        (2, set(BINDING)),
        (3, set(ARITY)),
        (4, set(TERM_COUNT)),
        (5, set(OCCURRENCE)),
    ):
        assert {cell[axis] for cell in CELLS} == values, axis


# --- the negative direction: no POSITION admits a value outside its language -------------------

# Each position gets the values that can actually REACH it. A single shared battery of `str`
# values makes one position vacuous: `compose_key` refuses a bare `str` interpolation outright
# ("may contain a separator"), so the SPLICE position would refuse all of them for an unrelated
# reason and test nothing.
_TEXTUAL = [
    "a,b",  # the arity separator
    "a#2",  # the occurrence sigil: the sharpest, it forges a coordinate
    "a:b",  # the tag separator
    "a;b",  # the term separator
    "a=b",  # the coordinate-name separator
    "a b",  # a space is in no atom kind
    "",  # the empty value addresses nothing
    "Foo_Bar",  # legal in a NAME atom, illegal as a TAG
    "2fa",  # digit-led: no delimiter, and still not an atom
]

POSITIONS: dict[str, tuple[Callable[[Any], Key], list[Any]]] = {
    "tag": (lambda v: compose_key(t"{Tag(v)}:{Segment('x')}"), _TEXTUAL),
    "coordinate": (lambda v: compose_key(t"ns:{Segment(v)}"), _TEXTUAL),
    "path atom": (lambda v: compose_key(t"ns:{Segment('a')}/{Segment(v)}"), _TEXTUAL),
    "named coordinate": (
        lambda v: compose_key(t"ns:{Segment('a')},k={Segment(v):default=0}"),
        _TEXTUAL,
    ),
    # A splice takes a VALUE, not text, so the battery is typed. `5` and `Segment(...)` are the
    # values an unchecked splice would make a term TAG. `Tag("a-b")` is the ACCEPTING case: a
    # term head names a namespace, so with the `Segment` rows refused it is the only value
    # keeping the anti-vacuity assert below off zero, and a battery that refuses everything
    # tests nothing.
    "spliced term": (
        # lint: terminal-hole — DELIBERATE: an unwrapped splice IS the subject here.
        lambda v: compose_key(t"ns:1;{v}"),
        [5, 0, Segment("Foo_Bar"), Segment("a-b"), Segment("12"), Tag("a-b")],
    ),
}
"""Every position a value can ENTER a key from the composer: the induction read as a set of holes.

`spliced term` is the subtle one: a non-`Key` value spliced after `;` with no check becomes a
term TAG, so `compose_key(t"ns;{5}")` would return `Key('ns;5')`.
"""


@pytest.mark.parametrize("position", list(POSITIONS))
def test_no_position_admits_a_value_outside_its_language(position):
    """The composer must REFUSE, or produce something that parses.

    Stated as an implication rather than as "must raise", because the two are not the same
    requirement and only the weaker one is true: a position may legitimately accept a value the
    grammar then renders differently. What must never happen is the composer minting text its own
    parser rejects, which would let `Key.stored()` hold bytes `Key.terms()`, `keymap.explain`,
    `graphview` and `split_occurrence` all choke on.
    """
    build, values = POSITIONS[position]
    accepted = 0
    for value in values:
        try:
            key = build(value)
        except ValueError, TypeError, KeySyntaxError:
            continue  # refused at the door: sound
        accepted += 1
        text = key.stored()
        parse(text)  # raises KeySyntaxError if the composer minted outside the language
        assert parse(text).render() == text, f"{position}: {value!r} -> {text!r}"
    # ANTI-VACUITY per position: a position that refuses EVERYTHING tests nothing.
    if position == "spliced term":
        assert accepted >= 1, "the splice battery must contain values the composer accepts"


def test_five_named_out_of_language_spellings_are_each_refused():
    """Named cases, kept beside the product that generalizes them.

    The product is what stops the NEXT drift; a named case is what makes the hazard teachable.
    Each would compose and then fail `parse` without its fence: two through the splice path,
    three through a `Tag` check that fences only `:` and `;`.
    """
    for forge in (
        lambda: compose_key(t"ns;{5}"),  # splice: an int as a term tag
        lambda: compose_key(t"ns;{Segment('Foo_Bar')}"),  # splice: a NAME atom is not a TAG
        lambda: compose_key(t"{Tag('Foo_Bar')}:{1}"),  # Tag: uppercase + underscore
        lambda: compose_key(t"{Tag('a,b')}:{1}"),  # Tag: the arity separator
        lambda: compose_key(t"{Tag('a#2')}:{1}"),  # Tag: the occurrence sigil
    ):
        # `TypeError` because two of the five are refused by TYPE at the term head rather than by
        # SHAPE at the grammar; the claim is that all five are refused, not where.
        with pytest.raises((ValueError, KeySyntaxError, TypeError)):
            forge()
