"""Pinned injectivity tests for ``op_key``: invariant I1 at the STRING boundary.

These assert the *class* the Lean gate (`formal/lean/Effective/Keys.lean`) proves for the
structured ``List Nat`` encoding but explicitly DEFERS for the string serialization
(Keys.lean:22-23). ``op_key`` (`effective/handlers/base.py`) is that runtime string
serialization, and two known cases threaten its injectivity:

| case                   | defect                              | defense                         |
|------------------------|-------------------------------------|---------------------------------|
| reserved arm namespace | a ``Step`` named ``ledger;X``       | ``op_key`` refuses a ``Step``   |
|                        | aliases an ``AppendLedgerRow``      | name in a reserved arm          |
|                        |                                     | namespace; pinned below         |
| same-arity gathers     | two gathers share ``gather:{len}``  | a gather's identity is          |
|                        |                                     | POSITIONAL: ``op_key(Gather)``  |
|                        |                                     | raises and the handlers key     |
|                        |                                     | leaves by ``gather:{g},{i};``   |

The gather case is pinned in
`tests/test_gather.py::test_two_same_arity_gathers_get_distinct_positional_keys`.

**A BARE TERMINAL HOLE IN THIS FILE IS DELIBERATE: do not sweep it.** Everywhere else in
``tests/`` a template's last hole is wrapped (``compose_key(t"extracted:{Segment(mid)}")``), so
that the terminal exemption can be deleted. This file is where the exemption is SPECIFIED, so the
remaining bare holes are the subject. They are three kinds, and only the first changes when the
mechanism lands:

- **Delimiter-bearing values**: ``t"artifact:{'message/rfc822:9f2c'}"`` (the MIME-type path
  case), ``t"fork:{Segment('c')};{'review:m1'}"``, ``t"event;{'review:m1'}"``, the three forged
  frames ``t"talk:{'a;b'}"`` (twice) / ``t"talk:{'x;y'}"``, and the ``xfail``'s own
  ``t"ns:{'a:b'}"``. A ``Segment`` refuses both delimiters, so none of these can be wrapped; they
  need SEMANTIC rewrites, driven by the ``/``-as-metacharacter change. The forged-frame three are
  the subtlest: each carries the frame delimiter ``;``, and ``compose_key`` refuses a bare ``str``
  coordinate by type before ``scope_prefix`` or ``frame_path`` sees it, so what they pin is that
  upstream refusal.
- **The empty segment**: ``{''}`` in ``test_a_static_separator_makes_it_injective_again`` and
  ``test_a_trailing_empty_segment_is_a_segment_not_an_absence``. ``Segment("")`` raises by
  design, so this is a different class from the delimiter one and is not waiting on anything.
- **Refusals that fire BEFORE the terminal is reached**: the adjacent-hole pair
  ``t"x:{'p'}{'q'}"`` and the format-spec ``t"x:{'a'!r}"``. Wrapping them would hide which rule
  is under test.

Everything else here is wrapped like the rest of ``tests/``. The two already-legal shapes, an
``int`` terminal and a spliced ``Key``, need no wrapper in any file, and since ``Segment``
refuses a non-``str``, wrapping one fails loudly with a message naming the right move.

**The ``compose_key`` templates written inside SOURCE STRINGS below are a fourth thing again**:
the lint fixtures from ``_scoped_scan`` down. They are parsed, never executed, so
``compose_key``'s runtime rules never see them. Some exist precisely to be UNWRAPPED:
``t"{T}:{name}:{suffix}"`` is the negative case asserting ``qualified-wraps-an-untyped-name``, and
its wrapped twin two lines later asserts the opposite. Wrapping either would invert its test. They
move when the REGISTRY learns the separator and loses its tails, which is a different change from
wrapping a value.
"""

import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest

from effective.api import GatherBranch, qualified_event_name
from effective.code import _action_key, _segment_key
from effective.domain import CallTool
from effective.govern import GateState
from effective.handlers.base import op_key, step_key
from effective.keys import Key, Segment, Tag, compose_key
from effective.keys.grammar import (
    METACHARACTERS,
    TERM_SEPARATOR,
    Hole,
    KeySyntaxError,
    ParsedKey,
)
from effective.ops import (
    AppendLedgerRow,
    AwaitEvent,
    Gather,
    LedgerRow,
    SleepUntil,
    Step,
    StoreArtifact,
    event_name,
)

# This whole module ATTACKS injectivity: every case constructs two things the
#  key must separate and tries to make them collide. A pass here means the attack failed,
#  which is evidence only because these pins are mutation-checked.
pytestmark = pytest.mark.adversarial


def _dop() -> CallTool[str]:
    return CallTool(name="t", result_schema=str)


def test_op_key_injective_across_arms() -> None:
    """A ``Step`` name in a substrate namespace is WRAPPED, so it cannot alias. No denylist.

    Every key begins with its arm, a Step's included, so an author's `ledger;x` composes
    `step;ledger;x` and the regions are disjoint **by construction**. The property pinned is
    DISJOINTNESS rather than refusal: a denylist is bounded by what it enumerates, and
    disjointness holds for names nobody thought to enumerate.
    """
    # `.stored()`, not `str()`: a `Key` is opaque, so `str()` yields the REPR and a `Step` named
    # `Key(_value='ledger:X')` would be testing nothing.
    substrate_names = [
        op_key(AppendLedgerRow(row=LedgerRow(event_id=Key.parse("x")))).stored(),  # ledger;x
        op_key(AwaitEvent(name=Key.parse("e"), schema=str)).stored(),  # event;e
        "sleep:0",  # the positional sleep namespace (op_key(SleepUntil) itself raises, below)
        op_key(StoreArtifact(value={"k": 1}, content_type="application/json")).stored(),
        "gather:2",  # the positional gather namespace (op_key(Gather) itself raises, below)
        "monitor:0",  # likewise positional
    ]
    for name in substrate_names:
        composed = op_key(Step(name=name, op=_dop())).stored()
        assert composed != name, f"a Step named {name!r} composed the substrate's own key"
        assert composed.startswith("step"), composed
        # and the substrate's name survives INSIDE, so the author's identity is not mangled
        assert name in composed, composed

    # An author name in no substrate namespace goes through the same arm, which is the point:
    # there is no second path and therefore no second rule to keep in sync.
    assert op_key(Step(name="tool:act", op=_dop())).stored() == "step;tool:act"


def test_op_key_of_a_gather_raises_no_content_key() -> None:
    """A ``Gather`` has no standalone op key: its identity is positional (the
    ``gather:{g},{i};`` path a handler assigns while walking the execution tree), not a
    content function of the op. ``op_key`` says so loudly rather than returning a misleading
    arity label that two distinct gathers would share."""
    with pytest.raises(ValueError, match="no standalone op key"):
        op_key(Gather(branches=(_dop, _dop)))


def _shape(*parts, fields, site="x:1"):
    """A `Shape` built the way the registry builds one: from a template SKELETON.

    `Shape(tag=..., slots=(Fld(...), Lit(...)), tail=...)` is gone with the flat grammar — a shape
    is now a parsed skeleton plus its field names, so a test states the template and lets the same
    parser the composer uses derive the structure. `...` marks a hole, in order."""
    from effective.keys.grammar import Hole, parse_skeleton
    from effective.keys.registry import Shape

    numbered, i = [], 0
    for part in parts:
        if part is ...:
            numbered.append(Hole(i))
            i += 1
        else:
            numbered.append(part)
    return Shape(skeleton=parse_skeleton(numbered), fields=tuple(fields), site=site, sites=(site,))


def _witness(shape):
    """A key this shape would mint, plus the bindings a decode must return.

    Rendered through `keymap._render_skeleton` — the same walk `Shape.label()` and the lint's
    borrowing check use — so the witness cannot drift from what the shape means. Values are
    `v0, v1, …` in hole order, which is also the order `fields` is recorded in."""
    from effective.keys.registry import _render_skeleton

    values = iter(f"v{i}" for i in range(len(shape.fields)))
    rendered = _render_skeleton(shape.skeleton, lambda: next(values))
    return rendered, dict(
        zip(shape.fields, (f"v{i}" for i in range(len(shape.fields))), strict=True)
    )


# --- compose_key refuses ADJACENT holes (I1) -----------------------------------------------
#
# `t"x:{a}{b}"` has no delimiter between the two values, so no escaping can make the split
# recoverable: ("p","q") and ("pq","") both flatten to `x:pq`. That is the one injectivity
# question the producer can settle STRUCTURALLY (from the template's shape) instead of by
# escaping bytes, and it is what makes `Keys.lean`'s by-construction discharge honest.


def test_adjacent_interpolations_are_refused():

    with pytest.raises(ValueError, match="adjacent with no delimiter"):
        compose_key(t"x:{'p'}{'q'}")


def test_the_refusal_also_covers_trusted_Key_segments():
    """A `Key` is a trusted SEGMENT, not a licence to skip the structural check — two
    adjacent `Key`s are just as unrecoverable as two raw values."""
    from effective.keys import Key

    with pytest.raises(ValueError, match="adjacent with no delimiter"):
        compose_key(t"x:{Key.parse('p')}{Key.parse('q')}")


def test_the_collision_the_refusal_prevents():
    """Show the class, so the pin says WHY: without the check these two distinct tuples
    produce one key, and escaping cannot help because there is no delimiter to escape."""

    def flat(a: str, b: str) -> str:
        return f"x:{a}{b}"  # what compose_key would have produced

    assert flat("p", "q") == flat("pq", "")


def test_a_static_separator_makes_it_injective_again():
    from effective.keys import Segment

    assert compose_key(t"x:{Segment('p')},{Segment('q')}") != compose_key(t"x:{Segment('pq')}")


def test_every_shipping_compose_key_template_still_composes():
    """Dogfood: the check is free — no call site in the repo violates it. `op_key` over every
    op arm is the exhaustive exercise of the shipped templates."""

    from effective.domain import CallTool
    from effective.handlers.base import op_key
    from effective.ops import AppendLedgerRow, AwaitEvent, Step, StoreArtifact

    ops = [
        Step(name="tool:x", op=CallTool(name="x", args={}, result_schema=dict)),
        AwaitEvent(Key.parse("review:m1"), dict),
        AppendLedgerRow(LedgerRow(event_id=Key.parse("reviewed:m1"))),
        StoreArtifact(value={"a": 1}, content_type="application/json"),
    ]
    assert len({op_key(o).stored() for o in ops}) == len(ops)


# --- the substrate's AUTHORITY namespaces are disjoint BY CONSTRUCTION ---------------------
#
# An author's Step name cannot reach the substrate's authority namespaces (an approval, a spend
# grant, a gate's park, a fork's identity). Those names ARE the authorization, so an alias there
# delivers an answer to the wrong question.
#
# Nothing FENCES it. A Step's key is `step:{name}` or `step;{name}`, so an author writing
# `approve;ledger;reviewed:m1` gets `step;approve;ledger;reviewed:m1`, which begins with `step`,
# and no substrate key does. The regions are disjoint because of their SHAPE, not because a tuple
# enumerated them, which is the separation-logic claim this repo makes in prose holding in the
# code. A list of forbidden names is bounded by what it enumerates; an arm term is bounded by
# nothing.


@pytest.mark.parametrize(
    "name",
    [
        "approve;ledger;reviewed:m1",  # the human-tier approval event for another op
        "budget-grant:r1,0",  # the measured trip's grant
        "fork:cf-a;review:m1",  # a fork child's event world
        "hyp:cf-a;reviewed:m1",  # a fork child's lineage-scoped ledger id
    ],
)
def test_a_step_in_an_authority_namespace_is_wrapped_and_cannot_alias(name):
    """The successor to the denylist pin: the name is ACCEPTED and made disjoint."""
    from effective.domain import CallTool
    from effective.handlers.base import op_key
    from effective.ops import Step

    op = Step(name=name, op=CallTool(name="x", args={}, result_schema=dict))
    composed = op_key(op).stored()
    assert composed != name, "an author's Step composed the substrate's own authority key"
    assert composed == f"step;{name}", composed
    # The authority key itself is what the substrate mints; nothing an author writes reaches it.
    assert not composed.startswith(name)


def test_a_step_name_that_is_not_a_KEY_is_still_refused():
    """A substrate-looking name that raises, for a reason unrelated to any denylist.

    `govern:spend:r1:0` carries separators, so `step_key` reads it as a KEY, and it is not one:
    a term holds one `:`. The refusal is the GRAMMAR's, not a reserved-name check, and it fires
    identically for `mytool:a:b:c` in no substrate namespace at all."""
    from effective.domain import CallTool
    from effective.handlers.base import op_key
    from effective.ops import Step

    for name in ("govern:spend:r1:0", "mytool:a:b:c"):
        op = Step(name=name, op=CallTool(name="x", args={}, result_schema=dict))
        with pytest.raises(ValueError, match="carries a separator, so it is read as a KEY"):
            op_key(op)


def test_author_namespaces_still_compose():
    """The fence must not cost the author anything — every real step-name prefix still works."""
    from effective.domain import CallTool
    from effective.handlers.base import op_key
    from effective.ops import Step

    for name in ["tool:x", "act:1", "react:edit", "skill:s,activate", "extract:m1", "code:run"]:
        op = Step(name=name, op=CallTool(name="x", args={}, result_schema=dict))
        # The ARM, not the bare name: every author prefix composes, none is refused, and the key
        # says which arm it is — which is what makes the fence unnecessary rather than lenient.
        assert op_key(op).stored() == f"step;{name}"


def test_the_reserved_authority_set_is_DERIVED_from_the_declarations_both_ways():
    """The set is derived, not curated, and the derivation runs in both directions.

    "Every namespace the substrate MINTS must be reserved" distinguishes nothing, since every
    identity routes through `compose_key`: `review:` and `extracted:` are composed by the
    substrate too, and they are author/domain names that need no fence. So the composition site
    DECLARES the kind (`AuthorityTag`), and this checks that
    (a) every declared authority namespace is reserved, and (b) no reserved authority prefix lacks
    a declaration — a fence guarding a namespace nothing mints is drift too."""
    from pathlib import Path as _P

    from effective.lint import check_authority_tags

    root = _P(__file__).parent.parent / "src"
    files = [f for tree in ("effective", "agent") for f in (root / tree).rglob("*.py")]
    offenders = [str(v) for v in check_authority_tags(files)]
    assert offenders == [], offenders


# --- the key source map --------------------------------------------------------------------
#
# The registry and the tag-uniqueness gate are ONE artifact: registering a shape is how you
# discover two shapes claiming a tag (the `processed:` collision that made naive positional
# composition unsound). These pin both jobs, plus the property that makes decoding possible at
# all: interior holes delimiter-free by type, terminal hole verbatim.


def _registry():
    from effective.lint import build_key_registry

    root = Path(__file__).parent.parent / "src"
    files = [f for tree in ("effective", "agent") for f in (root / tree).rglob("*.py")]
    return build_key_registry(files)


def test_no_namespace_is_owned_by_two_template_shapes():
    """The gate. Two shapes under one tag means a reader cannot tell which produced a key."""
    _shapes, problems = _registry()
    assert [str(p) for p in problems] == [], [str(p) for p in problems]


def test_every_shipping_namespace_is_registered_with_its_producing_site():
    """The map must cover the namespaces the substrate actually mints — if a tag is missing, a key
    bearing it cannot be explained, which is the whole point of the artifact."""
    from effective.keys.registry import KeyMap

    shapes, _ = _registry()
    keymap = KeyMap.from_shapes(shapes)
    for tag in ("event", "ledger", "artifact", "sleep", "approve", "govern", "fork", "hyp"):
        assert tag in keymap.variants, f"{tag!r} is not registered"
        assert all(":" in v.site for v in keymap.variants[tag])  # file:line
        assert any(v.fields for v in keymap.variants[tag]), f"{tag!r} registered no fields"


def test_a_key_decodes_back_into_its_named_fields_and_producing_line():
    """The payoff: read a key off the wire, recover how it was produced. The field names come from
    the template's own source EXPRESSIONS — which is what an f-string cannot carry."""
    from effective.keys.registry import KeyMap

    shapes, _ = _registry()
    keymap = KeyMap.from_shapes(shapes)

    forked = keymap.explain("fork:r-fork;review:m1")
    # `child_run_id`, not `self._child_run_id`: the fork namespace is composed by
    # `fork_event_name`, a named function with a domain parameter. The decoded field names come
    # from the template's source EXPRESSIONS, so moving a composer changes what a key says about
    # itself.
    assert forked.bindings == {"child_run_id": "r-fork", "name": "review:m1"}
    assert forked.site.endswith("absurd.py:219") or "absurd.py" in forked.site

    # the terminal hole keeps its delimiters — readable AND decodable, the §2 dependency
    assert keymap.explain("hyp:cf-a;reviewed:m1").bindings["event_id"] == "reviewed:m1"

    # A positional key decodes to its ordinal, and its producing line is the walk that assigned
    # it — the source map pointing at where the identity is DECIDED rather than where the op is
    # interpreted, which is what makes it useful for a sleep with no name of its own.
    slept = keymap.explain("sleep:3")
    assert slept.bindings == {"n": "3"}
    # `in`, not a prefix test: the registry records the site absolute under pytest and relative
    # standalone. `placing()` in `handlers/base.py` composes the sleep key, so the map points at
    # the walk that assigned it rather than at the counter.
    assert "handlers/base.py" in slept.site
    # a multi-field shape splits left-to-right, because interior holes cannot carry the delimiter
    # `govern` has arity 4: the op component is what makes an approval settle exactly one op.
    # The field names are `gate`/`run_id`/…, not `self.gate`/`self.run_id`/…: a template's hole
    # EXPRESSIONS are what the source map records, so binding the values to names before composing
    # is what turns a decode from "what the author typed" into "what the field is called".
    # Optional coordinates are NAMED on the wire and OMITTED at their default (§1b), so `pass_n=0`
    # is absent from the bytes and absent from the decode. `park_name` is asked for the spelling
    # rather than it being restated here.
    assert keymap.explain(
        GateState(
            gate="spend", run_id="r-1", op_key="step;tool:charge_card", occurrence=2
        ).park_name.stored()
    ).bindings == {
        "gate": "spend",
        "run_id": "r-1",
        "occurrence": "2",
        "op_key": "step;tool:charge_card",
    }


def test_the_occurrence_suffix_is_decoded_as_its_own_coordinate_not_swallowed():
    """`Key.occurrence`'s `#N` is appended AFTER composition, so no registered variant knows
    about it, and a trailing hole would absorb it in silence.

    That decode reports the op of `explain('approve;r1:tool:charge_card#2')` as
    `'r1:tool:charge_card#2'`, naming the wrong op for exactly the keys the coordinate exists to
    disambiguate, and it never raises. The source map is the first rung of the operator's
    diagnosis ladder, and a wrong answer there is worse than no answer.

    Pinned on BOTH producers of the suffix, because one rule covers them: the engines append it
    to a repeated `Step` name and the substrate to a repeated authority name."""
    from effective.keys.registry import KeyMap

    keymap = KeyMap.from_shapes(_registry()[0])

    approve = keymap.explain("approve;r1;tool:charge_card#2")
    assert approve.bindings["placed_key(op)"] == "r1;tool:charge_card", approve.bindings
    assert approve.occurrence == 2
    # `key` echoes the caller verbatim: a name pasted off a checkpoint row comes back as pasted.
    assert approve.key == "approve;r1;tool:charge_card#2"

    # Unsuffixed is the first occurrence and reports nothing — `Key.occurrence` is the identity
    # at n <= 1, so there is no suffix to read and none to invent.
    assert keymap.explain("approve;r1;tool:charge_card").occurrence is None

    # Multi-digit, so the pattern is not "one trailing character". Composed rather than spelled,
    # and asserted as a PROPERTY rather than a shape: what this test owns is that the suffix is
    # split off, not what `depth-grant:`'s fields happen to be — that belongs to
    # `test_a_key_decodes_back_into_its_named_fields_and_producing_line`. A literal goes red when
    # the namespace gains a field, for a reason with nothing to do with occurrences.
    from effective.budget import depth_grant_name

    repeated = depth_grant_name("r1", depth=1, generation=0).occurrence(17).stored()
    decoded = keymap.explain(repeated)
    assert decoded.occurrence == 17
    assert not any("#" in value for value in decoded.bindings.values()), decoded.bindings

    # Form (a) is UNTOUCHED and stays distinguishable: `govern:` carries an occurrence as a
    # template FIELD, which the registry can see, so it lands in `bindings` and the below-the-
    # grammar field stays `None`. Two spellings of one idea, told apart.
    governed = keymap.explain("govern:spend,r-1,occurrence=2;step;tool:charge_card")
    assert governed.bindings["occurrence"] == "2"
    assert governed.occurrence is None


def test_a_hash_inside_a_VALUE_is_unwritable_which_is_what_the_suffix_rests_on():
    """`#` means occurrence and nothing else, in one position and nowhere else.

    The occurrence is a suffix in the grammar, and the scheme rests on this: an atom charset that
    admits no `#` is why an author cannot spell an occurrence, and therefore why
    `step_key("tool:a").occurrence(2)` has exactly one producer. A `;occ:2` term spelling lacks
    the property, since `occ` would be an ordinary tag (the collision is pinned in the next test).

    **What re-opens it:** admitting `#` to `grammar.NAME`. Then a value can carry one, and
    `_peel_occurrence`, which takes everything before the LAST `#`, starts splitting keys at a
    byte the author chose."""
    from effective.keys.grammar import NAME, OCCURRENCE_SIGIL, Atom, KeySyntaxError

    assert OCCURRENCE_SIGIL not in NAME.pattern
    for tail in ("m1#draft", "m1#2x", "m1#"):
        with pytest.raises(KeySyntaxError, match="not a well-formed atom"):
            Atom.of(tail)


def test_the_occurrence_has_exactly_one_producer_which_the_term_spelling_did_not():
    """A term spelling of the occurrence puts it IN the language as an ordinary tag, so two
    producers mint one key, the engine's dup rule and an author's own structured step name:

        step_key("tool:a").occurrence(2)  ->  step;tool:a;occ:2
        step_key("tool:a;occ:2")          ->  step;tool:a;occ:2     COLLIDE

    The collision is a silent stale read: the second occurrence's thunk never runs and the author
    step's committed value is returned in its place. Under the suffix the author's name is just a
    name."""
    engine = step_key("tool:a").occurrence(2)
    author = step_key("tool:a;occ:2")
    assert engine.stored() == "step;tool:a#2"
    assert author.stored() == "step;tool:a;occ:2"
    assert engine != author


def test_round_trip_every_registered_shape_composes_and_decodes():
    """Property: for each registered shape, a synthetic key built to that shape decodes back to
    the values it was built from. Pins the compose/decode inverse across the WHOLE registry, so a
    new namespace cannot silently break decodability."""
    from effective.keys.registry import KeyMap

    shapes, _ = _registry()
    keymap = KeyMap.from_shapes(shapes)
    for tag, variants in keymap.variants.items():
        for shape in variants:
            key, expected = _witness(shape)
            assert keymap.explain(key).bindings == expected, (tag, shape.label())


def test_apply_and_unapply_are_inverses_over_every_form_a_key_ACTUALLY_takes():
    """The round-trip above, over the forms the substrate really ships rather than the bare one.

    **Ambiguity is a bug, not a tradeoff**: parsing is well defined, and both `apply` and
    `unapply` are tested to remove such bugs. If
    `compose_key` is `apply` and `explain` is `unapply`, then the property is that they are
    inverses, including for the two things applied to a key after composition:

    * a FRAME, prepended by `Key.prefixed` when a `scoped(...)` encloses the op;
    * an OCCURRENCE, appended by `Key.occurrence` when one identity is asked twice.

    Both live outside the composed template, so a bare-key round trip cannot see either. A wrong
    `unapply` raises nothing: it binds the whole frame of `rec:0;ledger;reviewed:m1` into the
    recursion index, or reports an occurrence as part of the op key.

    So the property is exercised over the CROSS PRODUCT of what can be applied (bare, framed,
    occurrence-suffixed, and both) for every registered namespace, because "which namespaces get
    framed" is not a stable list and picking one would be a hand-count."""
    from effective.keys import Key
    from effective.keys.registry import KeyMap

    keymap = KeyMap.from_shapes(_registry()[0])
    frames = ("", "rec:0;", "rec:0;d:2;")
    occurrences = (1, 2, 17)
    checked = 0

    for variants in keymap.variants.values():
        for shape in variants:
            # `_witness` renders through the same skeleton walk `Shape.label()` uses, so the key
            # under test cannot drift from what the shape means.
            witness, expected = _witness(shape)
            bare = Key.parse(witness)
            for frame in frames:
                for nth in occurrences:
                    applied = bare.occurrence(nth).prefixed(frame)
                    decoded = keymap.explain(applied.stored())

                    assert decoded.bindings == expected, (applied.stored(), shape.label())
                    assert decoded.occurrence == (nth if nth > 1 else None), applied.stored()
                    assert tuple(a.key for a in decoded.frame) == tuple(
                        atom for atom in frame.split(TERM_SEPARATOR) if atom
                    ), applied.stored()
                    # `key` echoes what the caller held, so an operator's paste comes back whole.
                    assert decoded.key == applied.stored()
                    checked += 1

    # Anti-vacuity: a registry that failed to load would iterate nothing and pass in silence.
    assert checked > 200, f"the cross product collapsed to {checked} cases"


def test_an_unregistered_namespace_is_refused_with_a_pointer_to_the_fix():
    from effective.keys.registry import KeyMap, UnknownTag

    with pytest.raises(UnknownTag, match="not registered"):
        KeyMap.from_shapes([]).explain("nosuch:x")


def test_a_template_written_as_ADJACENT_literals_is_read_whole():
    """Wrapping a long template across two t-string literals is legal in 3.14 and natural, so the
    scanner must read the GROUP, not just the first segment.

    Reading only the first segment drops fields: `govern`'s park template wrapped to fit the line
    limit registers 3 fields instead of 5, and `explain` mis-decodes every govern key."""
    from effective.lint import _compose_key_templates, _module_consts, _shape_of

    src = (
        "from effective.keys import Segment, compose_key\n"
        "def f(a, b, c):\n"
        "    return compose_key(\n"
        '        t"wrapped:{Segment(a)}:{b}"\n'
        '        t":{c}"\n'
        "    )\n"
    )
    found = list(_compose_key_templates(src))
    assert len(found) == 1, "the concatenation group should be read once, whole"
    # `_module_consts` carries (constructor, literal) per name; `_shape_of` wants name -> literal,
    # the same projection `build_key_registry` does.
    consts = {name: literal for name, (_ctor, literal) in _module_consts(src).items()}
    tag, fields = _shape_of(found[0][1], consts)
    assert tag == "wrapped"
    assert fields == ("a", "b", "c"), f"lost a field across the wrap: {fields}"


# --- the scanner's soundness ---------------------------------------------------------------
#
# The registry lint promises anything unresolvable is "reported as UNREGISTRABLE, never silently
# skipped". A literal read by indexing past a quote character parses a single- or triple-quoted
# template's tag as `t'event2` / `""event3` and drops it with no registration AND no violation,
# and `ruff format` keeping the repo double-quoted would hide that. These pin the quote-form
# independence and the loud refusals.


def _scan(source: str):
    """(registered tags, unregistrable-violation count) for one module's source."""
    import tempfile
    from pathlib import Path as _P

    from effective.lint import build_key_registry, check_authority_tags

    with tempfile.TemporaryDirectory() as d:
        path = _P(d) / "m.py"
        path.write_text(source)
        shapes, _ = build_key_registry([path])
        unregistrable = [v for v in check_authority_tags([path]) if v.rule == "unregistrable-tag"]
    return [s.tag for s in shapes], len(unregistrable)


_HEAD = "from effective.keys import compose_key\n"


@pytest.mark.parametrize(
    "quoting",
    ['compose_key(t"ev:{n}")', "compose_key(t'ev:{n}')", 'compose_key(t"""ev:{n}""")'],
)
def test_the_scanner_reads_a_template_regardless_of_quote_form(quoting):
    """Structural read (`string_content`), not quote-indexing — so the gate does not depend on the
    formatter's quote preference."""
    tags, unregistrable = _scan(f"{_HEAD}def f(n): return {quoting}\n")
    assert tags == ["ev"], f"{quoting} was not registered"
    assert unregistrable == 0


def test_a_compound_static_namespace_is_refused_loudly_not_dropped():
    """`t"a:b:{n}"` is injective on the wire (statics are fixed) but UNDECODABLE: `explain` finds
    the namespace by splitting at the first delimiter. So it must be refused, and refused
    *audibly*: a silent drop is the defect."""
    tags, unregistrable = _scan(f'{_HEAD}def f(n): return compose_key(t"a:b:{{n}}")\n')
    assert tags == []
    assert unregistrable == 1


def test_a_truly_dynamic_namespace_is_refused_loudly():
    source = (
        "from effective.keys import Tag, compose_key\n"
        'def f(ns, n): return compose_key(t"{Tag(ns)}:{n}")\n'
    )
    tags, unregistrable = _scan(source)
    assert tags == []
    assert unregistrable == 1


def test_the_gate_scans_every_tree_that_mints_a_key():
    """`examples/` holds a live producer, so a scan that skips it lets a drifting namespace
    there trip no gate anywhere."""
    from pathlib import Path as _P

    from effective.lint import build_key_registry

    root = _P(__file__).parent.parent
    files = [
        f
        for tree in ("src/effective", "src/agent", "examples")
        for f in (root / tree).rglob("*.py")
    ]
    shapes, problems = build_key_registry(files)
    assert [str(p) for p in problems] == []
    # The claim is about the SCAN PATHS, so it must not be anchored to one namespace that a
    # refactor can move out of `examples/`: that would measure where one producer happens to
    # live. Ask whether ANY registered site is under `examples/`.
    scanned = {site for shape in shapes for site in shape.sites}
    assert any("/examples/" in site for site in scanned), (
        "no examples/ producer reached the registry — the scan paths regressed"
    )


def test_a_format_spec_or_conversion_is_refused():
    """The spec slot is owned by the DSL, not the author: this repo reads specs as DIRECTIVES on
    the data axis (`effective.channels`: `{x:role=system}`, `cache`). The key grammar defines no
    directive for it, so a spec is intent the processor will not honor, and dropping it silently
    composes bytes the template does not say (`t"n:{n:03d}"` as `n:7`). Reserved, not forbidden in
    principle: if a namespace ever needs canonical rendering, `compose_key` imposes it, never the
    call site.

    An absent spec is `''` while an absent conversion is `None`: it falls out of `__format__`,
    where `''` is the natural empty spec. Testing both the same way makes every plain hole look
    like it carries a spec."""

    with pytest.raises(ValueError, match="format spec"):
        compose_key(t"n:{7:03d}")
    with pytest.raises(ValueError, match="conversion"):
        compose_key(t"x:{'a'!r}")
    # a plain hole is unaffected: the asymmetry handled correctly
    assert (
        compose_key(t"event;{Key.parse('review:m1'):domain=address}").stored() == "event;review:m1"
    )


def test_only_a_real_nesting_can_produce_a_framed_key():
    """Three execution structures must not compose one key.

    Unfenced, `scoped(t"talk:{'a;b'}")`, a real `scoped(a, scoped(b))`, and
    `step("talk:a;b;tool:x")` all produce `talk:a;b;tool:x`, and `ReplayHandler` binds one
    workflow's trace to another with no mismatch. The key has exactly ONE producer, which is what
    injectivity means.

    `;` is an ordinary term separator that the composer does not refuse. Two of the three routes
    are separated STRUCTURALLY, by where the arm sits:

        real nesting      talk:a;b;step;tool:x     frame, frame, arm, name
        forged step name  step;talk:a;b;tool:x     arm, then whatever the author called it

    So an author may write `step("talk:a;b;tool:x")` and it cannot alias a nesting, because the arm
    is in front of it. The third route is a refusal: `'a;b'` is a bare `str` in a coordinate,
    refused by TYPE rather than by a check that has to know what a frame is."""
    from effective.api import scoped, step
    from effective.domain import CallTool
    from effective.handlers.recording import RecordingHandler

    def leaf():
        return (yield from step("tool:x", CallTool(name="t", args={}, result_schema=object)))

    def forged_atom():
        return (yield from scoped(compose_key(t"talk:{'a;b'}"), leaf))

    def real_nesting():
        return (
            yield from scoped(
                compose_key(t"talk:{Segment('a')}"), lambda: scoped(compose_key(t"b"), leaf)
            )
        )

    def forged_step_name():
        return (
            yield from step("talk:a;b;tool:x", CallTool(name="t", args={}, result_schema=object))
        )

    handler = RecordingHandler(responses={"tool:x": 1})
    handler.run(real_nesting)
    assert [e.key.stored() for e in handler.trace] == ["talk:a;b;step;tool:x"]

    # The forged STEP NAME is legal and mints a DIFFERENT key: the arm leads it, so it cannot be
    # read as a nesting. Distinctness is the property.
    forged = RecordingHandler(responses={"talk:a;b;tool:x": 1})
    forged.run(forged_step_name)
    assert [e.key.stored() for e in forged.trace] == ["step;talk:a;b;tool:x"]
    assert forged.trace[0].key.stored() != handler.trace[0].key.stored()

    # The forged ATOM is refused, by TYPE: a bare `str` may carry a separator, so it cannot
    # fill a coordinate. No frame-specific rule is involved or needed.
    with pytest.raises(ValueError, match="which may contain a separator"):
        RecordingHandler(responses={"tool:x": 1}).run(forged_atom)


def test_the_frame_delimiter_is_refused_at_every_forging_position():
    """Every position where a VALUE could carry `;` is fenced, and the STATIC position
    deliberately is not.

    `;` is the term separator, so `t"a;b:{1}"` is an author writing a two-term template and the
    composer has no business refusing it. There is nothing to forge: a frame IS a leading term, so
    writing one is composing one. The forgery to stop is a VALUE smuggling a boundary past a
    check, and every value position below refuses."""
    from effective.keys import Segment, Tag, scope_prefix

    with pytest.raises(ValueError, match="frame delimiter"):
        Segment("a;b")
    with pytest.raises(ValueError, match=TERM_SEPARATOR):
        Tag("a;b")
    # A STATIC `;` composes: two terms, said out loud, by an author who typed the separator.
    assert compose_key(t"a;b:{1}").stored() == "a;b:1"
    # A `str` in a coordinate is refused by TYPE, which needs no frame-specific rule and says
    # something truer: the value may contain ANY separator, not just this one.
    with pytest.raises(ValueError, match="which may contain a separator"):
        scope_prefix(compose_key(t"talk:{'a;b'}"))
    # `/` is refused too, for a DIFFERENT reason: it is the PATH separator, so an atom holding
    # one is not one atom.
    with pytest.raises(ValueError, match="path delimiter"):
        Segment("a/b")


def test_a_frame_path_splits_back_to_the_atoms_that_built_it():
    """`frame_path` accumulation is injective over atom LISTS.

    That is the whole license for the concat being a render backend: the obligation is
    discharged one level down, in `scope_prefix`, which refuses an atom carrying `;` and then
    terminates with one. So every segment ends at exactly one `;`, and the split is exact.

    Asserted over a set of atoms chosen to be adversarial about it: one carrying the KEY
    delimiter in its terminal hole (legal, and it must not become a frame boundary), and one
    that is a bare tag with no holes at all (the atom `b` that makes `talk:a;b;` reachable,
    the nesting the await axis collides with two tests down)."""
    from effective.keys import Segment, frame_path, scope_prefix

    atoms = [
        compose_key(t"talk:{Segment('a')}"),
        compose_key(t"b"),
        compose_key(t"rev:{Segment('r2')},{Segment('m1')}"),
    ]
    path = ""
    for atom in atoms:
        path = frame_path(path, atom)

    assert path == "talk:a;b;rev:r2,m1;"
    assert path.split(TERM_SEPARATOR)[:-1] == [atom.stored() for atom in atoms]
    # `frame_path` appends the atom's `scope_prefix` and nothing else.
    assert frame_path("x;", atoms[0]) == f"x;{scope_prefix(atoms[0])}"


def test_frame_path_refuses_an_atom_that_would_forge_a_boundary():
    """The refusal lives in `scope_prefix` and `frame_path` inherits it, pinned so that a typed
    `FramePath` cannot quietly relocate the check and leave the accumulator open."""
    from effective.keys import frame_path

    # The refusal is the composer's TYPE gate (a `str` coordinate may carry any separator, not
    # just this one), and `frame_path` never sees the value: the atom cannot be built, so the
    # accumulator cannot be handed a forged one.
    with pytest.raises(ValueError, match="which may contain a separator"):
        frame_path("talk:a;", compose_key(t"talk:{'x;y'}"))


def test_an_author_cannot_forge_a_frame_on_the_await_axis():
    """A forged frame on the await axis, closed at the AUTHOR SURFACE.

    `_PrefixedCtx.step` and `.await_event` apply the same frame prefix one line apart, so a
    forged frame in an await name and a real nesting compose one park name. `event_name` refuses
    the frame delimiter in author text, which makes the collision unwritable from a workflow.

    **The ctx-level composition is ambiguous if handed such a name**, as the assertion below
    shows, and closing THAT means typing `AwaitEvent.name` through both engines. What this test
    pins is the reachable half: no workflow can mint one."""
    from effective.handlers.absurd import _PrefixedCtx
    from effective.keys import scope_prefix

    class Sink:
        """A `TaskContext` that only records the await names it is handed."""

        def __init__(self) -> None:
            self.seen: list[str] = []

        def step(self, name: Key, thunk: Callable[[], object], /) -> object:
            return thunk()

        def await_event(self, name: Key, /) -> None:
            self.seen.append(name.stored())

        def sleep_until(self, when: datetime, /, *, name: Key | None = None) -> None:
            return None

    from effective.ops import event_name

    # The two axes answer DIFFERENTLY. A Step is addressed by its op KEY and the arm wraps that,
    # so `step;b;c` cannot be read as a nesting and needs no fence. An await is addressed by its
    # EVENT NAME, which no arm touches, so nothing structural sits between an author's `b;c` and a
    # real `scoped(b, scoped(c))`: the fence stays until the address gets an arm of its own.
    assert op_key(Step(name="b;c", op=_dop())).stored() == "step;b;c"
    with pytest.raises(ValueError, match="frame delimiter"):
        event_name("b;c")

    # And the ctx composition it protects: fed the forged name directly, the two collide,
    # which is why the guard is at the surface and why typing the axis is the remaining half.
    outer = scope_prefix(compose_key(t"talk:{Segment('a')}"))
    inner = scope_prefix(compose_key(t"b"))
    sink = Sink()
    _PrefixedCtx(sink, outer).await_event(Key.parse("b;c"))
    _PrefixedCtx(_PrefixedCtx(sink, outer), inner).await_event(Key.parse("c"))
    assert sink.seen[0] == sink.seen[1], "unreachable from a workflow, still ambiguous here"


def test_a_mime_type_is_a_PATH_inside_one_coordinate_not_a_terminal_exemption():
    """A MIME type is read structurally, with no hole exempted from the rules.

    `message/rfc822` IS `type/subtype` per RFC 2045, so it is a two-atom PATH in one coordinate,
    and `/` means path everywhere rather than "frame boundary except here". The digest is
    `sha256-` prefixed, so its kind is legible without asking what the field means."""
    key = compose_key(
        t"artifact:{Segment('message')}/{Segment('rfc822')},{Segment('sha256-9f2c')}"
    )
    assert key.stored() == "artifact:message/rfc822,sha256-9f2c"
    terms = key.terms()
    assert len(terms) == 1, "one term: an artifact key names one thing"
    assert [c.path for c in terms[0].coordinates] == ["message/rfc822", "sha256-9f2c"]


# --- the dual of `scope_prefix`: stripping frames back off ---------------------------------


def test_unframed_finds_a_boundary_by_the_TAG_that_follows_it():
    """The rule, and the two wrong rules it exists instead of.

    `rsplit("/", 1)` would answer `rfc822:9f2c` for a deployed artifact key; a left-strip would
    answer the same. Neither end of the string is a reliable boundary, because a terminal hole may
    contain `/`. The tag is reliable (that is the key grammar's claim to recoverable position), so
    a boundary counts only where a KNOWN tag follows it, earliest match winning."""
    from effective.keys import unframed

    tags = ("ledger;", "artifact:", "event;", "gather:")
    assert unframed("ledger;x", tags=tags) == "ledger;x"  # no frame
    assert unframed("rec:0;ledger;x", tags=tags) == "ledger;x"  # one scope frame
    assert unframed("rec:1;fold:2;ledger;x", tags=tags) == "ledger;x"  # nested scopes
    assert unframed("rec:0;gather:0,1;ledger;x", tags=tags) == "gather:0,1;ledger;x"
    # earliest match wins, which is what protects the deployed artifact key
    assert unframed("artifact:message/rfc822:9f2c", tags=tags) == "artifact:message/rfc822:9f2c"
    # no known tag anywhere: not a frame, returned unchanged
    assert unframed("rec:0;tool:a", tags=tags) == "rec:0;tool:a"


def test_kind_of_reads_through_frames():
    """A `startswith(tag)` consumer is wrong once a scope frame can appear. This is the
    projection half; the measured bridge is the other."""
    from effective.graphview import kind_of

    assert kind_of("rec:0;ledger;x") == "ledger"  # a prefix test answers "step"
    assert kind_of("rec:1;fold:2;ledger;x") == "ledger"
    assert kind_of("rec:0;gather:0,1;event;q") == "await"
    assert kind_of("artifact:message/rfc822:9f2c") == "artifact"  # the `/` is not a frame
    assert kind_of("rec:0;tool:a") == "step"  # an author op stays a step


def test_an_author_may_not_park_on_a_substrate_authority_name():
    """An author may not name an await after something the substrate parks on. Absurd delivers
    events by name, first-emit-wins across the whole queue, so an unguarded
    `await_event("approve;tool:act")` parks on the name the permission layer's `human` tier parks
    on and consumes the approval meant for the gate.

    A composed `Key` is trusted because it came from `compose_key` and is registry-visible; that
    is how the substrate's own parks pass, without a trust-by-location rule.

    **The await axis is the one that needs this fence, and the asymmetry is worth stating.** A
    Step is addressed by its op KEY, and the arm wraps that (`step;approve:x` cannot be
    `approve:x`), so a Step needs no fence at all. An await is addressed by its EVENT NAME, the
    wire address an emitter delivers to, and the arm term wraps only the checkpoint key
    (`event;{name}`), never the address. So there is nothing structural between an author's
    `await_event("approve:x")` and the gate's park, and `RESERVED_AUTHORITY_TAGS` is the thing
    standing there. Retiring it needs the ADDRESS to gain an arm, which is a separate change; do
    not delete this test on the strength of the arm term alone."""
    from effective.keys import RESERVED_AUTHORITY_TAGS
    from effective.ops import event_name

    for reserved in RESERVED_AUTHORITY_TAGS:
        with pytest.raises(ValueError, match="reserved authority namespace"):
            event_name(f"{reserved}:mine")
        # BOTH spellings, which is what makes this a TAG rather than a prefix: a tuple holding
        # `"approve:"` goes stale the moment an arm mints `approve;…`, and a fence that holds only
        # via a separate `;` refusal is one edit from not holding.
        with pytest.raises(ValueError, match="reserved authority namespace"):
            event_name(f"{reserved};mine")

    assert event_name("review:m1").stored() == "review:m1"  # an author's own subject is untouched
    # The substrate's own, composed — `approve:x`, not `approve;x`. The `:` here is a TAG
    # separator, not a term separator, and an expected value is exactly where a mechanical rewrite
    # of one for the other lands unseen: no producer changes, and the assertion still passes.
    #
    # **Composed through the `APPROVE` constant, not a bare `t"approve:…"` static.** Both mint the
    # same bytes, and only the constant carries the `Scope` — which is what separates a
    # substrate-composed authority key from a `Key.parse`d one at this fence. A literal is a
    # shortcut that stops modelling production, and `just lint --authority-tags` requires the
    # constant at every real site.
    from effective.permission import APPROVE

    assert event_name(compose_key(t"{APPROVE}:{Segment('x')}")).stored() == "approve:x"


# --- namespaces as TAGGED UNIONS: separation as a structural property ----------------------
#
# One shape per tag, enforced by ARITY, is a proxy rather than the property: it is what a
# registry must fall back on when `Shape` records no statics and `explain` can only split
# positionally, which mis-binds any template with an interior literal. These pin the real rule —
# a namespace may own several variants when `separated` proves their languages disjoint.


def test_the_decoder_consumes_a_literal_instead_of_swallowing_it():
    """With only (tag, holes) recorded, a decode of `code:seg,0,foo` binds `seg='seg:0'`: the
    literal absorbed into the value. Recording the statics lets the decoder consume them."""
    from effective.keys.registry import KeyMap

    shape = _shape("ns:", ..., ",seg,", ..., fields=("name", "n"))
    keymap = KeyMap.from_shapes([shape])
    assert keymap.explain("ns:foo,seg,0").bindings == {"name": "foo", "n": "0"}


def test_two_variants_of_one_namespace_decode_to_their_own_fields():
    from effective.keys.registry import KeyMap

    keymap = KeyMap.from_shapes(
        [
            _shape("ns:", ..., ",seg,", ..., fields=("name", "n"), site="a:1"),
            _shape("ns:", ..., ",act,", ..., ",", ..., fields=("name", "j", "tool"), site="b:2"),
        ]
    )
    assert keymap.explain("ns:x,seg,0").bindings == {"name": "x", "n": "0"}
    assert keymap.explain("ns:x,act,3,read").bindings == {"name": "x", "j": "3", "tool": "read"}


def test_a_key_matching_no_variant_is_refused_by_name():
    from effective.keys.registry import KeyMap

    keymap = KeyMap.from_shapes(
        [_shape("ns:", ..., ",seg,", ..., fields=("name", "n"), site="a:1")]
    )
    with pytest.raises(ValueError, match="matches no registered variant"):
        keymap.explain("ns:x:other:0")


def test_separation_admits_a_discriminator_and_refuses_a_literal_facing_a_hole():
    """Both cases must be reached AT THE SAME INDEX. Comparing a 2-slot variant against a 1-slot
    one lets `zip` run out before the literal-vs-hole position, so the assertion holds for the
    wrong reason and a mutation that makes `Lit` facing `Fld` count as separation survives. The
    pair below puts them at index 1 on both sides."""
    from effective.keys.registry import separated

    seg = _shape("ns:", ..., ",seg,", ..., fields=("name", "n"))
    act = _shape("ns:", ..., ",act,", ..., fields=("name", "n"))
    hole = _shape("ns:", ..., ",", ..., ",", ..., fields=("name", "kind", "rest"))

    assert separated(seg, act)[0], "distinct literals at a fixed coordinate ARE separation"
    ok, why = separated(seg, hole)
    assert not ok, "a literal facing a hole is NOT separation"
    assert "no discriminating coordinate" in why
    # and the overlap is real: this key is in BOTH languages
    assert seg.decode("ns:x,seg,0") == {"name": "x", "n": "0"}
    assert hole.decode("ns:x,seg,0") == {"name": "x", "kind": "seg", "rest": "0"}


def test_the_registry_refuses_an_unseparated_pair_and_names_both_sites():
    """The probe is an OPTIONAL-COORDINATE overlap, deliberately.

    A flat-form pair (`ns:{n}:seg:{i}`) is refused by the grammar as unregistrable before
    `separated` is ever consulted, so it would measure the parser rather than the rule. The pair
    below is the live hazard: `ns:{n}` and `ns:{n},d={d:default=0}` BOTH render `ns:x`, because an
    optional at its default is omitted. One namespace, two variants, one string in both languages.

    This test pins the hazard at the REGISTRY level;
    `test_ARITY_SEPARATES_which_the_flat_grammar_could_not_say` carries the direct
    `separated`-level assertion."""
    from effective.lint import build_key_registry

    src = (
        "from effective.keys import Segment, compose_key\n"
        'def a(n): return compose_key(t"ns:{Segment(n)}")\n'
        'def b(n, d): return compose_key(t"ns:{Segment(n)},d={d:default=0}")\n'
    )
    path = Path("build/_sep_probe.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(src)
    try:
        _shapes, problems = build_key_registry([path])
        assert [p.rule for p in problems] == ["tag-variants-not-separated"]
        assert "not provably disjoint" in problems[0].message
    finally:
        path.unlink()


def test_a_hole_followed_by_a_static_is_INTERIOR_and_must_be_delimiter_free():
    """EVERY hole is delimiter-free by type, in every position, so there is no last-position
    special case.

    `skill:{s},activate` is an ordinary two-coordinate term, and the property that a delimiter in
    a value cannot slide a field boundary is held by `Segment` refusing one at construction."""
    assert compose_key(t"skill:{Segment('s')},activate").stored() == "skill:s,activate"
    with pytest.raises(ValueError, match="separator"):
        Segment("s,activate")


def test_a_readable_delimiter_bearing_NAME_survives_as_a_splice():
    """A readable author-facing name keeps its delimiters, because it is not a value.

    `fork:c;review:m1` composes and reads: a nested `Key` spliced in by induction carries it, so
    the structure is stated rather than smuggled as a raw `str` in a last hole."""
    from effective.keys import Key, Segment

    inner = Key.parse("review:m1")
    assert (
        compose_key(t"fork:{Segment('c')};{inner:domain=address}").stored() == "fork:c;review:m1"
    )
    # and the same name as a raw string is refused
    with pytest.raises(ValueError, match="no last-position exemption"):
        compose_key(t"fork:{Segment('c')};{'review:m1':domain=address}")


def test_the_skill_namespace_ships_as_a_union_of_two_variants():
    """The live payoff: one tag, `skill:`, owns two shapes."""
    from effective.keys.registry import KeyMap

    shapes, problems = _registry()
    assert problems == []
    keymap = KeyMap.from_shapes(shapes)
    labels = {v.label() for v in keymap.variants["skill"]}
    assert labels == {"skill:{name},activate", "skill:{name},refresh,{n}"}
    assert keymap.explain("skill:sk1,activate").bindings == {"name": "sk1"}
    assert keymap.explain("skill:sk1,refresh,2").bindings == {"name": "sk1", "n": "2"}


def test_a_variant_refuses_a_key_with_MORE_than_it_declares():
    """Separation's arity ground is only sound if a decode is EXACT.

    A `decode` that ignores trailing segments leaves every other pin green and lets
    `skill:{name},activate` swallow a longer sibling. The two ways to be "longer" are separable
    and both are refused: an extra COORDINATE on the term, and an extra TERM after it."""
    from effective.keys.grammar import Hole, parse_skeleton
    from effective.keys.registry import Shape

    shape = Shape(skeleton=parse_skeleton(["skill:", Hole(0), ",activate"]), fields=("name",))
    assert shape.decode("skill:x,activate") == {"name": "x"}
    assert shape.decode("skill:x,activate,extra") is None  # one coordinate too many
    assert shape.decode("skill:x,activate;tool:t") is None  # one term too many


def test_ARITY_SEPARATES_which_the_flat_grammar_could_not_say():
    """Different coordinate counts are different languages.

    A grammar with an absorbing tail cannot say this: a tail swallows any number of segments, so
    `depth-grant:` at 2 and 3 coordinates is *not* provably disjoint. A coordinate count read from
    the bytes makes it so. An empty coordinate is refused AT THE PARSE, which is asserted in
    `test_grammar.py::test_a_coordinate_at_its_default_is_omitted_not_left_empty`.

    **Read the name narrowly.** With only REQUIRED coordinates, declared count equals rendered
    count, so a rule comparing declared counts passes too. The optional-coordinate half is
    asserted below, in the same test, because the two halves are one property."""
    from effective.keys.registry import separated

    one = _shape("ns:", ..., fields=("a",))
    two = _shape("ns:", ..., ",", ..., fields=("a", "b"))
    ok, why = separated(one, two)
    assert ok, why
    assert "arit" in why, why  # 'arity' or 'arities': the ground, not the wording
    # and neither accepts the other's keys, which is what "disjoint" has to mean
    assert one.decode("ns:q,extra") is None
    assert two.decode("ns:q") is None

    # THE OTHER DIRECTION, and it is the one that bites. Both coordinates above are REQUIRED, so
    # declared count == rendered count and this pair separates under the sound rule AND under one
    # comparing declared counts. Make the second coordinate OPTIONAL and the counts
    # come apart: `ns:{a}` and `ns:{a},d={d:default=0}` both render `ns:v`, so they are NOT
    # disjoint and a rule comparing declared counts says they are.
    optional = _shape("ns:", ..., ",d=", ..., fields=("a", "d"))
    overlaps, why = separated(one, optional)
    assert not overlaps, why
    assert "no discriminating coordinate" in why, why


# --- THE POLICY: every key well formed, no exceptions ----------------------------------------
#
# A key is a typed, inductively formed path of `name:value` constructor groups. Names and values
# are delimiter-free and non-empty; a value that WANTS delimiters is not a value, it is a nested
# path, and nesting happens by INDUCTION inside `compose_key` — never by composing a path outside
# and handing the result in as opaque text. A caller whose natural format carries the delimiter
# picks a delimiter-free format (`sleep:`'s ISO timestamp swaps `:` for `;` — readable, reversible
# because ISO-8601 has no `;`). That is what removes the last exception.


def test_an_empty_string_is_not_a_name_or_a_value():
    """An empty segment is a value that is not a value, and admitting one makes `separated`'s
    arity ground unsound."""
    with pytest.raises(ValueError, match="empty"):
        Segment("")


def test_a_segment_refuses_a_non_str_so_an_opaque_key_cannot_be_laundered_into_one():
    """`Key` has no `__str__` so that flattening it is an explicit act. A `Segment` that took
    `value: object` and called `str(value)` would hand that protection back one class over: the
    repr becomes a segment, and the segment becomes an identity.

    A *tagged* key's repr contains a `:`, so the delimiter check refuses it: the right answer for
    the wrong reason. A **tag-only** key's repr does not, so it would compose as
    `x:Key(_value='tagonly', _scope=None):1`. Both are asserted, because a fix that only reddens
    the first leaves the reachable half open."""
    from effective.keys.grammar import KeySyntaxError

    tag_only = compose_key(t"tagonly")
    tagged = compose_key(t"ns:{Segment('abc')}")
    assert ":" not in repr(tag_only)  # the reachable half: nothing refused this repr
    assert ":" in repr(tagged)  # the other half, refused for the wrong reason

    for opaque in (tag_only, tagged):
        with pytest.raises(TypeError, match="not a str"):
            # ty: ignore[invalid-argument-type], DELIBERATE. The narrowed signature makes `ty`
            # reject this statically. The runtime check still matters: a t-string hole's value is
            # not type-checked against anything, so an untyped caller reaches `Segment` with `ty`
            # green. This asserts that second line of defense.
            Segment(opaque)

    # And the refusal is a narrowing, not a wall: the two types it names as belonging in a hole
    # DIRECTLY still compose there, which is what makes `Segment` the wrong wrapper rather than a
    # missing one. **In their own positions**: a `Key` is not an atom that looks like a key, it
    # is a term sequence, so it SPLICES after `;` and a
    # coordinate refuses it outright. `grammar.Kind` says the same thing from the lexer's side by
    # having no `key` kind at all.
    assert compose_key(t"x:{1};{tag_only:domain=any}").stored() == "x:1;tagonly"
    assert compose_key(t"x:{Segment('ab')},{1}").stored() == "x:ab,1"
    with pytest.raises(KeySyntaxError, match="not a well-formed atom"):
        compose_key(t"x:{tag_only},{1}")  # a coordinate holds an ATOM, and a key is not one


def test_a_segment_refuses_EVERY_metacharacter_the_grammar_declares():
    """The fence set is `grammar.METACHARACTERS`, and this asserts `Segment` covers all of it.

    A metacharacter missing from `Segment`'s fence is invisible end to end: `Atom.of` refuses it
    at fill time, so every composition test stays green while the brand lies
    (`Segment("s,activate")` would be two coordinates wearing a one-atom brand).

    Iterating the constant is what makes this a gate rather than a hand-written case per
    character. A new metacharacter with no fence reddens here on the commit that adds it."""
    for meta in METACHARACTERS:
        with pytest.raises(ValueError, match=re.escape(repr(meta))):
            Segment(f"a{meta}b")
        # …and terminally, which is where `#` actually appears and where a naive `in` check on a
        # value's interior would have passed it through.
        with pytest.raises(ValueError, match=re.escape(repr(meta))):
            Segment(f"ab{meta}2")


def test_the_empty_key_is_not_a_key():
    """The same policy one type up. `Segment("")` is refused above because an empty value is a
    value that is not a value; an empty KEY is the degenerate case of the injectivity this whole
    module is about: it addresses nothing, and every empty key is the same key, so it collides
    with itself wherever it is stored.

    Both doors are asserted, because `parse` is the one route with its own name and would be the
    tempting place to put a check that belongs on the constructor."""
    with pytest.raises(ValueError, match="empty key"):
        Key(ParsedKey(()))
    with pytest.raises(KeySyntaxError, match="empty key"):
        Key.parse("")


def test_a_key_refuses_anything_that_is_not_a_parsed_key():
    """`Segment`'s laundering hole, one type over and closed the same way.

    A `Key` that held text would store a repr for `Key(some_key)` and `'None'` for `Key(None)`, a
    name that reads as an absence and behaves as a perfectly good identity. `Key` holds TERMS, so
    neither can be constructed: text included, because text comes in through `parse`.

    Written as separate calls rather than a loop, and that is not a style choice. A loop over
    `(a_key, None, 7, "govern:r1")` widens the element to a union that `ty` does not flag, so the
    ignores below would be unnecessary and the comment claiming `ty` rejects these would be false
    while looking verified. `ty` reports `invalid-argument-type` on each form."""
    with pytest.raises(TypeError, match="not a ParsedKey"):
        # ty: ignore[invalid-argument-type], DELIBERATE: the same second line of defense
        # `Segment`'s twin above documents. `ty` rejects this statically, while a t-string hole's
        # value is not type-checked against anything.
        Key(compose_key(t"tagonly"))
    with pytest.raises(TypeError, match="not a ParsedKey"):
        Key(None)  # ty: ignore[invalid-argument-type] — DELIBERATE, as above
    with pytest.raises(TypeError, match="not a ParsedKey"):
        Key(7)  # ty: ignore[invalid-argument-type] — DELIBERATE, as above
    with pytest.raises(TypeError, match="not a ParsedKey"):
        # TEXT is refused too: text enters through the door that has a name, `Key.parse`.
        Key("govern:r1")  # ty: ignore[invalid-argument-type] — DELIBERATE, as above


def test_a_delimiter_bearing_value_is_refused_in_EVERY_position():
    """No terminal exemption: the LAST hole does not take the rest of the string.

    That exemption is only a coherent idea in a grammar that cannot say where a field ends. `,`
    ends a coordinate and `;` ends a term, so a value carrying a separator is not a value: it is a
    structure being flattened in, the Bobby Tables shape."""
    with pytest.raises(ValueError, match="delimiter"):
        compose_key(t"ns:{'a:b'}")


def test_a_closed_key_splices_in_any_position():
    """Induction. A `Key` is a CLOSED sequence of terms (its width is fixed by construction), so
    it carries no ambiguity wherever it sits, and the composer may splice it interior as well as
    terminal.

    **A splice is a `;`, and only a `;`.** A COORDINATE holds an atom, and an atom cannot contain
    a separator, so the composer refuses a key there (`t"artifact:{inner}"`); `;` is the operator
    that means "these terms extend the sequence". The tag here is the test's own: borrowing
    `artifact:` would put a second shape under a namespace production owns."""
    inner = compose_key(t"rfc822:{Segment('sha256-015abd')}")
    assert compose_key(t"wrap;{inner:domain=any}").stored() == "wrap;rfc822:sha256-015abd"
    assert (
        compose_key(t"wrap;{inner:domain=any};tail:{Segment('x')}").stored()
        == "wrap;rfc822:sha256-015abd;tail:x"
    )


def test_no_op_key_flattens_a_path_into_a_value():
    """The universal statement, over the arms that had exceptions. A key's segment count is a
    property of its STRUCTURE, and a value that smuggles delimiters breaks that.

    A sleep raises rather than appearing here: its wake time is DATA, so it is kept as a value
    instead of being respelled into an identity. What this covers is our own nested path composed
    outside the composer (`artifact_id`)."""
    from effective.ops import StoreArtifact

    artifact_key = op_key(StoreArtifact(value={"a": 1}, content_type="message/rfc822"))

    # `artifact:` is a NESTED path (content-type, digest), and what it must not be is a flattened
    # opaque value. ONE term, TWO coordinates: a `type/subtype` PATH and an algorithm-prefixed
    # digest, `sha256-` so its kind is legible.
    assert artifact_key.stored().startswith("artifact:message/rfc822,sha256-")
    assert len(artifact_key.stored().split(",")) == 2, artifact_key.stored()


# --- positional identity: a sleep is the n-th sleep, not its wake time ------------------------
#
# `op_key(Gather)` already RAISES and states the principle: a gather's identity is positional —
# the g-th gather in this frame — "not a content function of one op". `SleepUntil` is the same
# category. A sleep has no author-given name, so the only
# identity available to it is where the walk found it; its wake time is DATA, and it is already
# serialized as data (on Absurd the checkpoint's own state IS the ISO string — the same bytes the
# name carries; on SQLite nothing is written at all).
#
# Honest scope, stated once: two same-instant sleeps do NOT break a run today. The Absurd SDK's
# `#k` occurrence counter separates the checkpoints below our seam. The defect is that `op_key`
# is not the thing doing that work, which is why the collision below is real and the run is fine.


def test_op_key_of_a_sleep_raises_no_content_key():
    """A sleep has no standalone op key: its identity is the walk's ordinal, and its wake time
    is data. `op_key` says so loudly rather than returning a name two distinct sleeps share."""
    from datetime import UTC, datetime

    with pytest.raises(ValueError, match="no standalone op key"):
        op_key(SleepUntil(when=datetime(2026, 7, 25, 12, 0, tzinfo=UTC)))


def test_two_sleeps_in_one_frame_are_two_identities():
    """The attack: run a workflow that sleeps twice to the SAME instant, and try to make the
    recorded trace tell the two apart. It cannot, so replay has nothing to bind by.

    Driven through the recorder rather than `op_key` directly, deliberately: the walk is what
    will assign the coordinate, so the walk is what the pin must exercise. A step between the
    sleeps proves both were traversed."""
    from datetime import UTC, datetime

    from effective.api import sleep_until, step
    from effective.domain import CallTool
    from effective.handlers.recording import RecordingHandler

    wake = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)

    def wf():
        yield from sleep_until(wake)
        yield from step("tool:a", CallTool(name="a", args={}, result_schema=dict))
        yield from sleep_until(wake)
        return None

    handler = RecordingHandler(responses={"tool:a": {}})
    handler.run(wf)
    sleeps = [e.key.stored() for e in handler.trace if e.key.stored().startswith("sleep:")]

    assert len(sleeps) == 2, sleeps
    assert len(set(sleeps)) == 2, f"two distinct sleeps must not share one key: {sleeps}"


# --- the authority-SCOPE gate ---------------------------------------------------------------
#
# `--authority-tags` asks whether a namespace is FENCED. These ask the question a template cannot
# answer (do a shape's fields determine the op-occurrence?), which is why the namespace declares
# its `Scope` and the gate holds it to the declaration.
#
# Every pin here is MUTATION-CHECKED against its own fix: `_scoped_scan` runs the rule
# over a synthetic module, so each test can state the defect it guards *and produce it*, rather
# than asserting green over `src/` and hoping the rule can fail.


def _scoped_scan(source: str) -> list[str]:
    """The `--authority-scopes` rules that fire over one synthetic module."""
    import tempfile
    from pathlib import Path as _P

    from effective.lint import check_authority_scopes

    with tempfile.TemporaryDirectory() as d:
        path = _P(d) / "m.py"
        path.write_text(source)
        return [v.rule for v in check_authority_scopes([path])]


_SCOPE_HEAD = "from effective.keys import AuthorityTag, Key, Scope, Segment, compose_key\n"


def test_QUALIFIED_refuses_a_wrapped_hole_that_is_not_a_Key():
    """`QUALIFIED` says the occurrence question RECURSES into the key it wraps. A `str` there is
    nothing to recurse into, and launders a hand-rolled name into the wrapping namespace."""
    untyped = f'{_SCOPE_HEAD}T = AuthorityTag("t", scope=Scope.QUALIFIED)\n'
    untyped += (
        'def n(child: str, name: str) -> Key: return compose_key(t"{T}:{Segment(child)}:{name}")\n'
    )
    assert _scoped_scan(untyped) == ["qualified-wraps-an-untyped-name"]

    typed = untyped.replace("name: str)", "name: Key)")
    assert _scoped_scan(typed) == []


def test_a_declaration_with_no_scope_names_the_axis_it_is_missing():
    """Unreachable through a ty-clean tree (`scope` is a required keyword), so this covers the
    untyped consumer — and the message has to say WHICH axis, not echo a TypeError."""
    assert _scoped_scan(f'{_SCOPE_HEAD}T = AuthorityTag("t")\n') == ["undeclared-authority-scope"]


def test_the_walk_coordinates_EVERY_settlement_await():
    """`placing` applies `Key.occurrence` to every `Scope.SETTLEMENT` await, in every walk, with
    no namespace filter, so no namespace has to supply its own coordinate.

    This asserts the property at the level it holds: the DISPATCH. The namespace's declared
    `Scope` decides, so a new authority namespace inherits the coordinate by declaring
    `SETTLEMENT` and nothing has to remember to check it. A lint pin on the gate's output would
    stay green over a false claim about the runtime."""
    from pydantic import BaseModel

    from effective.budget import chain_grant_name, depth_grant_name
    from effective.handlers.base import placed_await_name, placing
    from effective.keys import FramePosition, Scope
    from effective.ops import AwaitEvent

    class _Ack(BaseModel):
        ok: bool = True

    for name in (depth_grant_name("r1", generation=0, depth=1), chain_grant_name("r1", 0)):
        assert name.scope is Scope.SETTLEMENT, name
        op, position = AwaitEvent(name=name, schema=_Ack), FramePosition()
        placed = []
        for _ask in range(3):
            with placing(op, position):
                placed.append(placed_await_name(op).stored())
        # Byte-preserving at the first ask; a coordinate on every one after it.
        assert placed == [name.stored(), f"{name.stored()}#2", f"{name.stored()}#3"], placed


def test_the_walk_leaves_an_ACCRUAL_await_alone():
    """The other arm of the same dispatch, and the reason it is not "always suffix": a
    `budget-grant:` answer RAISES the run's ceiling and must not be re-requested at the next op.
    First-emit-wins is the feature, so `ACCRUAL` gets no coordinate."""
    from pydantic import BaseModel

    from effective.budget import budget_grant_name
    from effective.handlers.base import placed_await_name, placing
    from effective.keys import FramePosition, Scope
    from effective.ops import AwaitEvent

    class _Ack(BaseModel):
        ok: bool = True

    name = budget_grant_name("r1", 0)
    assert name.scope is Scope.ACCRUAL, name
    op, position = AwaitEvent(name=name, schema=_Ack), FramePosition()
    for _ask in range(3):
        with placing(op, position):
            assert placed_await_name(op).stored() == name.stored()


# --- the scope scanner's reading of hard shapes --------------------------------------------


@pytest.mark.parametrize(
    "declaration",
    [
        'T = AuthorityTag(value="t", scope=Scope.SETTLEMENT)',
        'NAME = "t"\nT = AuthorityTag(NAME, scope=Scope.SETTLEMENT)',
        'S = Scope.SETTLEMENT\nT = AuthorityTag("t", scope=S)',
        'T = AuthorityTag("t", scope=Scope.SETTLEMENT if flag else Scope.ACCRUAL)',
    ],
)
def test_a_declaration_this_scanner_cannot_read_is_REPORTED_not_skipped(declaration):
    """A declaration the scanner cannot resolve is reported, never silently skipped, which is the
    contract `_shape_of` states one section up. The conditional is worse than silent if read
    naively: `rsplit('.', 1)[-1]` takes its last member, so a namespace whose true branch is
    `SETTLEMENT` resolves to the PERMISSIVE `ACCRUAL` and goes unchecked."""
    source = f"{_SCOPE_HEAD}flag = True\n{declaration}\n"
    source += 'def n(run: str, d: int) -> Key: return compose_key(t"{T}:{Segment(run)}:{d}")\n'
    assert _scoped_scan(source) == ["unresolvable-authority-declaration"]


def test_QUALIFIED_reads_every_hole_not_just_the_terminal_one():
    """The obligation is a property, not a position. Checking only the last hole passes
    `t"{T}:{name}:{suffix}"` with `name: str` and `suffix: Key`: the hand-rolled string in the
    wrapped slot, a `Key` in the tail. `fork:`/`hyp:` wrap in the terminal hole, so the tree alone
    cannot show the difference."""
    head = f'{_SCOPE_HEAD}T = AuthorityTag("t", scope=Scope.QUALIFIED)\n'
    interior = (
        head + 'def n(name: str, suffix: Key) -> Key: return compose_key(t"{T}:{name}:{suffix}")\n'
    )
    assert _scoped_scan(interior) == ["qualified-wraps-an-untyped-name"]

    # A `Segment(...)` hole is safe whatever the parameter says: the marker validates at
    # construction. Reading only the peeled field name reddens `fork:`, whose `child_run_id` is
    # `Segment`-wrapped in the template and `str` at the parameter.
    wrapped = head + (
        "def n(name: str, suffix: Key) -> Key:\n"
        '    return compose_key(t"{T}:{Segment(name)}:{suffix}")\n'
    )
    assert _scoped_scan(wrapped) == []


@pytest.mark.parametrize(
    "definition",
    [
        # a defaulted annotated parameter is a different tree-sitter kind
        'def n(c: Segment, name: Key = None) -> Key: return compose_key(t"{T}:{c}:{name}")',
        # a closure reads its OUTER function's parameters
        "def outer(name: Key):\n"
        '    def inner() -> Key: return compose_key(t"{T}:{Segment(c)}:{name}")\n'
        "    return inner",
    ],
)
def test_QUALIFIED_does_not_false_positive_on_a_legal_definition(definition):
    """No unactionable red ("annotate the wrapped parameter") when there is nothing to annotate
    or it is already annotated one frame out."""
    assert (
        _scoped_scan(f'{_SCOPE_HEAD}T = AuthorityTag("t", scope=Scope.QUALIFIED)\n{definition}\n')
        == []
    )


def test_an_unreadable_hole_yields_NO_verdict_rather_than_a_false_one():
    """`self.name` is the shape `govern:`'s own reference implementation uses, and there is no
    parameter to annotate. Silence is the honest answer; a red would be unactionable."""
    source = f'{_SCOPE_HEAD}T = AuthorityTag("t", scope=Scope.QUALIFIED)\n'
    source += (
        "class C:\n    def n(self) -> Key:\n"
        '        return compose_key(t"{T}:{Segment(self.c)}:{self.name}")\n'
    )
    assert _scoped_scan(source) == []


def test_an_AuthorityTag_survives_pickle_and_copy_and_refuses_mutation():
    """`__slots__` plus a keyword-only `__new__` breaks both protocols for a `Tag` subclass, and
    a bare slot lets `GOVERN.scope = ...` succeed: a mutable declaration, on a module-level
    singleton, under a gate."""
    import copy as _copy
    import pickle

    from effective.keys import AuthorityTag, Scope

    tag = AuthorityTag("govern", scope=Scope.SETTLEMENT)
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        revived = pickle.loads(pickle.dumps(tag, protocol))
        assert revived == tag, protocol
        assert revived.scope is Scope.SETTLEMENT, protocol
    assert _copy.copy(tag).scope is Scope.SETTLEMENT
    assert _copy.deepcopy(tag).scope is Scope.SETTLEMENT
    with pytest.raises(AttributeError):
        tag.scope = Scope.ACCRUAL  # ty: ignore[invalid-assignment] - the point of the test


def test_a_composed_key_retains_the_reach_its_leading_tag_declared():
    """The runtime reader of `AuthorityTag.scope`. `compose_key` is handed the TYPED tag as its
    leading interpolation, so the composed key keeps what that type knew, with no text-keyed
    lookup, which would be bounded by what it enumerates.

    Pinned on all three reaches plus the two negatives, because the negatives are where a
    mistake is silent: a plain-static namespace and a parsed key must report `None`, and `None`
    must not be read as an opt-out."""
    from effective.budget import budget_grant_name, chain_grant_name, depth_grant_name
    from effective.handlers.absurd import fork_event_name
    from effective.keys import Key, Scope, compose_key

    assert depth_grant_name("r1", depth=1, generation=0).scope is Scope.SETTLEMENT
    assert chain_grant_name("r1", 0).scope is Scope.SETTLEMENT
    assert budget_grant_name("r1", 0).scope is Scope.ACCRUAL
    assert fork_event_name("r-fork", Key.parse("review:m1")).scope is Scope.QUALIFIED

    # A namespace whose leading position is a plain static declares nothing. The tag is the test's
    # own: borrowing `event:` would put a second shape under a namespace production owns.
    assert compose_key(t"plain:{Segment('q')}").scope is None
    # And the READ boundary carries none: `parse` validates nothing, so it must not manufacture
    # a guarantee. `None` here means "not composed from an authority namespace", never "accrual".
    assert Key.parse("depth-grant:r1,1").scope is None
    assert Scope.ACCRUAL is not None  # the two are different answers, stated so


def test_the_declared_reach_survives_the_two_named_recompositions():
    """`prefixed` and `occurrence` are the only places a finished key is re-composed, and both
    must carry the reach: a `depth-grant:` inside a gather frame is still a settlement, and so is
    its second occurrence. A reach that fell off at a frame boundary would silently re-permit
    exactly the aliasing the coordinate exists to stop."""
    from effective.budget import depth_grant_name
    from effective.keys import Scope

    name = depth_grant_name("r1", depth=1, generation=0)
    assert name.prefixed("gather:0,1;").scope is Scope.SETTLEMENT
    assert name.occurrence(2).scope is Scope.SETTLEMENT
    assert name.occurrence(1).scope is Scope.SETTLEMENT  # identity at n <= 1, same object


def test_the_declared_reach_is_not_part_of_a_key_s_identity():
    """`_scope` is `compare=False`, so it touches neither equality nor hashing. That is required
    rather than tidy: keys are dict keys and set members throughout the substrate (seed maps,
    grant maps, checkpoint lookups), and a key that compared unequal to the same key read back
    off the wire would break every one of them silently."""
    from effective.budget import depth_grant_name
    from effective.keys import Key, Scope

    composed = depth_grant_name("r1", depth=1, generation=0)
    parsed = Key.parse(composed.stored())

    assert composed.scope is Scope.SETTLEMENT
    assert parsed.scope is None
    assert composed == parsed
    assert hash(composed) == hash(parsed)
    assert len({composed, parsed}) == 1
    assert {composed: "a"}[parsed] == "a"


def test_a_key_survives_pickle_and_copy_with_its_declared_reach():
    """The round trip `AuthorityTag`'s own pin exists for, one type down: a field on a
    `frozen`/`slots` dataclass is exactly the shape that breaks pickle.

    The `stored()` form is what must never drift; the reach riding along is a convenience, and it
    must not perturb the wire."""
    import copy as _copy
    import pickle

    from effective.budget import depth_grant_name
    from effective.keys import Scope

    name = depth_grant_name("r1", depth=1, generation=0)
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        revived = pickle.loads(pickle.dumps(name, protocol))
        assert revived.stored() == name.stored(), protocol
        assert revived.scope is Scope.SETTLEMENT, protocol
    assert _copy.copy(name).scope is Scope.SETTLEMENT
    assert _copy.deepcopy(name).scope is Scope.SETTLEMENT


def test_fork_event_prefix_is_the_prefix_fork_event_name_composes():
    """The RELATIONAL pin `hyp:`'s twin already has (`test_fork_ledger.py`, and
    `counterfactual.py:278` states why): two spellings of one namespace drift SILENTLY, so pin
    them against each other rather than against a literal."""
    from effective.handlers.absurd import fork_event_name, fork_event_prefix

    for child in ("c1", "r-fork", "cf-a"):
        name = fork_event_name(child, compose_key(t"review:{Segment('m1')}"))
        assert name.stored().startswith(fork_event_prefix(child)), (child, name.stored())


def test_a_scoped_sleep_is_still_engine_internal():
    """`is_engine_internal` reads through frames. Matching prefix markers on the RAW name, a
    sleep inside a `scoped(...)` (`rec:0;sleep:0`) does not start with its tag and reads as NOT
    engine-internal while its unscoped twin reads as internal.

    Three consumers depend on it: checkpoint reads; cross-engine key-sequence parity, where an
    unfiltered scoped sleep makes the two engines look divergent; and a fork seed, where
    `SeedingCtx.unconsumed()` would be non-empty and `run_fork` would refuse a correct fork.

    The negatives are pinned as hard as the positives, because over-filtering is the worse
    failure: a `ledger;` row IS a checkpoint the readers must keep, framed or not, and
    `artifact:message/rfc822:…` is a deployed key whose `/` is content rather than a frame."""
    from effective.checkpoints import is_engine_internal

    for internal in (
        "sleep:0",
        "rec:0;sleep:0",
        "gather:0,1;sleep:2",
        "$awaitEvent:q",
        "rec:0;$awaitEvent:q",
        "gather:0;wake-race:1,cond",  # the INFIX marker, matched mid-name either way
    ):
        assert is_engine_internal(internal) is True, internal

    for kept in (
        "rec:0;ledger;x",
        "tool:a",
        "rec:0;tool:a",
        "artifact:message/rfc822,sha256-9f2c",
    ):
        assert is_engine_internal(kept) is False, kept


def test_an_authority_occurrence_never_reaches_a_step_shaped_reader():
    """The constraint to pin before shipping `#N` on an authority name, and it is contingent
    rather than structural — so it is pinned rather than argued.

    Both bridges strip a trailing `#N` (`_DUP_SUFFIX.sub`), which is correct for the engines'
    duplicate-STEP suffix and wrong for an authority occurrence: one regex would mean two things,
    and stripping it would collapse two distinct grants back into one.

    It does not bite because the two never meet. A bare `depth-grant:r1:1#2` IS step-shaped to
    `is_step_checkpoint`, but it never reaches a bridge in that form: on Absurd the checkpoint is
    `$awaitEvent:depth-grant:r1:1#2`, which is engine-internal and filtered BEFORE the strip, and
    SQLite writes no await checkpoint at all.

    **Both halves of that are contingent**, which is why both are asserted here. If SQLite ever
    checkpoints awaits, or if the engine-internal filter stops running before the strip, the
    hazard is live and this test is where it surfaces."""
    from effective.budget import depth_grant_name
    from effective.checkpoints import is_engine_internal, is_step_checkpoint

    occurrence = depth_grant_name("r1", depth=1, generation=0).occurrence(2).stored()

    # The hazard, stated: bare, it looks like a Step to the readers that key on shape.
    assert is_step_checkpoint(occurrence) is True
    assert is_engine_internal(occurrence) is False

    # And the two reasons it never arrives bare.
    assert is_engine_internal(f"$awaitEvent:{occurrence}") is True
    assert is_step_checkpoint(f"$awaitEvent:{occurrence}") is False


def test_a_framed_key_decodes_to_its_frame_and_its_inner_identity():
    """A frame is applied by `Key.prefixed`, OUTSIDE the composer, so like `Key.occurrence`'s
    suffix it is invisible to every registered variant, and the decoder has to be told. One
    behavior for one concept, whether or not the frame's leading atom is a registered tag.

    The atoms decode too, each to its own producing line, because a frame atom is itself a
    composed key and its arity varies by namespace (`seed` none, `rec:` one, `fold:` two)."""
    from effective.keys.registry import KeyMap

    keymap = KeyMap.from_shapes(_registry()[0])

    framed = keymap.explain("rec:0;ledger;reviewed:m1")
    assert framed.bindings == {"row.event_id": "reviewed:m1"}, framed.bindings
    assert [atom.key for atom in framed.frame] == ["rec:0"]
    assert framed.frame[0].bindings == {"i": "0"}
    assert "combinators.py" in framed.frame[0].site

    # Nested scopes, outermost first, each decoded.
    # `depth` is an OPTIONAL coordinate, so it is named on the wire.
    nested = keymap.explain("rec:0;d:2;depth-grant:r,0,depth=1")
    assert [atom.key for atom in nested.frame] == ["rec:0", "d:2"]
    assert [atom.bindings for atom in nested.frame] == [{"i": "0"}, {"depth": "2"}]
    assert nested.bindings["depth"] == "1"

    # An atom no `src/` template registers is REPORTED, not dropped: a missing atom would
    # misstate which frame the operator is standing in.
    unknown = keymap.explain("q:0;depth-grant:r,0,depth=1")
    assert [atom.key for atom in unknown.frame] == ["q:0"]
    assert unknown.frame[0].site == ""
    assert unknown.bindings == {"run_id": "r", "generation": "0", "depth": "1"}

    # A frame and an occurrence are both outside the template, and both come off.
    both = keymap.explain("rec:0;approve;r1;tool:charge_card#2")
    assert both.occurrence == 2
    assert [atom.key for atom in both.frame] == ["rec:0"]
    assert both.bindings["placed_key(op)"] == "r1;tool:charge_card"


def test_a_gather_frame_and_a_mime_path_decode_with_no_special_case():
    """Two live keys that a frame decoder could take apart are unambiguous by construction:

    `gather:{g},{i};{qualified}` is a registered template whose frame is IN the grammar, and
    `artifact:message/rfc822,sha256-…` carries its MIME type as a two-atom PATH rather than as a
    value that happens to contain a slash. Neither needs a preference expressed."""
    from effective.keys.registry import KeyMap

    keymap = KeyMap.from_shapes(_registry()[0])

    gathered = keymap.explain("gather:0,1;event;q")
    assert gathered.frame == ()

    artifact = keymap.explain("artifact:message/rfc822,sha256-9f2c")
    assert artifact.frame == ()
    assert (artifact.bindings["kind"], artifact.bindings["subtype"]) == ("message", "rfc822")


_DOORS = [
    ("step_key", step_key, "tool:charge-card#2", "tool:charge-card", "step;tool:charge-card"),
    (
        "code/_segment_key",
        lambda n: _segment_key(n, True, 0),
        "act:t0#2",
        "act:t0",
        "code;seg:0;act:t0",
    ),
    (
        "code/_action_key",
        lambda n: _action_key(n, True, 1, "read_file"),
        "act:t0#2",
        "act:t0",
        "code;action:1;tool:read_file;act:t0",
    ),
    ("event_name(str)", event_name, "review:m1#2", "review:m1", "review:m1"),
    (
        "api/qualified_event_name",
        lambda n: qualified_event_name(GatherBranch(0, 1), name=n),
        "review:m1#2",
        "review:m1",
        "gather:0,1;review:m1",
    ),
]
"""Each door that parses an author's structured name: a name with an occurrence, the same name
without one, and the key the door stores for it."""
_DOOR_COLUMNS = ("door", "mint", "forged", "plain", "stored")
_DOOR_IDS = [d[0] for d in _DOORS]


@pytest.mark.parametrize(_DOOR_COLUMNS, _DOORS, ids=_DOOR_IDS)
def test_NO_AUTHOR_DOOR_lets_a_name_carry_an_occurrence(door, mint, forged, plain, stored):
    """The occurrence suffix has one producer, `Key.occurrence`.

    A splice never passes `Atom.of`, where a composed name's occurrence is refused, so each door in
    `_DOORS` refuses it itself. A door that admitted one would mint a key byte-identical to the
    engine's own second occurrence, and replay would serve that occurrence's value to another op:

        step_key("tool:charge-card#2")              ->  step;tool:charge-card#2
        step_key("tool:charge-card").occurrence(2)  ->  step;tool:charge-card#2

    The invariant is over the family: a door added to `_DOORS` is held to it as well.
    """
    with pytest.raises(ValueError, match="an occurrence is not an author's to write"):
        mint(forged)


@pytest.mark.parametrize(_DOOR_COLUMNS, _DOORS, ids=_DOOR_IDS)
def test_each_author_door_admits_its_name_without_an_occurrence(door, mint, forged, plain, stored):
    """Anti-vacuity for the refusals: a door that refused every name would pass them."""
    assert mint(plain).stored() == stored


@pytest.mark.parametrize("text", ["review:m1#2", "review:m1"])
def test_op_keys_event_arm_takes_a_key_and_will_not_parse_author_text(text):
    """The other route to `event;{name}`: `AwaitEvent.name` is a `Key`, minted by `event_name`.

    Text is turned away whether or not it carries an occurrence, which is what keeps this door
    shut: a route that parsed the plain name would parse the suffixed one too."""
    with pytest.raises(TypeError, match="mint it with `event_name"):
        op_key(AwaitEvent(name=text, schema=str))
    assert (
        op_key(AwaitEvent(name=event_name("review:m1"), schema=str)).stored() == "event;review:m1"
    )


def test_the_runtime_still_mints_the_occurrence_the_doors_refuse():
    """The refusal is at the author's door, so the grammar still carries the suffix."""
    assert step_key("tool:charge-card").occurrence(2).stored() == "step;tool:charge-card#2"


def test_the_COMPOSER_cannot_mint_a_key_its_own_PARSER_refuses():
    """One grammar, or it is not a grammar.

    A spelling the composer admits and `parse` rejects puts text in `Key.stored()`, the DURABLE
    form, that `Key.terms()`, `keymap.explain`, `graphview` and `split_occurrence` all choke on.

    Both paths are closed where every call passes: `Term.__post_init__` validates the tag (the
    splice path), and `Tag.__new__` validates against `grammar.TAG` (the marker path).
    """
    import pytest

    from effective.keys import Segment, Tag, compose_key
    from effective.keys.grammar import KeySyntaxError, parse

    # the SPLICE path: a non-`Key` splice would become a term tag with no check at all.
    # `TypeError` joins the tuple because the term-head refusal answers WRONG TYPE before the
    # grammar sees wrong shape: the marker constructors draw the same line, and what this test
    # asserts is that the composer refuses, not which layer gets there first.
    for forge in (
        lambda: compose_key(t"ns;{5}"),
        lambda: compose_key(t"ns;{Segment('Foo_Bar')}"),
    ):
        with pytest.raises((KeySyntaxError, ValueError, TypeError)):
            forge()

    # the MARKER path: `Tag` checks the whole grammar, not a denylist of `:` and `;`
    for bad in ("Foo_Bar", "a,b", "a#2", "a/b", "2fa", ""):
        with pytest.raises(ValueError, match="well-formed tag"):
            Tag(bad)

    # ANTI-VACUITY: the legitimate forms still compose, and what they mint still parses.
    # (A splice must DECLARE its domain, which is a separate rule.)
    spliced = Key.parse("tool:a")
    for good in (compose_key(t"{Tag('ns')}:{1}"), compose_key(t"ns;{spliced:domain=any}")):
        assert parse(good.stored()).render() == good.stored()


def test_Tag_refuses_a_NON_STR_the_way_Segment_does():
    """Opacity is only as strong as the narrowest constructor taking `object`.

    A `Tag(value: object)` that called `str(value)` would launder an opaque `Key`'s REPR into the
    LEADING position of an identity. `Segment` is narrowed for exactly this, and `AuthorityTag`
    inherits the narrowing.
    """
    import pytest

    from effective.keys import AuthorityTag, Key, Scope, Tag

    # `ty` rejects both at the call site, which is HALF the fix. The runtime check is the other
    # half: an untyped caller (a double, a dict round-trip, `Any`) reaches them anyway.
    for construct in (  # each DELIBERATE: the refusal is what is under test
        lambda: Tag(Key.parse("tagonly")),  # ty: ignore[invalid-argument-type]
        lambda: AuthorityTag(Key.parse("tagonly"), scope=Scope.SETTLEMENT),  # ty: ignore[invalid-argument-type]
    ):
        with pytest.raises(TypeError, match="takes a str"):
            construct()

    assert str(Tag("ns")) == "ns"  # anti-vacuity


def test_separated_does_not_treat_a_HOLE_tag_as_a_discriminator():
    """A wrong TRUE here lets one namespace mint two shapes it cannot tell apart.

    A hole is never a discriminator, because it matches whatever literal faces it, and that holds
    for a term's tag as much as for a coordinate. Two shapes reported disjoint while BOTH decode
    `ns:lit;tag:v` falsify `_decode_one`'s stated "at most one can match": the invariant the
    source map, `--key-borrowing` and `--key-literals` all rest on. No interpolated non-leading
    tag exists in `src/`, so this is the pin that keeps the hazard latent.
    """
    from effective.keys.registry import separated

    holey = _shape("ns:", ..., ";", ..., ":", ..., fields=("a", "T", "b"))
    literal = _shape("ns:", ..., ";tag:", ..., fields=("T", "b"))
    disjoint, why = separated(holey, literal)
    assert not disjoint, why
    # and the key they BOTH accept is the witness that a TRUE here would have been wrong
    assert holey.decode("ns:lit;tag:v") is not None
    assert literal.decode("ns:lit;tag:v") is not None

    # ANTI-VACUITY: two LITERAL tags still separate, which is the case the arm exists for.
    assert separated(
        _shape("ns:", ..., ";seg:", ..., fields=("a", "b")),
        _shape("ns:", ..., ";act:", ..., fields=("a", "b")),
    )[0]


# --- `event_name`'s decision table, exercised in its TOTALITY -------------------------------
#
# One row per ARM. The arms are the parameterization, which is the point of writing the function
# as a total `match` rather than an `isinstance` split: a table has to name its
# cases, so the cases become the test, and "did we cover it?" is answered by construction instead
# of by picking a couple of examples.
#
# The row named `parsed authority Key` is the arm an `isinstance` short-circuit cannot express, a
# `Key` that is NOT trustworthy, and the row below it (`composed authority Key`) is the case that
# refutes the naive fix of distrusting every `Key`.
_ARMS = [
    ("Key + op arm", lambda: Key.parse("event;review:m1"), "op ARM"),
    ("parsed authority Key", lambda: Key.parse("depth-grant:r1,0#2"), "carries no scope"),
    ("composed authority Key", lambda: _depth_grant(), None),
    ("ordinary Key", lambda: Key.parse("review:m1"), None),
    ("str + op arm", lambda: "event;review:m1", "op ARM"),
    ("str + authority tag", lambda: "depth-grant:r1,0", "reserved authority namespace"),
    ("str + frame delimiter", lambda: "a;b", "frame delimiter"),
    ("str + occurrence", lambda: "review:m1#2", "occurrence"),
    ("ordinary str", lambda: "review:m1", None),
    ("neither", lambda: 7, "not int"),
]


def _depth_grant():
    from effective.budget import depth_grant_name

    return depth_grant_name("r1", generation=0, depth=0)


@pytest.mark.parametrize(("arm", "build", "refusal"), _ARMS, ids=[a[0] for a in _ARMS])
def test_event_name_decides_every_arm_of_its_table(arm, build, refusal):
    """Every case the table names, including the two that only a table could tell apart.

    `parsed authority Key`: `depth-grant:r1,0#2` is WELL-FORMED, and awaiting it means parking
    on the name a substrate gate answers, which Absurd delivers first-emit-wins across the queue,
    so the author's park consumes the gate's answer. The discriminator is `Scope`:
    every authority namespace is declared as an `AuthorityTag`, so the substrate's own park names
    carry one and `Key.parse` never attaches one.

    `composed authority Key` is the row that keeps the fix honest: distrusting every `Key` would
    refuse the substrate's own grant, which is the mistake a two-arm type test invites.
    """
    from effective.ops import event_name

    if refusal is None:
        assert event_name(build()).stored()  # accepted, and it is a real key
    else:
        with pytest.raises((ValueError, TypeError), match=refusal):
            event_name(build())


ARMS_IN_THE_TABLE = 8
"""How many `case` arms `event_name`'s match has — pinned so adding one forces a row above.

**Rows outnumber arms (10 vs 8), deliberately.** Two arms carry more than one behaviour: `str()`
delegates to `authored_key`, which refuses an occurrence inside it, so `ordinary str` and
`str + occurrence` share an arm; and the two accepting `Key` rows (`composed authority` and
`ordinary`) exercise the same arm from opposite sides of the guard above it. A row-per-arm
equality would be tidier and would assert something false."""


def test_the_arms_table_covers_the_whole_decision_table():
    """Anti-vacuity for the parameterization — the table must track the match.

    Counted from the source, so ADDING an arm without a row fails here rather than silently
    shrinking the coverage of the test above. That is the failure mode a hand-written case list
    has and a generated one does not: nothing about `_ARMS` knows when the function grew.
    """
    import inspect

    from effective import ops

    arms = inspect.getsource(ops.event_name).count("\n        case ")
    assert arms == ARMS_IN_THE_TABLE, f"{arms} arms now; add rows to `_ARMS` and update this"
    assert len(_ARMS) >= arms


def test_the_key_gates_see_a_DOTTED_or_ALIASED_compose_key():
    """One helper feeds THREE rules, so one blind spot is shared three ways.

    Matching the callee text EXACTLY makes `keys.compose_key(...)` invisible to `--key-registry`,
    `--key-borrowing` and `--key-literals` at once. `lint._compose_key_templates` matches the last
    dotted component (`rsplit(".", 1)[-1]`, the convention `_identity_slots` uses). An ALIAS still
    defeats an AST rule, which is what such a rule is bounded by; the store gate is the backstop
    that is not.
    """
    from effective.lint import check_terminal_holes_source

    dotted = 'from effective import keys\ndef a(x): return keys.compose_key(t"budget-grant:{x}")\n'
    bare = "from effective.keys import compose_key\n"
    bare += 'def a(x): return compose_key(t"budget-grant:{x}")\n'
    assert len(check_terminal_holes_source(dotted, "x.py")) == 1, "a dotted callee must be scanned"
    assert len(check_terminal_holes_source(bare, "x.py")) == 1  # anti-vacuity: same finding bare
    # and something that is NOT compose_key is still ignored, so the match did not go greedy
    other = 'def a(x): return other.compose_keys(t"budget-grant:{x}")\n'
    assert check_terminal_holes_source(other, "x.py") == []


# --- the hole ordinals index ONE enumeration -------------------------------------------------
#
# `compose_key` splits a `Template` into `items`, and everything downstream that speaks of an
# interpolation speaks of it BY INDEX: the directive maps are keyed by one, `_expressions` and
# `_fill` are positional lists, and `_parts` mints the `Hole(index)` all three are read with.
# Three take `_interpolations(items)` and the fourth, `_parts`, counts, so this is the one
# agreement left and the only one worth a test.
#
# Parameterized over the PRODUCT of static/hole positions rather than over examples: the property
# is inductive over the item list, so the cases are `{static, hole}^n` for n in 1..5. A hand-picked
# list misses exactly the shapes nobody thought of.


def _shape_space(width: int) -> list[tuple[str, ...]]:
    from itertools import product

    return [shape for n in range(1, width + 1) for shape in product("SH", repeat=n)]


_ITEM_SHAPES = _shape_space(5)


@pytest.mark.parametrize("shape", _ITEM_SHAPES, ids=["".join(s) for s in _ITEM_SHAPES])
def test_hole_ordinals_index_the_one_enumeration(shape):
    """`Hole(index)` addresses `_interpolations(items)[index]`, for every arrangement of the two.

    The two walks are `_parts`, which hands out the ordinal by counting non-statics, and
    `_interpolations`, which builds the list that ordinal indexes. They agree because both filter
    the same predicate over the same list in the same order, and that sentence is the whole proof,
    so this is where it is checked rather than asserted in a docstring.

    **What it catches**, confirmed by mutation and by counting which shapes survive (the count is
    the evidence that the space is doing work rather than that 62 cases ran):

    - incrementing `_parts`' `index` on a static too reddens **42 of 62**, and the 20 survivors are
      exactly the `H*S*` shapes, where no hole has a static in front of it to be mis-counted;
    - `reversed(...)` in `_interpolations` reddens **42 of 62**, and those 20 survivors are exactly
      the shapes with fewer than two holes, where reversal is the identity.

    Both leave a different 20 standing for a different reason, which is what says the space
    separates the two walks rather than testing one of them twice.

    It does NOT catch a defect uniform across both walks, which is the standing limit of an
    agreement test (`tests/_walks.py` says the same of the interpreters)."""
    from string.templatelib import Interpolation

    from effective.keys.processor import _Directives, _interpolations, _parts

    items = [
        "x" if kind == "S" else Interpolation(f"v{position}", f"expr{position}")
        for position, kind in enumerate(shape)
    ]
    interpolations = _interpolations(items)
    parts = _parts(items, _Directives({}, {}))

    assert len(parts) == len(items), "a part per item — statics are carried, not dropped"
    holes = [(position, part) for position, part in enumerate(parts) if isinstance(part, Hole)]
    assert len(holes) == shape.count("H") == len(interpolations)
    for position, hole in holes:
        # the ordinal addresses the enumeration, and the enumeration holds the item at that
        # POSITION — identity, so a list that merely has the right length cannot pass
        assert interpolations[hole.index] is items[position]
    assert [hole.index for _, hole in holes] == list(range(len(interpolations))), "dense, in order"


def test_the_ordinals_survive_a_REAL_template_not_just_a_hand_built_list():
    """The domain above is the two functions' own, a `list[str | Interpolation]`. This is the
    domain `compose_key` actually hands them, and it is not the same set.

    **The product is a SUPERSET, and by more than the shape names suggest.** Two bounds, and the
    second is the sharp one:

    - `Template` normalizes (never two adjacent statics, and empty ones dropped), so a `SS` shape
      is unreachable. The space does not exercise an empty static either way: every `S` here is
      `"x"` (above).
    - `compose_key` **refuses adjacent interpolations** outright, before `_interpolations` is ever
      called. That takes out **31 of the 62** shapes (every one containing `HH`), and of the 31
      that survive, 19 open with a static and can carry a leading tag.

    None of that makes the space vacuous: `_parts` and `_interpolations` are pure functions of the
    list and both mutations still redden within the reachable shapes. It does mean the 62 measures
    the two walks, not the composer, so this case covers the live path."""
    from effective.keys.processor import _Directives, _interpolations, _parts

    template = t"artifact:{Segment('message')}/{Segment('rfc822')},{Segment('sha256-9f2c')}"
    items = list(template)
    interpolations = _interpolations(items)
    parts = _parts(items, _Directives({}, {}))

    assert [i.value for i in interpolations] == ["message", "rfc822", "sha256-9f2c"]
    holes = [part for part in parts if isinstance(part, Hole)]
    assert [hole.index for hole in holes] == [0, 1, 2]
    # and the composed key is the one the grammar documents, so the alignment is the live path
    assert compose_key(template).stored() == "artifact:message/rfc822,sha256-9f2c"


# --- the composer's walk, exercised one ARM at a time ----------------------------------------
#
# `_fill` and its two helpers are total `match`es closed by `assert_never`. `ty`
# proves no arm is MISSING; nothing static proves an arm is REACHED, and an arm no template can
# reach is a case invented rather than found. So the arms are the parameterization: one
# row per case, each a real `t"..."` rather than a hand-built skeleton, because the question is
# what the composer does with what `compose_key` is actually handed.
#
# Deleting an arm reddens exactly the rows that reach it, which is the property saying these rows
# discriminate rather than all riding one code path. Measured, three deletions of the six-row set:
# `_fill_tag`'s `Hole` arm reddens 1 (`SkeletonTerm, Hole tag`); `_fill_atom`'s `Atom` arm reddens
# 1 (`coordinate atom, literal`); `_fill`'s `Splice` arm reddens 2, both splice rows, because one
# arm serves both. Nothing reddens all six, which is what a shared code path would do.

_GOVERN = Tag("govern")

_WALK_ARMS = [
    ("SkeletonTerm, literal tag", lambda: t"ns:{Segment('v')}", "ns:v"),
    ("SkeletonTerm, Hole tag", lambda: t"{_GOVERN}:{Segment('v')}", "govern:v"),
    ("coordinate atom, literal", lambda: t"ns:lit,{Segment('v')}", "ns:lit,v"),
    ("coordinate atom, Hole", lambda: t"ns:{Segment('v')}", "ns:v"),
    (
        "Splice of a Key",
        lambda: t"event;{Key.parse('review:m1'):domain=address}",
        "event;review:m1",
    ),
    ("Splice of a bare Tag", lambda: t"ns:{Segment('v')};{_GOVERN}", "ns:v;govern"),
]


@pytest.mark.parametrize(("arm", "build", "stored"), _WALK_ARMS, ids=[a[0] for a in _WALK_ARMS])
def test_every_arm_of_the_composer_s_walk_is_reachable_from_a_template(arm, build, stored):
    """One row per case of `_fill` / `_fill_tag` / `_fill_atom`, each reached by composing.

    The two `SkeletonElement` arms, the two `SkeletonTerm.tag` arms, and the two `SkeletonAtom`
    arms: six rows for three tables, which is the product, not a sample. `assert_never` says the
    tables name every case in the union; this says every case the tables name is one a template
    can produce, and the two claims are independent."""
    assert compose_key(build()).stored() == stored


def test_a_value_out_of_the_union_is_LOUD_rather_than_falling_through():
    """An unmatched arm that falls through to a default path mis-dispatches silently. A `match`
    closed by `assert_never` converts that into a raise that NAMES the value, so the failure is a
    stack trace instead of a wrong answer.

    **This pins the RUNTIME channel.** The static one is fenced: the parameter is
    `atom: SkeletonAtom`, and the line below needs a `ty: ignore` to type-check at all, so a third
    grammar constructor is caught statically at the `match`. What is left to pin is the case `ty`
    is not in: a value arriving through an untyped edge, or a caller with the checker switched off.
    The static proof does not run in production."""
    from effective.keys.processor import _fill_atom

    class NotAnAtom:
        """A third `SkeletonAtom` constructor, as a future grammar change would introduce it."""

    with pytest.raises(AssertionError, match="unreachable"):
        _fill_atom(NotAnAtom(), [])  # ty: ignore[invalid-argument-type]


# --- claims about the walks, pinned so they can only be wrong once ---------------------------


def test_the_two_walks_agree_on_what_an_INTERPOLATION_IS_not_merely_on_ORDER():
    """`_interpolations` and `_parts` must apply the SAME predicate, not complementary ones.

    `test_hole_ordinals_index_the_one_enumeration` pins that the two walks hand out the same
    ordinals, over a space built from real `Interpolation`s, which is exactly the space in which
    `not isinstance(item, str)` and `case Interpolation()` are indistinguishable. They are not the
    same predicate: with one each, a duck-typed item is an interpolation to `_interpolations` and
    unreachable to `_parts`.

    Unreachable through `compose_key`, whose parameter is a `Template`, so this is the pin that
    keeps the agreement from being a coincidence of spelling. An
    agreement test whose two sides test different things is the failure mode `tests/_walks.py`
    names one level up: they agree, and they are wrong together.

    Reddens if either walk's predicate is written as the complement of the other's."""
    from effective.keys.processor import _Directives, _interpolations, _parts

    class NotAnInterpolation:
        """Neither arm of `str | Interpolation`: what a complement-spelled filter lets through."""

    items = ["ns:", NotAnInterpolation()]

    # `_parts` refuses it, loudly, by its closing arm
    with pytest.raises(AssertionError, match="unreachable"):
        _parts(items, _Directives({}, {}))  # ty: ignore[invalid-argument-type]
    # and `_interpolations` must refuse it too, as an EMPTY result, since it filters
    assert _interpolations(items) == [], (  # ty: ignore[invalid-argument-type]
        "`_interpolations` admitted an item `_parts` calls unreachable — the two walks are "
        "spelled as complements again, and the ordinal agreement is a coincidence"
    )


_SPLICE_DOMAIN = [
    ("arbitrary object", lambda: object(), "may contain a separator"),
    ("bool", lambda: True, "may contain a separator"),
    ("int", lambda: 7, "stands as a term's namespace"),
    ("Tag", lambda: Tag("govern"), None),
    ("Segment", lambda: Segment("q"), "stands as a term's namespace"),
]


@pytest.mark.parametrize(
    ("kind", "build", "refusal"), _SPLICE_DOMAIN, ids=[s[0] for s in _SPLICE_DOMAIN]
)
def test_what_actually_reaches_the_splice_s_non_Key_arm(kind, build, refusal):
    """The arm's LIVE domain, enumerated, because reading it gives the wrong answer.

    The arm is `Any`-typed (`Interpolation.value` is `Any` by the stdlib's signature), so `ty`
    cannot state its domain and only a probe can, which is what this row set is.

    **Three of the five rows refuse UPSTREAM of the arm and two do not, and the difference is the
    thing to keep.** An arbitrary object and a `bool` die on `compose_key`'s delimiter-freeness
    scan; the `int` and the `Segment` reach the arm and die on its own term-head check. So the
    live domain is one type, and widening any fence (upstream or in the arm) has to come
    past a row here."""
    template = t"ns:{Segment('v')};{build()}"
    if refusal is None:
        assert compose_key(template).stored().startswith("ns:v;")
    else:
        with pytest.raises((ValueError, KeySyntaxError, TypeError), match=refusal):
            compose_key(template)


def test_a_hole_may_NAME_a_term_which_is_the_splice_arm_s_twin():
    """`_fill_tag`'s `Hole` arm: a term head handed over as an interpolation rather than written.

    The twin of the splice arm above, and the reason both exist: a namespace is sometimes a
    constant, and forcing it to be a literal means spelling it twice. `t"{GOVERN}:{gate}"` is the
    live shape; this is the same shape one term in.

    **Only a `Tag` may name a term**, so the marker here is the legal one; the refusal of a
    `Segment` in the same position is `test_only_a_Tag_may_NAME_a_term`."""
    assert compose_key(t"ns:{Segment('v')};{Tag('forged')}:{Segment('x')}").stored() == (
        "ns:v;forged:x"
    )


# --- Only a `Tag` may head a term ----------------------------------------------------------


def test_only_a_Tag_may_stand_as_a_WHOLE_term():
    """The splice arm's refusal. `_fill`'s non-`Key` arm took anything and `str()`d it.

    The marker family assigns each marker one legal position (`keys.Tag`'s table): `Tag` names a
    namespace, `Segment` brands delimiter-free TEXT, `Key` is a finished composition. A whole-term
    splice is a namespace position, so only the first fits.

    **A POSITION rule, not a safety fix**, and overstating it is the trap. `Term.__post_init__`
    re-checks the head against the predicate `Tag.__new__` runs, so a `Segment` here widens the
    accepted TYPE and not the accepted LANGUAGE, and it cannot reach the LEADING position where
    authority lives (the test below holds that separately). Two markers agreeing on the bytes is
    not agreement on what the bytes mean.

    Mutation: drop the call in `_fill`'s splice arm and this reddens while the twin stays green,
    which is why they are two tests rather than one."""
    with pytest.raises(TypeError, match="stands as a term's namespace"):
        compose_key(t"ns;{Segment('tail')}")


def test_only_a_Tag_may_NAME_a_term():
    """The twin, in `_fill_tag`'s `Hole` arm.

    Same rule, second position: a term head with coordinates rather than a zero-arity one. It is
    a separate test because a one-arm fix passes every gate the other arm has.

    Mutation: drop the call in `_fill_tag`'s `Hole` arm and this reddens while the splice test
    stays green."""
    with pytest.raises(TypeError, match="stands as a term's namespace"):
        compose_key(t"ns:{Segment('v')};{Segment('forged')}:{Segment('x')}")


def test_a_Segment_cannot_OPEN_a_key_which_is_where_authority_lives():
    """The reason the refusal above is a position rule and not a forging fix.

    `_has_leading_tag` tests `isinstance(value, Tag)`, so the leading position is not reachable
    by a `Segment`, and `Key.scope` is retained from the leading value only, so a term head
    carries no authority to forge. Measured over a 2388-string corpus: 47 texts compose as a
    leading `Tag`, 0 as a leading `Segment`."""
    with pytest.raises(ValueError, match="must begin with a namespace TAG"):
        compose_key(t"{Segment('approve')}:{1}")
    assert compose_key(t"{Tag('approve')}:{1}").scope is None, (
        "a plain Tag declares no reach, so the leading fence is not what makes authority safe"
    )


def test_the_alias_the_refusal_LEAVES_standing():
    """What must survive: a namespace written as a static and the same one written as a `Tag`.

    Three templates can spell one key's head: a static, a `Tag`, and a `Segment`. The first two
    are this alias, and it is deliberate: it is the reason `Tag` is interpolable at all, so a
    namespace constant need not be spelled twice. Only the `Segment` is refused.

    So this row's job is to make an OVER-narrowing fix, one that starts refusing an interpolated
    `Tag`, fail here rather than in production."""
    assert (
        compose_key(t"ns;{Tag('tail')}").stored()
        == compose_key(t"ns;tail").stored()
        == ("ns;tail")
    )


# --- the two ABSORBED arms, which `ty` cannot guard -------------------------------------------
#
# A refined arm placed before a broad arm on the same class is correct, and it makes the checker
# blind: delete the refined arm and the match is still exhaustive, so `assert_never` proves
# nothing about it. `effective.lint.ordered_arm_report` lists them; these are their named pins,
# which is the only guard they have.


def test_a_composed_key_KEEPS_the_reach_its_AuthorityTag_declared():
    """`compose_key`'s refined `AuthorityTag` arm — absorbed by `case str() | Interpolation()`.

    Deleting it makes every composed authority key report `scope=None`, which reads as *this key
    was not composed from an authority namespace* — the permissive answer, on the axis where
    permissiveness is the whole risk. `ty` cannot see the deletion, so this row is the guard."""
    from effective.keys import Scope
    from effective.permission import APPROVE

    approved = compose_key(t"{APPROVE}:{Segment('r-1')}")
    assert approved.scope is Scope.SETTLEMENT, (
        "the leading AuthorityTag's reach must survive composition; None would read as "
        "'not an authority namespace' and every consumer treats that as no constraint"
    )
    assert compose_key(t"{Tag('plain')}:{Segment('x')}").scope is None, (
        "and a bare Tag must still declare nothing — the arm is not a blanket"
    )


def test_a_HOLE_supplied_tag_renders_from_the_holes_not_from_the_static_arm():
    """`keymap`'s refined `SkeletonTerm(tag=str())` arm — absorbed by the bare `SkeletonTerm`.

    The two arms differ in where the tag comes from: written in the template, or consumed from
    the hole stream FIRST, ahead of the coordinates. Deleting the refined arm sends a statically
    written tag down the hole path, which silently shifts every subsequent field by one — and
    `ty` cannot see it, because the broad arm still matches."""
    from effective.keys.grammar import parse_skeleton
    from effective.keys.registry import Shape

    written = Shape(
        skeleton=parse_skeleton(["gate-state:", Hole(0), ",", Hole(1)]),
        fields=("gate", "run_id"),
    )
    assert written.label() == "gate-state:{gate},{run_id}", (
        "a WRITTEN tag renders verbatim and the holes fill the coordinates. Down the broad arm "
        "the tag would be taken from the hole stream FIRST, consuming `gate` and shifting every "
        "field by one — silently, because the broad arm still matches"
    )
    assert written.tag == "gate-state"
