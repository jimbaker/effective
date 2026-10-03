"""The identity grammar, as a language (`effective.keys.grammar`).

    key        := term [';' term]*
    term       := tag [':' coordinate [',' coordinate]*]
    coordinate := atom ['/' atom]*

The parser is a LEAF, with no consumers, so what it asserts is the language itself rather than any
composer's use of it.

**The claim under test is that the parse needs no registry.** Every assertion below is about bytes
and separators alone; nothing here loads `build/key-registry.json`, and that is the property, not
an omission. The shipped grammar cannot make it — `keymap.explain` looks a tag up to learn its
arity and `keymap._unframe` tries decodes to find a frame boundary.

A separate file rather than an eleventh section of `test_op_key_injectivity.py`: that file is
1600+ lines organised around the OLD grammar's ten concerns, several of which this work deletes.
"""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from effective.govern import GOVERN
from effective.keys import (
    Key,
    Scope,
    Segment,
    Tag,
    compose_key,
    gather_prefix,
    race_choice,
    race_endings,
    race_prefix,
)
from effective.keys.frame import (
    FRAME_ARMS,
    _leading_frames,
    branch_frames,
    is_branch_frame,
    past_frames,
    split_frames,
)
from effective.keys.grammar import (
    Atom,
    Coordinate,
    Hole,
    KeySyntaxError,
    Kind,
    ParsedKey,
    Splice,
    Term,
    is_tag,
    kind_of,
    named,
    parse,
    parse_skeleton,
)

# --- atoms are typed, from the bytes alone -------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("0", Kind.INTEGER),
        ("7", Kind.INTEGER),
        ("42", Kind.INTEGER),
        ("0190c3e2-8f4a-7c1d-9b2e-1a2b3c4d5e6f", Kind.UUID),  # a uuid7
        ("sha256-9f2c1a", Kind.DIGEST),
        ("blake3-deadbeef", Kind.DIGEST),
        ("charge_card", Kind.NAME),
        ("r-78802aac", Kind.NAME),
        ("billing@example.com", Kind.NAME),
        ("message", Kind.NAME),
    ],
)
def test_an_atoms_kind_is_readable_from_its_bytes(text, kind):
    assert kind_of(text) is kind


@pytest.mark.parametrize("text", ["", "07", "a:b", "a,b", "a;b", "a/b", "-lead", "9f2c", "_x"])
def test_what_is_not_an_atom(text):
    """The separators are the load-bearing four: an atom that could contain one would make arity
    uncountable, which is the single property this grammar buys.

    `07` is refused because an integer has no leading zero — two spellings of seven would be two
    keys for one identity. `9f2c` is refused because it is neither an integer nor a name, and a
    bare hex digest is not a kind: §1d narrows a digest to `sha256-…` so that its kind is legible
    without asking anyone what the field means."""
    assert kind_of(text) is None
    with pytest.raises(KeySyntaxError, match="well-formed atom"):
        Atom.of(text)


def test_the_narrow_kinds_are_tried_before_name():
    """Not a style point — the kinds overlap as regexes and are disjoint only as intents. A uuid
    and a digest both satisfy `NAME`, so ordering is what assigns them. Reordering `kind_of`
    silently reclassifies every uuid in the system, and nothing else in this file would notice."""
    assert kind_of("0190c3e2-8f4a-7c1d-9b2e-1a2b3c4d5e6f") is Kind.UUID
    assert kind_of("sha256-9f2c1a") is Kind.DIGEST
    assert kind_of("sha999-9f2c1a") is Kind.NAME  # not a known algorithm -> an ordinary name


def test_a_digest_needs_an_algorithm_the_module_knows():
    """The kind is decidable only because the prefix comes from a set. `run-abc123` is a perfectly
    good name, and the only thing keeping it one is that `run` is not an algorithm."""
    assert kind_of("run-abc123") is Kind.NAME
    assert kind_of("sha256-abc123") is Kind.DIGEST


# --- the grammar ---------------------------------------------------------------------------


def test_the_designs_worked_examples_parse_and_round_trip():
    """§3c's table, which is the design's own statement of what the rewrite produces.

    One entry is corrected here and the correction is the finding: §3c gives
    `artifact:message/rfc822,9f2c`, whose last atom is a BARE hex digest — not in the language
    §1d defines two sections earlier, which narrows a digest to `sha256-<hex>`. The example shows
    the structural rewrite (`:` -> `,`) with the atom left un-migrated. Fix §3c, not this."""
    for text in [
        "approve;ledger;committed:casc-a81f990d",
        "gather:0,0;event;foo",
        "artifact:message/rfc822,sha256-9f2c",
        "rec:0;ledger;reviewed:m1",
        "depth-grant:r-78802aac,0#2",
        "govern:approve,r1;step;tool:charge-card",
        "depth-grant:r1,g",
        "depth-grant:r1,g,depth=2",
        "emit;ask:foo",
        "gather:0;wake-race:1,cond",
    ]:
        assert parse(text).render() == text, text


def test_each_metacharacter_carries_exactly_one_meaning():
    """`;` sequence, `,` arity, `/` path — which is what makes the notation self-describing to a
    reader who has never seen it, human or agent."""
    key = parse("rec:0;artifact:message/rfc822,sha256-9f2c")
    assert key.tags == ("rec", "artifact")
    frame, artifact = key.terms
    assert frame.arity == 1
    assert artifact.arity == 2  # `,` counted two coordinates
    assert artifact.coordinates[0].atoms == (
        Atom("message", Kind.NAME),
        Atom("rfc822", Kind.NAME),
    )  # `/` made the MIME type a two-atom PATH, not a value containing a slash


def test_arity_is_countable_with_no_registry():
    """The whole claim, stated as a test. Nothing is looked up; the separators answer it."""
    assert [t.arity for t in parse("gather:0,0;event;foo").terms] == [2, 0, 0]
    assert [t.arity for t in parse("govern:approve,r1,0,0;tool:charge-card").terms] == [4, 1]


def test_a_bare_tag_is_a_qualifier():
    """A term with no coordinates — `emit`, and the arm terms that wrap rather than identify."""
    key = parse("emit;ask:foo")
    assert key.terms[0] == Term("emit")
    assert key.terms[0].arity == 0


def test_a_foreign_tag_is_admitted_as_a_QUALIFIER():
    """§1f's original zero-arity shape, still legal: a bare foreign tag, then our terms after `;`.

    `$` is the declared marker rather than an accident. The `:`-joined form the SDK actually
    writes is the sibling case, pinned in `test_the_two_foreign_JOINS_stay_distinguishable`."""
    key = parse("$awaitEvent;ask:foo")
    assert key.terms[0] == Term("$awaitEvent", (), foreign=True)
    assert key.terms[1].tag == "ask"
    assert key.render() == "$awaitEvent;ask:foo"


def test_the_two_foreign_JOINS_stay_distinguishable():
    """A foreign tag may carry a payload after `:`, and the two JOINS stay distinct.

    The vendored SDK writes `f"$awaitEvent:{event_name}"`, so refusing a payload on a foreign tag
    would leave the engine's own rows outside the language. The two JOINS carry the same tags in
    the same order, and a structure that could not tell them apart would render one spelling for
    two different durable names."""
    sdk_style = parse("$awaitEvent:event:foo")  # what the SDK writes
    our_style = parse("$awaitEvent;event;foo")  # what our own composer would write

    assert sdk_style.render() == "$awaitEvent:event:foo"
    assert our_style.render() == "$awaitEvent;event;foo"
    assert sdk_style != our_style, "two durable names must not collapse to one structure"
    assert sdk_style.wrapped is not None  # the SDK wraps OUR key
    assert our_style.wrapped is None  # our own form is a qualifier, not a wrapper


def test_wake_race_stops_being_special():
    """`absurd.py:1575` hand-composes `{scope}gather:{g}:wake-race:{i}:{cond}`, which is why
    `checkpoints.is_engine_internal` is a prefix-OR-INFIX matcher. As ordinary terms it is a key
    like any other and that predicate collapses to "any term whose tag is in the foreign set"."""
    assert parse("gather:0;wake-race:1,cond").tags == ("gather", "wake-race")


# --- coordinates: required-positional, then optional-named ---------------------------------


def test_bare_is_required_and_named_is_optional():
    """The rule on the wire, with no marker of its own (§1b). The boundary is DERIVED, never
    written — there is no `/` on the wire and none in the template."""
    key = parse("depth-grant:r1,g,depth=2")
    [term] = key.terms
    assert [c.name for c in term.coordinates] == [None, None, "depth"]
    assert term.coordinates[2].path == "2"


def test_a_bare_coordinate_may_not_follow_a_named_one():
    """Because the required prefix is fixed-width, and that is what every discriminator and arity
    check depends on. A bare coordinate past the boundary would make the prefix's width depend on
    the value."""
    with pytest.raises(KeySyntaxError, match="follows a named one"):
        parse("depth-grant:r1,depth=2,g")


def test_a_coordinate_may_not_be_named_twice():
    with pytest.raises(KeySyntaxError, match="given twice"):
        parse("depth-grant:r1,depth=2,depth=3")


def test_a_coordinate_at_its_default_is_omitted_not_left_empty():
    """One identity, one spelling, or replay misses. An empty coordinate is a gap where an
    omission belongs."""
    with pytest.raises(KeySyntaxError, match="empty coordinate"):
        parse("depth-grant:r1,,2")


# --- what is not in the language -----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", "empty key"),
        ("a;;b", "empty term"),
        ("Tag:x", "well-formed tag"),
        ("snake_tag:x", "well-formed tag"),
        ("-lead:x", "well-formed tag"),
        ("tag:", "no coordinates"),
        ("tag:x//y", "well-formed atom"),
        ("tag:=2", "empty coordinate name"),
    ],
)
def test_refusals(text, match):
    with pytest.raises(KeySyntaxError, match=match):
        parse(text)


def test_a_tag_is_lower_kebab_and_the_charset_does_not_widen():
    """§1e: conventions are good. The three snake_case tags are not admitted by
    relaxing this — all three cease to exist under §3b."""
    assert parse("depth-grant:r1,0,1").tags == ("depth-grant",)
    for offender in ("child_done", "fork_done", "fork_sealed"):
        with pytest.raises(KeySyntaxError, match="well-formed tag"):
            parse(f"{offender}:c1")


# --- round-tripping ------------------------------------------------------------------------


def test_render_is_the_inverse_of_parse():
    """Both directions. `parse(render(k)) == k` is what lets a consumer hold the structure and
    hand back bytes; `render(parse(t)) == t` is what makes the wire form canonical."""
    for text in [
        "tool:foo",
        "emit;ask:foo",
        "gather:0,0;event;foo",
        "depth-grant:r1,0,depth=2",
        "artifact:message/rfc822,sha256-9f2c",
        "$awaitEvent;ask:foo",
    ]:
        key = parse(text)
        assert key.render() == text
        assert parse(key.render()) == key


def test_a_structure_built_by_hand_renders_to_the_form_that_parses_back():
    """The composer's direction (§8 step 6 will use it): decisions first, bytes last."""
    key = ParsedKey(
        (
            Term("gather", (Coordinate((Atom.of("0"),)), Coordinate((Atom.of("1"),)))),
            Term("event"),
            Term("tool", (Coordinate((Atom.of("charge-card"),)),)),
        )
    )
    assert key.render() == "gather:0,1;event;tool:charge-card"
    assert parse(key.render()) == key


# --- the display projection ----------------------------------------------------------------


def test_names_can_be_projected_onto_a_positional_key():
    """`named` is the source map's decode: a composed key read back into NAMED fields.

    Kept out of `parse` so the self-describing rendering is visibly a projection over the one
    grammar: the wire never carries a name for a required coordinate."""
    key = parse("depth-grant:r1,0,1")
    shown = named(key, {"depth-grant": ("run_id", "generation", "depth")})
    assert shown.render() == "depth-grant:run_id=r1,generation=0,depth=1"


def test_a_projection_does_not_overwrite_a_name_the_wire_carried():
    key = parse("depth-grant:r1,0,depth=2")
    shown = named(key, {"depth-grant": ("run_id", "generation", "OTHER")})
    assert [c.name for c in shown.terms[0].coordinates] == ["run_id", "generation", "depth"]


def test_a_projection_of_an_unknown_tag_leaves_it_positional():
    key = parse("mystery:a,b")
    assert named(key, {}).render() == "mystery:a,b"


# --- skeletons: the same grammar over a template ---------------------------------------------
#
# Two consumers have to read a `compose_key` template the same way: the composer, which fills
# holes with VALUES, and the lint, which fills them with source EXPRESSIONS and registers a shape.
# The registry is a source map only if it describes what the composer actually mints.


def _skeleton(*parts):
    """A template's statics and holes, with holes numbered in order — what a caller would build
    from `list(template)`."""
    numbered, index = [], 0
    for part in parts:
        if part is ...:
            numbered.append(Hole(index))
            index += 1
        else:
            numbered.append(part)
    return parse_skeleton(numbered)


def test_a_single_term_template_counts_its_coordinates():
    shape = _skeleton("budget-grant:", ..., ",", ...)
    [term] = shape.elements
    assert term.tag == "budget-grant"
    assert len(term.coordinates) == 2
    assert shape.holes == 2


def test_a_literal_discriminator_is_a_coordinate_like_any_other():
    """§1c's first row: variants differing by a LITERAL discriminator are several shapes under one
    tag. `skill:{name},activate` and `skill:{name},refresh,{n}` are the shipped pair."""
    activate = _skeleton("skill:", ..., ",activate")
    refresh = _skeleton("skill:", ..., ",refresh,", ...)
    assert len(activate.elements[0].coordinates) == 2
    assert len(refresh.elements[0].coordinates) == 3
    assert activate.holes == 1
    assert refresh.holes == 2


def test_a_hole_standing_as_a_whole_term_is_a_splice():
    """`;` means the value EXTENDS the sequence rather than filling a coordinate — composition by
    induction, which is what makes a nested key expressible at all."""
    shape = _skeleton("gather:", ..., ",", ..., ";", ...)
    gather, splice = shape.elements
    assert (gather.tag, len(gather.coordinates)) == ("gather", 2)
    assert isinstance(splice, Splice)
    assert splice.hole == Hole(2)


def test_a_tag_may_itself_be_a_hole():
    """How a namespace constant names itself — `t"{GOVERN}:{gate},{run_id}"`. The `AuthorityTag`
    is a typed value, which is what lets the composed key retain the reach it declared."""
    shape = _skeleton(..., ":", ..., ",", ...)
    [term] = shape.elements
    assert term.tag == Hole(0)
    assert len(term.coordinates) == 2


def test_a_coordinate_may_be_a_path_of_holes():
    """`artifact:{type}/{subtype},{digest}` — a MIME type is `type/subtype`, two atoms."""
    shape = _skeleton("artifact:", ..., "/", ..., ",", ...)
    [term] = shape.elements
    assert len(term.coordinates) == 2
    assert len(term.coordinates[0].atoms) == 2  # the path
    assert len(term.coordinates[1].atoms) == 1


def test_a_static_only_template_is_a_bare_qualifier():
    shape = _skeleton("seed")
    assert shape.elements[0].tag == "seed"
    assert shape.holes == 0


def test_the_OLD_flat_grammar_is_refused_and_the_message_says_so():
    """The migration signal, and the reason the registry rework could not wait for step 9: a
    second `:` in a term is the old grammar, where every field was colon-separated. The lint's own
    skeleton reader could not see this at all — it reported the NEW form as an error instead."""
    with pytest.raises(KeySyntaxError, match="OLD flat grammar"):
        _skeleton("tag:", ..., ":", ...)


def test_a_hole_glued_to_literal_text_has_no_position():
    """The `unslotted-segment` rule, now a property of the shared parser rather than a check the
    lint kept privately: the value and the suffix cannot be told apart on the wire."""
    with pytest.raises(KeySyntaxError, match="not one atom"):
        _skeleton("x:", ..., "b")


def test_a_skeleton_and_the_key_it_mints_agree_on_arity():
    """The property the two consumers exist to share. A skeleton's coordinate count is what `parse`
    counts in the rendered key — so a registry built from templates describes the keys minted."""
    shape = _skeleton("depth-grant:", ..., ",", ..., ",", ...)
    minted = parse("depth-grant:r1,0,1")
    assert len(shape.elements) == len(minted.terms)
    assert len(shape.elements[0].coordinates) == minted.terms[0].arity


# --- §1f: the foreign adapter -----------------------------------------------------------------


FOREIGN_SHAPES = [
    # (the bytes the engine writes, what our terms should be, what `wrapped` returns)
    ("$awaitEvent:review:m1", ("$awaitEvent", "review"), "review:m1"),
    (
        "$awaitEvent:rec:0;review:sp-85847d85",
        ("$awaitEvent", "rec", "review"),
        "rec:0;review:sp-85847d85",
    ),
    (
        "$awaitEvent:approve;ledger;processed:sha256-bdbddb70872b68d9",
        ("$awaitEvent", "approve", "ledger", "processed"),
        "approve;ledger;processed:sha256-bdbddb70872b68d9",
    ),
    (
        "$awaitEvent:budget-grant:mg-c0ab9919,0",
        ("$awaitEvent", "budget-grant"),
        "budget-grant:mg-c0ab9919,0",
    ),
    ("$awaitEvent:q-f4f1b0b5", ("$awaitEvent", "q-f4f1b0b5"), "q-f4f1b0b5"),
    # The OTHER payload kind: Absurd's own task id, which is not a term of ours at all.
    ("$awaitTaskResult:0197f3aa-1c2d-7e88-9a0b-2f3c4d5e6f70", ("$awaitTaskResult",), None),
    # A bare foreign tag — §1f's original zero-arity case, still legal.
    ("$awaitEvent", ("$awaitEvent",), None),
]


@pytest.mark.parametrize(
    ("text", "tags", "wrapped"), FOREIGN_SHAPES, ids=[s[0][:40] for s in FOREIGN_SHAPES]
)
def test_a_foreign_name_the_ENGINE_writes_parses_and_round_trips(text, tags, wrapped):
    """Every shape here is copied from `absurd.c_default`, not invented.

    The remainder of a foreign key is OURS, and the vendored SDK writes it after a `:`
    (`f"$awaitEvent:{event_name}"`) rather than as a term of its own. Refusing that join would
    leave the engine's own rows outside the language, invisibly, because every reader here is
    total.
    """
    parsed = parse(text)
    assert parsed.tags == tags
    assert parsed.render() == text, "a checkpoint name that does not round-trip is a lookup miss"
    assert (parsed.wrapped.render() if parsed.wrapped else None) == wrapped


def test_the_occurrence_belongs_to_the_CHECKPOINT_not_the_wrapped_event():
    """`$awaitEvent:review:m1#2` is the second checkpoint of a park on `review:m1`.

    Keeping the two together is what made `graphview` draw `$awaitEvent:q#2` and `$awaitEvent:q`
    as two nodes, each reporting occurrence 1 — the identity and its use-index have to separate
    for a repeat to fold onto the node it repeats.
    """
    parsed = parse("$awaitEvent:review:m1#2")
    assert parsed.occurrence == 2
    assert parsed.wrapped is not None
    assert parsed.wrapped.render() == "review:m1"  # NOT `review:m1#2`
    assert parsed.render() == "$awaitEvent:review:m1#2"


def test_a_foreign_payload_is_still_CHECKED():
    """Foreign does not mean unchecked, or the column stops being parseable for a new reason.

    The payload is read in whichever grammar fits — ours when its head is a well-formed tag, an
    atom otherwise — and something that is neither is refused. A `Key` REPR is the live example:
    it is exactly what leaked into 28 checkpoint names, and it does not parse.
    """
    with pytest.raises(KeySyntaxError):
        parse("$awaitEvent:Key(_value='child-done:x', _scope=None)")
    with pytest.raises(KeySyntaxError):
        parse("$awaitEvent:review:a b")  # a space is in no atom kind


@pytest.mark.parametrize("nibble", list("0123456789abcdef"))
def test_a_task_id_reads_the_SAME_WAY_whatever_hex_digit_it_OPENS_with(nibble):
    """The payload's reading may not depend on an accident of the uuid's first byte.

    `TAG` (`[a-z][a-z0-9-]*`) and `UUID` (`[0-9a-f-]`) are not disjoint, so a task id opening
    `a`-`f` matched the tag pattern in full and became a TAG of ours, while one opening `0`-`9`
    became a coordinate — one key form, two structures, 6 of 16. `render` round-tripped both,
    which is why no reader could see it and why this parametrizes over all sixteen instead of
    pinning an example. The single fixture above is `0197f3aa-…`; digit-led, like every uuid7,
    which is exactly the case that already worked.
    """
    text = f"$awaitTaskResult:{nibble}198f0c1-1234-7abc-8def-0123456789ab"
    parsed = parse(text)
    assert parsed.tags == ("$awaitTaskResult",), "the task id is Absurd's atom, not a term of ours"
    assert parsed.wrapped is None
    assert parsed.render() == text


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("a198f0c1-1234-7abc-8def-0123456789ab", Kind.UUID),  # letter-led, so it matches TAG
        ("0198f0c1-1234-7abc-8def-0123456789ab", Kind.UUID),  # digit-led, so it does not
        ("sha256-9f2c1a", Kind.DIGEST),  # always matches TAG, and was missed the first time
        ("blake3-abcdef", Kind.DIGEST),
    ],
)
def test_an_ATOM_kind_is_never_a_tag_however_it_is_spelled(text, kind):
    """`is_tag` asks `kind_of`, so it covers the kinds rather than a list of them.

    The first version excluded `UUID` by name and shipped one kind short: a digest is entirely
    `[a-z0-9-]` and leads with a letter, so it matched the tag charset in every case rather than
    six in sixteen. Asking which KIND the bytes are is the same question `kind_of` already answers
    for atoms, and it does not need extending when a fifth kind arrives.
    """
    assert kind_of(text) is kind
    assert not is_tag(text)
    with pytest.raises(KeySyntaxError, match="is an ATOM and not a tag"):
        Term(text)
    with pytest.raises(ValueError, match="is an ATOM and not a tag"):
        Tag(text)  # the third gate, and the one the first pass missed
    # ... and it reads as the foreign term's COORDINATE, which is what it is.
    assert parse(f"$awaitTaskResult:{text}").wrapped is None


def test_a_uuid_is_an_ATOM_and_may_not_be_a_tag():
    """Stated at the constructor, so the composing side cannot mint what the parser now refuses.

    `kind_of` already tries the narrow kinds before `NAME` because a uuid and a digest both match
    `NAME`; this is the same rule at the tag boundary. A uuid names one thing, and a tag names a
    namespace of things.
    """
    with pytest.raises(KeySyntaxError, match="is a uuid"):
        Term("a198f0c1-1234-7abc-8def-0123456789ab")
    Term("a198f0c1")  # not uuid-shaped, so still an ordinary lower-kebab tag


# --- `named` is a PROJECTION, so it may not merge two keys -------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "$awaitEvent:review:m1",  # the foreign `:` join, dropped with `wraps_payload`
        "$awaitEvent:q-f00968aa#2",  # and the occurrence, dropped with the rebuilt ParsedKey
        "step;tool:read#3",  # ours, so this was never only a foreign-bytes problem
        "depth-grant:r-1,0,depth=1",
    ],
)
def test_attaching_field_names_does_not_change_the_KEY(text):
    """`named` re-renders a key with its field names attached; the bytes may not move.

    It called itself "the `display()` projection" while collapsing `step;tool:read#3` onto
    `step;tool:read` and re-rendering `$awaitEvent:review:m1` as `$awaitEvent;review:m1` — bytes
    no producer wrote. A display that shows two ops as one key is a projection that lost.
    """
    assert named(parse(text), {}).render() == text


@pytest.mark.parametrize(
    ("fields", "why"),
    [
        ({"gather": ("branch",)}, "a list shorter than the arity would name only a prefix"),
        ({"gather": ("a;b",)}, "a metacharacter in the name"),
        ({"gather": ("a=b",)}, "the name/value separator in the name"),
        ({"gather": ("",)}, "an empty name"),
        ({"gather": ("max(d, limit)",)}, "a hole expression that carries a separator"),
    ],
)
def test_a_term_that_cannot_be_named_SAFELY_keeps_its_authored_spelling(fields, why):
    """`named` is total: each of these would otherwise render bytes `parse` refuses.

    Both holes are reachable from the registry rather than invented. `fields` holds one tuple per
    tag while several tags have variants of different arity, and a field NAME is a hole's source
    expression — the map already holds `epoch_atom(until)` and `str(index)`, so a separator
    inside one is a comma away.

    Skipping rather than raising because this is a DISPLAY projection: an unnamed key is a
    smaller loss than a crashed dashboard, and it is what an absent entry already renders.
    """
    rendered = named(parse("gather:0,1" if "branch" in str(fields) else "gather:0"), fields)
    assert "=" not in rendered.render(), why
    parse(rendered.render())  # the property: whatever it renders, we can read back


def test_naming_from_the_REGISTRY_never_forges_bytes_over_the_live_shape_space():
    """The caller the docstring names — a map built from `build/key-registry.json` — is safe.

    Three ways a caller could reconcile a tag whose variants disagree on arity (first, longest,
    shortest), over every key shape the registry can mint. The assertion is that naming either
    happens or is skipped, never renders something `parse` refuses.
    """
    registry = json.loads(Path("build/key-registry.json").read_text())["shapes"]
    picks = {
        "first": {t: tuple(v["variants"][0]["fields"]) for t, v in registry.items()},
        "longest": {
            t: max((tuple(x["fields"]) for x in v["variants"]), key=len)
            for t, v in registry.items()
        },
    }
    samples = ["gather:0,1", "approve:2,alice", "step;tool:read#3", "$awaitEvent:review:m1"]
    for how, fields in picks.items():
        for text in samples:
            rendered = named(parse(text), fields).render()
            parse(rendered)  # raises if the projection forged bytes
            assert how  # the pick is part of the case identity


# --- round-trip as a PROPERTY over generated keys ---------------------------------------------


def _generated_keys(rounds: int = 4000) -> list[ParsedKey]:
    """Structures across the shape space, built then rendered — so every case is in-language.

    Generated from the STRUCTURE outward rather than by making strings and hoping: a text
    generator mostly produces parse errors, and the interesting property is over keys that are
    real. Deterministic (`Random(7)`), so a failure is reproducible from the seed alone.
    """
    from random import Random

    rng = Random(7)
    tags = ["step", "event", "ledger", "tool", "rec", "gather", "budget-grant", "q-a1"]
    atoms = ["a", "m1", "0", "12", "sha256-9f2c", "epoch-1767225600.5", "x.y@z", "a_b", "A-b"]
    out: list[ParsedKey] = []
    for _ in range(rounds):
        terms: list[Term] = []
        # Frames BEFORE the foreign term, so its join is exercised away from the head as well.
        # The engine writes `gather:0,0;$awaitEvent:review:m1` under a gather, and a generator
        # that only ever puts the foreign term first cannot reach that shape.
        for _ in range(rng.choice([0, 0, 0, 1, 2])):
            frame = rng.choice(["gather", "rec", "d"])
            terms.append(Term(frame, (Coordinate((Atom.of(str(rng.randint(0, 2))),)),)))
        if rng.random() < 0.15:  # a foreign term, both joins, leading or framed
            terms.append(Term("$awaitEvent", (), foreign=True, wraps_payload=rng.random() < 0.5))
            if terms[-1].wraps_payload:  # a wrapping term needs a payload to wrap
                terms.append(Term(rng.choice(tags), ()))
        for _ in range(rng.randint(1, 3)):
            positional = tuple(
                Coordinate(tuple(Atom.of(rng.choice(atoms)) for _ in range(rng.randint(1, 2))))
                for _ in range(rng.randint(0, 2))
            )
            named = tuple(
                Coordinate((Atom.of(rng.choice(atoms)),), name=n)
                for n in rng.sample(["depth", "pass-n", "occurrence"], rng.randint(0, 2))
            )
            terms.append(Term(rng.choice(tags), positional + named))
        occurrence = rng.choice([None, None, None, 2, 3, 47])
        out.append(ParsedKey(tuple(terms), occurrence))
    return out


def test_render_and_parse_are_inverse_over_the_whole_shape_space():
    """Both directions, over generated keys rather than a handful of examples.

    The property is seeded so a failure reproduces. It covers what the example list cannot
    enumerate: atom
    kinds crossed with arity, `/` paths, named coordinates past the boundary, occurrence suffixes,
    foreign heads under BOTH joins, and every combination of them.
    """
    keys = _generated_keys()
    assert len(keys) > 1000, "the generator went vacuous"
    seen_text: dict[str, ParsedKey] = {}
    for key in keys:
        text = key.render()
        assert parse(text) == key, f"structure did not survive {text!r}"
        assert parse(text).render() == text, f"bytes did not survive {text!r}"
        # INJECTIVITY, over everything generated: one structure per spelling.
        if (clash := seen_text.get(text)) is not None:
            assert clash == key, f"{text!r} is minted by two DIFFERENT structures"
        seen_text[text] = key
    # anti-vacuity: the space really was exercised
    assert any("#" in t for t in seen_text)
    assert any(t.startswith("$") for t in seen_text)
    assert any("=" in t for t in seen_text)
    assert any("/" in t for t in seen_text)


def test_the_two_properties_1g_IS_and_that_nothing_else_pinned():
    """The occurrence grammar's headline guarantees, asserted where they are SPECIFIED.

    Break `#1`-has-no-wire-form, or let an interior splice keep its occurrence, and without these
    pins the entire infra-free suite stays green: the only other guards are durable-engine tests
    that fail for a downstream reason, so on a laptop with no Postgres the guarantees would have
    no guard at all.
    """
    from effective.keys import Key, compose_key

    # 1. `#1` has NO wire form: a first occurrence is the bare key, so two spellings cannot exist.
    with pytest.raises(KeySyntaxError, match="occurrence 1"):
        parse("a#1")
    assert Key.parse("review:m1").occurrence(1).stored() == "review:m1"

    # 2. An INTERIOR splice of a suffixed key refuses — an interior occurrence has no reader,
    #    and the old behaviour stranded it silently.
    suffixed = Key.parse("review:m1").occurrence(2)
    with pytest.raises(ValueError, match="occurrence"):
        compose_key(t"ns:1;{suffixed:domain=any};extra:1")
    # ANTI-VACUITY: the TERMINAL splice still hoists, which is the case §1g keeps.
    # lint: terminal-hole — `suffixed` is a `Key`, which the rule cannot see.
    assert compose_key(t"ns:1;{suffixed:domain=any}").stored() == "ns:1;review:m1#2"


# --- the CONTAINMENT partition, as a property over the same corpus ----------------------------


def test_split_frames_and_the_identity_reconstruct_the_key():
    """The partition is positional and total, so nothing may fall between the halves.

    Stated as a round-trip rather than as example frame tuples because the failure this guards is
    a term going missing — a walk that consumed a leading term and then classified it as neither
    a frame nor part of the identity would still look right on the machine tape, whose frames all
    carry coordinates. Over the generated corpus a bare tag in the leading run is reachable, and
    that is the case a hand-written list would not have held.
    """
    keys = [key.render() for key in _generated_keys()]
    assert len(keys) > 1000, "the generator went vacuous"
    wrapped = 0
    for key in keys:
        frames, identity = split_frames(key)
        halves = [t for frame in frames for t in parse(frame).terms] + list(parse(identity).terms)
        assert sorted(map(repr, halves)) == sorted(map(repr, parse(key).terms)), (
            f"the partition lost a term on {key!r}"
        )
        if any(term.wraps_payload for term in parse(key).terms):
            wrapped += 1
        else:
            # Without a wrapping term the halves are still POSITIONAL, which is the stronger
            # claim and the one most keys get.
            assert ";".join((*frames, identity)) == key
    assert wrapped > 100, "the generator stopped minting the case this test is about"


def test_past_frames_is_split_frames_with_a_filter_over_the_first_half():
    """The relationship the new function is FOR: one walk, two readers.

    `past_frames` keeps a frame when its tag is not in `drop`, or when it carries no coordinate at
    all (a bare `gather` qualifier frames nothing, so there is nothing to remove). Re-deriving its
    answer from the partition is what proves the two walks cannot drift — the alternative is a
    second implementation of the same term scan, which is the shape this repo keeps finding as
    drift between a writer and a reader.

    Four drop sets, including the empty one: with nothing to drop, `past_frames` must return its
    input unchanged, which is the arm a filter is easiest to get wrong. One set mixes an arm with
    scope tags, because the two arrive from different vocabularies and the walk tests only
    membership.
    """
    keys = [key.render() for key in _generated_keys()]
    for drop in (FRAME_ARMS, (*FRAME_ARMS, "rec", "fold", "d"), ("rec", "d"), ()):
        for key in keys:
            # By POSITION, as the walk filters: an identity term whose bytes echo a dropped
            # frame's (`rec:0;event;rec:0`) must survive, and a value filter would remove it.
            found = _leading_frames(key)
            assert found is not None, "every generated key is in the language"
            at_frames, _at_identity = found
            terms = parse(key).terms
            removed = {at for at in at_frames if terms[at].tag in drop and terms[at].coordinates}
            survivors = list(parse(past_frames(key, drop=drop)).terms)
            expected = [t for at, t in enumerate(terms) if at not in removed]
            assert list(map(repr, survivors)) == list(map(repr, expected)), (
                f"the two walks disagree on {key!r} at drop={drop!r}"
            )


def test_a_frame_run_ends_at_the_first_arm_that_is_not_a_frame():
    """The identity boundary, on the shapes the substrate actually mints.

    `gather` is in `ARM_TAGS` *and* in `FRAME_ARMS`, and that is the discrimination: a walk that
    stopped at "the first arm" would make every gather identify instead of wrap, flattening all
    branches into leaf labels. The last two rows are the pair that catches it: a gather LEADING is
    a frame, and a gather carried INSIDE an arm was never in the leading run and must be left
    alone.
    """
    assert split_frames("d:1;state:draft;step;tool:apply_fix") == (
        ("d:1", "state:draft"),
        "step;tool:apply_fix",
    )
    assert split_frames("step;tool:list_dir;turn:0") == ((), "step;tool:list_dir;turn:0")
    assert split_frames("gather:0,0;step;tool:a") == (("gather:0,0",), "step;tool:a")
    assert split_frames("event;gather:0,0;ev:r1") == ((), "event;gather:0,0;ev:r1")
    assert split_frames("race:0,1;gather:0,0;step;tool:a") == (
        ("race:0,1", "gather:0,0"),
        "step;tool:a",
    )


def test_race_and_gather_frames_never_share_bytes():
    """A race branch, a gather branch and a race's choice, at every small coordinate, are
    pairwise distinct, and each frame wraps the same identity the same way."""
    inner = Key.parse("step;tool:a")
    minted = [
        *(Key.parse(gather_prefix(g, i) + inner.stored()) for g in range(3) for i in range(3)),
        *(Key.parse(race_prefix(r, i) + inner.stored()) for r in range(3) for i in range(3)),
        *(race_choice(r) for r in range(3)),
    ]
    assert len({key.stored() for key in minted}) == len(minted)
    for r, i in ((0, 0), (1, 2)):
        framed = race_prefix(r, i) + inner.stored()
        assert split_frames(framed) == (
            (framed.removesuffix(";" + inner.stored()),),
            "step;tool:a",
        )
        assert past_frames(framed) == inner.stored()


def test_a_branch_frame_is_told_from_a_race_s_own_frames_by_arity():
    """Which frames name a BRANCH, on the shapes the substrate actually mints.

    `is_branch_frame` reads the coordinate count, so asserting it against the constants it is
    built from would be circular. The frames here come from the minters the handlers use, which
    is the only thing that says what arity each one really carries. A race is the discrimination:
    it frames its branches with two coordinates and its own choice and endings with one, so a
    reader keyed on the tag alone answers the same for both.
    """
    inner = Key.parse("step;tool:a").stored()
    assert is_branch_frame("race:0,1")
    assert is_branch_frame("gather:0,1")
    assert not is_branch_frame("race:0")
    assert branch_frames(race_prefix(0, 1) + inner) == ("race:0,1",)
    assert branch_frames(gather_prefix(0, 1) + inner) == ("gather:0,1",)
    assert branch_frames(race_prefix(0, 1) + gather_prefix(2, 0) + inner) == (
        "race:0,1",
        "gather:2,0",
    )
    # A race's own records wrap no branch, and neither does a scope or a bare identity.
    assert branch_frames(race_choice(0).stored()) == ()
    assert branch_frames(race_endings(0).stored()) == ()
    assert branch_frames("d:1;state:draft;step;tool:apply_fix") == ()
    assert branch_frames(inner) == ()
    # A scope INSIDE a branch keeps the branch: the innermost one is what an op ran in.
    assert branch_frames(gather_prefix(0, 1) + "d:3;" + inner) == ("gather:0,1",)


def test_the_partition_is_total_over_text_the_language_refuses():
    """Total over STRINGS, for the reason `past_frames` is: the banked corpus.

    `tests/test_banked_corpus.py` asserts that at least one legacy key does NOT parse and that
    every banked store stays readable, so a viewer whose frame split went through `parse` could
    not open half the corpus. These are shapes `parse` rejects; the partition still answers.
    """
    for refused in ("gather:0:0:", "cand:0,0;", "a b;step;tool:x", ""):
        with pytest.raises(KeySyntaxError):
            parse(refused)
        frames, identity = split_frames(refused)
        assert ";".join((*frames, identity)) == refused


# --- `Key.parse`'s journey ------------------------------------------------------------------
#
# The docstring shows four lines of behaviour; these are them. Same rule as `Key.scope`'s rows:
# the case set is named in prose, and every case has a name appearing in both the prose and the
# test, so a change to what `parse` promises reddens something rather than just aging a docstring.


def test_parse_refuses_only_the_empty_name():
    # The empty key addresses nothing and every empty key is the same key, so it collides with
    # itself wherever it is stored. That is the ONE thing parse can refuse without a grammar.
    with pytest.raises(ValueError, match="empty key addresses nothing"):
        Key.parse("")


def test_parse_refuses_text_the_grammar_would_refuse():
    """`Key` holds terms, so parsing is the read boundary doing its job.

    Under a lazy read, text that is not a key travels as one until something asks for its terms,
    and mostly nothing does. An eager parse makes the failure arrive where the text enters rather
    than wherever it is next used."""
    with pytest.raises(KeySyntaxError):
        Key.parse("not a key at all")


def test_a_parsed_key_is_weaker_than_the_composed_key_it_came_from():
    # THE hazard, and the reason `parse` is a named seam rather than a cast. The bytes survive the
    # wire and the guarantee does not: a composed key retained what its typed AuthorityTag
    # declared, and text cannot carry that. A consumer reading `.scope` off a parsed key to decide
    # authority gets `None` — silently the un-numbered, permissive answer.
    composed = compose_key(t"{GOVERN}:{Segment('r1')}")
    reparsed = Key.parse(composed.stored())
    assert composed.stored() == reparsed.stored()
    assert composed.scope is Scope.SETTLEMENT
    assert reparsed.scope is None


def test_parse_is_the_pydantic_validator_so_json_takes_the_same_door():
    # The door a caller takes WITHOUT SEEING IT, and the one the docstring never named: every Key
    # in a model field arriving from a checkpoint round-trip or a ledger row is parsed here.
    class Holder(BaseModel):
        key: Key

    holder = Holder.model_validate_json('{"key": "govern:r1"}')
    assert holder.key == Key.parse("govern:r1")
    assert holder.model_dump(mode="json") == {"key": "govern:r1"}
    with pytest.raises(ValidationError):
        Holder.model_validate_json('{"key": ""}')


# --- what the frame separator's docstring shows --------------------------------------------------


def test_a_frame_is_just_the_leading_term():
    # There is no second grammar to reconcile: a framed key parses like any other, and the
    # boundary is where a term ends rather than something a reader searches for.
    framed = "d:0;state:draft;gather:0,1;govern:r1"
    assert [term.tag for term in parse(framed).terms] == [
        "d",
        "state",
        "gather",
        "govern",
    ]
    assert split_frames(framed) == (("d:0", "state:draft", "gather:0,1"), "govern:r1")


def test_the_term_sequence_narrows_left_to_right():
    # The property the docstring's diagram shows, and the one a two-term example cannot: each
    # frame picks a smaller region of the run than the one before it, and the last term is the op.
    # Cutting the sentence that SAID this left it neither said nor shown, which is how a
    # compression turns into a loss.
    framed = "d:0;state:draft;gather:0,1;govern:r1"
    frames, op = split_frames(framed)
    assert op == "govern:r1"
    # each frame is a prefix of the one below it in the diagram, widest first
    assert frames == ("d:0", "state:draft", "gather:0,1")
    for narrower in range(1, len(frames) + 1):
        assert framed.startswith(";".join(frames[:narrower]))


def test_the_two_delimiters_are_refused_for_their_own_reasons():
    # `;` would forge a scope boundary; `/` would make arity uncountable. Different hazards, and
    # the messages say which — a reader who hits one should not have to guess which rule bit.
    with pytest.raises(ValueError, match="frame delimiter"):
        Segment("a;b")
    with pytest.raises(ValueError, match="path delimiter"):
        Segment("a/b")


# --- what `Tag`'s docstrings show ---------------------------------------------------------------


def test_a_tag_composes_as_a_literal_or_as_a_constant():
    # Both spellings exist because a namespace is sometimes a constant, and forcing the literal
    # form would mean writing it twice. The CONSTANT example in that docstring was dead for an
    # unknown time — carried forward across a rewrite in the OLD flat grammar `{tag}:{a}:{b}`,
    # which parse_skeleton refuses. Pinned so it cannot go dead again unnoticed.
    from effective.counterfactual import FORK_SCOPE

    assert compose_key(t"tool:{Segment('read_file')}").display() == "tool:read_file"
    # lint: terminal-hole  -- a `Key` in the wrapped hole is what QUALIFIED requires
    composed = compose_key(t"{FORK_SCOPE}:{Segment('c1')};{Key.parse('review:m1'):domain=address}")
    assert composed.display() == "hyp:c1;review:m1"


def test_a_tag_is_checked_against_the_grammar_not_a_denylist():
    # Every one of these is a byte the old two-character denylist (`:` and `;`) let through.
    for bad in ("a#2", "a,b", "A", "1a"):
        with pytest.raises(ValueError, match="not a well-formed tag"):
            Tag(bad)
    with pytest.raises(TypeError, match="takes a str, not Key"):
        Tag(Key.parse("event:x"))  # ty: ignore[invalid-argument-type]


# --- what `Scope`'s docstrings show --------------------------------------------------------------


def _asked_twice_key(name: Key) -> list[str]:
    from effective.handlers.base import placed_await_name, placing
    from effective.keys import FramePosition
    from effective.ops import AwaitEvent

    position, seen = FramePosition(), []
    for _ in range(2):
        op = AwaitEvent(name=name, schema=dict)
        with placing(op, position):
            seen.append(placed_await_name(op).display())
    return seen


def test_the_three_scopes_differ_where_the_walk_reads_them():
    # The table `Scope`'s docstring shows. Only SETTLEMENT makes a second ask a second question.
    from effective.budget import BUDGET_GRANT, DEPTH_GRANT
    from effective.counterfactual import FORK_SCOPE

    settlement = compose_key(t"{DEPTH_GRANT}:{Segment('r1')},{Segment('0')}")
    accrual = compose_key(t"{BUDGET_GRANT}:{Segment('r1')}")
    child_event = Key.parse("review:m1")
    # lint: terminal-hole  -- a `Key` in the wrapped hole is what QUALIFIED requires
    qualified = compose_key(t"{FORK_SCOPE}:{Segment('c1')};{child_event:domain=address}")
    assert _asked_twice_key(settlement) == ["depth-grant:r1,0", "depth-grant:r1,0#2"]
    assert _asked_twice_key(accrual) == ["budget-grant:r1", "budget-grant:r1"]
    assert _asked_twice_key(qualified) == ["hyp:c1;review:m1", "hyp:c1;review:m1"]


def test_a_qualified_key_recurses_the_occurrence_to_what_it_wraps():
    # QUALIFIED identifies no occurrence of its own, so the question does not vanish — the
    # wrapped key keeps its own. That claim was prose for months; this is what it looks like.
    from effective.counterfactual import FORK_SCOPE
    from effective.govern import GOVERN

    inner = compose_key(t"{GOVERN}:{Segment('r1')}").occurrence(2)
    # lint: terminal-hole  -- a `Key` in the wrapped hole is what QUALIFIED requires
    wrapped = compose_key(t"{FORK_SCOPE}:{Segment('c1')};{inner:domain=address}")
    assert inner.display() == "govern:r1#2"
    assert wrapped.display() == "hyp:c1;govern:r1#2"
    assert wrapped.scope is Scope.QUALIFIED


def test_segment_refuses_a_delimiter_an_empty_string_and_a_non_str():
    # The three refusals `Segment`'s docstring shows. The Key case is refused by TYPE: a delimiter
    # check alone would refuse a TAGGED key for the wrong reason (its repr contains a ':') and let
    # a tag-only key's repr sail through.
    with pytest.raises(ValueError, match="key delimiter"):
        Segment("a:b")
    with pytest.raises(ValueError, match="empty string"):
        Segment("")
    with pytest.raises(TypeError, match="not a str"):
        Segment(Key.parse("tagonly"))  # ty: ignore[invalid-argument-type]


def test_a_key_rides_pydantic_as_text_and_refuses_a_str_in_python_mode():
    # The asymmetry `__get_pydantic_core_schema__` shows: type the side where the guarantee is
    # enforced, parse the side where it is not.
    from pydantic import BaseModel, ValidationError

    class Row(BaseModel):
        event_id: Key

    key = compose_key(t"{GOVERN}:{Segment('r1')}")
    assert Row(event_id=key).model_dump(mode="json") == {"event_id": "govern:r1"}
    assert Row.model_validate_json('{"event_id": "govern:r1"}').event_id.stored() == "govern:r1"
    with pytest.raises(ValidationError):
        Row(event_id="govern:r1")


def test_prefixed_keeps_the_reach_and_occurrence_is_byte_preserving_at_one():
    # Two shown behaviours. The reach travelling with the identity is what makes a framed
    # depth-grant: still a settlement; occurrence(1) returning the bare key is what keeps every
    # recorded run from being orphaned by the suffix existing.
    key = compose_key(t"{GOVERN}:{Segment('r1')}")
    framed = key.prefixed("gather:0,1;")
    assert framed.display() == "gather:0,1;govern:r1"
    assert framed.scope is Scope.SETTLEMENT
    assert key.occurrence(1).display() == "govern:r1"
    assert key.occurrence(2).display() == "govern:r1#2"
    with pytest.raises(ValueError, match="is not a count"):
        key.occurrence(0)
    with pytest.raises(ValueError, match="already carries occurrence"):
        key.occurrence(2).occurrence(3)


def test_the_composer_refuses_a_raw_terminal_hole_and_adjacent_holes():
    # Both refusals `compose_key`'s docstring shows. The terminal one matters most: a raw
    # terminal hole is refused like any other, with no exemption for the last position.
    with pytest.raises(ValueError, match="may contain a separat"):
        # lint: terminal-hole  -- raw ON PURPOSE; this line asserts the composer refuses it, so
        # wrapping it in a Segment would delete the case. The lint and the test agree.
        compose_key(t"{Tag('ns')}:{Segment('a')};{'x:y'}")
    with pytest.raises(ValueError, match="adjacent"):
        compose_key(t"{Tag('ns')}:{Segment('p')}{Segment('q')}")


def test_a_qualified_wrapped_hole_must_hold_a_key():
    # What QUALIFIED's docstring shows. A Key there was composed, so its namespace was checked
    # when it was made; a str or a Segment is whatever the caller typed.
    from effective.counterfactual import FORK_SCOPE

    child_event = Key.parse("review:m1")
    # lint: terminal-hole  -- a `Key` in the wrapped hole is what QUALIFIED requires
    good = compose_key(t"{FORK_SCOPE}:{Segment('c1')};{child_event:domain=address}")
    assert good.display() == "hyp:c1;review:m1"
    with pytest.raises(TypeError, match="stands as a term's namespace"):
        compose_key(t"{FORK_SCOPE}:{Segment('c1')};{Segment('review-m1'):domain=address}")
    with pytest.raises(ValueError, match="is str, which may contain a separat"):
        # lint: terminal-hole  -- raw ON PURPOSE; the line asserts the composer refuses it
        compose_key(t"{FORK_SCOPE}:{Segment('c1')};{'review:m1':domain=address}")


@pytest.mark.parametrize("g", [0, 1, 12])
@pytest.mark.parametrize("i", [0, 1, 7])
def test_the_gather_prefix_is_minted_once_for_every_walk(g, i):
    """Every walk mints the gather frame through `gather_prefix`.

    The durable handler's `_branch_handler` and `_join`, the recorder's `run_branch` and the
    replay driver's `_drive` all apply the same frame. A walk that spells it can drift from the
    others, and checkpoint names cannot survive that drift.
    """
    assert gather_prefix(g, i) == f"gather:{g},{i};"

    # It goes through `scope_prefix`, so it inherits that function's refusal rather than
    # restating it: the frame separator is applied in the one place that applies one.
    prefix = gather_prefix(g, i)
    assert parse(prefix + "step;tool:read").tags == ("gather", "step", "tool")
