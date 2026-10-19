"""A key's structure is canonical: what it holds is what its stored form parses to.

`Key` equality is structural, and replay compares a live key with one read back through
`Key.parse`, so every route that builds a key, composition and prefixing included, has to land on
the structure `parse` gives its bytes.
"""

import pickle
from collections.abc import Callable
from itertools import product
from string.templatelib import Interpolation, Template

import pytest
from test_grammar_induction import ATOMS, CELLS, _compose

from effective.govern import GOVERN
from effective.graphview import from_keys, project
from effective.keys import Key, Segment, Tag, compose_key
from effective.keys.frame import (
    admits,
    leads_with_an_arm,
    past_frames,
    scope_prefix,
    split_frames,
)
from effective.keys.grammar import (
    Atom,
    Coordinate,
    Domain,
    KeySyntaxError,
    Kind,
    ParsedKey,
    Term,
    parse,
    split_occurrence,
)
from effective.keys.processor import gather_prefix

READ = (
    "$awaitEvent:ask:foo",
    "$awaitEvent",
    "$awaitTaskResult:0197f3aa-1c2d-7e88-9a0b-2f3c4d5e6f70",
    "gather:0,1;$awaitEvent:review:m1#2",
    "event;approve;step;tool:charge-card",
    "step;tool:a#3",
    "gather:0,0;gather:1,2;event;review:m1#2",
    "rec:1;govern:r1",
    "$awaitEvent:gather:0,1;$awaitEvent:foo",
    "n:a/b=1",
    "n:a:b=1",
    "n:*=1",
)
"""Keys read back off an engine, among them the foreign terms only `parse` mints, and names and
adjacent wrappers a stricter check could refuse."""


def _keys() -> list[Key]:
    composed = [
        key if occurrence is None else key.occurrence(occurrence)
        for atom, shape, binding, arity, terms, occurrence in CELLS
        for key in [_compose(ATOMS[atom], shape, binding, arity, terms)]
    ]
    review = compose_key(t"review:{Segment('m1')}")
    spliced = [
        # lint: terminal-hole: `review` is a `Key`.
        compose_key(t"event;{review:domain=address}"),
        # lint: terminal-hole: `review.occurrence(2)` is a `Key`.
        compose_key(t"event;{review.occurrence(2):domain=address}"),
        compose_key(t"{GOVERN}:{Segment('r1')}"),
    ]
    return [*composed, *spliced, *(Key.parse(text) for text in READ)]


KEYS = _keys()

PREFIXES = (
    "",
    gather_prefix(0, 1),
    gather_prefix(2, 0) + gather_prefix(0, 3),
    scope_prefix(compose_key(t"rec:{Segment('a-b')}")),
    scope_prefix(Key.parse("$awaitEvent:review:m1")),
)
"""What the frame layer puts in front of a key: nothing, one frame, two, and a scope atom."""

CORPUS = [key.prefixed(prefix) for key in KEYS for prefix in PREFIXES]


@pytest.mark.parametrize("key", CORPUS, ids=[key.stored() for key in CORPUS])
def test_every_key_holds_the_structure_its_stored_form_parses_to(key):
    assert Key.parse(key.stored()) == key


def _foreign_static_head() -> Template:
    review = compose_key(t"review:{Segment('m1')}")
    return t"$awaitEvent;{review:domain=any}"


def _foreign_static_inner() -> Template:
    return t"ns;$awaitEvent:{Segment('a')}"


def _foreign_tag_hole() -> Template:
    return Template(
        Interpolation(Tag("$awaitEvent"), "head_tag", None, ""),
        ":",
        Interpolation(Segment("a"), "a", None, ""),
    )


def _foreign_tag_splice() -> Template:
    return Template("ns;", Interpolation(Tag("$awaitEvent"), "lone_tag", None, ""))


@pytest.mark.parametrize(
    ("template", "names"),
    [
        (_foreign_static_head, "'\\$awaitEvent'"),
        (_foreign_static_inner, "'\\$awaitEvent'"),
        (_foreign_tag_hole, "'head_tag'"),
        (_foreign_tag_splice, "'lone_tag'"),
    ],
)
def test_the_composer_refuses_a_foreign_tag_at_a_term_head(template, names):
    """A foreign tag is another grammar's, so only reading its bytes mints one; the refusal names
    the tag a static wrote, or the hole that carried it."""
    with pytest.raises(ValueError, match=f"{names}.*foreign"):
        compose_key(template())


def test_a_scope_atom_carrying_an_occurrence_is_refused():
    with pytest.raises(ValueError, match="occurrence"):
        scope_prefix(Key.parse("rec:1#2"))


def test_a_key_prefixed_by_a_fragment_carrying_an_occurrence_is_refused():
    with pytest.raises(ValueError, match="occurrence"):
        Key.parse("govern:r1").prefixed("rec:1#2;")


def _reparsed(key: Key, prefix: str) -> Key:
    """`Key.prefixed` as the re-parse of its own rendering: the oracle."""
    return Key(parse(prefix + key.stored()), key.scope)


@pytest.mark.parametrize("prefix", PREFIXES)
def test_prefixing_agrees_with_reparsing_the_joined_text(prefix):
    for key in KEYS:
        prefixed, reparsed = key.prefixed(prefix), _reparsed(key, prefix)
        assert (prefixed.stored(), prefixed.terms(), prefixed.scope) == (
            reparsed.stored(),
            reparsed.terms(),
            reparsed.scope,
        )


@pytest.mark.parametrize(
    ("fragment", "rule"),
    [
        ("gather:0,1", "does not end in"),
        (";", "addresses nothing"),
        ("rec:1;;", "empty term"),
        ("rec:1#2;", "occurrence"),
        ("Rec:1;", "not a well-formed tag"),
    ],
)
def test_a_fragment_outside_the_language_is_refused(fragment, rule):
    with pytest.raises(ValueError, match=rule):
        Key.parse("govern:r1").prefixed(fragment)


@pytest.mark.parametrize(
    ("placement", "key", "is_one"),
    [
        ("govern:r1", "govern:r1", True),
        ("govern:r1#2", "govern:r1", True),
        ("gather:0,1;govern:r1#2", "gather:0,1;govern:r1", True),
        ("govern:r1#3", "govern:r1#2", False),
        ("govern:r1#2", "govern:r1#2", False),
        ("govern:r1", "govern:r1#2", False),
        ("govern:r2", "govern:r1", False),
        ("gather:0,1;govern:r1", "govern:r1", False),
    ],
)
def test_a_placement_is_an_occurrence_of_an_unnumbered_key(placement, key, is_one):
    assert Key.parse(placement).is_occurrence_of(Key.parse(key)) is is_one


def test_occurrence_agrees_with_stripping_the_suffix_off_the_text():
    for key in KEYS:
        base = key.unnumbered()
        numbered = (base, base.occurrence(2), base.occurrence(3))
        for placement, mine in product(numbered, numbered[:2]):
            text = split_occurrence(placement.stored())[0] == mine.stored()
            assert placement.is_occurrence_of(mine) is text, (placement, mine)


@pytest.mark.parametrize("domain", list(Domain))
def test_a_key_is_admitted_where_its_text_is(domain):
    for key in CORPUS:
        assert admits(domain, key) is admits(domain, key.stored()), key.stored()


def test_a_key_leads_with_an_arm_where_its_text_does():
    for key in CORPUS:
        assert leads_with_an_arm(key) is leads_with_an_arm(key.stored()), key.stored()


def test_templates_alike_but_for_a_hole_are_composed_apart():
    """A template's statics alone do not decide its skeleton: a spec or an expression can."""
    step = compose_key(t"step;tool:{Segment('a')}")
    # lint: terminal-hole: `step` is a `Key`.
    assert compose_key(t"event;{step:domain=any}").stored() == "event;step;tool:a"
    with pytest.raises(ValueError, match="domain=address"):
        # lint: terminal-hole: `step` is a `Key`.
        compose_key(t"event;{step:domain=address}")
    first = second = 7
    with pytest.raises(ValueError, match="'first'"):
        # lint: terminal-hole: `first` is an `int`.
        compose_key(t"n:{first:domain=any}")
    with pytest.raises(ValueError, match="'second'"):
        # lint: terminal-hole: `second` is an `int`.
        compose_key(t"n:{second:domain=any}")


def test_a_rendering_belongs_to_its_own_key():
    base = compose_key(t"tool:{Segment('a')}")
    assert base.stored() == "tool:a"
    assert base.occurrence(2).stored() == "tool:a#2"
    assert base.prefixed("rec:1;").stored() == "rec:1;tool:a"
    assert Key.parse("tool:b").stored() == "tool:b"
    assert base.stored() == "tool:a"


def _built() -> list[ParsedKey]:
    """Structures built from the constructors alone, over a product of their fields."""
    atoms = [
        built
        for text in ("2", "abc", "0197f3aa-1c2d-7e88-9a0b-2f3c4d5e6f70", "sha256-9f", "a b")
        for kind in Kind
        if (built := _or_refused(lambda text=text, kind=kind: Atom(text, kind))) is not None
    ]
    coordinates = [
        built
        for path in [(), *((atom,) for atom in atoms), tuple(atoms[:2])]
        for name in (None, "x", "", "a,b")
        if (built := _or_refused(lambda path=path, name=name: Coordinate(path, name))) is not None
    ]
    shapes = [
        (),
        *((c,) for c in coordinates),
        *((c, d) for c in coordinates[:6] for d in coordinates[:6]),
    ]
    terms = [
        built
        for tag in ("t", "$a")
        for shape in shapes
        for foreign, wraps in ((False, False), (True, False), (True, True))
        if (
            built := _or_refused(
                lambda tag=tag, shape=shape, f=foreign, w=wraps: Term(
                    tag, shape, foreign=f, wraps_payload=w
                )
            )
        )
        is not None
    ]
    heads = terms[:: max(1, len(terms) // 12)]
    wrapping = [term for term in terms if term.wraps_payload]
    ours = [term for term in terms if not term.foreign][:6]
    sequences = [
        *((term,) for term in terms),
        *((a, b) for a in heads for b in heads),
        *((w, o) for w in wrapping for o in ours),
    ]
    return [
        built
        for sequence in sequences
        for occurrence in (None, 1, 2)
        if (built := _or_refused(lambda s=sequence, o=occurrence: ParsedKey(s, o))) is not None
    ]


def _or_refused[T](build: Callable[[], T]) -> T | None:
    try:
        return build()
    except KeySyntaxError:
        return None


def test_a_structure_the_constructors_accept_reads_back_as_itself():
    misread = [built for built in _built() if _reads_back(built) != built]
    assert not misread, misread[:5]


def _reads_back(built: ParsedKey) -> ParsedKey | None:
    try:
        return parse(built.render())
    except KeySyntaxError:
        return None


def test_a_numbered_key_keeps_its_reach_without_its_occurrence():
    numbered = compose_key(t"{GOVERN}:{Segment('r1')}").occurrence(2)
    assert (numbered.unnumbered().stored(), numbered.unnumbered().scope) == (
        "govern:r1",
        numbered.scope,
    )


def test_a_parsed_key_pickled_before_its_rendering_slot_still_renders():
    pickled = Key.parse("gather:0,1;govern:r1#2").structure()
    assert pickle.loads(pickle.dumps(pickled)).render() == "gather:0,1;govern:r1#2"
    legacy = ParsedKey.__new__(ParsedKey)
    legacy.__setstate__([pickled.terms, pickled.occurrence])  # ty: ignore[invalid-argument-type]
    assert legacy.render() == "gather:0,1;govern:r1#2"


PROJECTED = (
    "$awaitEvent:gather:0,1;$awaitEvent:foo",
    "n:a/b=1",
    "n:a:b=1",
    "n:*=1",
)
"""Keys the parser reads whose projections or names a stricter check could refuse."""


@pytest.mark.parametrize("text", PROJECTED)
def test_a_parsed_key_composes_and_projects_as_before(text):
    key = Key.parse(text)
    assert key.stored() == text
    # lint: terminal-hole: `key` is a `Key`.
    assert compose_key(t"event;{key:domain=any}").terms() == (Term("event"), *key.terms())
    for domain in Domain:
        assert admits(domain, key) is admits(domain, text)
    assert split_frames(text) == split_frames(key.stored())
    assert past_frames(text) == past_frames(key.stored())


@pytest.mark.parametrize("base", [Atom, Coordinate, Term, ParsedKey, Key])
def test_a_key_structure_has_no_subclass(base):
    with pytest.raises(TypeError, match="subclass"):
        type("Child", (base,), {})


@pytest.mark.parametrize(
    ("cls", "state"),
    [
        (Atom, ["2", Kind.NAME]),
        (Coordinate, [(), None]),
        (Term, ["t", (), True, False]),
        (ParsedKey, [(Term("n"),), 1]),
    ],
)
def test_a_restored_structure_checks_its_form(cls, state):
    restored = cls.__new__(cls)
    with pytest.raises(KeySyntaxError):
        restored.__setstate__(state)


def test_a_coordinate_name_carries_no_occurrence_sigil():
    """Its key's unnumbered rendering would read the sigil as an occurrence."""
    with pytest.raises(KeySyntaxError, match="coordinate name 'a#'"):
        Key.parse("n:a#=1#2")


def test_a_projection_drops_a_foreign_terms_coordinate():
    graph = project(from_keys("r1", ["$awaitTaskResult:foo,1", "$awaitTaskResult:foo,2"]))
    assert [node.key for node in graph.nodes] == ["$awaitTaskResult:foo"]


def test_a_restored_key_carries_every_field():
    restored = Key.__new__(Key)
    with pytest.raises(ValueError, match="fields"):
        restored.__setstate__((parse("tool:a"),))
