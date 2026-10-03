"""The identity grammar as a language: parse, render, typed atoms.

    key        := term [';' term]*
    term       := tag [':' coordinate [',' coordinate]*]
    coordinate := atom ['/' atom]*

Each metacharacter carries
exactly one meaning — ``;`` sequence, ``,`` arity, ``/`` path — so a key reads as a sequence of
NARROWING coordinates: frames, then wrappers, then the arm and its identity, then any refinement.

**The property this module exists for: the parse needs no registry.** An atom cannot contain a
separator, so arity is countable from the bytes and a reader always knows where a field ends — no
consumer has to look a tag up to learn its shape, and none has to *try decodes* to find a frame
boundary. That is what lets ONE reader serve every consumer: a grammar whose fields are only
findable via a registry forces each consumer to carry its own partial reader, and partial readers
disagree.

**A leaf, deliberately.** It imports nothing from ``effective`` — only ``re``, ``dataclasses`` and
``enum`` — so it can be verified before anything consumes it, and so the module that owns the
identity TYPES (``effective.keys``) can build on the module that owns the LANGUAGE rather than the
other way round.

**A leading term is a FRAME, and a frame sequence narrows left to right**::

    d:0;state:draft;gather:0,1;govern:r1
    │   │           │          └ the op itself
    │   │           └ which gather branch within that
    │   └ which state within that level
    └ which level of the trampoline

So a framed key is not a second grammar to reconcile. It parses like any other, and the frames are
the terms before the last::

    parse(...).tags    ->  ("d", "state", "gather", "govern")
    split_frames(...)  ->  (("d:0", "state:draft", "gather:0,1"), "govern:r1")
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, assert_never

from pydantic_core import core_schema

TERM_SEPARATOR = ";"
ARITY_SEPARATOR = ","
PATH_SEPARATOR = "/"
TAG_SEPARATOR = ":"
COORDINATE_NAME_SEPARATOR = "="
FOREIGN_SIGIL = "$"

OCCURRENCE_SIGIL = "#"
r"""The suffix marking one identity asked more than once: `…#2`.

**Part of the LANGUAGE, and the same bytes both durable engines already store.** The Absurd SDK
mints it for a duplicate checkpoint name (`name if count == 1 else f"{name}#{count}"`,
`absurd_sdk:904`/`:1148`) and `SqliteTaskContext` mirrors the identical counter, so the one place
this could be spelled differently is the one place it must not be. Spelled as a term (`;occ:2`),
a fork-seed lookup stops byte-matching what Absurd stored and a `#\d+$` strip on the SQLite
bridge goes dead. A spelling the substrate translates at a seam is a seam that rots.

**A suffix, not a term, because it qualifies the WHOLE key**, which is how every reader treats
it: `registry.explain` peels it before dispatching, `Explanation.occurrence` is its
own field "because it is not a template field", and `graphview` folds on it. One slot on
`ParsedKey`, so `#N#M` cannot be represented and `#1` has no wire form.

The other property a term cannot give: `#` is in NO atom charset, so an author cannot write
one. As a term, `occ` would be an ordinary tag: `step_key("tool:a;occ:2")` would mint the same
key as `step_key("tool:a").occurrence(2)`, two producers for one identity, which is the
injectivity violation this grammar exists to prevent."""

PROJECTION_SIGIL = "*"
"""What a coordinate a PROJECTION dropped renders as, and the one byte no producer may write.

A projection is a quotient over a stored key: it answers "which coordinates did this run and
another agree on?" by replacing the ones a view drops. Substitution rather than deletion, so the
arity survives and a projection of a two-coordinate key stays a two-coordinate reading; and a byte
no atom may hold, so *a coordinate a VIEW may drop, the RECORD never may* is enforced by the
grammar rather than by a convention. `KeyMap.project` is the only producer, and it renders into a
`ProjectedKey`, which has no durable form.

Listed in `METACHARACTERS` for the rule stated there: a new byte that means something is a new
fence. That is what stops `Segment("*")` from branding a forged projection as an atom."""

METACHARACTERS = (
    TERM_SEPARATOR,
    ARITY_SEPARATOR,
    PATH_SEPARATOR,
    TAG_SEPARATOR,
    OCCURRENCE_SIGIL,
    PROJECTION_SIGIL,
)
"""Every byte that means something — the closed set an ATOM may not contain.

This exists to be **enumerated by other modules**. `Segment.__new__` fences each of these with
its own error message; a byte that means something but is not fenced lets a value carry structure
while wearing a one-atom brand.

The fences stay hand-written, because an error message that names the culprit *and* the fix is
worth more than a loop. What this constant buys is **coverage as a testable claim**:
`test_op_key_injectivity` iterates it, so adding a metacharacter without adding its fence reddens
immediately. A new byte in the language is a new fence here."""

TAG = re.compile(r"^[a-z][a-z0-9-]*$")
"""Lower-kebab, no underscores. The charset does not widen to admit snake_case tags."""

FOREIGN_TAG = re.compile(r"^\$[A-Za-z][A-Za-z0-9]*$")
"""A tag minted by another grammar: Absurd's ``$awaitEvent``. ``$`` is the declared marker."""

INTEGER = re.compile(r"^(0|[1-9][0-9]*)$")
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.@-]*$")

DIGEST_ALGORITHMS = ("sha256", "sha512", "blake3")
"""The extension point for the digest kind. A digest is ALGORITHM-PREFIXED rather than bare
hex, and the prefix has to come from a known set or the kind is not decidable: ``run-abc123`` is a
perfectly good name, and only knowing that ``run`` is not an algorithm keeps it one."""

DIGEST = re.compile(rf"^(?:{'|'.join(DIGEST_ALGORITHMS)})-[0-9a-f]+$")


class Kind(Enum):
    """An atom's type, inferable from its bytes, so a reader needs no registry to know it.

    Each kind has a canonical rendering that is injective and lexically self-identifying, which is
    what lets a lexer tag every atom.

    **There is no `key` kind.** Its lexical tell would be *contains `;`*, and `;` is the term
    separator, so no atom can contain one. A nested key is spliced as TERMS into the sequence, as
    in `fork:{child};<key>`. `key` is a type the COMPOSER dispatches on when a `Key` lands in a
    hole, a shape this lexer never meets."""

    INTEGER = "integer"
    UUID = "uuid"
    DIGEST = "digest"
    NAME = "name"


def kind_of(text: str) -> Kind | None:
    """The atom's kind from its bytes alone; ``None`` when the text is not a well-formed atom.

    Order matters and is not arbitrary — the kinds are not disjoint as regexes, only as intents.
    A UUID and a digest both match ``NAME``, so the narrow kinds are tried first and ``NAME`` is
    the fallthrough. Reordering this silently reclassifies every uuid in the system."""
    if INTEGER.match(text):
        return Kind.INTEGER
    if UUID.match(text):
        return Kind.UUID
    if DIGEST.match(text):
        return Kind.DIGEST
    if NAME.match(text):
        return Kind.NAME
    return None


def is_tag(text: str) -> bool:
    """Whether ``text`` is one of OUR tags: lower-kebab, and not a narrower kind first.

    ``TAG`` is not disjoint from the atom kinds it shares a charset with. Every character of a
    canonical uuid is in ``[a-z0-9-]``, so a task id opening ``a``-``f`` matches the lower-kebab
    pattern in full while one opening ``0``-``9`` does not; ``sha256-9f2c1a`` matches it always.
    That is exactly `kind_of`'s narrow-kinds-first problem — a uuid and a digest both match
    ``NAME``, so the narrow kinds are tried first — arriving at the tag boundary, and it takes the
    same answer rather than a second one: **a tag is text whose kind is `NAME`**. A uuid names one
    thing and a digest names one artifact; only a name names a namespace of things.

    Asking `kind_of` rather than excluding the kinds by hand is the difference between a rule and
    a list. The first version of this function said ``not UUID.match(text)`` and was one kind
    short the day it shipped — the shape CLAUDE.md warns about, at the scale of two.

    Used by all three gates that admit a tag: `Term.__post_init__` and `marker.Tag` on the
    composing side, `_parse_foreign` on the reading side."""
    return bool(TAG.match(text)) and kind_of(text) is Kind.NAME


class KeySyntaxError(ValueError):
    """The text is not in the language.

    RAISED, where ``registry.Shape.decode`` returns ``None``, and the asymmetry is the point: a
    decode is total because a tag owns several variants and "not mine" is a value the caller
    dispatches on. Here there is exactly one grammar, so "not in the language" is an error and
    there is nobody to hand it to."""


@dataclass(frozen=True, slots=True)
class Atom:
    """One typed component of a coordinate."""

    text: str
    kind: Kind

    @classmethod
    def of(cls, text: str) -> Atom:
        if (kind := kind_of(text)) is None:
            raise KeySyntaxError(
                f"{text!r} is not a well-formed atom: an atom is an integer (no leading zero), a "
                f"canonical lowercase uuid, an algorithm-prefixed digest "
                f"({DIGEST_ALGORITHMS[0]}-…), or a name matching {NAME.pattern}. It may not "
                f"contain a separator — that is what makes arity countable without a registry."
            )
        return cls(text, kind)

    def render(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class Coordinate:
    """One argument of a term: a path of atoms, optionally named.

    Most coordinates are a single atom; a MIME type is two (`message/rfc822` IS `type/subtype` per
    RFC 2045, so splicing it by induction is the correct reading rather than an accommodation).

    **`name` is what makes a coordinate optional**, and the rule has no marker of its own:
    *bare = required, named = optional*. The required prefix is therefore fixed-width with no
    omissions in it, which is what any discriminator or arity check depends on."""

    atoms: tuple[Atom, ...]
    name: str | None = None

    @property
    def path(self) -> str:
        return PATH_SEPARATOR.join(atom.text for atom in self.atoms)

    def render(self) -> str:
        if self.name is None:
            return self.path
        return f"{self.name}{COORDINATE_NAME_SEPARATOR}{self.path}"


@dataclass(frozen=True, slots=True)
class Term:
    """A tag and its coordinates — a term reads as a CALL, ``tag:arg,arg``.

    A term with no coordinates is a bare tag: a **qualifier** (``emit``, and the arm terms that
    wrap rather than identify)."""

    tag: str
    coordinates: tuple[Coordinate, ...] = ()
    foreign: bool = field(default=False)
    """A tag minted by another grammar, admitted by the foreign-tag adapter."""

    wraps_payload: bool = field(default=False)
    """A FOREIGN term whose payload follows after `:` rather than as a term after `;`.

    **It exists to keep the grammar INJECTIVE, which is the one thing this cannot give up.**
    `$awaitEvent:ask:foo` (what the SDK writes — `f"$awaitEvent:{event_name}"`) and
    `$awaitEvent;ask:foo` (what our own composer would write) carry the same tags in the same
    order, so without this flag they parse to one structure and `render` can only return one of
    them. Two distinct durable names collapsing to one is exactly the collision the arc exists to
    forbid — and for a checkpoint name, rendering the other spelling is a lookup that misses."""

    def __post_init__(self) -> None:
        """The tag is checked HERE, so nothing can build a term outside the language.

        `parse` checks before constructing, which is why this looks redundant; the composer does
        not. Without this check `compose_key(t"ns;{5}")` would return `Key('ns;5')` and
        `compose_key(t"ns;{Segment('Foo_Bar')}")` would return `Key('ns;Foo_Bar')`: a non-`Key`
        splice becomes a term tag, and `parse` refuses both. A composer that mints what its own
        parser rejects has stopped being one grammar.

        Validating at the DATACLASS rather than at each construction site is the point: the
        constructor is the one place every path passes, so the rule is bounded by the type rather
        than by which call sites someone remembered."""
        if not is_tag(self.tag) and not FOREIGN_TAG.match(self.tag):
            if (kind := kind_of(self.tag)) in (Kind.UUID, Kind.DIGEST):
                raise KeySyntaxError(
                    f"{self.tag!r} is a {kind.value}, so it is an ATOM and not a tag. It names "
                    f"one thing rather than a namespace of things, which is a tag's job — so it "
                    f"belongs in a COORDINATE: `tag:{self.tag}`. (Some of these do match "
                    f"{TAG.pattern} as well; that overlap is why the kind is asked and not just "
                    f"the charset.)"
                )
            raise KeySyntaxError(
                f"{self.tag!r} is not a well-formed tag: lower-kebab matching {TAG.pattern}, or a "
                f"foreign tag matching {FOREIGN_TAG.pattern}. A term's tag names a namespace, so "
                f"a value belongs in a COORDINATE — `tag:{self.tag}` rather than `{self.tag}`."
            )

    @property
    def arity(self) -> int:
        return len(self.coordinates)

    def render(self) -> str:
        if not self.coordinates:
            return self.tag
        args = ARITY_SEPARATOR.join(c.render() for c in self.coordinates)
        return f"{self.tag}{TAG_SEPARATOR}{args}"


@dataclass(frozen=True, slots=True)
class ParsedKey:
    """A key: a sequence of narrowing terms, and at most one occurrence.

    `occurrence` is the `#N` suffix — `None` for a first occurrence, which is unrepresented on the
    wire exactly as the Absurd SDK writes it (`name if count == 1 else f"{name}#{count}"`)."""

    terms: tuple[Term, ...]
    occurrence: int | None = None

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(term.tag for term in self.terms)

    @property
    def wrapped(self) -> ParsedKey | None:
        """The key a FOREIGN term wraps — what `graphview` and `parked` should ask.

        `None` when this key is not foreign-headed, or when the foreign payload is in the other
        runtime's own vocabulary rather than ours (`$awaitTaskResult:{uuid}`, whose payload rides
        as the foreign term's coordinate). So a reader asks one question and gets our key or
        nothing, instead of slicing a prefix off the text and hoping.

        **`registry._wrapped_payload` is the one consumer**, reached through
        `_peel`. Everywhere else the prefix is spelled
        as text: `graphview.KINDS`, `checkpoints.ENGINE_INTERNAL` and `checkpoints.NON_STEP` hold
        it in three tables, and `handlers/absurd.py` reproduces it a fourth time as a LOOKUP key,
        which is the consequential one: that site re-creates the SDK's own mint, so a drift there
        is a checkpoint miss rather than a mislabel. `bridge_absurd` reaches the tables through
        `is_engine_internal`, and `parked` works on the raw name. One referent spelled once per
        reader, which is the shape this package warns about elsewhere; retyping those readers onto
        terms is what would collapse them here.

        **The occurrence does NOT come along, and that is the useful part.**
        `$awaitEvent:review:m1#2` is the SECOND checkpoint of a park on the event `review:m1` —
        the event's name is `review:m1`, and `#2` counts the checkpoint. Handing both back
        together is what made `graphview` draw `$awaitEvent:q#2` and `$awaitEvent:q` as two
        distinct nodes, each reporting occurrence 1: the identity and its use-index have to
        separate for a repeat to fold onto the node it repeats."""
        match self.terms:
            case (Term(foreign=True, wraps_payload=True), *rest) if rest:
                return ParsedKey(tuple(rest))
            case _:
                return None

    def render(self) -> str:
        """The flat form. An f-string/`str.join` backend below a processor that has already made
        every structural decision — the one position where eager format-then-concat is exactly
        right, and the single exemption in "no f-string composes a `Key`"."""
        flat = _render_terms(self.terms)
        if self.occurrence is None:
            return flat
        return f"{flat}{OCCURRENCE_SIGIL}{self.occurrence}"


def _render_terms(terms: tuple[Term, ...]) -> str:
    """Join terms, giving a payload-wrapping FOREIGN term the `:` its own grammar chose.

    The SDK writes `f"$awaitEvent:{event_name}"`, so rendering that join our way would produce
    bytes the engine never wrote, and for a checkpoint name that is a lookup that misses. The
    recursion re-reads the head at every step, so the join is decided per term rather than only
    for the first::

        $awaitEvent:review:m1              a wrapping term at the head
        gather:0,0;$awaitEvent:review:m1   and one under a frame, which reaches here too
    """
    match terms:
        case (Term(foreign=True, wraps_payload=True) as head, *rest) if rest:
            return f"{head.tag}{TAG_SEPARATOR}{_render_terms(tuple(rest))}"
        case (head, *rest):
            tail = _render_terms(tuple(rest))
            return f"{head.render()}{TERM_SEPARATOR}{tail}" if tail else head.render()
    return ""  # the base case: a one-term key recurses to here


def _parse_foreign(tag: str, separator: str, rest: str, chunk: str) -> tuple[Term, ...]:
    """The foreign-tag ADAPTER: admit a tag minted by another grammar, and read its payload.

    A foreign tag carries a payload: the vendored SDK writes `f"$awaitEvent:{event_name}"` and
    `f"$awaitTaskResult:{task_id}"`, and the two payloads are not the same kind of thing. Refusing
    them leaves those rows unparseable, invisible to a total reader that returns the text
    unchanged.

    So the payload is read, and the reading is chosen by asking whether its head is a tag of OURS:

        $awaitEvent:review:m1     the payload's head is OUR tag  -> our TERMS follow
        $awaitTaskResult:{uuid}   the payload is a bare atom     -> the foreign term's COORDINATE

    The second is honest about whose vocabulary a task id is: it is Absurd's, it is a well-formed
    atom, and pretending it is a term of ours would invent a tag nobody minted.

    **The question is `is_tag`, not `TAG.match`.** The two payload charsets overlap: a canonical
    uuid is entirely `[a-z0-9-]`, so one opening `a`-`f` matches the lower-kebab tag pattern in
    full and would take the wrong branch, while one opening `0`-`9` would not. `TAG.match` decides
    the structure by a hex nibble.

    The corpus rarely shows that case: `$awaitTaskResult` is written only by the SDK's
    `await_task_result`, which this repo does not call, and every task id we mint is a `uuid7`,
    whose leading nibble is `0` until the ms clock passes 2**44. The field takes whatever the
    caller passes, so the format itself admits either nibble.
    """
    if not separator:
        return (Term(tag, (), foreign=True),)
    head, inner_separator, inner_rest = rest.partition(TAG_SEPARATOR)
    if not is_tag(head):
        # The other runtime's own payload (a task uuid). It still has to be a well-formed atom —
        # foreign does not mean unchecked, or the column stops being parseable for a new reason.
        return (Term(tag, _parse_coordinates(rest, chunk), foreign=True),)
    wrapped = Term(head, _parse_coordinates(inner_rest, chunk)) if inner_separator else Term(head)
    return (Term(tag, (), foreign=True, wraps_payload=True), wrapped)


def _parse_coordinates(rest: str, chunk: str) -> tuple[Coordinate, ...]:
    """The arguments of one term, left to right, enforcing the required-then-optional boundary."""
    out: list[Coordinate] = []
    seen: set[str] = set()
    for argument in rest.split(ARITY_SEPARATOR):
        if not argument:
            raise KeySyntaxError(
                f"empty coordinate in {chunk!r}: a coordinate at its default is OMITTED, never "
                f"written as a gap, so one identity keeps exactly one spelling."
            )
        name, marked, path = argument.partition(COORDINATE_NAME_SEPARATOR)
        if not marked:
            name, path = None, argument
        elif not name:
            raise KeySyntaxError(f"empty coordinate name in {argument!r}")
        if name is None and seen:
            raise KeySyntaxError(
                f"bare coordinate {argument!r} follows a named one in {chunk!r}: bare means "
                f"REQUIRED and named means OPTIONAL, so every bare coordinate belongs to the "
                f"fixed-width prefix. A bare one past the boundary would make the prefix's width "
                f"depend on the value."
            )
        if name is not None:
            if name in seen:
                raise KeySyntaxError(f"coordinate {name!r} given twice in {chunk!r}")
            seen.add(name)
        if not path:
            raise KeySyntaxError(f"coordinate {argument!r} has no value in {chunk!r}")
        out.append(Coordinate(tuple(Atom.of(a) for a in path.split(PATH_SEPARATOR)), name=name))
    return tuple(out)


def parse(text: str) -> ParsedKey:
    """Parse the flat form. Single pass, no registry, no ambiguity.

    Contrast the two readers this replaces: ``registry.explain`` looks the tag up to learn its
    arity, and ``registry._unframe`` tries decodes to find a frame boundary. Both questions are
    answered here by the separators alone.

    **The occurrence suffix comes off FIRST**, before any term is read — it is a property of the
    whole key, not of its last term, which is why it is one slot on `ParsedKey` rather than
    something a term could carry. `#N#M` is therefore a parse error by construction: the second
    `#` lands inside an atom, where no charset admits it."""
    if not text:
        raise KeySyntaxError("the empty key addresses nothing")
    text, occurrence = _peel_occurrence(text)
    terms: list[Term] = []
    for chunk in text.split(TERM_SEPARATOR):
        if not chunk:
            raise KeySyntaxError(f"empty term in {text!r}")
        tag, separator, rest = chunk.partition(TAG_SEPARATOR)
        if FOREIGN_TAG.match(tag):
            terms.extend(_parse_foreign(tag, separator, rest, chunk))
            continue
        if not TAG.match(tag):
            raise KeySyntaxError(
                f"{tag!r} is not a well-formed tag: lower-kebab, matching {TAG.pattern} — or a "
                f"foreign tag matching {FOREIGN_TAG.pattern}."
            )
        if not separator:
            terms.append(Term(tag))  # a bare tag: a qualifier
            continue
        if not rest:
            raise KeySyntaxError(
                f"term {chunk!r} ends in {TAG_SEPARATOR!r} with no coordinates. A term with no "
                f"arguments is written as the bare tag {tag!r}."
            )
        terms.append(Term(tag, _parse_coordinates(rest, chunk)))
    return ParsedKey(tuple(terms), occurrence)


def _peel_occurrence(text: str) -> tuple[str, int | None]:
    """`("tool:a", 2)` for `"tool:a#2"`; `(text, None)` when the key names a first occurrence.

    End-anchored and digits-only, and **one slot**: everything before the last `#` is the key, so
    a second `#` cannot be peeled and lands inside an atom, where no charset admits it. That is
    what makes `#N#M` a parse error rather than a rule somebody has to remember.

    A first occurrence is UNREPRESENTED, matching the Absurd SDK byte for byte
    (`name if count == 1 else f"{name}#{count}"`, `absurd_sdk:904`/`:1148`) — so `#1` is refused
    rather than accepted-and-normalized, or one identity would have two spellings."""
    base, sigil, nth = text.rpartition(OCCURRENCE_SIGIL)
    if not sigil:
        return text, None
    if not base:
        raise KeySyntaxError(f"{text!r} is an occurrence suffix with no key in front of it")
    if not INTEGER.match(nth):
        raise KeySyntaxError(
            f"{text!r} ends in {OCCURRENCE_SIGIL!r} followed by {nth!r}, which is not a count. "
            f"The occurrence suffix is {OCCURRENCE_SIGIL}N with N an integer — and it is the only "
            f"thing {OCCURRENCE_SIGIL!r} means, in the only position it may appear."
        )
    if (count := int(nth)) < 2:
        raise KeySyntaxError(
            f"{text!r} spells occurrence {count}, which has no wire form. A first occurrence is "
            f"the bare key, so {OCCURRENCE_SIGIL}1 would be a second spelling of one identity; "
            f"counts below 1 are a counting bug, not a name."
        )
    return base, count


# --- skeletons: the same grammar, over a template rather than over text --------------------
#
# A `compose_key` template is statics interleaved with holes, and TWO consumers have to read that
# structure the same way: the composer, which fills holes with VALUES and renders; and the lint,
# which fills them with the author's source EXPRESSIONS and registers a shape. The key registry
# is a source map only if it describes what the composer actually mints, so both read the
# template through one parser. A second reader that split on `:` would miss the other separators
# and report `budget-grant:{run_id},{trip}` as a hole glued to literal text.
#
# So the skeleton parse lives here, beside the text parse, and both consumers call it.


class Domain(Enum):
    """What a SPLICE admits — the narrowing the grammar cannot infer for itself.

    A splice's value is a finished `Key`, i.e. ANY key, so `event;{name}` accepts a step's identity
    as far as the parser is concerned. What rules it out is not syntax but POSITION: an await's
    payload is the address an emitter delivers to, and no producer emits an event named after a
    checkpoint. `event;step;tool:foo` is therefore well-formed and unmintable.

    **The declaration lives at the composition site**, as a `format_spec` directive
    (`t"event;{name:domain=address}"`), and not in a table the composer consults. A table would be
    bounded by what it enumerates, and only 6 of the substrate's 16 splices are even preceded by an
    arm tag; the other ten sit under `emit`, `approve`, `govern`, `fork`, `hyp`, `code`, and a
    `gather:` frame. (The count is the `domain=` declarations inside a t-string in `src/`: a splice
    must declare, so the declarations ARE the sites.) A directive is bounded by what the composer
    sees, which is every template anywhere. It is also STATIC, so the registry and the lint read
    the same declaration the composer enforces, from the template rather than from a second reader.

    **Two values and an escape, deliberately small.** `ADDRESS` and `IDENTITY` are duals over the
    closed `ARM_TAGS` set; `ANY` is the named exit, so an author who means "any key" says so where
    the next reader can see it rather than by omission."""

    ADDRESS = "address"
    """The payload is a WIRE ADDRESS — an event name, an `event_id`, a step's author name. It must
    not open with an arm, because a key that does is an op's own identity."""

    IDENTITY = "identity"
    """The payload is an op's own identity — what `op_key` mints. It MUST open with an arm.
    `gather:{g},{i};{qualified}` wraps one, and `gather:0,0;foo` names no op."""

    ANY = "any"
    """Deliberately unconstrained. A visible opt-out, not a default."""


@dataclass(frozen=True, slots=True)
class Hole:
    """A template interpolation, in the position the grammar gives it.

    `index` is its ordinal among the template's interpolations, so a consumer can pair it with
    whatever it holds — a runtime value for the composer, a source expression for the lint.

    `domain` is what a SPLICE in this position admits, read off the interpolation's `format_spec`.
    **It rides the `Hole` rather than the `Splice`** because `parse_skeleton` is handed a
    `Sequence[str | Hole]` and has no other channel — a `Splice` is minted from a lone `Hole`, so
    anything a splice needs to know has to arrive inside one. Both producers fill it: the composer
    from `Interpolation.format_spec`, the lint from the AST's `format_specifier` node, which is
    what lets one declaration serve four readers. It is `None` on every non-splice hole, where it
    means nothing.

    **`domain` does not COMPARE**, the same exemption `Key._scope` takes. A domain constrains the
    PRODUCER, not the language: `event;{name:domain=address}` and `event;{name:domain=any}` mint
    exactly the same wire shape, and a decoder cannot tell them apart because there is nothing to
    tell. Comparing it made `Skeleton` equality disagree — `_same_variant` stopped merging the
    three `event;` sites into one variant, and `separated` then reported the namespace as owning
    two overlapping shapes. The registry was right to complain and the field was wrong to be in
    the comparison."""

    index: int
    domain: Domain | None = field(default=None, compare=False)


type SkeletonAtom = Atom | Hole
type Part = str | Hole


@dataclass(frozen=True, slots=True)
class SkeletonCoordinate:
    atoms: tuple[SkeletonAtom, ...]
    name: str | None = None


@dataclass(frozen=True, slots=True)
class SkeletonTerm:
    """One term of a template. `tag` is a `Hole` when the template opens with a `Tag`
    interpolation (`t"{GOVERN}:{gate}"`), which is how a namespace constant names itself."""

    tag: str | Hole
    coordinates: tuple[SkeletonCoordinate, ...] = ()


@dataclass(frozen=True, slots=True)
class Splice:
    """A hole occupying a WHOLE term — `t"approve;{op_key(op)}"`.

    The value is a finished `Key`, and it EXTENDS the sequence rather than filling a coordinate:
    that is what `;` means, and what makes composition inductive. Distinguished from a
    `SkeletonTerm` whose tag is a `Hole` by the presence of a `:` — a hole followed by coordinates
    is naming a namespace, a hole standing alone is contributing terms.

    **The one ambiguity, stated rather than hidden:** a bare `t"{SOME_TAG}"` with no coordinates
    parses as a `Splice` too, because nothing in the statics distinguishes it. No such template
    exists today. The composer resolves it by the value's TYPE (a `Tag` becomes a zero-arity
    term); a static reader cannot, and should say so rather than guess."""

    hole: Hole


type SkeletonElement = SkeletonTerm | Splice


@dataclass(frozen=True, slots=True)
class Skeleton:
    elements: tuple[SkeletonElement, ...]

    @property
    def holes(self) -> int:
        """How many interpolations this skeleton accounts for — the composer's arity check."""
        seen: set[int] = set()
        for element in self.elements:
            match element:
                case Splice(hole):
                    seen.add(hole.index)
                case SkeletonTerm(tag, coordinates):
                    if isinstance(tag, Hole):
                        seen.add(tag.index)
                    for coordinate in coordinates:
                        seen.update(a.index for a in coordinate.atoms if isinstance(a, Hole))
                case unreachable:
                    assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
        return len(seen)


def _split(parts: list[Part], separator: str) -> list[list[Part]]:
    """Split a statics-and-holes sequence on a separator, keeping holes intact.

    The separator only ever appears in a STATIC — a hole is a value, and a value cannot contain a
    separator (that is the property the whole grammar rests on). So splitting the statics and
    carrying the holes across is exact, not an approximation."""
    out: list[list[Part]] = [[]]
    for part in parts:
        match part:
            case Hole():
                out[-1].append(part)
            case str():
                pieces = part.split(separator)
                out[-1].append(pieces[0])
                for piece in pieces[1:]:
                    out.append([piece])
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
    return [[p for p in group if p != ""] for group in out]


def parse_skeleton(parts: Sequence[Part]) -> Skeleton:
    """The grammar over a template: statics interleaved with `Hole`s.

    Mirrors `parse` element for element, so a skeleton and the key it mints cannot disagree about
    where a field ends."""
    parts = list(parts)
    if not parts:
        raise KeySyntaxError("an empty template composes no key")
    elements: list[SkeletonElement] = []
    for chunk in _split(parts, TERM_SEPARATOR):
        if not chunk:
            raise KeySyntaxError("empty term in template")
        # lint: totality(selection) — a compound test on the chunk's LENGTH, not a dispatch on
        # its element: `len(chunk) == 1 and isinstance(...)` asks about the list.
        if len(chunk) == 1 and isinstance(chunk[0], Hole):
            elements.append(Splice(chunk[0]))
            continue
        head, *rest = _split(chunk, TAG_SEPARATOR)
        if len(head) != 1:
            raise KeySyntaxError(f"a term's tag must be one static or one Tag hole, got {head!r}")
        tag = head[0]
        if isinstance(tag, str) and not TAG.match(tag) and not FOREIGN_TAG.match(tag):
            raise KeySyntaxError(f"{tag!r} is not a well-formed tag in a template")
        if not rest:
            elements.append(SkeletonTerm(tag))
            continue
        if len(rest) > 1:
            raise KeySyntaxError(
                f"a term carries at most one {TAG_SEPARATOR!r}, and this one has "
                f"{len(rest)} — which is the OLD flat grammar, where every field was separated "
                f"by a colon. Coordinates of one term are separated by {ARITY_SEPARATOR!r}, and "
                f"a nested key is its own term after {TERM_SEPARATOR!r}."
            )
        flattened = [p for group in rest for p in group]
        coordinates = tuple(
            _skeleton_coordinate(argument) for argument in _split(flattened, ARITY_SEPARATOR)
        )
        elements.append(SkeletonTerm(tag, coordinates))
    return Skeleton(tuple(elements))


def _skeleton_coordinate(argument: list[Part]) -> SkeletonCoordinate:
    if not argument:
        raise KeySyntaxError("empty coordinate in template")
    name, argument = _coordinate_name(argument)
    atoms: list[SkeletonAtom] = []
    for atom in _split(argument, PATH_SEPARATOR):
        if len(atom) != 1:
            raise KeySyntaxError(
                f"{atom!r} is not one atom: a hole glued to literal text has no position in this "
                f"grammar, because the value and the suffix cannot be told apart on the wire. Put "
                f"a separator between them, or compose the value first."
            )
        match atom[0]:
            case Hole() as hole:
                atoms.append(hole)
            case str() as text:
                atoms.append(Atom.of(text))
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
    return SkeletonCoordinate(tuple(atoms), name)


def _coordinate_name(argument: list[Part]) -> tuple[str | None, list[Part]]:
    """A leading `name=` on a coordinate, which is always the template author's own static."""
    first = argument[0]
    if not isinstance(first, str) or COORDINATE_NAME_SEPARATOR not in first:
        return None, argument
    name, _, remainder = first.partition(COORDINATE_NAME_SEPARATOR)
    if not name:
        raise KeySyntaxError("empty coordinate name in template")
    return name, ([remainder] if remainder else []) + argument[1:]


def split_occurrence(text: str) -> tuple[str, int | None]:
    """`("ns:a,b", 2)` for `"ns:a,b#2"`; `(text, None)` when it names a first occurrence.

    **One reader for every caller.** `registry` asks this to decode a key and `graphview` asks it
    to fold a repeat back onto its node. A per-caller `#(\\d+)$` regex is a second grammar for the
    one part of a key that is not a term: when the spelling changes, a caller left unmigrated
    raises nothing, reads a second occurrence as the first, and stops folding a cycle. This reads
    the suffix off the PARSE, so there is one reader whatever the spelling becomes.

    Total, like every reader here: text that is not in the language carries no occurrence."""
    try:
        parsed = parse(text)
    except KeySyntaxError:
        return text, None
    if parsed.occurrence is None:
        return text, None
    return ParsedKey(parsed.terms).render(), parsed.occurrence


def named(key: ParsedKey, fields: dict[str, tuple[str, ...]]) -> ParsedKey:
    """Re-render a positional key with its field names attached — the `display()` projection.

    `fields` maps tag -> field names, which is what the registry holds. Kept separate from `parse`
    so that the self-describing rendering is visibly a PROJECTION rather than a second grammar: the
    names are documentation attached to a decode, and the key on the wire never carries them for a
    required coordinate.

    **The rebuild preserves every field of the input.** Dropping `wraps_payload` would re-render
    `$awaitEvent:review:m1` as `$awaitEvent;review:m1`, bytes no producer wrote and a different
    valid key; dropping `occurrence` would collapse `step;tool:read#3` onto `step;tool:read`. With
    `fields={}` it is the identity on every distinct name in `absurd.c_default`: 7,126 of 7,126
    on 2026-08-29, a count that grows with every suite run, so recompute it rather than trust it.

    **What it is still NOT is injective, and the first caller inherits that.** Naming a REQUIRED
    coordinate spells it the way an OPTIONAL one is spelled, and the grammar keeps those distinct,
    so `named(parse("approve:2,alice"))` and `named(parse("approve:generation=2,who=alice"))`
    render alike — and `approve` has both spellings live. That is inherent to attaching names, not
    a defect in the rebuild: a display that must round-trip should render from the input, and one
    that wants names should not also be a key.

    **TOTAL, per term: a term is named only when the whole term can be named safely, and keeps
    its authored spelling otherwise.** Two ways naming goes wrong, and both emit bytes `parse`
    then rejects. A field list SHORTER than the arity names a PREFIX, leaving a bare coordinate
    after a named one; the registry supplies exactly this, because `fields` holds one tuple per
    tag while 9 tags have variants that disagree on arity — a caller building the map from the
    registry meets it on 157 live keys. And a field NAME is a hole's source expression, not an
    identifier, so it can carry a separator: the map already holds `epoch_atom(until)` and
    `str(index)`, and one `compose_key(t"…,{max(a, b)}")` would put a comma in one.

    Skipping rather than raising because this is a DISPLAY. Showing an unnamed key is a smaller
    loss than crashing a dashboard, and it is the behavior an absent registry entry already
    has — so the degradation is one a caller can already read."""
    out: list[Term] = []
    for term in key.terms:
        names = fields.get(term.tag, ())
        if len(names) < len(term.coordinates) or not all(map(_is_field_name, names)):
            names = ()
        out.append(
            Term(
                term.tag,
                tuple(
                    Coordinate(c.atoms, name=c.name if c.name is not None else _at(names, i))
                    for i, c in enumerate(term.coordinates)
                ),
                foreign=term.foreign,
                wraps_payload=term.wraps_payload,
            )
        )
    return ParsedKey(tuple(out), key.occurrence)


def _at(names: tuple[str, ...], index: int) -> str | None:
    return names[index] if index < len(names) else None


def _is_field_name(text: str) -> bool:
    """Whether `text` can be rendered as a coordinate's name and read back.

    A name is written `name=value`, so it has to be non-empty and carry neither a metacharacter
    nor the `=` that separates it from its value. Nothing checks this at `Coordinate`, because
    `parse` only ever supplies names it just read out of a key — the exposure is `named`, whose
    names come from a registry of source expressions."""
    return bool(text) and not any(m in text for m in (*METACHARACTERS, COORDINATE_NAME_SEPARATOR))


class Scope(Enum):
    """How far one answer in an authority namespace reaches: the axis `AuthorityTag` declares.

    The walk reads it to decide whether asking twice is asking twice
    (`handlers.base.placing`)::

        depth-grant:r1,0    settlement   asked twice ->  depth-grant:r1,0  depth-grant:r1,0#2
        budget-grant:r1     accrual      asked twice ->  budget-grant:r1   budget-grant:r1
        hyp:c1;review:m1    qualified    asked twice ->  hyp:c1;review:m1  hyp:c1;review:m1

    Why it is DECLARED rather than derived: `depth-grant:{run_id},{generation},depth={depth}` is
    a perfectly injective function of its fields and looks identical in kind to
    `budget-grant:{run_id},{trip}`, whose run-scoping is the intended feature. The difference is
    semantic and is not in the tree, so a template cannot answer it and the namespace says
    A lint then checks what was said, which a reader's choice of accessor could never
    be — that is the whole gain over the two `govern` accessors this replaced."""

    SETTLEMENT = "settlement"
    """One answer settles exactly ONE op-occurrence, so the name carries a coordinate the
    SUBSTRATE assigns. Either form discharges it and both are live: an interior template field
    (`govern`'s `occurrence` coordinate), or `Key.occurrence`'s suffix (`approve:`, whose counter
    is a per-run cell, so the first ask stays byte-identical). A field changes bytes and a suffix
    does not; neither is preferred."""

    ACCRUAL = "accrual"
    """One answer is in force for a declared scope (a run, a generation, a chain) ON PURPOSE.
    `budget-grant:` is the exemplar: a grant raises the run's ceiling, so one delivered while
    gating op1 is still in force at op2 and must NOT be re-requested. Also the honest home for a
    namespace addressing a per-run state cell rather than a question (`gate-state:`)."""

    QUALIFIED = "qualified"
    """This namespace wraps ANOTHER composed key and relocates it: a fork child's event world
    (`fork:{child};{name}`), a lineage-scoped ledger id (`hyp:{child};{event_id}`). It adds a
    lineage coordinate and identifies no occurrence of its own, so the occurrence question does
    not vanish — it recurses to the wrapped key, which keeps its own::

        govern:r1#2                    a settlement, on its second ask
        hyp:c1;govern:r1#2             wrapped, and the occurrence rides along

    So the wrapped hole must hold a `Key`, and the composer enforces it::

        wrapped = t"{FORK_SCOPE}:{Segment('c1')};{X:domain=address}"

        X = Key.parse("review:m1")   ->  hyp:c1;review:m1
        X = Segment("review-m1")     ->  TypeError: stands as a term's namespace and is a Segment
        X = "review:m1"              ->  ValueError: is str, which may contain a separator

    A `Key` there came out of `compose_key`, so its own namespace was checked when it was made. A
    `str` or a `Segment` is whatever the caller typed, and the composed key would then name a
    child's event under a namespace nobody registered."""


@dataclass(frozen=True, slots=True)
class Key:
    """A composed op-key: the finished identity of an op, and **opaque by construction**.

    There is no `__str__`, and that is the mechanism rather than an omission::

        key = compose_key(t"{GOVERN}:{Segment('r1')}")
        f"{key}"      -> "Key(_parsed=ParsedKey(...))"   # the repr, not the name
        key.stored()  -> "govern:r1"                 # the durable form, asked for
        key.display() -> "govern:r1"                 # the human form, may be elided

    An accidental flatten therefore cannot survive contact with its consumer: as data it
    mismatches, as display it renders visibly wrong. The one quiet position is a write to the
    append-only ledger, where a wrong name is simply a row nobody will ever match — which is why
    `LedgerRow.event_id` is typed `Key` rather than `str`, so a flatten there cannot be written.

    **Exactly two positions may treat a `Key` as text, and neither is a call site you write.**
    `compose_key`'s splice arm, which owns the grammar; and serde — the `sqlite3` adapter and the
    psycopg dumper, registered once against the driver, so the consumer that most legitimately
    wants text never asks for it.

    Everything else goes through a **named exit**, because a bare `.value` carries no intent and
    becomes the hatch everyone reaches for. `stored()` is the durable/wire form and must never
    drift; `display()` is the human/projection form and may be elided or truncated. Both are
    greppable, and by the rule above they should stay rare: a `stored()` outside those two
    positions is a finding that wants a reason, not a routine conversion.

    Not a `str` subclass, which is permanently flattened — the same protection PEP 750 designs
    into `Template`, declined. CLAUDE.md states that as the general rule.
    """

    _parsed: ParsedKey
    _scope: Scope | None = field(default=None, compare=False)
    """The reach the leading `AuthorityTag` declared, retained from composition — see `scope`."""

    def __post_init__(self) -> None:
        """The empty key is refused HERE rather than in `parse`, because every route into the type
        passes through the constructor and only one of them is named `parse`.

        A key with no terms addresses *nothing*, and every such key is the same key, so it
        collides with itself wherever it is stored — the degenerate case of the injectivity this
        whole grammar exists to hold.

        The type check rides along because the two failures are the same failure: a `Key` built
        from anything but a `ParsedKey` would carry a value that cannot render, and `stored()` is
        what replay binds to.
        """
        if not isinstance(self._parsed, ParsedKey):
            raise TypeError(
                f"Key: value is {type(self._parsed).__name__}, not a ParsedKey. A key is composed "
                f"by `compose_key` or read back by `Key.parse`; it holds the TERMS it was built "
                f"from, so text has to come in through `parse`."
            )
        if not self._parsed.terms:
            raise ValueError(
                "Key: the empty key addresses nothing, and every empty key is the same key — so "
                "it collides with itself wherever it is stored. Compose a real name, or carry the "
                "absence as `None` in a `Key | None` field."
            )

    @property
    def scope(self) -> Scope | None:
        """How far one answer to THIS key reaches, or `None` where nothing declared a reach.

        **What it is for**, and there is one consumer: `handlers.base.placing` numbers an await's
        occurrences only where the namespace declared `SETTLEMENT`. One approval settles one op,
        so asking twice is two questions and the second gets its own name; a grant accrues, so
        both asks wait on the same name and one answer serves both::

            GOVERN       = AuthorityTag("govern",       scope=Scope.SETTLEMENT)
            BUDGET_GRANT = AuthorityTag("budget-grant", scope=Scope.ACCRUAL)

            # the same key, awaited twice under one walk:
            SETTLEMENT   govern:r1        govern:r1#2      <- two questions
            ACCRUAL      budget-grant:r1  budget-grant:r1  <- one, asked again
            None         review:r1        review:r1

        `None` takes the un-numbered path too, which is exactly why it is not `ACCRUAL`: same
        behaviour, different fact. `ACCRUAL` was *declared*; `None` means this key was never
        composed from an authority namespace — an `event:`/`ledger:` key, an author's `Step`
        name, anything back through `Key.parse`. So test for the `Scope` you want:
        `is not Scope.ACCRUAL` reads as "settles" and is wrong for every key that declared
        nothing.

        Set by `compose_key` from the typed `AuthorityTag` in the leading hole, never re-derived
        from the leading text."""
        return self._scope

    @classmethod
    def parse(cls, text: str) -> Key:
        """Text back into a `Key`. The READ boundary, and it validates.

        Text that is not in the language raises here, at the door, rather than travelling as an
        opaque name until something asks for its terms::

            Key.parse("govern:r1")           -> a Key
            Key.parse("not a key at all")    -> KeySyntaxError
            Key.parse("")                    -> KeySyntaxError

        So catch `KeySyntaxError` around the parse, not around a later `.terms()`.

        Who calls it, and with what::

            checkpoints, bridge_absurd, bridge_sqlite   a checkpoint name read off the engine
            parked                                      a wake-event name read off the engine
            fork, agent.runtime                         a task param: params["fork_point"]
            govern, permit_tuning                       an op key read back off a stored row
            __get_pydantic_core_schema__                every `Key` in a model field, from JSON

        The last one writes no call: it is the validator this class installs, so a `Key` arriving
        from a checkpoint round-trip or a ledger row is parsed here whether or not anyone meant
        to.

            composed = compose_key(t"{GOVERN}:{Segment('r1')}")
            composed.scope                   -> Scope.SETTLEMENT
            Key.parse(composed.stored()).scope -> None              # same bytes, no guarantee

        The last pair is what a caller has to know: a parsed key carries the bytes and not the
        guarantee. A composed key retained what its typed `AuthorityTag` declared; text cannot.
        Use a parsed key as a name to store, compare and splice, not as evidence about what minted
        it. `compose_key` is where the guarantee comes from.

        Checking more here needs the grammar rather than a registry: a lookup could only know the
        REGISTERED namespaces, while `code:`, `skill:`, `react:`, `emit:` and task ids are live and
        unregistered."""
        return cls(parse(text))

    def terms(self) -> tuple[Term, ...]:
        """This key's terms, for splicing it into another key by INDUCTION.

        A field read: the structure is the truth and `stored()` is derived from it."""
        return self._parsed.terms

    def stored(self) -> str:
        """The durable/wire form — a checkpoint name, an event name, a DB parameter, JSON.

        Replay binds to this, so a change to how it renders orphans every recorded run. Distinct
        from `display` for that reason: one may be prettified and this one may not."""
        return self._parsed.render()

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        """Make `Key` a first-class value to Pydantic: the THIRD serde position, declared once on
        the type like the two driver registrations, never at a call site::

            Row(event_id=k).model_dump(mode="json")   ->  {"event_id": "govern:r1"}
            Row.model_validate_json('{"event_id": "govern:r1"}')  ->  a Key
            Row(event_id="govern:r1")                 ->  ValidationError

        Without this a `Key` would serialize structurally as `{"_parsed": {...}}`: a silent
        durable-format change, and the leak the opaque type exists to prevent.

        **The asymmetry is the rule in one place.** Python mode refuses a `str` where a `Key`
        belongs, which is the writer guarantee enforced at runtime as well as by `ty`. JSON mode
        parses the text, because a row read back off the wire has no `Key` to offer and this is
        the read boundary (see `parse`). Type the side where the guarantee is enforced; parse the
        side where it is not.

        **Scope, because it is easy to over-read:** this governs a `Key` in a MODEL FIELD.
        `to_jsonable_python` on an ad-hoc dict does not consult it, so a bare `Key` in an untyped
        mapping still serializes structurally. That is fine today (no step result or checkpoint
        value is key-valued), and it is why a ledger row is a typed model rather than a `dict`:
        the type is what makes the serialization correct.
        """
        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(
                cls.parse, core_schema.str_schema()
            ),
            python_schema=core_schema.is_instance_schema(cls),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda key: key.stored()
            ),
        )

    def prefixed(self, prefix: str) -> Key:
        """Apply a handler-side scope prefix, keeping the reach::

            k.prefixed("gather:0,1;")  ->  gather:0,1;govern:r1   (still a settlement)

        The ONE named place a key is re-composed by a scope, and the seam a ratified `scoped(...)`
        replaces when it is built. This is what `gather:{g},{i};` does to a branch's
        op keys and what `_PrefixedCtx` does on the durable path.

        **It deliberately does NOT go through `compose_key`.** A scope prefix is glued *ahead* of a
        tag, and `compose_key` requires the leading position to BE the tag. That mismatch is what
        blocks the scope-composing sites, and the reason scope is ratified to leave the key
        entirely rather than be absorbed by the composer. Applying a frame path to a finished
        identity is still composition, so it goes through a name rather than an f-string, which
        would compose the *repr*.
        """
        # The reach travels with the identity: a framed `depth-grant:` is still a settlement.
        # The prefix is still TEXT here — a `;`-terminated fragment the frame layer builds — so it
        # is parsed back with the key it fronts. Taking terms directly is the next step.
        return Key(parse(prefix + self._parsed.render()), self._scope)

    def occurrence(self, count: int) -> Key:
        """The suffix an engine appends when one name runs more than once::

            k.occurrence(1)  ->  govern:r1      # byte-identical to the bare key
            k.occurrence(2)  ->  govern:r1#2
            k.occurrence(0)  ->  is not a count
            …#2.occurrence(3) ->  already carries occurrence 2

        **`#N` is the grammar's one suffix**: `key := term (';' term)* ('#' INTEGER)?`. `#` is in
        no atom kind, so an author cannot write an occurrence into a name anywhere: the
        coordinate has exactly one producer, this method, checked where every composition already
        passes (`Atom.of`). It byte-matches what the vendored Absurd SDK's duplicate-checkpoint
        rule appends, because the two are the same rule: one counting, one spelling, no
        translation at the engine seam.

        **A suffix rather than a term**, because an occurrence is not a narrowing coordinate of the
        identity: it is a use-index assigned to a *finished* identity by the runtime, after
        composition. Its producer is disjoint from the authors, so its syntax is too. A terminal
        splice hoists it, since the n-th ask of a name is the n-th instance of the event identity;
        an interior splice refuses, because an interior occurrence has no reader.

        Omitted at its default of 1, so the short form is the canonical one and the long one is
        unwritable. The base key must be IN the language, which is the feature: minting `#N` onto
        anything at all is how the spelling that sat BELOW the grammar stayed silent.
        """
        parsed = self._parsed
        if parsed.occurrence is not None:
            raise ValueError(
                f"{self.stored()!r} already carries occurrence {parsed.occurrence}, and a key has "
                f"exactly one. Two counters reaching the same name is a counting bug — the three "
                f"producers each apply once per name per attempt — so this refuses rather than "
                f"minting a second suffix the grammar could not even represent."
            )
        if count < 1:
            raise ValueError(
                f"occurrence({count}) is not a count. Every producer here is 1-BASED (the SDK's "
                f"`count = get(name, 0) + 1`, and both mirrors of it), so a value below 1 means a "
                f"counter was never incremented — which silently returned the FIRST occurrence's "
                f"key and served its committed value to a later op."
            )
        if count == 1:
            return self  # a first occurrence is the bare key — unrepresented, as the SDK writes it
        return Key(replace(parsed, occurrence=count), self._scope)

    def display(self) -> str:
        """The human/projection form — a Mermaid label, an SVG ``data-node-id``, a log line.

        Separate from `stored` so a projection may elide or truncate without any risk of that
        choice reaching the durable record. Identical to `stored` today; it is *allowed* to
        diverge, and the type says which one a call site meant."""
        return self._parsed.render()


@dataclass(frozen=True, slots=True)
class ProjectedKey:
    """A quotient taken over a key: `Key`'s sibling, and never a `Key`.

    A projection answers "which coordinates did two runs agree on?" by replacing the ones a view
    drops with `PROJECTION_SIGIL`. `KeyMap.project` is the only producer. What comes back is not a
    name anything can be stored under, and the type is what says so::

        key.stored()        # the durable form
        projected.stored()  # AttributeError: there is no durable form

    **The missing method IS the rule.** *A coordinate a VIEW may drop, the RECORD never may*, and
    three mechanisms hold it rather than a convention: `LedgerRow.event_id` is typed `Key`, so `ty`
    refuses this at the write; the sqlite3 adapter and the psycopg dumper are registered against
    `Key` by exact type, so a driver raises rather than coercing; and `parse` refuses the rendered
    text, which covers the `str`-typed positions types do not reach. A SUBCLASS of `Key` would
    defeat the middle one by dispatching to a `stored()` that does not exist, so this is a sibling
    deliberately.

    `source` is the TEXT it was taken from rather than a `ParsedKey`, because a projection has to
    work on keys the grammar cannot parse: most banked checkpoint names do not, and a
    quotient that shut half a corpus out would be no use to the cross-run comparison it exists
    for. Holding the source is also what makes composition a set union rather than a re-reading:
    projecting a projection re-projects the SOURCE under the union of the two drop sets, so

        project(project(k, A), B) == project(k, A | B)

    holds by construction. Re-reading the rendered text could not do that, since `*` is outside
    the language by design."""

    source: str

    drop: frozenset[str]
    """The roles this projection was taken UNDER, which is a request rather than an effect.

    Named for the parameter that supplied it, and deliberately not `dropped`: an empty diff
    between `source` and `display()` has two causes, and only `claimed` tells them apart.
    `RunGraph.dropped` is the other word, holding AXIS labels for a reader, `"occurrence"` among
    them, which is no role at all."""

    claimed: bool
    """Did a variant decode the INNERMOST term of `source`, the one carrying the op's identity?

    Innermost, because every op carries an arm and every arm is registered as a trailing splice
    binding any payload, so "some variant decoded a term" is true of every recorded key whatever
    sits behind the arm. Frames are a prefix, so a key under a stranger's scope is claimed when
    its own op is registered, and the frame it could not read is projected or kept on its own::

        nosuchtag:whatever                  claimed=False   no variant reads the identity
        gather:0,1;ev:x    drop=(Index,)    claimed=False   the frame folded; `ev:` is a stranger
        task:r1            drop=(Run,)      claimed=True    read, and a `Subject` is not a `Run`
        govern:…,r9        drop=(Run,)      claimed=True    read, and the run id is gone

    So `claimed=False` does not mean nothing moved: five of this repo's corpus keys project and
    report False, because the map read their frames and not their identity. What it means is that
    a cross-run comparison calling two runs equivalent owes the reader the claimed fraction, since
    False on both sides says the registry could not rule on the op rather than that the runs
    agreed."""

    _text: str

    def display(self) -> str:
        """The human/projection form, and the only form. A node label, a group-by key, a diff.

        Named to match `Key.display` so a caller reads the same word for the same intent; there is
        deliberately no `stored` beside it."""
        return self._text

    def __str__(self) -> str:
        """Refused for the reason `Key` refuses it: a flatten must be an explicit act, and the one
        quiet place a wrong name survives is a write to a durable store."""
        raise TypeError(
            f"ProjectedKey has no string form: it is a quotient, not a name. "
            f"Call `.display()` for the projected text, or `.source` for the key it was taken "
            f"from. Got {self._text!r}."
        )
