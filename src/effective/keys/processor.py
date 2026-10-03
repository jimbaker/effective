"""The t-string processor: `compose_key`, and the refusals that make a key injective.

PEP 750's own word for a function that consumes a `Template`. This one reads the statics as the
key's syntax and the holes as its values, and every `_refuse_*` here is a case the grammar cannot
represent, caught where the template still has structure to inspect.
"""

from dataclasses import dataclass
from itertools import pairwise
from string.templatelib import Interpolation, Template
from typing import Any, NoReturn, assert_never

from effective.keys.frame import admits, scope_prefix
from effective.keys.grammar import (
    COORDINATE_NAME_SEPARATOR,
    OCCURRENCE_SIGIL,
    PATH_SEPARATOR,
    TERM_SEPARATOR,
    Atom,
    Coordinate,
    Domain,
    Hole,
    Key,
    ParsedKey,
    Skeleton,
    SkeletonAtom,
    SkeletonCoordinate,
    SkeletonTerm,
    Splice,
    Term,
    parse,
    parse_skeleton,
)
from effective.keys.marker import AuthorityTag, Index, Segment, Tag


def _has_leading_tag(items: list[str | Interpolation]) -> bool:
    """Whether the template opens with a legible namespace: a non-empty static, or a `Tag`."""
    if not items:
        return False
    match items[0]:
        case str() as static:
            return bool(static)
        case Interpolation() as interpolation:
            return isinstance(interpolation.value, Tag)
        case unreachable:
            assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead


def _interpolations(items: list[str | Interpolation]) -> list[Interpolation]:
    """The template's interpolations, in the order that NUMBERS them.

    **`Hole.index` is a position in THIS list**, so `_directives`, `_expressions`, `_fill` and
    `compose_key`'s delimiter scan all read it rather than each running their own `enumerate`.
    The one remaining agreement is with `_parts`, which assigns the ordinal by counting the same
    predicate over the same list in the same order, and
    `test_hole_ordinals_index_the_one_enumeration` checks it.

    **Same predicate, spelled the same way.** This once filtered `not isinstance(item, str)` while
    `_parts` matched `case Interpolation()`, so the two walks disagreed about what an
    interpolation IS: a duck-typed item was one to `_parts` and unreachable to this. Spelling one
    walk's test as the complement of the other's makes the agreement a coincidence."""
    return [item for item in items if isinstance(item, Interpolation)]


def _parts(items: list[str | Interpolation], directives: _Directives) -> list[str | Hole]:
    """A template as the grammar sees it: statics verbatim, each interpolation a `Hole`.

    `index` is the ordinal `_interpolations` hands out — the same walk, the same predicate, the
    same order — so `Hole(index)` addresses `_interpolations(items)[index]`.

    The hole carries its declared `domain` through, because `parse_skeleton` mints a `Splice` from
    a lone `Hole` and that is the only channel a splice has."""
    out: list[str | Hole] = []
    index = 0
    for item in items:
        match item:
            case str():
                out.append(item)
            case Interpolation():
                out.append(Hole(index, directives.domains.get(index)))
                index += 1
            case unreachable:
                assert_never(
                    unreachable
                )  # reached by a test that DELIBERATELY escapes the annotation, so no pragma
    return out


DEFAULT_DIRECTIVE = "default"
"""An OPTIONAL coordinate's fallback, on a NAMED coordinate: `depth={depth:default=0}`.

The name is not decoration. Bare means required, so a bare hole carrying a `default=` is refused:
a required coordinate cannot be omitted, which leaves its default unreachable.

The spec slot is **owned by the DSL, not the author** — the same stance `effective.channels` takes
on the data axis (`{x:role=system}`). It carries what a TYPE cannot say, and an optional
coordinate's default is exactly that: `0` the value and `0` the default are the same `int`, and
only the template can say which one is the fallback."""

DOMAIN_DIRECTIVE = "domain"
"""What a SPLICE admits: `{name:domain=address}` (`grammar.Domain`).

Like `DEFAULT_DIRECTIVE`, this carries what a TYPE cannot say. A splice's value is a `Key`, which
is to say ANY key, so the parser cannot tell an await's ADDRESS from an op's own IDENTITY. Only the
position can, and only the template knows the position. Declared here rather than in a
table the composer consults, so the registry and the lint read the same statement."""


@dataclass(frozen=True)
class _Directives:
    """What a template's `format_spec`s declare, keyed by interpolation index.

    Two maps rather than two functions, because both come off one walk of the interpolations and
    the walk is where a spec is refused."""

    defaults: dict[int, str]
    domains: dict[int, Domain]


def _directives(interpolations: list[Interpolation]) -> _Directives:
    """Parse every `format_spec` in the template: the compose side of the optional-coordinate
    rules and of the domain.

    Keyed by the ordinal `_interpolations` assigns, which is what lets `_parts` read a hole's
    declared domain back out by `Hole.index`.

    Refuses any other directive rather than dropping it silently: a dropped spec makes
    `t"n:{n:03d}"` compose `n:7`, bytes differing from what the template visibly says. The spec
    grammar is `directive=value` throughout, so there is one shape to learn."""
    defaults: dict[int, str] = {}
    domains: dict[int, Domain] = {}
    for index, interpolation in enumerate(interpolations):
        if interpolation.conversion is not None:
            raise ValueError(
                f"compose_key: interpolation {interpolation.expression!r} carries a conversion "
                f"({interpolation.conversion!r}). The key DSL defines no conversion — a key's "
                f"rendering is the atom kind's, not the call site's."
            )
        if not interpolation.format_spec:
            continue
        directive, marked, value = interpolation.format_spec.partition(COORDINATE_NAME_SEPARATOR)
        # An `if` chain, not a `match`: a bare name in a `case` value pattern CAPTURES rather than
        # compares, so `case DEFAULT_DIRECTIVE:` would match every directive. Dotting the name to
        # dodge that would mean importing this module into itself.
        if not marked:
            _refuse_unknown_directive(interpolation)
        elif directive == DEFAULT_DIRECTIVE:
            defaults[index] = value
        elif directive == DOMAIN_DIRECTIVE:
            try:
                domains[index] = Domain(value)
            except ValueError as exc:
                raise ValueError(
                    f"compose_key: interpolation {interpolation.expression!r} declares "
                    f"{DOMAIN_DIRECTIVE}={value!r}, which is not a splice domain. The domains are "
                    f"{', '.join(d.value for d in Domain)} — an ADDRESS is what an emitter "
                    f"delivers to, an IDENTITY is what `op_key` mints, and `any` is the "
                    f"deliberate opt-out."
                ) from exc
        else:
            _refuse_unknown_directive(interpolation)
    return _Directives(defaults, domains)


def _refuse_unknown_directive(interpolation: Interpolation) -> NoReturn:
    """The spec slot is the DSL's, so an unrecognized one is refused rather than dropped."""
    raise ValueError(
        f"compose_key: interpolation {interpolation.expression!r} carries the format spec "
        f"{interpolation.format_spec!r}. The key DSL defines two directives — "
        f"{DEFAULT_DIRECTIVE}={{value}}, an OPTIONAL coordinate's fallback, and "
        f"{DOMAIN_DIRECTIVE}={{domain}}, what a SPLICE admits. Render the value before composing; "
        f"if a NAMESPACE needs canonical formatting, impose it in `compose_key`, never at the "
        f"call site."
    )


def _expressions(interpolations: list[Interpolation]) -> list[str]:
    """Each hole's SOURCE EXPRESSION, by interpolation index — so a refusal can name the culprit.

    The thing only a `Template` carries: an f-string has already thrown the expression away by the
    time anything can complain about the value it produced."""
    return [interpolation.expression for interpolation in interpolations]


def _refuse_a_directive_out_of_position(
    skeleton: Skeleton, directives: _Directives, expressions: list[str]
) -> None:
    """Each directive belongs to ONE position, and the two swap places exactly.

    `default=` is a COORDINATE's, and `domain=` is a SPLICE's. A splice cannot be omitted at a
    default — it contributes terms, and omitting them changes the key's arity, which is what makes
    it readable without a registry. A coordinate cannot declare a domain — it holds one atom, whose
    kind the lexer already reads from its bytes.

    Refused rather than ignored, for the reason an unknown spec is: the slot is the DSL's, and a
    declaration that is silently dropped reads exactly like one that is enforced."""
    for element in skeleton.elements:
        match element:
            case Splice(hole=hole) if hole.index in directives.defaults:
                raise ValueError(
                    f"compose_key: the splice {expressions[hole.index]!r} carries a "
                    f"{DEFAULT_DIRECTIVE}=, which only an OPTIONAL COORDINATE can have. A splice "
                    f"contributes TERMS; omitting them at a default would change how many terms "
                    f"the key has, and arity is what makes it readable without a registry. "
                    f"Declare {DOMAIN_DIRECTIVE}= instead."
                )
            case Splice():
                continue
            case SkeletonTerm(coordinates=coordinates):
                pass
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
        # a domain is a SPLICE's declaration, so a hole filling a coordinate may not carry one
        for atom in (a for c in coordinates for a in c.atoms if isinstance(a, Hole)):
            if atom.domain is not None:
                raise ValueError(
                    f"compose_key: interpolation {expressions[atom.index]!r} declares "
                    f"{DOMAIN_DIRECTIVE}={atom.domain.value}, but it fills a COORDINATE, not a "
                    f"splice. A domain says which keys may extend the sequence here; a "
                    f"coordinate holds one atom, whose kind the lexer reads from its bytes."
                )


def _refuse_misplaced_defaults(skeleton: Skeleton, defaults: dict[int, str]) -> None:
    """*bare = required, named = optional*, enforced by the COMPOSER rather than by a lint.

    All three checks here keep the required prefix FIXED-WIDTH, which is what every arity check
    and every discriminator depends on: a defaulted coordinate must be named, a named coordinate
    must be defaulted, and no bare coordinate may follow a named one.

    The design assigned the third to a lint (Python's own no-non-default-after-a-default rule). It
    is here instead because a lint is bounded by what it SCANS and a composer is bounded by
    nothing: a template built anywhere, by anyone, meets this."""
    for element in skeleton.elements:
        match element:
            case Splice():
                # a splice contributes whole TERMS, so it has no coordinate to be misplaced
                continue
            case SkeletonTerm(coordinates=coordinates):
                pass
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
        optional_seen = False
        for position, coordinate in enumerate(coordinates):
            defaulted = any(
                atom.index in defaults for atom in coordinate.atoms if isinstance(atom, Hole)
            )
            # A decision table over the three axes that decide a coordinate's legality: how many
            # ATOMS it has, whether it is NAMED, and whether it is DEFAULTED. Eight cells; these
            # four raise and the rest fall through, so a reader can check the table is total by
            # counting rather than by following four independent conditions.
            match coordinate.atoms, coordinate.name, defaulted:
                case [_, _, *_], _, True:
                    raise ValueError(
                        f"compose_key: a defaulted coordinate is one atom, not a "
                        f"{len(coordinate.atoms)}-atom path — a default has to be comparable to "
                        f"the whole coordinate to know whether to omit it."
                    )
                case _, None, True:
                    raise ValueError(
                        f"compose_key: a coordinate with a {DEFAULT_DIRECTIVE}= is OPTIONAL, and "
                        f"an optional coordinate is named on the wire — write `name={{value:"
                        f"{DEFAULT_DIRECTIVE}=…}}`. Bare means required, and a required "
                        f"coordinate cannot be omitted, so its default is unreachable."
                    )
                case _, str() as name, False:
                    raise ValueError(
                        f"compose_key: coordinate {name!r} is named, so it is OPTIONAL, but "
                        f"declares no {DEFAULT_DIRECTIVE}=. Without one there is no rule for when "
                        f"to omit it, so one identity would have two spellings and replay would "
                        f"miss."
                    )
                case _, _, False if optional_seen:
                    raise ValueError(
                        f"compose_key: required coordinate at position {position} follows an "
                        f"optional one. The required prefix must be fixed-width — Python's own "
                        f"no-non-default-after-a-default rule, and for the same reason."
                    )
            optional_seen = optional_seen or defaulted


def _fill(
    skeleton: Skeleton,
    interpolations: list[Interpolation],
    defaults: dict[int, str],
    expressions: list[str],
) -> ParsedKey:
    """Put the values into the parsed skeleton: the composer's whole job, once parsing is done.

    Each hole resolves by TYPE, which is the module docstring's type system doing its work — a
    `Tag` names a namespace, a `Key` splices its terms in by induction, and anything else is an
    atom whose kind the lexer reads from the rendering.

    A `Key` also resolves by DOMAIN, the narrowing a type cannot do: every key is the same type,
    and what separates an await's address from an op's identity is the position it lands in. The
    template said which; this is where the value has to agree.

    **A spliced key's OCCURRENCE hoists, and only from the last position.** The suffix
    qualifies a whole key, so splicing `q#2` terminally makes the composed key the 2nd instance of
    that identity, which is what it means: the n-th ask of `q` IS the n-th instance of `event;q`.
    That is the live path — `handlers/base.py` splices `placed_await_name(op)` into `event;{…}` on
    every repeat of a settlement await. Anywhere else the suffix would land mid-sequence, where
    `#` is in no atom charset, so it refuses rather than stranding the coordinate the way the `occ`
    term did silently.

    **The splice's non-`Key` arm admits a `Tag` and nothing else**, checked here by
    `_refuse_a_non_tag_term_head` rather than left to upstream fences
    (`test_what_actually_reaches_the_splice_s_non_Key_arm` holds the enumeration). The upstream
    fences do most of the work (an arbitrary object and a `bool` die on the delimiter-freeness
    scan, an `int` on `Term`'s tag validation), but "nothing delivers a bad value today" is a
    statement about the callers, and this arm's parameter is `Any`. The check states the arm's own
    contract instead.

    Two corrections earned that shape, both from reading rather than running, and both worth
    keeping because the arm invites exactly this. An early draft called the arm a laundering hole;
    it never was — nothing reached it that could launder. A later one recorded the `Segment` case
    as an open question; it had already been ruled and pinned. **What survives both is the
    measurement:** the live domain was `Tag | Segment`, and the ruling takes it
    to `Tag`."""
    values = [interpolation.value for interpolation in interpolations]
    terms: list[Term] = []
    occurrence: int | None = None
    last = len(skeleton.elements) - 1
    for index, element in enumerate(skeleton.elements):
        match element:
            case Splice(hole=hole):
                spliced = values[hole.index]
                # The VALUE's type resolves the one ambiguity `Splice` documents, because the
                # parser cannot: a lone hole is a splice either way, and only the runtime value
                # says whether it extends the sequence or stands as a zero-arity qualifier.
                if isinstance(spliced, Key):
                    _refuse_a_wrong_splice(element, spliced, expressions[hole.index])
                    parsed = parse(spliced.stored())
                    if parsed.occurrence is not None:
                        _refuse_a_hoisted_occurrence(parsed, expressions[hole.index], index, last)
                        occurrence = parsed.occurrence
                    terms.extend(parsed.terms)
                else:
                    _refuse_a_non_tag_term_head(spliced, expressions[hole.index])
                    terms.append(Term(str(spliced)))
            case SkeletonTerm(tag=tag, coordinates=coordinates):
                terms.append(
                    Term(
                        _fill_tag(tag, values, expressions),
                        tuple(
                            filled
                            for coordinate in coordinates
                            if (filled := _fill_coordinate(coordinate, values, defaults))
                            is not None
                        ),
                    )
                )
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
    return ParsedKey(tuple(terms), occurrence)


def _refuse_a_hoisted_occurrence(
    parsed: ParsedKey, expression: str, index: int, last: int
) -> None:
    """An occurrence hoists out of a splice only from the template's LAST element.

    Split out of `_fill` so its splice arm reads as one decision; the rule is stated there."""
    if index == last:
        return
    raise ValueError(
        f"compose_key: the splice {expression!r} carries occurrence {parsed.occurrence}, and it "
        f"is not the template's last element. The suffix qualifies a WHOLE key, so from here it "
        f"would have to sit mid-sequence, where {OCCURRENCE_SIGIL!r} is in no atom's charset. "
        f"Splice it last, or splice the base key and apply `.occurrence()` to the result."
    )


def _refuse_a_wrong_splice(element: Splice, spliced: Key, expression: str) -> None:
    """Is this the RIGHT key for this position — the reachability check the type cannot make.

    An arm's payload is declared `Key`, i.e. any key, so `event;{name}` admits a step's checkpoint
    identity as far as the parser is concerned. Nothing emits an event named after a checkpoint, so
    `event;step;tool:foo` is well-formed and *unmintable*, and it shipped in the design's own
    for a day, and is why `KeyMap.explain` decoded a key no producer makes.

    **Undeclared is refused, not waved through.** A default of "anything" would leave exactly the
    hole this closes, and would make every future splice inherit it silently. `domain=any` is the
    way to mean anything, where the next reader can see that someone decided it."""
    if element.hole.domain is None:
        raise ValueError(
            f"compose_key: the splice {expression!r} declares no {DOMAIN_DIRECTIVE}=. A splice "
            f"takes a `Key`, and every key is the same type — only the POSITION says whether this "
            f"one is an ADDRESS an emitter delivers to or an op's own IDENTITY, and only the "
            f"template knows the position. Write "
            f"`{{{expression}:{DOMAIN_DIRECTIVE}={Domain.ADDRESS.value}}}`, "
            f"`…={Domain.IDENTITY.value}`, or `…={Domain.ANY.value}` if it really is either."
        )
    if not admits(element.hole.domain, spliced.stored()):
        expected = (
            "opens with an op arm, so it is an op's own identity and not an address"
            if element.hole.domain is Domain.ADDRESS
            else "opens with no op arm, so it is an address and not an op's own identity"
        )
        raise ValueError(
            f"compose_key: the splice {expression!r} is declared "
            f"{DOMAIN_DIRECTIVE}={element.hole.domain.value}, but {spliced.stored()!r} "
            f"{expected}. "
            f"Nothing mints the key this would compose, so nothing would ever look it up."
        )


def _refuse_a_non_tag_term_head(value: Any, expression: str) -> None:
    """A term head names a NAMESPACE, and only a `Tag` may name one.

    **One decision, two arms** — `_fill`'s non-`Key` splice, where a lone hole stands as a
    zero-arity term, and `_fill_tag`'s `Hole`, where a hole heads a term with coordinates. Written
    once because the pilot that surfaced this found four spellings of one enumeration in this same
    function, and a rule spelled twice is the next one of those.

    **What this is NOT is a safety fix, and saying so is what keeps it from being over-trusted.**
    `Term.__post_init__` re-checks the head against the very predicate `Tag.__new__` runs —
    `TAG.match(...) or FOREIGN_TAG.match(...)` — so a `Segment` here widened the accepted TYPE and
    not the accepted LANGUAGE; nothing was aliased and no authority was forgeable, because a
    `Segment` cannot reach the LEADING position at all (`_has_leading_tag` takes a `Tag`) and
    `Key.scope` is retained from the leading value only. It is a position rule: `Tag` claims *this
    text names a namespace* and `Segment` claims *this text carries no delimiter*, and a term head
    needs the first claim. Two markers agreeing on the bytes is not the same as agreeing on what
    the bytes mean.

    A `Key` never arrives here — it splices by induction, which is a different arm."""
    if isinstance(value, Tag):
        return
    raise TypeError(
        f"compose_key: {expression!r} stands as a term's namespace and is a "
        f"{type(value).__name__}. A term head takes a `Tag`, which claims the text NAMES a "
        f"namespace; `Segment` claims only that it carries no delimiter, which is a different "
        f"thing to know about the same bytes. Write `Tag({expression})` if it really is a "
        f'namespace, or put the value in a coordinate — t"…:{{{expression}}}" — if it is data.'
    )


def _fill_coordinate(
    coordinate: SkeletonCoordinate, values: list[Any], defaults: dict[int, str]
) -> Coordinate | None:
    """One coordinate's atoms, or `None` when it sits at its default and is therefore OMITTED.

    **A coordinate at its default is always omitted**, enforced here rather than asked of the
    author, so the short form is *the* canonical form and the long one is unwritable. That is what
    keeps one identity to one spelling, and replay from missing. Present optionals render in
    declaration order for free: the template IS the declaration order, and this walk preserves
    it."""
    match coordinate.atoms:
        case [Hole() as hole] if hole.index in defaults and (
            f"{values[hole.index]}" == defaults[hole.index]
        ):
            return None
        case _:
            return Coordinate(
                tuple(_fill_atom(atom, values) for atom in coordinate.atoms), coordinate.name
            )


def _fill_tag(tag: str | Hole, values: list[Any], expressions: list[str]) -> str:
    """A term's namespace, which a template may write or interpolate::

        t"govern:{gate}"      the tag written out
        t"{GOVERN}:{gate}"    a namespace constant naming itself

    The second keeps the tag a typed value at the call site instead of a repeated literal.

    **Only the interpolated arm has something to prove.** A written tag arrives from the parser as
    a plain `str`, so `case Tag()` in the pattern would reject every statically-written namespace
    in the repo; an interpolated one carries a claim about its own type. That is why the check sits
    inside the `Hole` arm's body rather than above the `match`."""
    match tag:
        case str():
            return tag
        case Hole(index=index):
            value = values[index]
            _refuse_a_non_tag_term_head(value, expressions[index])
            return str(value)
        case unreachable:
            assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead


def _fill_atom(atom: SkeletonAtom, values: list[Any]) -> Atom:
    """A literal atom stands as written; a `Hole` becomes the rendering of its value.

    Split out of `_fill_coordinate`'s comprehension so the two arms of `SkeletonAtom` are a
    decision `ty` can close rather than a conditional expression it cannot."""
    match atom:
        case Atom():
            return atom
        case Hole(index=index):
            return Atom.of(f"{values[index]}")
        case unreachable:
            assert_never(
                unreachable
            )  # reached by a test that DELIBERATELY escapes the annotation, so no pragma


def gather_prefix(g: int, i: int) -> str:
    """The prefix branch *i* of gather *g* contributes to every key minted inside it.

    `scope_prefix`'s sibling for the frame the SUBSTRATE mints rather than one an author scopes.
    Every walk (the durable handler, the recorder, the replay driver) mints the frame here, so the
    gather arm's tag and the tag, arity and term separators are spelled once.

    Composed and then handed to `scope_prefix`, so the frame separator is applied in the one place
    that applies it and inherits its refusal of a frame carrying the delimiter."""
    return scope_prefix(compose_key(t"gather:{Index(g)},{Index(i)}"))


def race_prefix(r: int, i: int) -> str:
    """The prefix branch *i* of race *r* contributes to every key minted inside it.

    `gather_prefix`'s sibling, with the race's own tag and ordinal, so a race branch and a gather
    branch at the same coordinates never share a key."""
    return scope_prefix(compose_key(t"race:{Index(r)},{Index(i)}"))


def race_choice(r: int) -> Key:
    """The checkpoint that holds race *r*'s choice: its winners, or its impossibility."""
    return compose_key(t"race:{Index(r)};choice")


def race_endings(r: int) -> Key:
    """The checkpoint that holds how each of race *r*'s branches ended, written at its barrier."""
    return compose_key(t"race:{Index(r)};endings")


def compose_key(template: Template) -> Key:
    """The sole op-key producer, which is what lets a lint prove the grammar and what discharges
    `Keys.lean`'s serialization-injectivity assumption by construction rather than by hope.

    **Nothing here escapes anything.** A composed key is injective because of the
    template's SHAPE and the TYPES it was handed::

        compose_key(t"{Tag('ns')}:{Segment('a')};{'x:y'}")     ->  refused: str may contain a
                                                                   separator
        compose_key(t"{Tag('ns')}:{Segment('p')}{Segment('q')}") -> refused: adjacent holes

    Three rules, and each removes an ambiguity rather than hiding one:

    - the leading position is a namespace TAG (a non-empty static, or a `Tag`), so a reader always
      knows which template produced a string;
    - every interpolation is delimiter-free BY TYPE: a `Segment`, an `int`, or a `Key` spliced as
      its own terms, and the terminal hole is not exempt;
    - consecutive interpolations must be separated by a non-empty static, because `t"x:{a}{b}"`
      makes `("p","q")` and `("pq","")` the same bytes and no escaping can recover the split.

    A namespace is a tagged union: one tag may own several shapes, and the registry admits them
    only when `registry.separated` proves their languages disjoint from structure alone. Typed
    holes make that proof possible. Were a hole allowed a separator, `t"processed:{message_id}"`
    given `"m1,r2"` and `t"processed:{a},{b}"` given `("m1", "r2")` would both mint
    `processed:m1,r2`.
    """
    # ANNOTATED: `list(template)` is `list[str | Interpolation]` by PEP 750, and saying so is
    # what lets the two `match items[0]` decisions below be closed rather than merely written.
    items: list[str | Interpolation] = list(template)
    if not _has_leading_tag(items):
        first = "(empty)" if not items else getattr(items[0], "expression", items[0])
        raise ValueError(
            f"compose_key: a key template must begin with a namespace TAG — a non-empty static "
            f'(t"mytag:{{...}}") or a `Tag` interpolation (t"{{Tag(MY_TAG)}}:{{...}}") — but this '
            f"one begins with {first!r}. An untyped interpolation is invisible to the shape "
            f"registry, which is what makes a key's position recoverable."
        )
    for left, right in pairwise(items):
        if not isinstance(left, str) and not isinstance(right, str):
            raise ValueError(
                f"compose_key: interpolations {left.expression!r} and {right.expression!r} are "
                f"adjacent with no delimiter between them, so the key is not injective "
                f"({{'p','q'}} and {{'pq',''}} would collide). Put a static separator between "
                f"them, or compose one of them into a single value first."
            )
    # The format-spec slot is **owned by the DSL, not by the author**: the processor gives it a
    # meaning, not the t-string.
    # The key DSL defines two directives, `default=` (an optional coordinate's fallback) and
    # `domain=` (what a splice admits), and refuses every other spec rather than dropping it in
    # silence: a dropped `t"n:{n:03d}"` composes `n:7`, bytes the template does not say. Both
    # carry what a TYPE cannot: which value is a fallback, and which keys may extend the sequence.
    #
    # Sentinel asymmetry: an absent spec is `''` while an absent conversion is `None`. It falls
    # out of `__format__`. Testing both the same way makes every plain hole look like it carried
    # a spec.
    # The ONE enumeration of the holes. Everything below that speaks of an interpolation by index —
    # the directive maps, the source expressions, the runtime values, and the `Hole` ordinals
    # `_parts` mints — addresses a position in this list.
    interpolations = _interpolations(items)
    directives = _directives(interpolations)
    # EVERY hole must be delimiter-free: there is no terminal exemption.
    #
    # It existed because the last hole "took the rest of the string", which is only a coherent
    # idea in a grammar that cannot say where a field ends. This one can: `,` ends a coordinate
    # and `;` ends a term, so a value carrying a separator is not a value at all — it is a
    # structure being flattened in, which is the Bobby Tables shape the whole arc is about.
    #
    # What replaces it is INDUCTION. A delimiter-bearing value was always one of three things,
    # and each now has a spelling: a nested KEY (splice it with `;`), a PATH (compose the atoms
    # with `/`), or a value that should never have carried a delimiter (wrap it in `Segment`,
    # which refuses one at construction). `artifact:{type}/{subtype},{digest}` is the worked
    # example of all three at once: a path, a delimiter-free subtype, and a digest coordinate.
    for interpolation in interpolations:
        value = interpolation.value
        if isinstance(value, Tag | Key):
            # a Tag names the namespace; a Key is a CLOSED structure, spliced by induction
            continue
        if not isinstance(value, Segment | int) or isinstance(value, bool):
            raise ValueError(
                f"compose_key: interpolation {interpolation.expression!r} is "
                f"{type(value).__name__}, which may contain a separator — so the composed key "
                f"would not be injective. Wrap it: Segment({interpolation.expression}). There is "
                f"no last-position exemption: a value that WANTS delimiters is a nested key "
                f"(splice it after {TERM_SEPARATOR!r}), or a path (compose its atoms with "
                f"{PATH_SEPARATOR!r}) — never one flattened value."
            )
    # The composed key RETAINS the reach its leading tag declared (`Key.scope`). The template
    # handed us the `AuthorityTag` itself — a typed value, not the text of one — so this reads a
    # guarantee rather than re-deriving it by matching bytes against a registry. A namespace whose
    # leading position is a plain static or a bare `Tag` declares nothing, and gets `None`.
    match items[0]:
        case Interpolation(value=AuthorityTag() as tag):
            declared = tag.scope
        case str() | Interpolation():
            declared = None
        case unreachable:
            assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
    # PARSED, not joined. The statics are a term of the little language (`parse_skeleton`), the
    # holes are its values, and `stored()` is the rendering — which is what makes the module
    # docstring's claim true of the implementation rather than of its intent.
    #
    # The f-string/`str.join` backend survives INSIDE `render`, which is the one exempt position:
    # eager format-then-concat is exactly right once every structural decision is already made.
    skeleton = parse_skeleton(_parts(items, directives))
    expressions = _expressions(interpolations)
    _refuse_a_directive_out_of_position(skeleton, directives, expressions)
    _refuse_misplaced_defaults(skeleton, directives.defaults)
    parsed = _fill(skeleton, interpolations, directives.defaults, expressions)
    return Key(parsed, declared)


# `_refuse_forged_frames` is GONE. It refused the frame delimiter in a template
# STATIC, on the grounds that a frame is never something a key template supplies. Under this
# grammar a frame IS a leading term and `;` is the term separator, so a static carrying one is the
# author composing a sequence — which is exactly what `event;{key}` and `approve;{op_key}` do.
#
# The fence did not disappear, it moved to where the untrusted values are: `Segment` refuses every
# separator at construction, `step_key` refuses one in an author's step name, and `scope_prefix`
# refuses an atom that would forge a boundary. A template is written by the substrate; a value
# comes from anywhere.
