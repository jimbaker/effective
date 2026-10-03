"""The key **source map** — read a composed key and recover exactly how it was produced.

On the wire a key is a string (checkpoint names, ledger `event_id`s, event names) in the grammar
of `effective.keys.grammar`: terms separated by `;`, a tag before `:`, coordinates separated by
`,`. A `Template` knows more than the string does: it knows the **source expression of every
hole**. Recording each template's shape once turns a key into a decodable record, linked back to
the line that minted it:

    hyp:r-fork;review:m1  ->  the `compose_key` line in `counterfactual.fork_scoped`
                              {'child_run_id': 'r-fork', 'event_id': 'review:m1'}

**The same artifact as the enforcement.** Registering a shape is how two shapes claiming one tag
are discovered, and two shapes under one tag are a collision unless `separated` proves them
disjoint. So the registry is built once, by the lint, and serves both purposes.

**Built statically, by the lint** (`effective.lint --key-registry`), not at runtime: a source scan
gets `file:line` by construction, costs nothing in production, and fails the gate on a collision.
The map is derived, so it is gitignored (`build/key-registry.json`) and regenerated, never edited.

**Decodability rests on a grammar with no escaping.** An escaped key such as
`hyp:r-fork,review%3Am1` still decodes mechanically, but it defeats the glance. An interior hole is
delimiter-free by type and a nested key is its own term, so `explain` splits left to right and a
human reads the same structure the parser sees.

Operational use: a production diagnosis starts from a key found in a live run. Without the map
that key is matched against a human's memory of the codebase; with it the key names its own
producing line.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, assert_never

MAP_NAME = Path("build/key-registry.json")
"""Where the derived map sits, relative to a tree root rather than to anyone's shell."""

DEFAULT_PATH = Path(__file__).resolve().parents[3] / MAP_NAME
"""The map in THIS checkout, found from the module rather than from the working directory.

A library whose answer depends on where python was started is not a library: `fold_cycles` took a
`FileNotFoundError` from `/tmp` while working from the repo root, and a reader has no way to see
that from the call. `load` still falls back to a working-directory path, which is what a caller
with a map of their own building gets."""

# The grammar itself, so a decode cannot drift from a compose — this module now holds NO reader
# of its own, and imports nothing from `effective.keys`. A frame is a leading TERM, so even
# un-framing is a walk over parsed terms rather than a second reading of the delimiters.
from effective.keys.frame import unmintable  # noqa: E402
from effective.keys.grammar import (  # noqa: E402
    ARITY_SEPARATOR,
    COORDINATE_NAME_SEPARATOR,
    FOREIGN_TAG,
    OCCURRENCE_SIGIL,
    PATH_SEPARATOR,
    PROJECTION_SIGIL,
    TAG_SEPARATOR,
    TERM_SEPARATOR,
    Atom,
    Coordinate,
    Hole,
    KeySyntaxError,
    ParsedKey,
    ProjectedKey,
    Skeleton,
    SkeletonAtom,
    SkeletonCoordinate,
    SkeletonTerm,
    Splice,
    parse,
    parse_skeleton,
    split_occurrence,
)
from effective.keys.marker import Role  # noqa: E402


@dataclass(frozen=True)
class Shape:
    """One VARIANT of a namespace: the template structure of the keys it mints.

    **A skeleton plus its field names, and nothing else.** The structure comes from
    `grammar.parse_skeleton`, the SAME reader `compose_key` uses, so a registered shape and the
    keys the composer mints cannot disagree about where a field ends. The registry is a source
    map only if it describes what is actually minted.

    `fields` are the holes' source expressions in order, which is what lets a decode return NAMED
    values (`{'run_id': 'r1', 'depth': '2'}`) and what makes the map a source map rather than a
    list of patterns. Two templates under one tag may differ in field names and still be one
    variant — `budget-grant:{run_id},{trips}` and `budget-grant:{run_id},{i}` are one protocol
    written twice; `sites` records them all.

    No hole absorbs the rest of the key: a coordinate count is read from the bytes, which is what
    makes arity a sound ground for `separated`."""

    skeleton: Skeleton
    fields: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    """The ROLE each coordinate declares, parallel to `fields` and `""` where none is declared.

    A marker (`Name`, `Run`, `Index`, `Subject`) says what the coordinate means, and the lint
    reads it off the mint site. That declaration is the half a projection cannot recover from the
    bytes: the text of `hyp:c1;reviewed:m1` says where the coordinates end, and only the shape
    says that the first one identifies an execution. A fold drops by `(tag, position)` looked up
    here, which is what lets `govern`'s single term keep its gate name and drop its run id.

    Persisted, unlike a splice's `domain=`: a role describes the LANGUAGE rather than one
    producer, so every site minting a variant declares the same role for each of its
    coordinates."""

    site: str = ""
    sites: tuple[str, ...] = ()

    @property
    def tag(self) -> str:
        """The leading term's tag — the namespace. A `Hole` here means the template opens with a
        `Tag` interpolation the lint resolved to its constant before constructing this."""
        match self.skeleton.elements[0]:
            case SkeletonTerm(tag=str() as tag):
                return tag
            case _:
                return ""

    @property
    def arity(self) -> int:
        return len(self.fields)

    @property
    def template(self) -> str:
        """The shape with its holes anonymous, which is the form two variants are compared in.

        `budget-grant:{run_id},{trips}` and `budget-grant:{run_id},{i}` are one protocol written
        twice, and both render `budget-grant:{},{}`. Sorting a population by this groups the
        families: `just key-view`.
        """
        return _render_skeleton(self.skeleton, lambda: "{}")

    def label(self) -> str:
        """The shape as a reader sees it: the template with each hole named."""
        holes = iter(self.fields)
        return _render_skeleton(self.skeleton, lambda: f"{{{next(holes, '?')}}}")

    def decode(self, key: str) -> dict[str, str] | None:
        """Bindings if `key` is in THIS variant's language, else `None`.

        Total by design: a tag may own several variants, so "not mine" has to be a value the
        caller dispatches on rather than an exception — the asymmetry `grammar.KeySyntaxError`
        documents from the other side.

        Structural, over the PARSED key. The old implementation consumed delimiter-separated
        segments and could not tell a coordinate from a nested key; this one walks two structures
        the same parser produced.

        **Membership only — REACHABILITY is asked one level up**, in `explain`, and deliberately
        not here. A splice binds every remaining term, so this cannot refuse an unmintable payload
        anyway; and `explain` unframes before it dispatches, so by the time a shape sees a key the
        evidence is already peeled off. `grammar.unmintable` therefore runs on the whole key,
        before the frame comes off."""
        try:
            parsed = parse(key)
        except KeySyntaxError:
            return None
        return _bind(self.skeleton, parsed, self.fields)


def _render_skeleton(skeleton: Skeleton, hole: Any) -> str:
    """Render a skeleton, calling `hole()` for each interpolation, in order."""
    parts: list[str] = []
    for element in skeleton.elements:
        match element:
            case Splice():
                parts.append(hole())
            case SkeletonTerm(tag=str() as tag, coordinates=coordinates):
                parts.append(_render_term(tag, coordinates, hole))
            case SkeletonTerm(coordinates=coordinates):
                # The tag is itself a hole (`t"{GOVERN}:{gate}"`), and it is consumed FIRST —
                # argument evaluation orders it ahead of the coordinates', which is the order a
                # reader pairs `fields` in.
                parts.append(_render_term(hole(), coordinates, hole))
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
    return TERM_SEPARATOR.join(parts)


def _render_term(tag: str, coordinates: tuple[SkeletonCoordinate, ...], hole: Any) -> str:
    """One term. A coordinate whose holes come back `None` was OMITTED and renders as nothing.

    `None` is how a projection says "the author never wrote this one", which is not the same as
    "a drop set removed it": an optional renders only when it was passed, so re-rendering it would
    put a coordinate on the wire that no producer minted. Every hole is still drawn from `hole()`,
    omitted or not, because the field names were consumed in step when the key was bound."""
    if not coordinates:
        return tag  # a bare tag: a qualifier
    rendered = []
    for coordinate in coordinates:
        atoms = [_render_atom(atom, hole) for atom in coordinate.atoms]
        if any(atom is None for atom in atoms):
            continue
        name = f"{coordinate.name}{COORDINATE_NAME_SEPARATOR}" if coordinate.name else ""
        rendered.append(name + PATH_SEPARATOR.join(atoms))
    if not rendered:
        return tag
    return f"{tag}{TAG_SEPARATOR}{ARITY_SEPARATOR.join(rendered)}"


def _render_atom(atom: SkeletonAtom, hole: Any) -> str:
    match atom:
        case Atom(text=text):
            return text
        case Hole():
            # Named rather than left to a wildcard: `SkeletonAtom` has exactly these two arms, so
            # the `_` that stood here WAS the hole arm and said so nowhere.
            return hole()
        case unreachable:
            assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead


def _bind(skeleton: Skeleton, parsed: ParsedKey, fields: tuple[str, ...]) -> dict[str, str] | None:
    """Walk a skeleton against a parsed key, binding each hole to what sits in its position.

    A decision table over `(skeleton element, what is left of the key)`: every arm is a shape the
    grammar can produce, and `None` is "this variant's language does not contain the key" — a
    value the caller dispatches on, never an error."""
    out: dict[str, str] = {}
    names = list(fields)
    terms = list(parsed.terms)
    last = len(skeleton.elements) - 1

    def take() -> str:
        return names.pop(0) if names else "?"

    for index, element in enumerate(skeleton.elements):
        match element:
            case Splice() if index == last and terms:
                # A splice consumes EVERY remaining term: the value was a finished key, and `;`
                # means it extended the sequence. This is the inverse of composition-by-induction,
                # and it is why a splice may only be LAST — anything after it has no boundary.
                # `ParsedKey.render`, not `;`.join: a foreign term joins its payload with
                # `:`, so joining by hand reports the field's value as bytes no producer wrote.
                out[take()] = ParsedKey(tuple(terms)).render()
                return out
            case Splice():
                return None  # not last, or nothing left for it to absorb
            case SkeletonTerm() if not terms:
                return None  # the skeleton wants a term the key does not have
            case SkeletonTerm(tag=tag, coordinates=coordinates):
                term = terms.pop(0)
                match tag:
                    case str() if tag != term.tag:
                        return None  # a literal tag that disagrees
                    case Hole():
                        out[take()] = term.tag  # the tag IS the field (`{GOVERN}`)
                if _bind_coordinates(coordinates, term.coordinates, out, take) is None:
                    return None
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
    return out if not terms else None


def _bind_coordinates(
    want: tuple[SkeletonCoordinate, ...],
    got: tuple[Coordinate, ...],
    out: dict[str, str],
    take: Any,
) -> Any:
    """One term's coordinates. `None` means this variant does not match.

    **A shape spans a RANGE of wire arities.** An optional coordinate is named and is OMITTED at
    its default, so `govern:`'s four-coordinate template renders two, three or four of them.
    Comparing declared count to rendered count would refuse every key that left an optional out,
    which is most of them, since the default is the common case. `separated` reads the same fact
    from the other side, through `_arity_range`.

    A refusal here is silent: `explain` falls back to `_unframe` when no variant matches, so
    `govern:approve,r1;step;tool:charge-card` would peel down to `step;tool:charge-card` and
    report `step_key`'s line, a confident answer naming the wrong producer.

    Required coordinates are positional and must all be present; `compose_key` guarantees they form
    a fixed-width prefix, so the walk below is unambiguous. An omitted optional consumes its field
    NAME without binding a value, or every later field would pair with the wrong hole."""
    remaining = list(got)
    for expected in want:
        actual = remaining[0] if remaining else None
        if expected.name is not None and (actual is None or actual.name != expected.name):
            # An optional left out at its default. Its holes still have to be consumed from
            # `fields`, or `occurrence` would bind to `pass_n`'s name one position later.
            for atom in expected.atoms:
                # lint: totality(filter) — one arm is selected to be counted and the other
                # does nothing, so there is no second arm to name.
                if isinstance(atom, Hole):
                    take()
            continue
        if actual is None or expected.name != actual.name:
            return None
        remaining.pop(0)
        if len(expected.atoms) != len(actual.atoms):
            return None
        for atom, value in zip(expected.atoms, actual.atoms, strict=True):
            match atom:
                case Atom(text=text) if text != value.text:
                    return None  # a literal that disagrees kills the match
                case Hole():
                    out[take()] = value.text
                # an `Atom` that AGREES falls through both arms: matched, nothing to bind
    return out if not remaining else None


def separated(a: Shape, b: Shape) -> tuple[bool, str]:
    """Are two variants' languages PROVABLY disjoint, decided on the structure alone?

    **Arity is now a sound ground, where it was not.** The old rule had to say "arity, WHEN
    NEITHER ABSORBS", because a tail could swallow any number of segments — so an arity-2 and an
    arity-3 variant of `depth-grant:` were not provably disjoint and the registry refused a union
    it should have admitted. With coordinates counted from the bytes, different counts mean
    different languages, full stop.

    **It walks the whole term sequence.** A discriminator the grammar puts in plain sight at a
    later term counts: `code;{name};seg:{n}` and `code;{name};action:{j};tool:{x}` differ at term 1
    by tag, arity and term count, so they are disjoint. A `Splice` still stops the walk, because it
    absorbs every remaining term and nothing past it has a fixed position.

    Conservative by design: an unproven pair is refused, so a `False` here is "not shown disjoint",
    never "shown to collide"."""
    if a.tag != b.tag:
        return True, "different tags"
    # `strict=False` on purpose: unequal lengths are a RESULT here (different term counts are a
    # separation), decided after the walk rather than raised during it.
    for position, pair in enumerate(zip(a.skeleton.elements, b.skeleton.elements, strict=False)):
        match pair:
            case (Splice(), _) | (_, Splice()):
                return False, f"a splice at term {position} absorbs every remaining term"
            # `str() as x` — BOTH tags must be LITERALS. A `Hole` tag matches whatever literal
            # faces it (`_bind`), so it discriminates nothing, and comparing two holes compares
            # their interpolation ORDINALS, an artifact of the template. `_discriminator` makes
            # the same argument for coordinates ("a hole is never a discriminator"). Otherwise
            # `ns:{a};{T}:{b}` and `ns:{T};tag:{b}` would be reported disjoint while BOTH decode
            # `ns:lit;tag:v`, which falsifies `_decode_one`'s "at most one can match", the
            # invariant the source map and the key lints rest on. No interpolated non-leading
            # tag exists in the tree.
            case (SkeletonTerm(tag=str() as x), SkeletonTerm(tag=str() as y)) if x != y:
                return True, f"different tags at term {position}"
            case (SkeletonTerm(coordinates=p), SkeletonTerm(coordinates=q)) if _arity_disjoint(
                p, q
            ):
                return True, f"disjoint arities at term {position}, countable from the bytes"
            case (SkeletonTerm(coordinates=p), SkeletonTerm(coordinates=q)) if (
                index := _discriminator(p, q)
            ) is not None:
                return True, f"a literal discriminator at term {position}, coordinate {index}"
    if len(a.skeleton.elements) != len(b.skeleton.elements):
        return True, "different term counts, countable from the bytes"
    return False, "no discriminating coordinate, and the arities agree"


def _arity_range(coordinates: tuple[SkeletonCoordinate, ...]) -> tuple[int, int]:
    """The coordinate counts a term can RENDER: required-only at the low end, all of them high.

    **The declared count is not the arity, because a named coordinate is optional by
    construction** — `compose_key` refuses a named one with no default and a defaulted one with no
    name — and an optional AT its default is omitted. So `probe:{x},d={d:default=0}` renders one
    coordinate or two, and a term has a RANGE rather than a count."""
    return sum(1 for c in coordinates if c.name is None), len(coordinates)


def _arity_disjoint(
    left: tuple[SkeletonCoordinate, ...], right: tuple[SkeletonCoordinate, ...]
) -> bool:
    """Can no rendering of one term have the same coordinate count as any rendering of the other?

    Comparing DECLARED counts is only sound while every coordinate is required. Once one can be
    omitted it is not: `probe:{x}` and `probe:{x},d={d:default=0}` declare different counts and
    both render `probe:v`. A registry that admits that pair lets one namespace mint two shapes it
    cannot tell apart, which is the collision this whole rule exists to refuse — so the question
    has to be asked over the RANGE each term can render, not over what it declares."""
    low_left, high_left = _arity_range(left)
    low_right, high_right = _arity_range(right)
    return high_left < low_right or high_right < low_left


def _discriminator(
    left: tuple[SkeletonCoordinate, ...], right: tuple[SkeletonCoordinate, ...]
) -> int | None:
    """The first coordinate where BOTH sides lead with a literal and the literals differ.

    A hole is never a discriminator — it matches whatever literal faces it, which is precisely the
    overlap `separated` exists to refuse. The pattern says that structurally: an arm requiring
    `Atom` on both sides simply does not match when either is a `Hole`."""
    # `strict=False`: arities may now legitimately differ here (overlapping ranges reach this
    # arm), and a discriminator in the shared prefix is still a discriminator.
    for index, pair in enumerate(zip(left, right, strict=False)):
        match pair:
            case (
                SkeletonCoordinate(atoms=(Atom(text=x), *_)),
                SkeletonCoordinate(atoms=(Atom(text=y), *_)),
            ) if x != y:
                return index
    return None


@dataclass(frozen=True)
class Explanation:
    """A decoded key: where it was composed, and the named values it carries.

    `caveats` is non-empty when the decode is not certain — see `_LEGACY_SHAPES`. It is on the
    result rather than raised, because the decode is still the best available reading; the caller
    (an operator, a diagnosis ladder) needs the reading *and* the doubt."""

    key: str
    site: str
    bindings: dict[str, str]
    caveats: tuple[str, ...] = ()
    occurrence: int | None = None
    """Which occurrence of this identity the key names, when it carries `Key.occurrence`'s `#N`.

    **Its own field rather than a binding, because it is not a template field.** `govern:` puts an
    occurrence IN its shape (`govern:{gate},{run_id},pass-n={n},occurrence={k};{op_key}`), so
    `bindings["occurrence"]` is already taken there and means something structurally different —
    a hole the registry can see, at a position the composer chose. This one is the grammar's
    terminal SUFFIX, appended to a finished identity by the runtime, which is why the decoder has
    to split it off before it can bind anything. Two readings of one idea, kept distinguishable at
    the surface that reports them.

    `None` means the key names the first (or only) occurrence: `Key.occurrence` is the identity
    at n <= 1, so an unsuffixed key and `#1` are the same string and there is nothing to report."""

    frame: tuple[Explanation, ...] = ()
    """The `scoped(...)` atoms this key ran inside, outermost first — each DECODED, not spelled.

    **A frame path is a sequence of composed keys, so it is decoded as one.** Every atom is
    itself a `compose_key` template (`t"rec:{i}"`, `t"d:{depth}"`, `t"fold:{lv},{k}"`), which is
    why this is not a list of strings and not `(label, index)` pairs: the arity varies by
    namespace — `seed` carries no hole, `rec:` one, `fold:` two — so the only correct reading is
    to ask the registry, per atom: a label and an index, like `t"rec:{g}"`, and the prefix decides
    how many such indexes there are.

    If `compose_key` is `apply`, this whole function is `unapply` — and a frame path is the
    `unapplySeq` case, each atom a fixed-arity `unapply` inside it.

    Empty for an unframed key, which is the common case and moves nothing. An atom the registry
    does not know is reported with an empty `site` and no bindings rather than dropped: an author
    may `scoped(...)` on a namespace no `src/` template registers, and a silently missing atom
    would misstate which frame the operator is standing in."""

    def __str__(self) -> str:
        pairs = ", ".join(f"{name}={value!r}" for name, value in self.bindings.items())
        nth = f", occurrence={self.occurrence}" if self.occurrence is not None else ""
        path = "".join(f"{atom.key}/" for atom in self.frame)
        under = f"\n    inside {path!r}" if self.frame else ""
        suffix = "".join(f"\n    ! {c}" for c in self.caveats)
        return f"{self.key}  ({self.site})  {pairs}{nth}{under}{suffix}"


class UnknownTag(KeyError):
    """The key's leading namespace is not in the registry — so nothing can be said about its
    shape. Either the registry is stale (regenerate: `just key-registry`) or the key was minted
    outside `compose_key`, the sole producer of a durable identity."""


@dataclass(frozen=True)
class KeyMap:
    """The registry, keyed by namespace tag."""

    variants: dict[str, tuple[Shape, ...]]
    """Every registered variant of each namespace. A tag maps to a TUPLE because a namespace is a
    tagged union — `code:seg,{n},{name}` and `code:action,{j},{name};tool:{tool}` are two variants
    of one namespace, admitted because `separated` proves their languages disjoint."""

    @classmethod
    def load(cls, path: Path | str | None = None) -> KeyMap:
        """Read the source map: THIS checkout's by default, then the working directory's.

        **A quotient taken through this map is only as current as the map**, so the file's version
        is part of any answer about whether two runs agree. It is rebuilt by `just lint`, which is
        why an in-repo caller never sees it stale; a caller somewhere else is told what to run
        rather than handed a `FileNotFoundError` naming a path it has no reason to recognise.

        An INSTALLED copy has neither, because the map is derived and the wheel ships no `build/`
        tree; packaging the map with the wheel is open work. Until then, pass a map from
        `from_shapes`."""
        for candidate in [Path(path)] if path is not None else [DEFAULT_PATH, MAP_NAME]:
            if candidate.exists():
                return cls._from_json(json.loads(candidate.read_text()))
        raise FileNotFoundError(
            f"no key source map at {path or DEFAULT_PATH}: it is DERIVED from the tree rather "
            f"than committed, so a checkout builds it and an INSTALLED copy has none yet "
            f"(packaging it is open work). In a checkout run `just key-registry`; anywhere "
            f"else pass a `KeyMap` built with `from_shapes` and ignore the path above."
        )

    @classmethod
    def _from_json(cls, raw: dict[str, Any]) -> KeyMap:
        return cls(
            variants={
                tag: tuple(_shape_from_json(tag, v) for v in entry["variants"])
                for tag, entry in raw["shapes"].items()
            }
        )

    @classmethod
    def from_shapes(cls, shapes: list[Shape]) -> KeyMap:
        grouped: dict[str, list[Shape]] = {}
        for shape in shapes:
            grouped.setdefault(shape.tag, []).append(shape)
        return cls(variants={tag: tuple(vs) for tag, vs in grouped.items()})

    def _decode_one(self, key: str) -> tuple[str, dict[str, str]] | None:
        """`(site, bindings)` for the first variant whose language contains `key`, else `None`.

        The `unapply` of `compose_key`, and total for the same reason Scala's is: a tag owns
        several variants, so "not mine" has to be a value the caller dispatches on. At most one
        can match — `separated` proves the variants pairwise disjoint before the registry admits
        them — so trying them in order is a lookup, not a guess."""
        for shape in self.variants.get(_leading_tag(key), ()):
            if (bindings := shape.decode(key)) is not None:
                return shape.site, bindings
        return None

    def _unframe(self, key: str) -> tuple[str, tuple[Explanation, ...]]:
        """Split a `scoped(...)` frame off the front — `(inner, atoms-outermost-first)`.

        **A walk over terms.** A frame IS a leading TERM, so where the frame ends is answered by
        the grammar: drop terms from the left until the remainder decodes, and what you dropped
        is the frame.

        The registry decides where to stop, because a frame atom and an identity are both
        well-formed terms and only ownership can tell them apart. The SPLIT is exact and only the
        STOPPING RULE consults anything."""
        try:
            parsed = parse(key)
        except KeySyntaxError:
            return key, ()
        terms = list(parsed.terms)
        atoms: list[Explanation] = []
        while terms:
            # `ParsedKey.render` for the same reason as the splice arm above — and this one
            # runs FIRST, so a hand-rolled join here hands the splice an already-forged key.
            inner = ParsedKey(tuple(terms)).render()
            if self._decode_one(inner) is not None:
                return inner, tuple(atoms)
            atoms.append(self._atom(terms.pop(0).render()))
        return key, ()

    def _atom(self, atom: str) -> Explanation:
        """One frame atom, decoded — `Explanation.frame`'s element type."""
        match self._decode_one(atom):
            case (site, bindings):
                return Explanation(key=atom, site=site, bindings=bindings)
            case _:
                return Explanation(key=atom, site="", bindings={})

    def project(self, key: str | ProjectedKey, *, drop: tuple[type[Role], ...]) -> ProjectedKey:
        """π on a STORED key: every coordinate whose registered role is in `drop`, replaced by `*`.

        **The quotient the tape can answer on its own.** The bytes of `govern:review,r9,pass-n=2`
        say where the coordinates end and nothing about what they mean; the shape says the second
        identifies an execution. A projection joins the two and needs neither the minting process
        nor a per-tag table, so one TERM can keep its gate name and drop its run id, which is
        the case a per-tag drop set structurally cannot express.

        **A dropped coordinate renders as `PROJECTION_SIGIL`.** Substitution holds the arity, so a
        projection of `tag:{a},{b}` stays a two-coordinate reading of a two-coordinate key, and a
        reader can see that a coordinate was there. Deletion says nothing in the place it removed.

        **Projecting a projection re-projects its SOURCE**, under the union of the two drop sets,
        which is what makes `project(project(k, A), B) == project(k, A | B)` hold rather than be
        tested. The rendered text is never re-read: `*` is outside the key language, so reading it
        back would be reading a grammar nothing writes.

        A key no variant claims comes back whole, which is the answer an unruled scope already
        gets. A coordinate's meaning is settled where it is minted, so a key this map has never
        been told about keeps every coordinate it carries."""
        match key:
            case ProjectedKey(source=source, drop=already):
                roles = already | {cls.role for cls in drop}
            case str() as source:
                roles = frozenset(cls.role for cls in drop)
            case unreachable:
                assert_never(unreachable)
        base, occurrence = split_occurrence(source)
        projected, claimed = self._project_walk(base, set(roles))
        text = projected if occurrence is None else projected + OCCURRENCE_SIGIL + str(occurrence)
        return ProjectedKey(source=source, drop=frozenset(roles), claimed=claimed, _text=text)

    def _project_walk(self, base: str, roles: set[str]) -> tuple[str, bool]:
        """Project every term of `base` that some variant claims, left to right.

        **Not `_unframe`, and the difference is what each reads.** `_unframe` parses first, so a
        key outside the language comes back whole and no frame is reached. This splits TEXT, which
        is the totality `fold_cycles` is held to: a caller scrubs run ids to `{run}` before
        comparing two runs (`agent.lineage.canonical`), and a scrubbed key must still fold. Over
        the banked corpus **465 of 828 names do not parse** (2026-09-12; the corpus re-banks
        routinely, so recount).

            gather:0,0;{run}:m   ->  gather:*,*;{run}:m

        Both walk left to right for the first suffix a variant claims, so the stopping rule is the
        same and only the reader differs.

        An unclaimed term is emitted as it stands and the walk continues; a term that decodes as a
        WHOLE ends it.

        **What the registry claims is the whole reach**, including behind `step;`, whose payload is
        the author's own key. `step;code:action,3,run;tool:fetch` projects its action index
        because `code:action,{},{};tool:{}` is a registered variant declaring that coordinate an
        `Index` (`code._action_key`), so the substitution is the author's own declaration read
        back. `step;myown:thing` and `step;ledger:x` come back whole. That is `STEP_ARM`'s rule
        holding rather than bending: it forbids a structural rule REFUSING on what an author put
        behind the arm, and a projection refuses nothing.

        **A wrapping FOREIGN term is peeled and its payload projected** (`_peel`), because the
        payload is OUR terms: a branch's park reads `$awaitEvent:gather:0,1;ev:x`, so the
        coordinate saying which branch sits inside the name Absurd wrote."""
        match self._peel(base, roles):
            case (str() as peeled, bool() as claimed):
                return peeled, claimed
            case None:
                pass  # not a wrapper: the term walk below answers
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
        terms = base.split(TERM_SEPARATOR)
        out: list[str] = []
        # The LAST segment decides, because frames are a prefix and the identity is the question.
        # `task:t7;step;myown:3` has a claimed frame in front of a namespace nobody registered, and
        # reporting the frame's answer would say the map ruled on a key it could not read.
        claimed = False
        for index, term in enumerate(terms):
            rest = TERM_SEPARATOR.join(terms[index:])
            if _has_foreign(term):
                # A wrapper reaches here too, under any frame whose shape carries no trailing
                # splice: `gather:{},{};{}` recurses into one and `d:{}` does not, so asking here
                # is what keeps the peel from depending on which frame happens to precede it.
                match self._peel(rest, roles):
                    case (str() as peeled, bool() as under):
                        out.append(peeled)
                        claimed = under
                    case None:
                        out.append(rest)  # the other runtime's own payload, in its own vocabulary
                        claimed = False
                break
            if self._decode_one(rest) is not None:
                projected, claimed = self._project_term(rest, roles)
                out.append(projected)
                break
            projected, claimed = self._project_term(term, roles)
            out.append(projected)
        return TERM_SEPARATOR.join(out), claimed

    def _peel(self, text: str, roles: set[str]) -> tuple[str, bool] | None:
        """A wrapping foreign term's payload, projected, with the head kept as its SOURCE wrote it.

        A wrapping foreign term is a BARE TAG, which is the same fact `ParsedKey.wrapped`
        dispatches on: `$awaitEvent` has empty coordinates and the terms after it are its payload,
        where `$awaitTaskResult:{uuid}` holds its payload AS a coordinate and wraps nothing. So
        everything projectable sits in the payload and the head is bytes to keep.

        **Two readers, because a scrubbed key has no terms to read.** `wrapped` answers from the
        parse and is the authority. A caller comparing two runs rewrites run ids to `{run}` first
        (`agent.lineage.canonical`), which puts the key outside the language, and a park behind a
        wrapper would then keep the branch coordinate its unwrapped sibling drops. So off-language
        text peels by tag. It needs no guard beyond that: only a term some variant claims is ever
        substituted, so a payload in another vocabulary comes back byte for byte with `claimed`
        False, which is the answer `wrapped` gives on the parseable side."""
        payload: str | None
        try:
            payload = _wrapped_payload(parse(text))
        except KeySyntaxError:
            payload = _foreign_tail(text)
        match payload:
            case str():
                return _project_tail(text, payload, lambda: self._project_walk(payload, roles))
            case None:
                return None
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead

    def _project_term(self, key: str, roles: set[str]) -> tuple[str, bool]:
        """One key, re-rendered through the SHAPE that minted it, or returned whole if unclaimed.

        The flag is whether the map reached the INNERMOST key, which is the only reading that
        answers anything. Every op carries an arm (`op_key`), and every arm's variant is a
        trailing splice binding any payload at all, so "some variant decoded this" is true of
        every recorded key whatever sits behind the arm: `step;myown:3` decodes as `step;{}` while
        `myown:3` is a namespace nobody registered and its integer coordinate declares no role.

        Re-rendering keeps the projection from becoming a second reader of the grammar: the same
        `_render_skeleton` the registry labels shapes with puts the delimiters back.

        **A trailing SPLICE is projected too, by recursion.** `ledger;{event_id}` binds a whole
        key as one field, and that key carries roles of its own; leaving it verbatim was why a
        `fetched:{run_id},{index}` row kept a branch index the fold had every reason to drop.
        `ledger;`, `event;`, `step;` and `gather:{g},{i};` are all this shape.

        **An OMITTED optional stays omitted.** An optional coordinate renders only when the author
        passed it (`_bind_coordinates`), so `govern:review,r9;…` binds three of six fields. Filling
        the gaps from the skeleton would put coordinates on the wire that the producer never wrote,
        and marking them `*` would report a drop nobody asked for."""
        for shape in self.variants.get(_leading_tag(key), ()):
            if len(set(shape.fields)) != len(shape.fields):
                continue  # colliding field names cannot be re-rendered by name
            bindings = shape.decode(key)
            if bindings is None:
                continue
            values = iter(
                [
                    None
                    if field not in bindings
                    else (PROJECTION_SIGIL if role in roles else bindings[field])
                    for field, role in zip(shape.fields, shape.roles, strict=True)
                ]
            )
            rendered = _render_skeleton(shape.skeleton, partial(next, values, None))
            return self._recurse_into_splice(shape, bindings, rendered, roles)
        return key, False

    def _recurse_into_splice(
        self, shape: Shape, bindings: dict[str, str], rendered: str, roles: set[str]
    ) -> tuple[str, bool]:
        """Re-project a trailing splice's payload inside an already-rendered term.

        A term with no splice reported its own decode and is claimed; one WITH a splice reports
        its payload's answer, because binding a payload says nothing about reading it."""
        match shape.skeleton.elements[-1], shape.fields:
            case Splice(), (*_, last):
                payload = bindings.get(last, "")
            case _:
                return rendered, True
        match _project_tail(rendered, payload, lambda: self._project_walk(payload, roles)):
            case (str() as spliced, bool() as claimed):
                return spliced, claimed
            case None:
                return rendered, True
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead

    def to_json(self) -> str:
        return json.dumps(
            {
                "_": "DERIVED by `effective.lint --key-registry`. Do not edit.",
                "shapes": {
                    tag: {"variants": [_shape_to_json(v) for v in variants]}
                    for tag, variants in sorted(self.variants.items())
                },
            },
            indent=2,
        )

    def explain(self, key: str) -> Explanation:
        """Decode `key` into the named values it carries, and name its producing line.

        Dispatch over the namespace's VARIANTS, each decoding by structural recursion (see
        `Shape.decode`). At most one can match: the registry admits a set of variants only when
        `separated` proves them pairwise disjoint, so trying them in order is a lookup rather than
        a guess — no ordering-dependent "first plausible" reading.

        **The occurrence suffix is split off FIRST.** `#N` is appended after composition
        (`Key.occurrence`, the coordinate that keeps one settlement from settling the next), so
        no registered variant knows about it, and a trailing hole would swallow it in silence:

            approve:r1:tool:charge_card    ->  placed_key(op)='r1:tool:charge_card'
            approve:r1:tool:charge_card#2  ->  placed_key(op)='r1:tool:charge_card#2'

        That raises nothing: the source map is the first rung of the operator's diagnosis ladder,
        and it would report the *wrong op* for exactly the keys the coordinate exists to
        disambiguate. Splitting first makes `#N` a named part of what a key carries.

        One rule covers both producers of the suffix, because there is only one idea: the engines
        append it to a repeated STEP name (`sqlite.py`, and Absurd's own dup-name rule) and the
        substrate appends it to a repeated AUTHORITY name. Both mean *the n-th occurrence of this
        identity*, and both come from `Key.occurrence`. The bridges strip it with the same
        anchored pattern (`bridge_sqlite.py`, `bridge_absurd.py`), so the three readers agree."""
        # `key` keeps echoing what the caller passed — an operator who pasted a name off a
        # checkpoint row should see it back verbatim — while the decode runs on the base.
        base, occurrence = split_occurrence(key)
        # REACHABILITY, before unframing, because unframing is what destroys the evidence.
        # `_unframe` peels terms from the left until the remainder decodes, so it happily strips
        # `gather:0,0` AND `event` off `gather:0,0;event;step;tool:foo` and reports `step_key`'s
        # line — a producer, a source map entry and a caveat-free reading for a key that says
        # *await an event whose name is a checkpoint*. Nothing mints it. The composer's splice
        # `Domain` refuses this one template at a time; this refuses the finished bytes, which is
        # the half a per-hole declaration cannot reach because the frame in front may be honest.
        #
        # WELL-FORMED and unmintable, both words load-bearing. Text that is not in the language at
        # all keeps falling through to the tag lookup below, whose message says *that* — a reader
        # handed `ns:x:other:0` needs "no registered variant", not a lecture about op arms.
        if _in_the_language(base) and (why := unmintable(base)) is not None:
            raise UnknownTag(
                f"key {key!r} is well-formed and UNMINTABLE: {why}. A source map that answered "
                f"this would name a producer for a key no producer makes."
            )
        inner, frame = self._unframe(base)
        tag = _leading_tag(inner)
        variants = self.variants.get(tag)
        if not variants:
            raise UnknownTag(
                f"key {key!r} has namespace {tag!r}, which is not registered. Regenerate the map "
                f"(`just key-registry`), or this key was not minted by `compose_key`."
            )
        for shape in variants:
            bindings = shape.decode(inner)
            if bindings is None:
                continue
            # No caveat for a bound value containing a delimiter: a value cannot contain a
            # separator under this grammar, so the condition is unreachable by construction.
            caveats: tuple[str, ...] = ()
            return Explanation(
                key=key,
                site=shape.site,
                bindings=bindings,
                caveats=caveats,
                occurrence=occurrence,
                frame=frame,
            )
        shown = " | ".join(v.label() for v in variants)
        raise ValueError(
            f"key {key!r} matches no registered variant of {tag!r} ({shown}). Either the map is "
            f"stale (`just key-registry`) or the key was composed by a different template."
        )


def _shape_to_json(shape: Shape) -> dict[str, Any]:
    """A variant on the wire, as the TEMPLATE it came from with each hole written `{}`.

    Round-tripped through `parse_skeleton`, so the artifact cannot encode a shape the grammar
    could not produce — where the old form listed slots and a `tail`, an encoding with its own
    rules and its own drift. `{}` is unambiguous: `{` is not a legal atom character.

    **A splice's `domain=` is deliberately NOT persisted.** It constrains the producer, not the
    language, so it is not part of the shape a decoder needs — and one variant may be written at
    several sites (four mint `event;{}`) which need not declare the same domain. Writing one of
    them here would put a second source of truth in a derived file, where the losing site's
    declaration is invisibly gone. The declaration lives in the template; `compose_key` enforces
    it and `lint._declared_domain` reads it from source when a scan wants it."""
    return {
        "template": shape.template,
        "fields": list(shape.fields),
        "roles": list(shape.roles),
        "site": shape.site,
        "sites": list(shape.sites or (shape.site,)),
    }


def _shape_from_json(tag: str, raw: dict[str, Any]) -> Shape:
    text = raw["template"]
    pieces = text.split("{}")
    parts: list[Any] = [pieces[0]]
    for index, piece in enumerate(pieces[1:]):
        parts.extend([Hole(index), piece])
    return Shape(
        skeleton=parse_skeleton([p for p in parts if p != ""]),
        fields=tuple(raw.get("fields", [])),
        roles=tuple(raw.get("roles", [])),
        site=raw["site"],
        sites=tuple(raw.get("sites", [raw["site"]])),
    )


def _wrapped_payload(parsed: ParsedKey) -> str | None:
    """The payload a foreign term wraps, read off the parse."""
    match parsed.wrapped:
        case ParsedKey() as wrapped:
            return wrapped.render()
        case None:
            return None


def _foreign_tail(base: str) -> str | None:
    """The text after a foreign tag, for a key the grammar refuses."""
    head, sigil, tail = base.partition(TAG_SEPARATOR)
    if sigil and tail and FOREIGN_TAG.match(head):
        return tail
    return None


def _project_tail(
    text: str, payload: str, project: Callable[[], tuple[str, bool]]
) -> tuple[str, bool] | None:
    """Replace `text`'s trailing `payload` with its projection, keeping the head's bytes.

    The one move `_peel` and `_recurse_into_splice` both make, and they had drifted: the splice
    arm dropped the `claimed` its recursion computed. A head kept as its SOURCE wrote it carries a
    separator this code never has to choose, `$awaitEvent:`'s `:` included.

    `None` when the payload is absent or does not end the text, which is the caller's signal to
    emit what it already had. No registered shape reaches that arm: a parse renders its own bytes
    back and a binding is a substring of what it bound. It stands against a shape whose render
    normalises, where the alternative is a key assembled at an offset this function computed."""
    if not payload or not text.endswith(payload):
        return None
    inner, claimed = project()
    return text[: len(text) - len(payload)] + inner, claimed


def _has_foreign(key: str) -> bool:
    """Does any term carry a tag another grammar minted: the one shape a `;` join would forge."""
    try:
        return any(term.foreign for term in parse(key).terms)
    except KeySyntaxError:
        return False


def _in_the_language(key: str) -> bool:
    """Does `key` parse at all — the question that must be asked before "is it reachable?"."""
    try:
        parse(key)
    except KeySyntaxError:
        return False
    return True


def _leading_tag(key: str) -> str:
    """The namespace, from the bytes — the text before the first separator that ends a tag."""
    for index, char in enumerate(key):
        if char in (TAG_SEPARATOR, TERM_SEPARATOR):
            return key[:index]
    return key


def explain(key: str, *, path: Path | str = DEFAULT_PATH) -> Explanation:
    """Convenience: load the map and decode one key. For more than one key, hold a `KeyMap`."""
    return KeyMap.load(path).explain(key)


__all__ = [
    "DEFAULT_PATH",
    "Explanation",
    "KeyMap",
    "Shape",
    "UnknownTag",
    "explain",
    "separated",
]
