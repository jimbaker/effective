"""The markers a call site hands `compose_key`, and the two readers of an author's own name.

A marker is a promise checked at CONSTRUCTION, so the composer can read it off the value's
`str`-ness and never re-check: `Tag` promises the text names a namespace, `Segment` promises it
carries no separator, `AuthorityTag` adds the reach one answer has. `Key`, the output, is not a
marker and is not a `str` — it lives in `effective.keys.grammar` with the language it is a term of.
"""

from typing import ClassVar, Protocol, Self, runtime_checkable

from effective.keys.grammar import (
    ARITY_SEPARATOR,
    FOREIGN_TAG,
    OCCURRENCE_SIGIL,
    PATH_SEPARATOR,
    PROJECTION_SIGIL,
    TAG,
    TAG_SEPARATOR,
    TERM_SEPARATOR,
    Key,
    Kind,
    Scope,
    is_tag,
    kind_of,
    parse,
)


@runtime_checkable
class Role(Protocol):
    """What a coordinate role promises: a word for what the coordinate MEANS.

    Declared per COORDINATE rather than per tag, because one term mixes them:
    `govern:{gate},{run_id},pass-n={n}` names a gate, identifies a run and counts a pass.

    Structural, because the roles are two families: a `Segment` is a `str` and an `Index` is an
    `int`, with no useful common ancestor. A drop set is typed over this, so
    `KeyMap.project(key, drop=(Run,))` is checked and a misspelling reads as an error."""

    role: ClassVar[str]


class Tag(str):
    """A namespace tag: the leading terminal of a key, naming which grammar the key belongs to
    (`event`, `ledger`, `govern`, `hyp`, …). The module docstring tables the marker family.

    The leading position takes a non-empty static **or** a `Tag`, and both spellings exist because
    a namespace is sometimes a constant::

        compose_key(t"tool:{Segment(name)}")
        ->  tool:read_file                           # a literal, the ordinary case

        compose_key(t"{FORK_SCOPE}:{Segment(child)};{event_id:domain=address}")
        ->  hyp:c1;review:m1                         # a constant, spelled ONCE

    Forcing the literal form would mean writing every such namespace twice, once for readers and
    once for the composer, which is the drift `FORK_SCOPE`'s own comment warns about. What the
    composer needs from either is that the namespace be legible to it, so `registry` can register
    the shapes it mints, which a `Tag` gives it and a bare `str` does not."""

    __slots__ = ()

    def __new__(cls, value: str) -> Self:
        """A `str`, and one the GRAMMAR accepts: checked against `grammar.TAG`, not a denylist::

            Tag("a#2")  ->  is not a well-formed tag: lower-kebab matching ^[a-z]…
            Tag("a,b")  ->  is not a well-formed tag
            Tag("A")    ->  is not a well-formed tag
            Tag(a_key)  ->  TypeError: takes a str, not Key

        The type check holds because opacity is only as strong as the narrowest constructor
        accepting `object`. `str(a_key)` renders a `Key`'s REPR, and here that would land in the
        leading position of an identity.

        Checking against the grammar rather than fencing named bytes is bounded by the language
        instead of by what a reader remembered: every refusal above is a byte a two-character
        denylist let through."""
        if not isinstance(value, str):
            raise TypeError(
                f"Tag() takes a str, not {type(value).__name__}. A `Key` here would render its "
                f"REPR into the leading position of an identity — pass the tag's TEXT."
            )
        if not is_tag(value) and not FOREIGN_TAG.match(value):
            if (kind := kind_of(value)) in (Kind.UUID, Kind.DIGEST):
                raise ValueError(
                    f"Tag({value!r}) is a {kind.value}, so it is an ATOM and not a tag. It names "
                    f"one thing rather than a namespace of things — pass it as a `Segment` in a "
                    f"COORDINATE instead: t\"{{Tag('ns')}}:{{Segment(value)}}\"."
                )
            raise ValueError(
                f"Tag({value!r}) is not a well-formed tag: lower-kebab matching {TAG.pattern}, or "
                f"a foreign tag matching {FOREIGN_TAG.pattern}. Write the colon as a static — "
                f"t\"{{Tag('ns')}}:{{…}}\" — and keep every separator out of the tag itself."
            )
        return super().__new__(cls, value)


class Segment(str):
    """A key segment that cannot carry a delimiter, checked once at construction::

        Segment("a:b")   ->  contains the key delimiter ':'
        Segment("")      ->  an empty string is not a name or a value
        Segment(a_key)   ->  TypeError: is Key, not a str

    This is what lets `compose_key` do no escaping at all. **Structure alone would
    not be enough**: with two free holes and one delimiter, `tag:{a}:{b}` cannot recover
    `("m1", "x:r2")` from `("m1:x", "r2")`: both flatten to the same bytes. Holes that *cannot*
    contain a delimiter remove the ambiguity instead of hiding it, so escaping becomes a type
    obligation discharged where the value is created, and `ty` checks it at every boundary that
    takes a `Segment`.

    **The rule is uniform: the terminal hole is not exempt.** A name and a value are
    delimiter-free and non-empty in EVERY position, and delimiters appear only where the structure
    puts them.

    What the old exemption was really carrying, which is the part worth keeping: every user of it
    was a nested PATH being flattened in (`artifact_id`'s `{ct}:{digest}`, an await name, an
    `op_key`), and those splice as `Key`s by induction. A value that seems to *want* a delimiter
    is usually not a value: `sleep:`'s ISO timestamp looked like one and was DATA in an identity
    slot, which the positional key removes rather than respells. Where the value really is foreign
    and uncontrolled it DIGESTS (`handlers.base.digest_atom`; the worked case is an email
    provider's immutable message id, whose charset the provider does not guarantee).
    """

    __slots__ = ()

    def __new__(cls, value: str) -> Self:
        """A `str`, non-empty, delimiter-free. Anything else is refused here.

        **`Self`, not `Segment`**, so a role subclass keeps its own type: `Run("r1")` is a `Run`
        to `ty`, which is what makes a declared coordinate role worth declaring.

        **`str` and not `object`**, because `text = str(value)` on an arbitrary value is an
        implicit flatten of the one type that declines `__str__` precisely so that flattening it
        is an explicit act::

            compose_key(t"x:{Segment(compose_key(t'tagonly'))}:{1}")
            ->  "x:Key(_value='tagonly', _scope=None):1"

        A delimiter check catches that only sometimes: a *tagged* key's repr contains a `:` and is
        refused for the wrong reason, while a tag-only key's repr does not and passes. The check
        is an allowlist rather than a `Key`-shaped refusal, so an unknown type is refused by
        default.

        **An empty segment is refused** because it is an ambiguity rather than an absence:
        `Segment("")` let an arity-1 and an arity-2 variant both decode from one string while
        `separated` called them disjoint. An absent field is the absence of a GROUP, which the
        path expresses by not having it.
        """
        if not isinstance(value, str):
            raise TypeError(
                f"Segment({value!r}) is {type(value).__name__}, not a str. A `Segment` brands "
                f"TEXT as delimiter-free, so a non-str would have to be flattened first — and "
                f"for an opaque type that flatten yields the REPR, which would then be composed "
                f"into an identity. A `Key` and an `int` each belong in the hole DIRECTLY "
                f"(a `Key` splices by induction, an `int` cannot carry a delimiter), so wrap "
                f"neither. Anything else: render it at the call site, where the flatten shows."
            )
        text = str(value)
        if not text:
            raise ValueError(
                "Segment(''): an empty string is not a name or a value. A key is a path of "
                "`name:value` groups, and an empty segment is a value that is not one — it also "
                "makes two shapes of different arity decode the same key. Express an absent "
                "field by omitting its group, not by binding it to nothing."
            )
        if TAG_SEPARATOR in text:
            raise ValueError(
                f"Segment({text!r}) contains the key delimiter {TAG_SEPARATOR!r}: an interior key "
                f"segment must be delimiter-free or the composed key is not injective. "
                f"Use a {TAG_SEPARATOR!r}-free value (e.g. "
                f"{text.replace(TAG_SEPARATOR, '-')!r}). A value that WANTS a separator is a "
                f"nested key, so splice it as one, or a path, so compose its atoms."
            )
        if TERM_SEPARATOR in text:
            raise ValueError(
                f"Segment({text!r}) contains the frame delimiter {TERM_SEPARATOR!r}, which would "
                f"forge a `scoped(...)` boundary inside a key. Use a {TERM_SEPARATOR!r}-free "
                f"value (e.g. {text.replace(TERM_SEPARATOR, '-')!r}), or express the nesting as "
                f"a real `scoped(...)`, which is what the frame is for."
            )
        if PATH_SEPARATOR in text:
            raise ValueError(
                f"Segment({text!r}) contains the path delimiter {PATH_SEPARATOR!r}. A `Segment` "
                f"is ONE ATOM, and every separator is refused for one reason: an atom that could "
                f"hold one would make arity uncountable. A genuine path — a MIME type, a scope "
                f"chain — is several atoms, composed as one."
            )
        if ARITY_SEPARATOR in text:
            # The fourth separator, and the last to be fenced. `Segment` refused the
            # other three and let `,` through, so `Segment("s,activate")` was a two-coordinate
            # value wearing a one-atom brand. It was never *composable* — `Atom.of` refuses it at
            # fill time — but that is the check arriving in the wrong place: `Segment` exists so a
            # value is "validated once where it is created, not re-escaped at every composition"
            # (`compose_key`'s own words), and a promise the type does not keep is worse than no
            # promise.
            raise ValueError(
                f"Segment({text!r}) contains the arity separator {ARITY_SEPARATOR!r}, which ends "
                f"a COORDINATE. A `Segment` is ONE ATOM, so this value would either be refused "
                f"later by the lexer or, in a position that skipped it, add a coordinate the "
                f"template does not declare. Two coordinates are written as two holes."
            )
        if OCCURRENCE_SIGIL in text:
            # The fifth fence, and the one the grammar had not yet grown when the other four were
            # written. `#N` is a suffix, so `#` joined the set of
            # bytes that mean something, and this constructor did not hear about it —
            # `Segment("ask#2")` branded a name-plus-occurrence as one atom. Same shape as `,`
            # above and the same remedy: fence it where the value is CREATED, not where it is
            # composed. The general rule this pair now evidences: a new metacharacter in the
            # grammar is a new fence here, and the fences are what make the brand mean anything.
            raise ValueError(
                f"Segment({text!r}) contains the occurrence sigil {OCCURRENCE_SIGIL!r}, which "
                f"marks a REPEAT of a whole key and may appear only at the very end. A `Segment` "
                f"is ONE ATOM, so this value would forge an occurrence the producer never "
                f"counted. Use `Key.occurrence(n)`, which is the one way to say it."
            )
        if PROJECTION_SIGIL in text:
            raise ValueError(
                f"Segment({text!r}) contains the projection sigil {PROJECTION_SIGIL!r}, which "
                f"marks a coordinate a VIEW dropped. Only `KeyMap.project` writes one, and what "
                f"it returns is a `ProjectedKey` with no durable form, so a `Segment` holding "
                f"one would put a quotient into the record. Compose the value the run actually "
                f"had; take the projection afterwards."
            )
        return super().__new__(cls, text)


class Name(Segment):
    """A coordinate that NAMES a distinct position: a subagent, a state, a bench task's label.

    The role a fold keeps. `Segment` promises the text carries no delimiter; a `Name` says what
    the coordinate MEANS.

    Two of one name are two positions, so a fold that merged them would merge nodes the program
    distinguishes: `sub:planner` and `sub:critic` run different prompts and tools, where
    `tests/_funnel`'s `talk:amber` and `talk:birch` run one lane over different data."""

    __slots__ = ()
    role: ClassVar[str] = "name"


class Run(Segment):
    """A coordinate that identifies an EXECUTION: a run id, a task id, a fork child's id.

    The role a cross-run comparison erases. Two runs of one program agree everywhere except here,
    so asking whether they unify under scope renaming is answered by dropping exactly these
    coordinates, while a fold keeps them because within one run they name distinct positions.

    **This is what `agent.lineage.canonical` rewrote text to find.** It took scrub tokens from the
    caller and `str.replace`d them, which reaches into content and cannot tell a coordinate from a
    substring; the role says it at the mint instead."""

    __slots__ = ()
    role: ClassVar[str] = "run"


class Subject(Segment):
    """A coordinate carrying the DOMAIN's own value: a message digest, a request id.

    The default role, kept by every fold here, since dropping it would merge rows about
    different subjects. `project(key, drop=(Subject,))` will do it, and nothing in this repo asks
    for that. Spelled rather than implied, so a reader can see the author considered
    the question."""

    __slots__ = ()
    role: ClassVar[str] = "subject"


class Index(Segment):
    """A coordinate that COUNTS repetitions: a gather branch, a depth, a generation, a round.

    The role a fold drops.

    **An `int` or a `str`.** A coordinate is an index when its values index re-executions of ONE
    position. `rec:0` and `rec:1` run the same leaf; `tests/_funnel`'s `talk:amber` and
    `talk:birch` run the same lane. `talk:`'s values are names, so a class admitting only integers
    would leave a fold no way to drop them.

    **Index-versus-name is a tell, not the test**, and a machine's `d:{n}` is where the tell
    misleads. A coding machine's levels 3 and 4 may run DIFFERENT states, which looks like naming:
    the `state:` frame nested inside says which code runs, and `d:` counts only how many times the
    trampoline has gone round. They are minted together (`d:{n};state:{name}`) so each carries
    one job. Without the counter a re-entered `state:` frame is not injective, since a backedge
    (`TEST -> DRAFT -> TEST`) mints two identical keys: a repeated scope frame gets no occurrence
    suffix.

    On the tape an index is what keeps re-executions of one position apart, which is what makes a
    repeated key injective. `fold_cycles` drops it to recover the program's shape: `rec:0;leaf`
    and `rec:1;leaf` become one node carrying a count of 2.

    The `int` arm flattens, and it does so here rather than in `Segment`, which admits only a
    `str` so that a `Key` cannot arrive through `__str__`."""

    __slots__ = ()
    role: ClassVar[str] = "index"

    def __new__(cls, value: str | int) -> Self:
        return _numbered(cls, value)


def _numbered[T: Segment](cls: type[T], value: str | int) -> T:
    """The constructor `Index` and `Ordinal` share: a number, or a name standing in for one.

    Shared by CALL rather than by inheritance, because the roles dispatch by class pattern and a
    subclass answers its parent's arm. `case Index():` would take an `Ordinal` and re-mint it as
    an `Index`, which is the declaration silently changing on the way through."""
    match value:
        case bool():
            # `bool` is an `int`, and `True` counts nothing. Refused above the int arm, which
            # would otherwise render it `True` and pass the atom rule as a name.
            raise TypeError(f"{cls.__name__}({value!r}) is a bool, which counts nothing")
        case int():
            return Segment.__new__(cls, str(value))
        case str():
            return Segment.__new__(cls, value)
        case _:
            raise TypeError(
                f"{cls.__name__}({value!r}) is {type(value).__name__}: "
                f"a {cls.role} is an int or a str"
            )


class Ordinal(Segment):
    """A coordinate that numbers a DISTINCT position of its kind: the k-th sleep in its frame.

    `Index`'s sibling, and the distinction is which question the number answers. An `Index` counts
    re-executions of one position, so a fold drops it and `rec:0;leaf` and `rec:1;leaf` become one
    node. An `Ordinal` numbers positions that are genuinely different, so a fold keeps it: the
    second sleep in a workflow is not the second execution of the first.

    **A fold that drops one manufactures a fact.** `sleep:{n}` takes its number
    from `FramePosition`, which counts sleeps within a frame. Drop it and a workflow that sleeps,
    works, sleeps again and works again folds its two sleeps together, so the graph gains an edge
    from the second body back to the sleep: a loop over a tape that has none. A fold may show
    fewer nodes and never fewer facts, so it keeps an `Ordinal`.

    **The number alone cannot say which this is**, which is why the mint declares it. The same
    `FramePosition` counter reads as an ordinal in straight-line code and as an index inside a
    loop, and the bytes are identical. `gather:{g}` is the case where that bites hardest and is
    deliberately still an `Index`: dropping matches every pinned tape, and a coordinate nothing
    can decide keeps the behavior it has until something measures it.

    A SIBLING of `Index`, sharing its constructor by call: the roles dispatch by class pattern, so
    a subclass would answer `case Index():` and be re-minted as the thing it exists to differ
    from."""

    __slots__ = ()
    role: ClassVar[str] = "ordinal"

    def __new__(cls, value: str | int) -> Self:
        return _numbered(cls, value)


class AuthorityTag(Tag):
    """A `Tag` whose namespace carries AUTHORITY: an approval, a spend grant, a gate's park, a
    counterfactual's identity. The declaration is the point.

    Once every identity composes through `compose_key`, "the substrate mints this namespace" stops
    distinguishing anything — `review:` and `extracted:` are substrate-composed too, and they are
    author/domain names that need no fencing. What needs fencing is the narrow set whose names ARE
    the authorization, because an author-controlled `Step` landing on one is not a wrong checkpoint
    but an answer delivered to the wrong question. That is not visible syntactically, so the
    composition site declares it, and `--authority-tags` checks `RESERVED_AUTHORITY_TAGS` equal to
    the declarations in both directions.

    **`scope` is REQUIRED** (see `Scope`), because a default makes "somebody forgot"
    indistinguishable from a considered `ACCRUAL`, and a named opt-out is the whole value of
    declaring. Omitting it is a `ty` error at the declaration site, which is where the guarantee is
    written.

    **The attribute is dereferenced on every composition and every walk**, and this docstring said
    the opposite for months, which is most of why a later reader took `Scope` for an annotation
    and went looking for a lint to stand behind it. The chain::

        AuthorityTag(scope=…)  ->  compose_key reads `items[0].value.scope`  ->  Key._scope
                               ->  placing(): `if name.scope is Scope.SETTLEMENT`
                               ->  Key.occurrence(position.next_await(name))

    So the declaration is what makes the walk apply an occurrence coordinate at all, and
    `Scope.ACCRUAL` is the other branch rather than a blank. It reaches the runtime because the
    template handed `compose_key` a TYPED VALUE rather than the text of one: PEP 750 on the
    identity axis, and the reason `Key.scope` can be trusted where a text-keyed registry lookup
    could not.

    `scope` is read-only (the property has no setter) and survives `pickle` and `copy`, which a
    `str` subclass with a keyword-only `__new__` owes those protocols through
    `__getnewargs_ex__`.
    """

    __slots__ = ("_scope",)

    _scope: Scope

    def __new__(cls, value: str, *, scope: Scope) -> Self:
        tag = super().__new__(cls, value)
        tag._scope = scope
        return tag

    @property
    def scope(self) -> Scope:
        """How far one answer in this namespace reaches. Read-only — see the class docstring."""
        return self._scope

    # `__getnewargs_ex__` serves protocols >= 2 (which reconstruct through `__new__`, so the
    # keyword has to be supplied); `__getstate__`/`__setstate__` serve 0 and 1 and `copy`, which
    # rebuild through `str.__new__` and then restore state. A `str` subclass with a custom
    # constructor owes both, and defining `__slots__` without `__getstate__` is a hard pickle error
    # rather than a silent lossy copy.
    def __getnewargs_ex__(self) -> tuple[tuple[str], dict[str, Scope]]:
        return (str(self),), {"scope": self._scope}

    def __getstate__(self) -> Scope:
        return self._scope

    def __setstate__(self, state: Scope) -> None:
        self._scope = state


RESERVED_AUTHORITY_TAGS = (
    "approve",
    "govern",
    "gate-state",
    "budget-grant",
    "generation-grant",
    "depth-grant",
    "round-grant",
    "chain-grant",
    "fork",
    "hyp",
)
"""What an AUTHOR may not name an AWAIT: the tags whose names ARE the authorization, an approval,
a grant, a gate's park, a fork's identity.

An event delivered by one of these answers a substrate question, and Absurd delivers by name
first-emit-wins across the queue, so an author's await landing on one consumes it. A `Step` needs
no such fence: its key opens with its own arm, so `step;approve:x` cannot be `approve:x` and the
two regions are disjoint by construction. An await keeps the fence because it is addressed by its
EVENT NAME, which no arm touches; see
`tests/test_op_key_injectivity.py::test_an_author_may_not_park_on_a_substrate_authority_name`.

Matched as a parsed leading TAG, not as a `"approve:"`-style prefix, so there is no spelling to go
stale. The arm tags are not here at all: an author's `ledger;x` composes `step;ledger;x` and is
disjoint by construction rather than by enumeration."""


def carries_structure(name: str) -> bool:
    """Does an author-supplied name denote a TERM SEQUENCE (`True`), or one atom (`False`)?

    Every wrapper of an author's name has to answer this, and both ask here so the two cannot
    answer differently. A wrapper supplies the tag, so a bare atom becomes a COORDINATE of it
    while a structured name splices in as its own terms::

        carries_structure("extract_ticket")    -> False
        step_key(...)                          -> step:extract_ticket
        _segment_key(..., seg=0)               -> code:seg,0,extract_ticket

        carries_structure("tool:charge-card")  -> True
        step_key(...)                          -> step;tool:charge-card
        _segment_key(..., seg=0)               -> code;seg:0;tool:charge-card

    The test is for a separator, not for "does it parse". A bare lower-kebab name parses perfectly
    well as a zero-arity tag, so a parse-first rule would spell `classify` one way and
    `extract_ticket`, which cannot be a tag at all, another: one kind of thing with two
    spellings, which is how a key scheme drifts.

    `/` is deliberately not in the set, because a path is structure inside ONE coordinate rather
    than a sequence of terms. So a name carrying one takes the single-atom path, where `Segment`
    then refuses it: `step_key("message/rfc822")` raises rather than composing."""
    return any(separator in name for separator in (TAG_SEPARATOR, TERM_SEPARATOR, ARITY_SEPARATOR))


def authored_key(name: str) -> Key:
    """An author's own name, turned into a `Key` without letting the author write an occurrence.

    This is the AUTHOR boundary; `Key.parse` is the READ boundary, and the difference is `#N`::

        Key.parse("tool:charge-card#2")      ->  tool:charge-card#2   # the engine wrote it
        authored_key("tool:charge-card#2")   ->  refused: an occurrence is not the author's

    A checkpoint name out of a store legitimately carries `#N`, because the engine put it there.
    An author may not. `step_key("tool:charge-card").occurrence(2)` composes
    `step;tool:charge-card#2`, so an author writing that suffix by hand would name the engine's
    second ask and be served a value committed for a different op, with nothing to signal it.

    The occurrence has exactly one producer, `Key.occurrence`, and the composer holds every other
    route shut: `Atom.of` refuses `#` in an atom, so the only way past is a SPLICE, which does not
    go through `Atom.of`. Every site that splices an author's name calls this function
    (`handlers.base.step_key`, `code`'s segment and action minters, `ops.event_name`, and
    `api.qualified_event_name`, the emitter's side of an await). A check repeated per call site
    is a denylist bounded by the call sites somebody remembered.

    A terminal splice may still hoist an occurrence out of the key it splices, and
    `placed_await_name` relies on that: the n-th ask of an await name IS the n-th instance of that
    identity, and there the substrate composed the suffix. This refusal is about who wrote it.

    The returned `Key` did not come from `compose_key`, so it carries no `scope` and the shape
    registry never saw it. It is a name to splice, not evidence about what minted it."""
    parsed = parse(name)
    if parsed.occurrence is not None:
        raise ValueError(
            f"the name {name!r} carries occurrence {parsed.occurrence}, and an occurrence is not "
            f"an author's to write — it is a use-index the RUNTIME assigns to a finished name, "
            f"with exactly one producer. Writing one here would alias the engine's own count for "
            f"that name and serve its committed value to a different op. Name the identity "
            f"({name.rsplit(OCCURRENCE_SIGIL, 1)[0]!r}) and let the substrate count it."
        )
    return Key(parsed)
