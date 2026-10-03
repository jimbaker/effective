"""Typed channels: PEP 750 t-string interpolations as bidirectional input/output channels.

An interpolation's ``value`` is a Channel object, its ``expression`` is the
field name, and the channel's type carries the write-back protocol. Input
channels render into the prompt; output channels declare what they need and
consume the response. Pydantic deserialization is the *default* convention;
``Gated`` runs a constraint check on the same seam (a neurosymbolic gate),
returning ``Repair`` to trigger a bounded re-prompt. Resolution lives in the
caller.
"""

import re
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain, count, groupby
from string.templatelib import Interpolation, Template
from typing import Any, Literal, Never, Protocol, assert_never, cast, get_args, runtime_checkable

from pydantic import TypeAdapter


class Output:
    """An ``Annotated`` marker that makes a *reusable type* an output channel.

    ``Money = Annotated[Decimal, Ge(0), Output]`` declares — once, as a contract on
    the type — the schema, the constraint (via ``annotated_types``/``AfterValidator``)
    and the direction. Interpolating such a type is the same as interpolating a
    ``Field``/``Gated`` object, but the guard rides the *type* (intrinsic) rather than
    a per-site lambda, and the repair reason is Pydantic's. ``render`` recognises it by
    the marker in the ``Annotated`` metadata.
    """


@dataclass(frozen=True)
class Done[T]:
    value: T


@dataclass(frozen=True)
class Repair:
    reason: str


type Resolution[T] = Done[T] | Repair


@runtime_checkable
class Channel[T](Protocol):
    """What an Interpolation.value holds: a typed slot in a prompt template."""

    def render(self) -> str: ...
    def write(self, name: str, response: Mapping[str, Any]) -> Resolution[T]: ...


_MISSING = object()


_adapter = lru_cache(maxsize=None)(TypeAdapter)  # build a TypeAdapter once per annotation


def _read[T](schema: Any, name: str, response: Mapping[str, Any]) -> Resolution[T]:
    """Pull ``name`` from the response and validate it against ``schema``: ``Done[T]`` or
    ``Repair``.

    One validator serves every schema a channel declares: a model, a scalar, ``list[Model]``, a
    union, an ``Annotated`` type whose constraints ride it. Real LLM output omits fields or sends
    the wrong shape, and a breach is a ``pydantic.ValidationError`` (a ``ValueError``), so it
    becomes a recoverable ``Repair`` to re-prompt with rather than a crash.
    """
    if (raw := response.get(name, _MISSING)) is _MISSING:
        return Repair(f"missing field: {name}")
    try:
        return Done(_adapter(schema).validate_python(raw))
    except (ValueError, TypeError) as exc:
        return Repair(f"invalid {name}: {exc}")


def output_annotation(value: Any) -> Any | None:
    """Return ``value`` if it is an ``Annotated[...]`` type carrying the ``Output``
    marker (so it is an output channel), else ``None``. The whole alias is returned
    because ``TypeAdapter`` understands it directly (constraints included)."""
    args = get_args(value)
    return value if args and any(meta is Output for meta in args[1:]) else None


@dataclass
class Field[T]:
    """Default output channel: deserialize ``response[name]`` into ``T``."""

    schema: type[T]

    def __post_init__(self) -> None:
        _adapter(self.schema)  # a schema pydantic cannot describe fails where it is written

    def render(self) -> str:
        return ""

    def write(self, name: str, response: Mapping[str, Any]) -> Resolution[T]:
        return _read(self.schema, name, response)


@dataclass
class Gated[T]:
    """Output channel that runs a constraint check on write-back."""

    schema: type[T]
    predicate: Callable[[T], bool]
    reason: str = "constraint violated"

    def __post_init__(self) -> None:
        _adapter(self.schema)

    def render(self) -> str:
        return ""

    def write(self, name: str, response: Mapping[str, Any]) -> Resolution[T]:
        match _read(self.schema, name, response):
            case Repair() as repair:
                return repair
            case Done(value):
                return Done(value) if self.predicate(value) else Repair(self.reason)
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead


@dataclass(frozen=True)
class TypedField[T]:
    """Output channel synthesised from an ``Output``-annotated *type*.

    The schema, constraints and any canonicalisation all ride the ``Annotated``
    alias (intrinsic), validated through one cached ``TypeAdapter``. This is the
    type-as-contract form of ``Field``/``Gated``: no per-site predicate, the repair
    reason is Pydantic's. ``Field``/``Gated`` objects stay for the genuinely *dynamic*
    predicate case (a per-call ``allowed`` set)."""

    annotation: Any

    def render(self) -> str:
        return ""

    def write(self, name: str, response: Mapping[str, Any]) -> Resolution[T]:
        return _read(self.annotation, name, response)


def as_channel(value: Any) -> Channel[Any] | None:
    """Classify an interpolation *value* as an output channel, or ``None`` (input).

    A ``Channel`` object is itself; an ``Output``-annotated type synthesises a
    ``TypedField``; anything else is an input rendered in place."""
    if isinstance(value, Channel):
        return value
    if (annotation := output_annotation(value)) is not None:
        return TypedField(annotation)
    return None


@dataclass(frozen=True)
class FormGate:
    """A cross-field constraint over the fully-resolved values (cf. Django's
    ``Form.clean``).

    `Field`/`Gated` validate one field each — a `Gated` predicate sees only its
    own value. A `FormGate` sees them all together, for invariants no single field
    can express ("not both X and Y empty"; "if X is set, Y must parse"; "amount ≤
    cap-for-this-category"). It runs *after* every field channel resolves and, on
    breach, returns a `Repair` that composes with the same re-prompt loop.
    """

    predicate: Callable[[Mapping[str, Any]], bool]
    reason: str


class SkillPin(Protocol):
    """A recorded content pin threaded into a render: what a pinned ``Skill``
    node resolves from instead of the registry. Structurally
    ``effective.skills.Pin``; the walk stays decoupled from the ops layer."""

    @property
    def name(self) -> str: ...
    @property
    def body(self) -> str: ...


@dataclass(frozen=True)
class Skill:
    """A named ``Template`` fragment, a node kind of the walk:
    interpolating ``{skill("pdf-processing")}`` splices the skill's disclosed
    body through the same recursive merge as a nested ``Template``, with the
    seam keyed by the *skill name* (you can see what composed).

    Unpinned, the body resolves lazily at ``render`` against the ``registry``
    argument — **whatever snapshot the deployment passed to this render**:
    defined, but unrecorded, and on the durable path a fresh worker after a
    park renders against *its* registry (worker-ambient). Use pins for
    anything durable. With ``pin=`` (an activation / refresh result the
    workflow threads explicitly), the body is the pin's *recorded* content: no
    registry consulted, replay-exact by construction."""

    name: str
    pin: SkillPin | None = None


def skill(name: str, pin: SkillPin | None = None) -> Skill:
    """The slot value for a skill disclosure: ``t"...{skill('da-mcp-query')}..."``
    (run-start snapshot) or ``t"...{skill('x', pin=p)}..."`` (pinned)."""
    return Skill(name, pin)


class Address(Protocol):
    """A durable record's name, as `effective.keys.Key` spells it; the walk stays decoupled from
    the key grammar."""

    def stored(self) -> str: ...


@dataclass(frozen=True)
class Citation:
    """Text quoted from a durable record, carrying the record's address.

    Interpolated, it renders as a data block labelled with the address, so a reader can decode the
    label to the row the text came from; the fence is the type's, and an explicit `:data` spec is
    refused. The address goes to the model and nothing the model writes becomes one: a caller
    reads the address off its own run, never out of a reply."""

    text: str
    address: Address


def cite(text: str, address: Address) -> Citation:
    """The slot value for quoting a record: ``t"...{cite(summary, session.outcome_id)}..."``."""
    return Citation(text, address)


class SkillResolver(Protocol):
    """What ``render`` needs from a registry: the disclosed body for a name.
    ``effective.skills.SkillRegistry`` satisfies this structurally; the walk
    stays decoupled from the filesystem loader (which owns load-time I/O)."""

    def body(self, name: str) -> Template: ...


type Role = Literal["system", "user", "assistant"]

_ROLES: frozenset[str] = frozenset(get_args(Role.__value__))  # a type stmt needs __value__


class ChannelError(Exception):
    """A structural error in a channel template, raised at ``render``: early and
    located, where the alternative is a silent mis-render at call time."""


class CacheOrderError(ChannelError):
    """The cache iron-rule is breached: a volatile (``nocache``) segment precedes a
    ``cache`` segment in the same role block, which poisons every downstream cache
    point (the cache iron-rule). Raised at ``render`` — located, early."""


class ChannelCollisionError(ChannelError):
    """Two interpolations across the composed tree claim the same channel name: a
    located render-time error, never a silent overwrite."""


class ChannelMismatchError(ChannelError):
    """The composed channels and the declared ``output`` model's fields disagree:
    every field needs a channel and vice-versa."""


class SkillResolutionError(ChannelError):
    """A ``Skill`` node cannot be disclosed: no registry was passed to ``render``,
    or the registry has no skill of that name — located at render, never a
    silent hole in the context."""


class SkillCycleError(ChannelError):
    """A skill's disclosed body (transitively) discloses itself — composition
    must stay a finite applicative tree."""


class DataSpecError(ChannelError):
    """A ``:data`` spec on a hole the prompt composes: a nested template, a disclosed skill, a
    citation or a declared channel. Only a value from outside the prompt is quoted, so the spec
    names a category error, raised where it was written."""


class IndependenceError(ChannelError):
    """The independence invariant: no node's rendering may read
    another node's *resolved* value. Within one render that is impossible by
    construction (channels resolve after the walk), so the detectable breach is
    a ``Done``/``Repair`` from a *prior* ``resolve`` arriving as an input — a
    turn boundary (control axis) masquerading as composition. The check covers
    **direct** input values (a resolution buried in a container renders as its
    repr and is on the author, like any other stringified object). Thread the
    unwrapped value deliberately if the dependence is the point: that is the
    monadic turn axis, and it belongs to the agent loop, not the template."""


_CONVERT: Mapping[str, Callable[[Any], str]] = {"r": repr, "s": str, "a": ascii}
"""The three conversions PEP 750 admits, keyed by the letter `Interpolation` carries. Applied
before formatting, as an f-string does, so `{text!r:data}` quotes the repr rather than the text."""

DATA_SPEC = "data"
"""The ``format_spec`` token marking a hole as content from OUTSIDE the prompt.

Trust is extrinsic, which is why it rides the slot beside role and cache rather than the channel:
the same `str` is authored prose in one template and a command's output in another, and only the
author of the site knows which. `effective.sql` draws the same line one grammar over, where a hole
binds as a parameter unless its spec says it is an identifier."""

DATA_MARK = "data"
END_MARK = "end"
"""The words a data block opens and closes with. `effective.envelope` writes the reply leg in the
same marker alphabet, so a model meets one convention in both directions."""

FENCES = (("[[", "##", "]]"), ("<<", "@@", ">>"), ("{{", "%%", "}}"), ("((", "~~", "))"))
"""The delimiters a block carries, in the order a render tries them.

The bracket and the mark change together, so a block nested inside another is read in a
different alphabet rather than counted. Repeating one bracket at every level is the `repeated
bracket` topology (`wiki/concepts/recursion-shapes.md`), which aliased spawn names, ledger rows
and approvals on the control axis.

No member shares a metacharacter with JSON, shell or code in a position that matters, so a body
needs no escaping at any depth."""

_MARK = re.compile(r"[\[<{(]{2}\s*(\S+)\s+(?:data|end)\b", re.IGNORECASE)
"""A delimiter of this family in the text, whatever bracket and mark it carries.

Spelled out rather than composed from the table, because a pattern built from a value is a
composition with a grammar of its own; `test_channel_data_spec.py` pins that the two agree."""


def _fenced(label: str, text: str) -> str:
    """`text` between delimiters that neither it nor `label` uses, with nothing escaped.

    The first fence whose mark is unused is chosen, so the same content always picks the same
    fence. Content carrying every fence falls back to the first counted mark it does not use, so a
    pathological input stays a block. `label` is one line, since a reader takes the opener as
    one."""
    used = {found.group(1) for found in _MARK.finditer(text + label)}
    counted = (("[[", "%" + str(n), "]]") for n in count())
    left, mark, right = next(fence for fence in chain(FENCES, counted) if fence[1] not in used)
    opened = f"{left} {mark} {DATA_MARK} {label} {mark} {right}"
    closed = f"{left} {mark} {END_MARK} {mark} {right}"
    return f"{opened}\n{text}\n{closed}"


def _declared(hole: Interpolation) -> Iterator[_Directive]:
    """The one directive a hole declares, or nothing."""
    if (directive := parse_directive(hole.format_spec or "")) is not None:
        yield directive


def directives(template: Template) -> Iterator[_Directive]:
    """Every directive a template declares, its nested templates included.

    Asked STRUCTURALLY, because inferring one from rendered output cannot tell an explicit
    `role=user` from an undirected hole, nor an explicit `nocache` from the default. A review
    measured both being accepted where they should be refused."""
    for item in template:
        match item:
            case str():
                continue
            case Interpolation(value=Template() as nested) as hole:
                yield from _declared(hole)
                yield from directives(nested)
            case Interpolation() as hole:
                yield from _declared(hole)
            case unreachable:
                assert_never(unreachable)


def _refuse_data(expression: str, kind: str) -> Never:
    raise DataSpecError(
        f"{{{expression}:{DATA_SPEC}}} marks {kind}, which the prompt composes rather than "
        f"quotes. Drop the :{DATA_SPEC} spec."
    )


@dataclass(frozen=True)
class _Directive:
    """The extrinsic intent parsed from an interpolation's ``format_spec``: a role override, a
    cache flag, and whether the value is data from outside the prompt. Never the type (type is
    intrinsic, on the channel)."""

    role: Role | None = None
    cache: bool | None = None
    data: bool = False


def parse_directive(spec: str) -> _Directive | None:
    """Parse a ``format_spec`` as a ``role=…;cache|nocache;data`` directive, or ``None``
    if it is an ordinary Python format spec (``.2f``, ``>10``) for an input value.

    A spec is a directive iff every ``;``-token is ``cache``/``nocache``/``data``/``role=R``.
    A ``role=`` token with an unknown role raises (located DX) rather than silently
    falling through to formatting.

    **A spec that mixes a directive with anything else raises**, because falling through would
    drop the directives silently: a review measured `{value:data;.2f}` reaching `__format__` with
    no fence at all, which is the one failure a guardrail may not have."""
    spec = spec.strip()
    if not spec:
        return None
    role: Role | None = None
    cache: bool | None = None
    data = False
    tokens = [token.strip() for token in spec.split(";") if token.strip()]
    directed = any(
        token in ("cache", "nocache", DATA_SPEC) or token.startswith("role=") for token in tokens
    )
    for tok in tokens:
        if tok == "cache":
            cache = True
        elif tok == "nocache":
            cache = False
        elif tok == DATA_SPEC:
            data = True
        elif tok.startswith("role="):
            r = tok.removeprefix("role=").strip()
            if r not in _ROLES:
                raise ValueError(f"unknown role directive {r!r} (use system|user|assistant)")
            role = cast(Role, r)  # narrowed by the _ROLES membership check above
        elif directed:
            raise ValueError(
                f"{tok!r} is not a directive, and this spec already carries one; mixing them "
                f"would drop the directives silently. Format the value before interpolating it."
            )
        else:
            return None  # an ordinary Python format spec, not a directive
    return _Directive(role, cache, data)


@dataclass(frozen=True)
class _Segment:
    role: Role
    text: str
    cache: bool


def _scoped(directive: _Directive | None, role: Role, cache: bool) -> tuple[Role, bool]:
    """The (role, cache) one interpolation renders under: its directive's
    overrides where present, else the inherited pair."""
    if directive is None:
        return role, cache
    return (
        directive.role if directive.role is not None else role,
        directive.cache if directive.cache is not None else cache,
    )


@dataclass(frozen=True)
class _Seam:
    """One seam entry on the walk's stream: ``name`` maps to the rendered text of
    exactly that node (for a composite, the joined text of just its sub-tree)."""

    name: str
    text: str


@dataclass(frozen=True)
class _ChannelDecl:
    """One declared output channel on the walk's stream: the field ``name`` and
    its channel object. Collision-checked at the reduction."""

    name: str
    channel: Channel[Any]


type _Item = _Segment | _ChannelDecl | _Seam


def _walk(
    template: Template,
    role: Role,
    cache: bool,
    *,
    registry: SkillResolver | None,
    disclosing: frozenset[str],
) -> Generator[_Item, None, str]:
    """Flatten a ``Template`` into one stream of role/cache-tagged ``_Segment``s,
    ``_ChannelDecl``s, and ``_Seam``s — and **return** this sub-tree's joined
    text, so ``yield from`` composes the stream while the return channel carries
    the sub-tree boundary a flat stream erases (the seam).

    A per-interpolation ``format_spec`` directive scopes role/cache to that
    interpolation's value (static prose and un-directived slots keep the inherited
    ``role``/``cache`` — ``user``/volatile by default, or the parent's under
    composition). ``disclosing`` carries the skill names on the current disclosure
    path (the cycle guard)."""
    parts: list[str] = []
    for item in template:
        # lint: totality(total) — the `continue` fall-through IS the `Interpolation` arm and
        # the body below handles it in full, so the walk is already total. Converting means
        # indenting
        # ~30 lines to gain no check the code does not already make.
        if isinstance(item, str):
            parts.append(item)
            yield _Segment(role, item, cache)
            continue
        directive = parse_directive(item.format_spec or "")
        seg_role, seg_cache = _scoped(directive, role, cache)
        seam_key = item.expression
        # Set by every arm, and read once below: a `:data` hole quarantines a value from outside
        # the prompt, and composition is the prompt's own structure. Naming it HERE rather than in
        # a second function is the fix for a drift a review found twice, once by adding a class
        # pattern and once by adding a guarded capture that a source-reading test could not see.
        composed: str | None = None
        match item.value:
            case Template() as sub:
                composed = "a nested template"
                # composition: recurse into the sub-prompt, inheriting
                # role/cache; the sub-tree's text comes back on the return channel
                rendered = yield from _walk(
                    sub, seg_role, seg_cache, registry=registry, disclosing=disclosing
                )
            case Skill() as node:
                composed = "a disclosed skill"
                # disclosure (step 6): resolve the body and ride the same recursive
                # walk, seam keyed by the skill *name*. The body inherits role/cache
                # — disclosed-as-uncached by default, so it sits after the cached
                # catalog as the volatile tail.
                seam_key = node.name
                rendered = yield from _disclose(
                    node, registry=registry, role=seg_role, cache=seg_cache, disclosing=disclosing
                )
            case Citation(text, address):
                composed = "a citation, fenced by its type"
                rendered = _fenced(address.stored(), text)
                yield _Segment(seg_role, rendered, seg_cache)
            case Done() | Repair():
                raise IndependenceError(
                    f"interpolation {item.expression!r} renders a "
                    f"{type(item.value).__name__} — a prior resolution re-entering a "
                    "render breaches independence (a turn boundary, not composition)"
                )
            case value if (channel := as_channel(value)) is not None:
                composed = "a declared channel"
                # declare before rendering: the reduction drives this generator
                # lazily, so a colliding name raises here and the duplicate's
                # render() never runs (matching the pre-streaming behavior)
                yield _ChannelDecl(seam_key, channel)
                rendered = channel.render() or f"<<{seam_key}>>"
                yield _Segment(seg_role, rendered, seg_cache)
            case value:
                # the directive consumed the spec; otherwise it is a Python format spec
                fmt = "" if directive is not None else (item.format_spec or "")
                converted = value if item.conversion is None else _CONVERT[item.conversion](value)
                rendered = format(converted, fmt)
                if directive is not None and directive.data:
                    rendered = _fenced(" ".join(item.expression.split()), rendered)
                yield _Segment(seg_role, rendered, seg_cache)
        if directive is not None and directive.data and composed is not None:
            _refuse_data(item.expression, composed)
        # the one emission tail: every node contributes its text and its seam
        # (a composite's segments already streamed out of the recursive walk)
        parts.append(rendered)
        yield _Seam(seam_key, rendered)
    return "".join(parts)


def _disclose(
    node: Skill,
    *,
    registry: SkillResolver | None,
    role: Role,
    cache: bool,
    disclosing: frozenset[str],
) -> Generator[_Item, None, str]:
    """Resolve a ``Skill`` node's body — from its pin (recorded content) when
    pinned, else from the registry snapshot — and walk it as a sub-tree. Every
    failure is a located render error: a missing registry, unknown name, or a
    pin/name mismatch is a ``SkillResolutionError``; a self-disclosure is a
    ``SkillCycleError``."""
    if node.name in disclosing:
        raise SkillCycleError(f"skill {node.name!r} discloses itself (via {sorted(disclosing)})")
    if node.pin is not None:
        if node.pin.name != node.name:
            raise SkillResolutionError(
                f"pin for skill {node.pin.name!r} threaded into skill {node.name!r}"
            )
        body: Template = Template(node.pin.body)
    elif registry is None:
        raise SkillResolutionError(
            f"skill {node.name!r} interpolated but render() got no registry"
        )
    else:
        try:
            body = registry.body(node.name)
        except LookupError as exc:
            raise SkillResolutionError(str(exc)) from None
    return (
        yield from _walk(body, role, cache, registry=registry, disclosing=disclosing | {node.name})
    )


def _check_cache_order(segments: Sequence[_Segment]) -> None:
    """The volatile-last cache iron-rule: within a role, no cached segment may follow
    a volatile one. Authoritative enforcement (the static ``just lint`` rule only sees
    single literals; cache flags compose dynamically across nested templates)."""
    seen_volatile: dict[Role, bool] = {}
    for seg in segments:
        if seg.cache and seen_volatile.get(seg.role):
            raise CacheOrderError(
                f"cached segment after a volatile one in role {seg.role!r}: {seg.text[:40]!r}"
            )
        if not seg.cache:
            seen_volatile[seg.role] = True


def _coalesce(segments: Sequence[_Segment]) -> Iterator[Message]:
    """Merge consecutive segments sharing ``(role, cache)`` into messages, after the
    volatile-last check: ``groupby`` the runs of one scope, then join each run's text
    once. (``groupby`` groups *consecutive* runs — exactly "coalesce adjacent"; a
    sort/dict grouping would fuse non-adjacent runs, a bug.) The last ``cache=True``
    message is the cache breakpoint the caller maps to a provider's ``cache_control``."""
    _check_cache_order(segments)
    for (role, cache), run in groupby(segments, key=lambda s: (s.role, s.cache)):
        yield Message(role, "".join(seg.text for seg in run), cache)


def _resolve_channels(
    channels: Mapping[str, Channel[Any]],
    response: Mapping[str, Any],
    form_gates: Sequence[FormGate],
) -> dict[str, Any] | Repair:
    """Write each output channel, then run the cross-field ``form_gates`` over the
    whole resolved dict. The first ``Repair`` (field-level, then form-level)
    short-circuits — a field you couldn't read is never form-checked."""
    results: dict[str, Any] = {}
    for name, channel in channels.items():
        match channel.write(name, response):
            case Repair() as repair:
                return repair
            case Done(value):
                results[name] = value
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
    for gate in form_gates:
        if not gate.predicate(results):
            return Repair(gate.reason)
    return results


@dataclass(frozen=True)
class Message:
    """One provider-neutral context segment. ``cache`` is the volatility boundary
    (the static/volatile seam); the *caller* maps it to a provider's
    ``cache_control`` so ``effective.channels`` stays provider-neutral."""

    role: Role
    content: str
    cache: bool = False


@dataclass(frozen=True)
class Prompt[S]:
    """The typed result of ``render``: the context (``messages``), the decode
    spec (``channels``), the ``seams`` surface, and the *declared* output signature
    ``S``. ``resolve`` recovers ``S`` end-to-end, so a composed prompt stays
    ty-legible even though the underlying ``Template`` is untyped."""

    messages: list[Message]
    channels: dict[str, Channel[Any]]
    seams: dict[str, str]
    output: type[S]

    def resolve(self, response: object, *, form_gates: Sequence[FormGate] = ()) -> S | Repair:
        """Resolve the channels (field + form-gate), then build and validate the
        declared ``output`` model. A field, form-gate, or model-level breach is a
        ``Repair``, uniformly re-promptable, and so is an answer that is not a mapping."""
        # lint: totality(coercion): a model's raw answer is untyped, so anything but a mapping
        # of fields is a Repair to re-prompt with.
        match response:
            case Mapping():
                values = _resolve_channels(self.channels, response, form_gates)
            case _:
                return Repair("the answer is not a mapping of field names to values")
        if isinstance(values, Repair):
            return values
        try:
            return _adapter(self.output).validate_python(values)
        except (ValueError, TypeError) as exc:  # model-level validators -> Repair
            return Repair(f"output {self.output.__name__}: {exc}")


def check_channels(channels: Mapping[str, Channel[Any]], output: type[Any]) -> None:
    """The signature validator: every field of the declared ``output`` model
    has a channel and vice-versa. Run inside ``render`` (early, located) and reusable
    as a test helper. Skipped for a non-Pydantic ``output`` (no fields to introspect)
    — the typed contract is then the wrapper alone."""
    fields = getattr(output, "model_fields", None)
    if fields is None:
        return
    declared = set(fields)
    present = set(channels)
    if declared != present:
        raise ChannelMismatchError(
            f"channels {sorted(present)} != fields of {output.__name__} {sorted(declared)} "
            f"(missing channels {sorted(declared - present)}; "
            f"extra channels {sorted(present - declared)})"
        )


def render[S](
    template: Template, *, output: type[S], registry: SkillResolver | None = None
) -> Prompt[S]:
    """The channel processor: a function on ``Template`` (tdom's ``html(t"…")``,
    psycopg3's ``execute(t"…")``), aimed at the context DSL. Walk the strings as
    prose and the interpolations as typed channels / inputs, and produce a typed
    ``Prompt[S]``. ``registry`` resolves any ``Skill`` nodes (their disclosed
    bodies splice through the same composition path). It performs no I/O, so a
    workflow calls it between yields; the determinism boundary holds for values
    whose ``__format__`` renders the same bytes on every run.

    This is the walk's one reduction: partition the stream into the ordered
    segments, the collision-checked channels, and the seams surface."""
    segments: list[_Segment] = []
    channels: dict[str, Channel[Any]] = {}
    seams: dict[str, str] = {}
    for item in _walk(template, "user", False, registry=registry, disclosing=frozenset()):
        match item:
            case _Segment():
                segments.append(item)
            case _Seam(name=name, text=text):
                seams[name] = text
            case _ChannelDecl(name=name, channel=channel):
                if name in channels:
                    raise ChannelCollisionError(f"duplicate channel name {name!r}")
                channels[name] = channel
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
    messages = list(_coalesce(segments))  # raises CacheOrderError before the signature check
    check_channels(channels, output)
    return Prompt(messages=messages, channels=channels, seams=seams, output=output)
