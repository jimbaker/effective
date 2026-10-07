"""Frames and the arms that bound them — one module, because the two are one idea.

A frame is a leading term, and a frame sequence ENDS where the op's own key begins: at the first
arm that is not itself a frame. So `past_frames` cannot be written without `ARM_TAGS`, and
`admits` cannot answer "is this an address" without stripping frames first. Splitting these into
`keys.frame` and `keys.arm` produced an import cycle in both directions — the import graph
reporting a fact about the domain rather than an accident of layout.

`ARM_TAGS` being CLOSED is what makes the questions here decidable: "could anything mint this?" is
a check over a finite alphabet, not a search.

`FramePosition` hands out the ordinals for ops identified by POSITION rather than by name — a
gather is the g-th gather in its frame, and nothing else can say which.
"""

from dataclasses import dataclass, field

from effective.keys.grammar import (
    ARITY_SEPARATOR,
    TAG_SEPARATOR,
    TERM_SEPARATOR,
    Domain,
    Key,
    KeySyntaxError,
    ParsedKey,
    parse,
)

ARM_TAGS = ("step", "event", "ledger", "artifact", "sleep", "gather", "race", "monitor")
"""The CLOSED set of op arms: the tags the substrate puts at the head of an op's own key.

Closed is what makes a rule over this set structural rather than a denylist: it is the op set
(`WorkflowOp`), which is closed by design, so a new arm is a change to the op set itself.
Compare `RESERVED_AUTHORITY_TAGS`, which is derived from the `AuthorityTag` declarations and
checked both ways by a lint precisely because it is not closed.

`monitor` names no op in the set. It stays reserved for the arrivals monitor the name is ruled
for, so no author claims it first."""


GATHER_ARM = "gather"
"""The arm that FRAMES — `STEP_ARM`'s sibling, and named for the same reason.

A gather's key is punctuation-shaped (`gather:{g},{i};`), and a reader that spells the tag and
its separators inline is one more place a grammar change has to reach. Spelling it once means a
grammar change moves one constant instead of a regex somebody has to find."""

RACE_ARM = "race"
"""The arm that frames a race's branch, `race:{r},{i};`, with its own ordinal beside a gather's.

A race also records a choice under `race:{r};…`, which carries one coordinate where a branch frame
carries two, so the two shapes never share bytes."""

FRAME_ARMS = (GATHER_ARM, RACE_ARM)
"""Arm tags that WRAP rather than identify — a frame, not an op's own key.

`gather` is reserved in `ARM_TAGS` like every other op's tag, and then never leads an identity:
`op_key(Gather())` raises, because a gather is placed by the walk and its BRANCHES are what get
keyed. What `gather:{g},{i};…` does is frame whatever follows, and a frame is TRANSPARENT to the
address/identity question — `gather:0,0;ev:r1` is an address (a branch's park, which `parked.py`
mints) and `gather:0,0;event;foo` is an identity, and the leading tag is the same in both.

So a domain rule looks PAST a frame; only `leads_with_an_arm` — the primitive, whose contract is
the leading tag and nothing else — does not."""

STEP_ARM = "step"
"""The arm whose payload is AUTHOR text, and therefore the one no REFUSAL may read past.

`step;{name}` wraps whatever the author called their step, and *that is the point*: the arm is
why no denylist is needed, because `step;ledger:x` is disjoint from `ledger:x` by construction. A
rule that refused an author's `ledger:x` inside a step's payload would put a denylist back, one
layer down.

A PROJECTION does read past it, on a different authority: `step;code:seg,0,answer` is a key
`code.py` composed, so its `seg` coordinate carries the role that site declared and a fold drops
it. What the arm bounds is what the SUBSTRATE may refuse, never what a payload's own mint says
about itself.

**And the precondition, because a shape match is not provenance.** The projection reads an
author's payload through whatever variant its bytes fit, which is right when the author IS the
registered site and a guess otherwise. A `tag:{}` variant declaring `Index` is where that bites:
`step;ask:alice` and `step;ask:bob` fit `ask:{}`, so a fold merges two steps an author meant to
keep apart and reports a cycle over a straight line. The registry knows how many there are, so
count them rather than naming one::

    uv run python -c "
    from effective.keys.registry import KeyMap
    print(sorted(t for t, vs in KeyMap.load().variants.items()
                 for v in vs if v.template == f'{t}:{{}}' and v.roles == ('index',)))"
    # 2026-09-15: ask compact d gen rec search

Six of the thirty single-hole variants, and every one an ordinary English word an author reaches
for. `code:action,{},{};tool:{}` and `code:seg,{},{}` carry a second term or a literal, so neither
can be hit by accident. Which of the six to rule out is an open decision."""

ADDRESS_ARMS = ("event", "ledger", "emit")
"""The bare qualifiers whose payload is a WIRE ADDRESS — the arms a reachability rule can read.

An await's payload is the event name an emitter delivers to; a ledger's is an `event_id`; an
emit's is the name being delivered to. All three are addresses somebody else must be able to
produce, so a key claiming one is a *checkpoint* claims something nothing does.

**`step` is deliberately absent** (see `STEP_ARM`), and so are `artifact`/`sleep`/`gather`/
`race`/`monitor`: those identify by COORDINATES, so they have no bare form and no payload to
police.
`emit` is here but not in `ARM_TAGS`: it is not an op's arm, it is the emitter's side of the same
address, and it carries the same obligation."""


def _leading_frames(key: str) -> tuple[tuple[int, ...], int] | None:
    """Indices of the leading FRAME terms and where the identity begins, or `None` off-language.

    One walk for both readers: `split_frames` names the halves, `past_frames` filters the first
    one in place.

    A wrapping FOREIGN term belongs to the identity, not to the frames — `$awaitEvent:` carries
    the key it wrapped, so the frames behind it are frames of *that* key, and a park inside a
    branch is still inside the branch.

    A frame carries a COORDINATE, so a bare `gather` qualifier stays an ordinary term. Removing
    that test also removes a refusal: the composer starts accepting `event;gather;foo` at a
    `domain=address` splice."""
    try:
        terms = parse(key).terms
    except KeySyntaxError:
        return None
    frames: list[int] = []
    at = 0
    while at < len(terms) - 1:
        term = terms[at]
        if term.foreign and term.wraps_payload:
            at += 1
            continue
        if term.tag in ARM_TAGS and term.tag not in FRAME_ARMS:
            break
        frames.append(at)
        at += 1
    return tuple(frames), at


def past_frames(key: str, *, drop: tuple[str, ...] = FRAME_ARMS) -> str:
    """`key` with the leading frame terms in `drop` REMOVED — the frames it does not know are kept
    where they stand, and the op's own identity is left whole.

    `drop` is the CALLER's vocabulary, the same shape `keys.unframed` takes and for the same
    reason: which frames to remove is a question about what the caller is asking. Every `src/`
    caller asks the domain question and passes `FRAME_ARMS`, the arms that WRAP.

    **It FILTERS the leading frame run; it does not truncate at the first frame it cannot drop.**
    Frames nest in any order — `agent.contrastbench` wraps a trial in `task:{id}` and a `descend`
    mints `d:{depth}` inside it — so a truncating walk would let one frame it must keep hide every
    frame behind it, and the projection would report dropping a coordinate that survived.

    **The identity boundary is an ARM, which is what makes the walk terminate honestly.** Frames
    precede the op's own key and `ARM_TAGS` is CLOSED, so the first arm that is not itself a frame
    ends the run — after which nothing is touched, and a `gather:` coordinate carried INSIDE an arm
    (`event;gather:0,0;ev:r1`, which is what a branch's park registers) is left alone because it
    was never in the leading run. A key with no arm at all is a bare authored name and comes back
    whole.

    **A frame whose tag is not in `drop` is KEPT, deliberately**, which is the safe direction.
    Nothing can infer that a stranger's `scoped(compose_key(t"vendor:{id}"))` counts executions
    rather than names them, so the walk keeps a coordinate it does not understand instead of
    merging positions that differ. The enumeration that needs to be closed is `ARM_TAGS`; what a
    coordinate COUNTS is a separate question, answered per coordinate by the role declared at its
    mint (`keys.marker.Index`).

    **Total over STRINGS, not just over the language.** A caller legitimately hands it text that
    is deliberately not a key: `effective.lineage.canonical` rewrites a scope token to the hole
    `{run}`, and `gather:0,2;ledger;{run}:m-fan` must still fold. Such text falls back to a walk
    over the term separator alone, where a frame boundary needs no more than that — a term ends
    at `;`, and there are no terms to read a wrapper from."""
    if (found := _leading_frames(key)) is None:
        return _past_frames_text(key, drop)
    frames, _identity_at = found
    parsed = parse(key)
    terms = parsed.terms
    removed = {at for at in frames if terms[at].tag in drop and terms[at].coordinates}
    kept = tuple(t for at, t in enumerate(terms) if at not in removed)
    return ParsedKey(kept, parsed.occurrence).render()


def _past_frames_text(key: str, drop: tuple[str, ...]) -> str:
    """`past_frames` for text the language refuses — the same walk over separators.

    Off-language text has no terms, so there is no wrapper to recognise; a `;` is all a frame
    boundary ever needed. Most of the banked and `build/` names arrive here; the count and
    the command that recomputes it are in `registry.KeyMap._project_walk`."""
    rest, kept = key, []
    while True:
        head, separator, tail = rest.partition(TERM_SEPARATOR)
        tag, marker, _coordinates = head.partition(TAG_SEPARATOR)
        if not separator or (tag in ARM_TAGS and tag not in FRAME_ARMS):
            return TERM_SEPARATOR.join([*kept, rest])
        if tag not in drop or not marker:
            kept.append(head)
        rest = tail


def split_frames(key: str) -> tuple[tuple[str, ...], str]:
    """`key` partitioned into its leading FRAME terms and the identity they wrap.

    The containment relation, which `past_frames` computes and then throws away: it walks the same
    leading run and returns only what survived a `drop`, so every consumer that wanted the frames
    themselves re-derived them by parsing the key again. `graphview.Node.frames` is the first
    reader that needs them as data rather than as a filter.

    **The partition loses no TERM**, and where nothing wraps it is positional too::

        TERM_SEPARATOR.join((*frames, identity)) == key    # no wrapping term

    A wrapping FOREIGN term rides with the identity while the frames behind it do not, so the two
    halves do not reconstruct that key by joining: the wrapper joins its payload with `:`, and
    the frames it stood in front of now precede it. `past_frames` drops exactly the frames named
    here that match its `drop`, in place. Both are pinned in `tests/test_grammar.py` over the
    generated corpus, which mints the wrapped case.

    **A term with no coordinate rides in `frames`, and that is a position rather than a claim.**
    `past_frames` keeps such a term whatever the caller asked to drop — a bare `gather` qualifier
    frames nothing, so there is nothing to remove — but it sits in the leading run all the same,
    ahead of the identity, and a partition that reconstructs its input has nowhere else to put it.
    A renderer reading `frames` as tree levels gets the level; a caller reading it as "these carry
    coordinates" must check, which is why the walk below tests `marker` rather than assuming it.

    Total over STRINGS for the same reason `past_frames` is: a key the language refuses still
    partitions, on a walk over the term separator alone. That walk is the FALLBACK — reading a
    wrapper needs terms — and the banked corpus is why it exists, since a `parse`-based split
    raises on most of its names."""
    if (found := _leading_frames(key)) is None:
        rest, frames = key, []
        while True:
            head, separator, tail = rest.partition(TERM_SEPARATOR)
            tag, _marker, _coordinates = head.partition(TAG_SEPARATOR)
            if not separator or (tag in ARM_TAGS and tag not in FRAME_ARMS):
                return tuple(frames), rest
            frames.append(head)
            rest = tail
    at_frames, at_identity = found
    parsed = parse(key)
    terms = parsed.terms
    # A `#N` counts the whole key, and a frame never carries one, so it rides on the identity.
    wrappers = tuple(t for at, t in enumerate(terms) if at < at_identity and at not in at_frames)
    return (
        tuple(terms[at].render() for at in at_frames),
        ParsedKey(wrappers + terms[at_identity:], parsed.occurrence).render(),
    )


def is_branch_frame(frame: str) -> bool:
    """Does one frame name a BRANCH, rather than wrap something else?

    | the frame        | carries                          |
    |------------------|----------------------------------|
    | `gather:{g},{i}` | a gather's ordinal, and a branch |
    | `race:{r},{i}`   | a race's ordinal, and a branch   |
    | `race:{r}`       | a race, framing its own choice   |

    So the arity decides, not the tag."""
    tag, marker, coordinates = frame.partition(TAG_SEPARATOR)
    return tag in FRAME_ARMS and bool(marker) and ARITY_SEPARATOR in coordinates


def branch_frames(key: str) -> tuple[str, ...]:
    """The frames of the innermost BRANCH `key` runs in, outermost first, empty on the main line.

    Every arm in `FRAME_ARMS` answers alike, since `is_branch_frame` decides each frame, so one
    added there needs no new reader. String in, string out, for `split_frames`'s reason: banked
    names the language would refuse still have to answer."""
    frames, _identity = split_frames(key)
    innermost = -1
    for at, frame in enumerate(frames):
        if is_branch_frame(frame):
            innermost = at
    return frames[: innermost + 1]


def leads_with_an_arm(key: str) -> bool:
    """Does `key` open with an op ARM — i.e. is it a checkpoint identity rather than an ADDRESS?

    The reachability question the grammar can answer on its own. An arm's payload is an ADDRESS:
    an await's payload is the event name an emitter delivers to, a ledger's is an `event_id`. A
    key that OPENS with an arm is an op's own identity, and no producer ever delivers an event to
    one — so `event;step;tool:foo` is well-formed and unmintable.

    **Only the LEADING tag, deliberately.** `event;approve;step;tool:charge-card` is real and
    legitimate: the permission layer composes the address `approve;{placed_key}`, and the placed
    key may itself be a step's. What makes that an address is that it opens with `approve`. A rule
    that refused an arm anywhere in the payload would refuse the substrate's own approvals.

    Total: text that is not in the language leads with nothing."""
    try:
        return parse(key).terms[0].tag in ARM_TAGS
    except KeySyntaxError, IndexError:
        return False


def admits(domain: Domain, payload: str) -> bool:
    """Is `payload` in `domain` — the one predicate the composer, the registry and the lint share.

    Decidable because `ARM_TAGS` is CLOSED: the key space is a qualification scheme over a finite
    tag alphabet, not a Turing-complete one, so "could anything mint this?" is a check rather than
    a search.

    **The leading tag decides, once frames are off the front.** Reading deeper than that would
    refuse the substrate's own approvals — `event;approve;step;tool:charge-card` is real, and its
    payload contains an arm. Reading less than that refuses a framed address: `gather:0,0;ev:r1`
    is a branch's park, which `parked.py` mints, and its leading tag is an arm's."""
    stripped = past_frames(payload)
    match domain:
        case Domain.ADDRESS:
            return not leads_with_an_arm(stripped)
        case Domain.IDENTITY:
            return leads_with_an_arm(stripped)
        case Domain.ANY:
            return True


def unmintable(key: str) -> str | None:
    """Why no producer could mint `key`, or `None` if one could — over the WHOLE key.

    The splice `Domain` decides one hole at composition time; this decides a finished key, at any
    reader, from its bytes alone. Both are needed and neither subsumes the other. A composer sees
    the template and can refuse before the bytes exist; a registry, a lint and an `explain` see
    only the bytes, and a key can reach them from a frame that is itself perfectly legitimate —
    `gather:0,0;event;step;tool:foo` is a good frame around a bad address, and no hole in
    `gather_frame` is wrong.

    **The rule: a bare ADDRESS ARM's payload may not itself open with an arm.** Nothing emits an
    event named after a checkpoint, so `event;step;tool:foo` is well-formed and unreachable.

    **And the scan STOPS at the first bare `step`**, because everything after one is a name the
    author chose and no structural claim survives it. `step;event;q` is a step somebody named
    `event;q` — odd, and minted by `step_key` the moment they ask for it. Reading past the step arm
    is how a reachability rule turns back into the denylist the arm term replaced.

    Decidable because both tag sets are CLOSED: the key space is a qualification scheme over a
    finite alphabet, not a Turing-complete one, so "could anything mint this?" is a check and not a
    search. Total: text that is not in the language gets its parse error back."""
    try:
        terms = parse(key).terms
    except KeySyntaxError as exc:
        return f"it is not in the language: {exc}"
    for index, term in enumerate(terms[:-1]):
        if term.coordinates:
            continue  # a frame or an identifying arm — it wraps, it does not address
        if term.tag == STEP_ARM:
            return None  # author territory from here on
        if term.tag in ADDRESS_ARMS:
            # `ParsedKey.render`, not `;`.join: a foreign term joins its payload with `:`, so a
            # hand-rolled join would quote bytes no producer wrote in the refusal below.
            payload = ParsedKey(tuple(terms[index + 1 :])).render()
            if leads_with_an_arm(past_frames(payload)):
                return (
                    f"the payload of the {term.tag!r} arm is {payload!r}, which opens with an op "
                    f"arm — an arm wraps an ADDRESS, and nothing emits an event named after a "
                    f"checkpoint, so no producer mints this"
                )
    return None


def scope_prefix(atom: Key) -> str:
    """The prefix a ``scoped(...)`` atom contributes to every key minted inside it: ``{atom};``.

    One named place, because the durable handler's ``_run_scoped``, the recorder's, the replay
    driver's and ``qualified_event_name`` on the emitter side all apply it. Spelling that join
    separately in each is how a key scheme drifts, and a scope prefix that drifts is a checkpoint
    nobody can find again. It returns ``str`` rather than ``Key`` because
    a prefix is a *fragment* — half of a name, which is precisely what ``Key`` refuses to be —
    and its one consumer is ``Key.prefixed``.

    **Which `f"{prefix}{name}"` sites are a render backend and which are composition** turns on
    the TYPE OF THE RIGHT OPERAND: a frame atom is backend (`frame_path`), a finished identity is
    composition (`Key.prefixed`), and an untyped name is a finding.
    """
    text = atom.stored()
    if TERM_SEPARATOR in text:
        # A frame may not contain the frame delimiter. Checked HERE rather than at `scoped()`
        # because this is the chokepoint every applier goes through — the durable handler, the
        # recorder, the replay driver and `qualified_event_name` — so no path can bypass it.
        #
        # `Segment` already refuses the delimiter, so nothing that COMPOSES its way here can
        # carry one. What can is the FORGE: this takes a `Key`, and `Key("a;b")` constructs —
        # a deliberate cast is caught by neither `ty` nor `--key-composition`. So this is not
        # defence in depth behind another check; it is the place a forged frame is caught, and
        # the parameter's type is not evidence that the value was composed.
        raise ValueError(
            f"scope atom {text!r} contains the frame delimiter {TERM_SEPARATOR!r}: a frame "
            f"cannot contain the character that separates frames, or this atom composes the same "
            f"path as a real nesting and replay cannot tell them apart. Nest the scopes instead — "
            f"`scoped(a, lambda: scoped(b, body))` — which is what expresses a path."
        )
    return f"{text}{TERM_SEPARATOR}"


def frame_path(path: str, atom: Key) -> str:
    """Extend a frame ``path`` by one ``atom`` — the accumulator ``scope_prefix`` is the
    single-atom rule of, and the dual of ``unframed``.

    Every segment ends at ``;``, which ``scope_prefix`` refuses inside an atom, so the whole path
    splits back to its atom list exactly. That is what licenses the concatenation as a render
    backend. A gather branch has no author atom, since ``op_key(Gather)`` raises, so
    ``gather_prefix`` composes its positional ``gather:{g},{i}`` atom and applies the same
    ``scope_prefix``.

    ``str`` in and out: a path is a fragment, which is what ``Key`` refuses to be.
    """
    return f"{path}{scope_prefix(atom)}"


@dataclass
class FramePosition:
    """The ordinals one THREAD OF CONTROL hands out to the ops whose identity is **positional**.

    Some ops are identified by a name the author chose (``Step``), some by their content
    (``StoreArtifact``). A ``Gather`` is identified by neither: it is *the g-th gather in this
    thread of control*, which is why ``op_key(Gather)`` refuses and lets the walk assign the
    coordinate. Three arms are in that category::

    - a **gather** is the g-th gather in the thread;
    - a **race** is the r-th race, counted apart from the gathers so each frame's tag names its
      ordinal's series;
    - a **sleep** is the n-th sleep. It has no author-given name, and its wake time is DATA
      rather than identity, so it stays a VALUE: the checkpoint's own state on Absurd, and
      ``available_at`` on both engines, which is what the scheduler reads.

    **Per thread of control, and the reason is concurrency, not taste.** Gather branches run under
    an ``asyncio.TaskGroup`` on the recording path and on threads when the durable ctx is
    ``concurrent_safe``, so a counter shared across branches would be handed out in *schedule*
    order: a different assignment on every run, and replay would look up a key the record does not
    have. A thread of control is single-threaded by construction, so its ordinals are
    schedule-independent. A gather branch starts a fresh one; a ``scoped`` body continues its
    parent's, because a scope is a namespace in the same thread, and a scope entered twice must not
    hand out one ordinal twice (`RecordingHandler._scope`, `DurableHandler._run_scoped`,
    `ReplayHandler._drive`, and the fork walks). **Every interpreter must agree on the numbering,
    or a recorded key stops matching the replayed one.**

    A thread's ordinal is safe *because* every key it names is qualified by the thread's path (its
    gather frames and scopes) before it reaches a store: the ordinal composes with the path, and
    the pair is globally injective. Such a counter is NOT safe for anything that escapes as a
    value; an artifact id is returned into the workflow and dereferenced globally, which is why it
    is content-addressed rather than counted. A positional ordinal never leaves the key.

    One object rather than an int on each handler: the numbering is a rule all three
    interpreters must agree on, and a rule three handlers implement separately is a rule three
    handlers can drift on."""

    gather: int = 0
    race: int = 0
    sleep: int = 0
    awaits: dict[Key, int] = field(default_factory=dict)
    """Per-NAME, unlike the three above, because an await's identity is *named* rather than
    positional — two asks on one name are the same question asked twice, and two asks on
    different names are unrelated. Counting them together would make the n-th ask depend on what
    else the thread happened to await."""

    def next_await(self, name: Key) -> int:
        """The n-th ask on ``name`` in this thread of control, **1-based** unlike the positional
        ordinals above, so that ``Key.occurrence`` is the identity at the first ask and a name
        asked once keeps the bytes it has always had. Nothing recorded is orphaned; only a repeat
        moves.

        This is the coordinate a ``Scope.SETTLEMENT`` namespace is declared to need: one answer
        settles ONE op-occurrence, and without it two occurrences compose one name and the first
        answer settles the second. Applied by ``placing`` rather than by the composer, because
        the two answerers of a ``descend`` grant compose the name independently — the substrate's
        arm and an author-supplied ``Grantor`` — and only the walk sits above both."""
        self.awaits[name] = self.awaits.get(name, 0) + 1
        return self.awaits[name]

    def next_gather(self) -> int:
        """The g of ``gather:{g},{i};``, post-increment, so a thread's first gather is 0."""
        g = self.gather
        self.gather += 1
        return g

    def next_race(self) -> int:
        """The r of ``race:{r},{i};``, post-increment, so a thread's first race is 0."""
        r = self.race
        self.race += 1
        return r

    def next_sleep(self) -> int:
        """The n of ``sleep:{n}``, post-increment, so a thread's first sleep is 0.

        Returns the ORDINAL and lets the caller compose, as ``next_gather`` does. Minting a key
        here would import the composer, and that import closes a cycle: the composer asks
        `admits`, which is defined above."""
        n = self.sleep
        self.sleep += 1
        return n


def unframed(key: str, *, tags: tuple[str, ...]) -> str:
    """Strip leading FRAMES off a qualified key, returning the canonical name: the dual of
    `scope_prefix`, and the only sound one.

    **Why this is not a right-split, and not a left-strip either.** A key's own terminal text may
    contain a separator (`artifact:message/rfc822,sha256-9f2c` carries a `/` inside a coordinate,
    since a MIME type IS `type/subtype`), so neither end of the string is a reliable boundary.
    What IS reliable is the tag: the key grammar's whole claim to recoverable position is that a
    canonical name *begins with a namespace tag*. So a frame boundary is only recognized where a
    known tag follows it.

    The rule: return the key from the EARLIEST position that both begins a known tag and sits at a
    frame boundary (offset 0, or just after a `TERM_SEPARATOR`). Earliest wins, so an artifact key
    matches at 0 and is returned whole rather than mistaken for a framed `rfc822:…`. No match
    returns the key unchanged, which is the right default: an unrecognized shape is not a frame.

    ``tags`` is the CALLER's vocabulary, deliberately, because "a known tag" is not global: the
    graph projection asks about node kinds, the measured bridge asks which checkpoints are
    non-Steps, and those are different questions over different tag sets. Passing it in is what
    keeps this one function honest for both instead of privileging one caller's list.

    **Not the same question as `grammar.past_frames`, and the difference is deliberate.** This asks
    *what op is this, whatever it sits under*, so it looks past every frame down to a known op tag.
    `past_frames` asks *which leading terms are frames of the kind I named*, and drops exactly
    those, which is what lets `fold_cycles` keep a `sub:{name}` scope while dropping a `rec:{i}`.

    **Why not `startswith(tag)`:** a prefix test cannot see a scope frame in front of the tag it
    is looking for, so `kind_of("rec:0;ledger;x")` answers `step` under one and `ledger` under
    this. A tag has no spelling to drift; a prefix carries a separator, and the separator is
    exactly what a grammar change moves."""
    boundaries = (0, *(i + 1 for i, ch in enumerate(key) if ch == TERM_SEPARATOR))
    return next((key[at:] for at in boundaries if key[at:].startswith(tags)), key)
