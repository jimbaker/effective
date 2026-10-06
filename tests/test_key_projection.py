"""`KeyMap.project`: the quotient a projection can take over STORED keys.

ROLE: adversarial. The role declared at the mint is only worth declaring if something
downstream reads it back off the tape, and these are that reading. What they pin is the join: the
bytes say where a key's coordinates END, the registry says what each one MEANS, and neither alone
is enough to drop a run id while keeping a gate's name in the same term.
"""

import operator
from functools import partial
from itertools import pairwise

import pytest

from effective.graphview import fold_cycles, from_keys, project, regroup
from effective.keys import Index, Name, Ordinal, Run, Subject
from effective.keys.grammar import (
    PROJECTION_SIGIL,
    TAG_SEPARATOR,
    KeySyntaxError,
    parse,
)
from effective.keys.registry import KeyMap, _foreign_tail, _project_tail, _wrapped_payload

pytestmark = pytest.mark.adversarial

GOVERN = "govern:review,r9,generation=0,pass-n=1,occurrence=0;step;tool:pay"


@pytest.fixture(scope="module")
def keymap() -> KeyMap:
    """The REAL registry, built from `src/`, because a synthetic one would pin the test's own
    idea of what production declares."""
    return KeyMap.load()


def test_one_term_keeps_its_name_and_drops_its_run(keymap):
    """The case a per-tag drop set structurally cannot express.

    `govern`'s single term carries a gate's name and a run identity, so a per-TAG drop set has
    only two answers, both wrong: keep the term and the two runs stay apart, drop it and the gate
    goes with it. Per-coordinate roles have the third."""
    # Spelled as a TRANSFORMATION of the stored key rather than retyped, so the assertion says
    # which coordinate moved and cannot drift from the key it is about.
    dropped = "," + PROJECTION_SIGIL + ","
    assert keymap.project(GOVERN, drop=(Run,)).display() == GOVERN.replace(",r9,", dropped)


def test_two_runs_of_one_gate_become_one_key(keymap):
    """The quotient itself: `~_H` computed over tape text, with no minting process in the room."""
    other = GOVERN.replace(",r9,", ",r-other,")
    assert other != GOVERN
    assert (
        keymap.project(other, drop=(Run,)).display()
        == keymap.project(GOVERN, drop=(Run,)).display()
    )


def test_dropping_a_name_keeps_the_run(keymap):
    """The drop set is read, not assumed. Reversing it reverses which coordinate survives, which
    is what shows the projection is consulting roles rather than positions.

    Both names go, and the second is the splice being entered: `govern` binds `step;tool:pay` as
    one field, `tool:{name}` declares a name inside it, and a coordinate's role does not depend on
    how deeply the key it sits in was spliced."""
    gate_and_tool = GOVERN.replace(":review,", ":" + PROJECTION_SIGIL + ",").replace(
        ":pay", ":" + PROJECTION_SIGIL
    )
    assert keymap.project(GOVERN, drop=(Name,)).display() == gate_and_tool


def test_a_projection_lies_outside_the_key_language(keymap):
    """*A coordinate a VIEW may drop, the RECORD never may.*

    The grammar refuses `*`, so the language of projections and the language of keys are disjoint:
    whatever a projection reaches, it arrives as a view."""
    projected = keymap.project(GOVERN, drop=(Run,)).display()
    with pytest.raises(KeySyntaxError):
        parse(projected)


def test_an_unregistered_key_comes_back_whole(keymap):
    """The same default as an unruled scope: a key no variant claims is KEPT.

    A coordinate's meaning is settled at the mint site, so that is where the coverage gate asks
    for it; this function only reads what the registry was told."""
    assert keymap.project("nosuchtag:whatever", drop=(Run,)).display() == "nosuchtag:whatever"


def test_an_undeclared_coordinate_survives_the_drop(keymap):
    """`govern`'s three integer coordinates declare no role yet, and a fold that dropped them
    because it could not read one would merge generations the program distinguishes."""
    kept = keymap.project(GOVERN, drop=(Run, Name, Subject)).display()
    assert "generation=0,pass-n=1,occurrence=0" in kept


def test_the_occurrence_suffix_survives_the_projection(keymap):
    """`#N` is the grammar's terminal suffix rather than a template field, so no shape knows about
    it and the projection has to put it back. `fold_cycles` drops it separately, on purpose: it is
    a different quotient, and stacking them is the caller's choice."""
    assert keymap.project("task:r1#3", drop=(Name,)).display() == "task:r1#3"
    assert (
        keymap.project("task:r1#3", drop=(Subject,)).display() == "task:" + PROJECTION_SIGIL + "#3"
    )


def test_a_frame_atom_is_projected_too(keymap):
    """A `scoped(...)` frame is a sequence of composed keys, so each atom decodes on its own. A
    run id in a frame is as much a run id as one in the payload."""
    framed = "hyp:c1;reviewed:m1"
    dropped = ":" + PROJECTION_SIGIL + ";"
    assert keymap.project(framed, drop=(Run,)).display() == framed.replace(":c1;", dropped)


def test_the_other_runtimes_own_payload_is_left_whole(keymap):
    """`$awaitTaskResult:{uuid}` carries its payload as the foreign term's COORDINATE, in a
    vocabulary that is not ours, so `ParsedKey.wrapped` answers `None` and there is nothing to
    project. `claimed` is what says the map could not rule rather than that it found nothing."""
    foreign = "$awaitTaskResult:0f2b1c3d-4e5f-4a6b-8c9d-0e1f2a3b4c5d"
    projected = keymap.project(foreign, drop=(Run, Subject, Index))
    assert projected.display() == foreign
    assert projected.claimed is False


def test_an_ordinal_survives_the_drop_that_takes_an_index(keymap):
    """The two numbers a key can carry, and the reason `Index` alone could not say which.

    `sleep:{n}` numbers the k-th sleep in its frame, so two sleeps are two positions and a fold
    that merged them would draw an edge back from the second to the first: a loop the program
    does not contain. `rec:{i}` numbers re-executions of ONE position, so a fold merges them and
    recovers the loop the program does. The bytes are identical in both."""
    assert keymap.project("sleep:0", drop=(Index,)).display() == "sleep:0"
    assert keymap.project("rec:0", drop=(Index,)).display() == "rec:" + PROJECTION_SIGIL
    assert keymap.project("sleep:0", drop=(Ordinal,)).display() == "sleep:" + PROJECTION_SIGIL


def test_two_straight_line_sleeps_do_not_fold_into_a_cycle(keymap):
    """The graph a reader would see if the sleep's coordinate were declared `Index`.

    Declared `Index`, the two sleeps merge and the folded graph runs
    `sleep -> a -> sleep -> b`, which reports a loop over a tape that has none. `Ordinal` keeps
    the fold from inventing that fact: a projection may hide what the record holds, never add to
    it."""
    tape = ["sleep:0", "step:a", "sleep:1", "step:b"]

    def position(key: str) -> str:
        return keymap.project(key, drop=(Index,)).display()

    folded = regroup(from_keys("r1", tape), position, dropped=("index",))
    assert [n.key for n in folded.nodes] == tape
    # The edges spelled out rather than `has_cycle`, which answers yes-or-no where the defect is
    # WHICH edge: a back-edge into the first sleep is what a merged pair would have drawn.
    assert [(e.src, e.dst) for e in folded.edges] == list(pairwise(tape))


def test_a_projection_composes_with_itself_by_union(keymap):
    """The law that made `ProjectedKey` a type rather than a string.

    Taken as text, a second projection reads a key carrying `*`, which no variant claims, so the
    drop set silently did nothing. Holding the SOURCE makes the second call re-project the key it
    started from under both drop sets, so the law is how the type is built rather than a property
    it happens to have."""
    once = keymap.project(GOVERN, drop=(Index,))
    assert keymap.project(once, drop=(Run,)) == keymap.project(GOVERN, drop=(Index, Run))
    assert keymap.project(once, drop=(Run,)).drop == frozenset({"index", "run"})


def test_the_map_is_found_from_the_module_not_from_the_shell():
    """A library whose answer depends on where python was started is not a library. `fold_cycles`
    raised `FileNotFoundError` from any working directory but the repo root, which no caller can
    see from the call, and every `src/` consumer inherited it (`tui/app.py`, `graphlayout/demo`).

    An INSTALLED copy still has no map, because the wheel ships no derived `build/` tree; that is
    packaging the key registry, and this pins only the checkout half."""
    import subprocess
    import sys

    from effective.keys.registry import DEFAULT_PATH

    assert DEFAULT_PATH.is_absolute()
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "from effective.keys.registry import KeyMap; print(len(KeyMap.load().variants))",
        ],
        cwd="/",
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert int(done.stdout) > 0


def _step(tag: str, coordinate: str) -> str:
    """The key an author's `step("<tag>:<coordinate>")` mints, composed rather than spelled."""
    from effective.domain import CallTool
    from effective.handlers.base import op_key
    from effective.ops import Step

    name = TAG_SEPARATOR.join((tag, coordinate))
    return op_key(Step(name=name, op=CallTool(name="t", args={}, result_schema=str))).stored()


def _folds_behind_the_arm(keymap: KeyMap) -> list[str]:
    """Every `tag:{}` variant declaring `Index`, which is the family an author's step name can
    collide with. Derived, so a new one joins the pin instead of the blind spot."""
    return sorted(
        tag
        for tag, variants in keymap.variants.items()
        for v in variants
        if v.template == f"{tag}:{{}}" and v.roles == ("index",)
    )


def test_an_authors_step_name_is_read_through_whatever_shape_it_fits(keymap):
    """CHOSEN, not inherited, and the ruling on step payload provenance is owed.

    A projection reads behind the `step;` arm because it must. A code run's action index travels
    there as author text (`code:action,{},{};tool:{}` under `step;`), so a walk that stopped at the
    arm would make the substrate's own coordinate invisible.

    The cost is that a shape match stands in for provenance: an author's two differently-named
    steps merge, and the graph reports a cycle over a tape that has none. Parameterized over the
    DERIVED family rather than over the one example that found it, so a new single-hole `Index`
    tag reddens this instead of joining the blast radius unremarked."""
    from effective.graphview import fold_cycles, from_keys

    family = _folds_behind_the_arm(keymap)
    assert family, "vacuous: no single-hole Index variant, so this pins nothing"
    for tag in family:
        # Through `op_key`, so the witness is a key an author's `step(...)` really mints.
        names = [_step(tag, "alpha"), _step(tag, "beta")]
        folded = fold_cycles(from_keys("r1", names))
        assert [n.count for n in folded.nodes] == [2], tag
        assert folded.nodes[0].key == keymap.project(names[0], drop=(Index,)).display(), tag
        assert folded.cyclic is True, tag  # over a straight line, which is the harm

    # The one that is read correctly, because the author IS the registered site.
    actions = ["step;code:action,0,run;tool:fetch", "step;code:action,1,run;tool:fetch"]
    folded_actions = fold_cycles(from_keys("r1", actions))
    assert [(n.key, n.count) for n in folded_actions.nodes] == [
        ("step;code:action,*,run;tool:fetch", 2)
    ]

    # And a step whose name no variant fits is left alone and named as unread.
    plain = fold_cycles(from_keys("r1", ["step;review", "step;approve"]))
    assert len(plain.nodes) == 2
    assert plain.unclaimed == ("step;review", "step;approve")  # tape order, not sorted


def test_a_projected_key_has_no_durable_form(keymap):
    """*A coordinate a VIEW may drop, the RECORD never may*, held by the type rather than a rule.

    `Key.stored()` is the durable form; the absence of it here is what says a quotient has none,
    and `LedgerRow.event_id: Key` is where `ty` turns that into a refusal."""
    projected = keymap.project(GOVERN, drop=(Run,))
    assert not hasattr(projected, "stored")
    assert projected.source == GOVERN

    # The absence is the weak half: an implicit flatten is the route a quotient reaches a store by,
    # and every one of these is a spelling of `__str__`.
    for flatten in (str, "{}".format, lambda p: f"{p}", partial(operator.mod, "%s")):
        with pytest.raises(TypeError, match="no string form"):
            flatten(projected)


def test_a_spliced_payload_is_projected_too(keymap):
    """`ledger;{event_id}` binds a whole key as one field, and that key has roles of its own.

    Left verbatim, the two ledger rows of one `append_ledger` inside one gather stayed two program
    points because the branch index sat inside the payload, where no drop set reached it."""
    assert keymap.project("ledger;fetched:r1,0", drop=(Index,)).display() == "ledger;fetched:r1,*"
    # The pair written out, not interpolated: a key is composed or it is quoted, and a loop that
    # builds one hides which two strings the assertion is actually about.
    first = "gather:0,0;ledger;fetched:r1,0"
    second = "gather:0,1;ledger;fetched:r1,1"
    branches = {keymap.project(k, drop=(Index,)).display() for k in (first, second)}
    assert len(branches) == 1, "one call site executed twice is one program point"


def test_an_optional_left_out_stays_out(keymap):
    """`govern:`'s optionals render only when the author passed them, so most stored keys are
    short. Requiring every field to bind returned those whole, which is the common spelling going
    unprojected; filling them from the skeleton would be worse, since it would put coordinates on
    the wire that no producer wrote."""
    short = "govern:review,r9;step;tool:pay"
    assert keymap.project(short, drop=(Run,)).display() == "govern:review,*;step;tool:pay"
    assert keymap.project("machine:r1", drop=(Run,)).display() == "machine:" + PROJECTION_SIGIL


def test_an_unclaimed_term_does_not_hide_the_terms_behind_it(keymap):
    """`rank:` is a fixture's own tag; the `gather:` and `rec:` coordinates behind it are the
    substrate's, and a tag this map was never told about is no reason to stop reading."""
    projected = keymap.project("rank;gather:0,0;rec:0;step:shortlist", drop=(Index,))
    assert projected.display() == "rank;gather:*,*;rec:*;step:shortlist"


def test_a_key_the_grammar_refuses_still_projects_its_claimed_frames(keymap):
    """The property that earns a second walk beside `_unframe`, which parses before it reads.

    `effective.lineage.canonical` rewrites a run id to `{run}` so two runs compare, and `{` is in
    no atom charset. Most banked names do not parse, so a projection that required one would
    shut half the corpus out of the comparison it exists for."""
    with pytest.raises(KeySyntaxError):
        parse("gather:0,0;{run}:m")
    assert keymap.project("gather:0,0;{run}:m", drop=(Index,)).display() == "gather:*,*;{run}:m"


def test_a_wrapping_foreign_term_keeps_its_own_join_and_yields_its_payload(keymap):
    """A branch's park carries the coordinate saying which branch INSIDE the name Absurd wrote.

    `ParsedKey.wrapped` is what separates the vendor's tag from our terms, and `TAG_SEPARATOR` is
    the join its grammar chose where ours is `;`. Emitting the wrong one is the forge
    `--forged-join` refuses, and keeping the payload unread leaves two branches of one gather in
    two classes."""
    first = "$awaitEvent:gather:0,0;ev:x"
    second = "$awaitEvent:gather:0,1;ev:x"
    branches = {keymap.project(k, drop=(Index,)).display() for k in (first, second)}
    assert branches == {"$awaitEvent:gather:*,*;ev:x"}


def test_a_projection_that_drops_nothing_is_the_identity(keymap):
    """A key whose coordinates no drop set names comes back byte for byte.

    The separators are the ones the source wrote, including the `:` a foreign term's own grammar
    chose, because the walk splices the projected payload back into the original bytes rather than
    re-deriving where a separator goes."""
    for key in ("$awaitEvent:review:m1", "gather:0,0;$awaitEvent:review:m1", "step;tool:pay"):
        assert keymap.project(key, drop=()).display() == key


def test_a_frame_survives_a_payload_no_variant_claims(keymap):
    """A park behind a combinator frame, which is what an Absurd tape carries.

    The frame atom decodes on its own; whether the identity behind it does is a different
    question, and conflating them left a `d:`-framed await unprojected."""
    assert (
        keymap.project("d:0;$awaitEvent:review:m1", drop=(Index,)).display()
        == "d:*;$awaitEvent:review:m1"
    )


UNCHANGED = (
    ("the map never heard of this key", "nosuchtag:whatever", False),
    ("read, and a Subject is not a Run", "task:r1", True),
)


@pytest.mark.parametrize(
    ("key", "claimed"),
    [(key, claimed) for _id, key, claimed in UNCHANGED],
    ids=[i for i, _k, _c in UNCHANGED],
)
def test_an_unchanged_projection_says_whether_the_map_read_it(keymap, key, claimed):
    """`drop` is the request and the diff is empty either way, so `claimed` is what separates a
    coverage gap from a fact about the run.

    A cross-run comparison that cannot tell them apart reports two runs equivalent when neither
    key decoded."""
    projected = keymap.project(key, drop=(Run,))
    assert projected.display() == projected.source
    assert projected.claimed is claimed


FRAMED_PARK = (
    (
        "a frame whose shape carries a trailing splice",
        ("gather:0,0;$awaitEvent:gather:0,0;ev:x", "gather:0,1;$awaitEvent:gather:0,1;ev:x"),
        "gather:*,*;$awaitEvent:gather:*,*;ev:x",
    ),
    (
        "a frame whose shape carries none",
        ("d:0;$awaitEvent:gather:0,0;ev:x", "d:1;$awaitEvent:gather:0,1;ev:x"),
        "d:*;$awaitEvent:gather:*,*;ev:x",
    ),
)


@pytest.mark.parametrize(
    ("tape", "folded"),
    [(tape, folded) for _id, tape, folded in FRAMED_PARK],
    ids=[i for i, _t, _f in FRAMED_PARK],
)
def test_a_park_behind_any_frame_reaches_its_branch_coordinate(keymap, tape, folded):
    """Which frame precedes a park decides nothing about whether its branch projects.

    The peel is reachable two ways, through a shape's trailing splice and through the walk's own
    foreign stop, and only one of them existed: a `gather:` frame recursed into the park while a
    `d:` frame emitted it whole, so two branches of one gather stayed two classes behind one frame
    and became one behind another."""
    assert {keymap.project(key, drop=(Index,)).display() for key in tape} == {folded}


def test_a_scrubbed_park_folds_like_its_unwrapped_sibling(keymap):
    """`canonical` rewrites a run id to `{run}` so two runs compare, which puts the key outside the
    language. `ParsedKey.wrapped` reads a parse, so a park behind a wrapper had no peel and kept
    the branch coordinate its unwrapped sibling drops."""
    with pytest.raises(KeySyntaxError):
        parse("$awaitEvent:gather:0,0;{run}:m")
    branches = {
        keymap.project(key, drop=(Index,)).display()
        for key in ("$awaitEvent:gather:0,0;{run}:m", "$awaitEvent:gather:0,1;{run}:m")
    }
    assert branches == {"$awaitEvent:gather:*,*;{run}:m"}


def test_a_fold_reports_the_keys_its_map_never_claimed(keymap):
    """`dropped` names what was ASKED for and `unclaimed` what was reached, so a fold over keys the
    registry has never heard of cannot report a quotient it did not take.

    The witness carries an ARM, which is the only shape production mints: `op_key` puts `step;` on
    every op and that variant is a trailing splice binding any payload, so a flag set by the
    outermost decode would be True here and the field would be empty on every recorded tape."""
    tape = ["gather:0,0;step;tool:a", "gather:0,1;step;tool:a", "step;myown:3", "mine:own"]
    folded = fold_cycles(from_keys("r1", tape), drop=(Index,), keymap=keymap)
    assert folded.dropped == ("occurrence", "index")
    assert folded.unclaimed == ("step;myown:3", "mine:own")
    assert fold_cycles(from_keys("r1", tape[:2]), drop=(Index,), keymap=keymap).unclaimed == ()

    # `()` says the map reached every key, so the projection that consults no map says `None`.
    assert project(from_keys("r1", tape)).unclaimed is None


def test_a_payload_that_is_not_the_texts_tail_splices_nothing():
    """`_project_tail` replaces a SUFFIX, so anything else is refused rather than spliced at a
    position it computed. Unit-level because no registered shape reaches the refusal: a parse
    renders its own bytes back and a binding is a substring of what it bound."""
    assert _project_tail("a;b", "b", lambda: ("X", True)) == ("a;X", True)
    assert _project_tail("a;b", "", lambda: ("X", True)) is None
    assert _project_tail("a;b", "a", lambda: ("X", True)) is None


def test_the_two_peel_readers_cannot_disagree_about_what_projects(keymap):
    """`_peel` reads a wrapper two ways, because a scrubbed key has no terms: `wrapped` answers
    from the parse, and text the grammar refuses peels by tag. They part on exactly one shape, a
    foreign term whose payload is a single ATOM, and an atom carries no tag for either to
    substitute. Both of these are shapes something mints: Absurd writes the first, and
    `effective.lineage.canonical` rewrites a run id to `{run}` to make the second."""
    absurd = "$awaitTaskResult:0192f0c0-0000-7000-8000-000000000000"
    assert _wrapped_payload(parse(absurd)) is None
    assert _foreign_tail(absurd) == "0192f0c0-0000-7000-8000-000000000000"
    assert keymap.project(absurd, drop=(Index,)).display() == absurd

    scrubbed = "$awaitTaskResult:{run}"
    with pytest.raises(KeySyntaxError):
        parse(scrubbed)
    assert keymap.project(scrubbed, drop=(Index,)).display() == scrubbed

    # A payload carrying a tag reaches both readers, and both hand back the same text.
    wrapping = "$awaitEvent:gather:0,1;step;tool:a"
    assert _wrapped_payload(parse(wrapping)) == _foreign_tail(wrapping)
    assert (
        keymap.project(wrapping, drop=(Index,)).display() == "$awaitEvent:gather:*,*;step;tool:a"
    )


ARM_LED = (
    ("an arm over a namespace nobody registered", "step;myown:3", "step;myown:3", False),
    (
        "an arm over one the substrate mints",
        "step;code:action,0,run;tool:fetch",
        "step;code:action,*,run;tool:fetch",
        True,
    ),
    (
        "a frame over an unregistered identity",
        "task:t7;step;myown:3",
        "task:t7;step;myown:3",
        False,
    ),
    ("a frame that FOLDS over one", "gather:0,1;ev:x", "gather:*,*;ev:x", False),
    (
        "a frame that folds over a registered identity",
        "gather:0,1;step;tool:a",
        "gather:*,*;step;tool:a",
        True,
    ),
    ("no arm at all", "mine:own", "mine:own", False),
)


@pytest.mark.parametrize(
    ("key", "display", "claimed"),
    [(key, display, claimed) for _id, key, display, claimed in ARM_LED],
    ids=[i for i, _k, _d, _c in ARM_LED],
)
def test_the_innermost_term_decides_whether_the_map_read_a_key(keymap, key, display, claimed):
    """Every arm variant is a trailing splice that binds any payload, so a flag set by the
    outermost decode is True for every key `op_key` mints and answers nothing. The last segment
    decides, because frames are a prefix and the identity is the question.

    The two columns are INDEPENDENT, which is what the fourth row is for: a frame can fold while
    the identity behind it goes unread, so `claimed=False` never means nothing moved."""
    projected = keymap.project(key, drop=(Index,))
    assert (projected.display(), projected.claimed) == (display, claimed)
