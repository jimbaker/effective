"""Pins for the coordinate roles: `Name`, `Run`, `Subject`.

A `Segment` promises delimiter-freedom; a role also declares what the coordinate MEANS, which is
what a projection reads. These pin the three properties that make the declaration worth having:
the roles compose, they inherit every refusal rather than restating it, and the registry records
the FIELD behind the wrapper so `explain` still names a producer.
"""

from pathlib import Path

import pytest

from effective.keys import Index, Key, Name, Role, Run, Segment, Subject, compose_key
from effective.keys.registry import KeyMap
from effective.lint import (
    check_coordinate_roles,
    check_coordinate_roles_source,
    check_python_codegen,
    registry_of,
)

ROLES = (Name, Run, Subject)


def test_a_role_composes_where_a_segment_does():
    # Spelled one per line rather than parametrized over the class: `--terminal-holes` reads
    # SOURCE TEXT, so an interpolated `{role(...)}` is an unwrapped terminal to it however the
    # value is typed. Writing the marker is also what an author would write.
    assert compose_key(t"tag:{Name('v1')}").stored() == "tag:v1"
    assert compose_key(t"tag:{Run('v1')}").stored() == "tag:v1"
    assert compose_key(t"tag:{Subject('v1')}").stored() == "tag:v1"


@pytest.mark.parametrize("role", ROLES)
def test_a_role_is_a_segment_and_keeps_its_own_type(role):
    # `Segment.__new__` returns `Self`, so the declaration survives into the type checker's view.
    value = role("r1")
    assert isinstance(value, Segment)
    assert type(value) is role


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize(
    ("bad", "fence"),
    [
        ("a:b", "key delimiter"),
        ("a;b", "frame delimiter"),
        ("a,b", "arity separator"),
        ("a/b", "path delimiter"),
        ("a#2", "occurrence sigil"),
        ("", "empty string"),
    ],
)
def test_a_role_inherits_every_refusal(role, bad, fence):
    # Inherited, not restated: a role that re-implemented the fence could drift from it. The
    # `match` names WHICH fence fired, so a role silently passing one value to another's branch
    # would fail here rather than read as green.
    with pytest.raises(ValueError, match=fence):
        role(bad)


@pytest.mark.parametrize("role", ROLES)
def test_a_role_refuses_a_non_str(role):
    with pytest.raises(TypeError):
        role(7)


def test_one_term_may_mix_roles():
    """The reason the declaration is per coordinate rather than per tag.

    Production's own case is `govern` (`src/effective/govern.py:239`), whose single term carries a
    gate's name, a run identity and two indices, so no per-tag table can rule on it. This invents
    its own tag: a second shape under a production namespace is invisible to `--key-registry`, and
    writing that namespace's template here would be the same borrow one layer over, which
    `just key-check` reads as prose."""
    shapes, problems = _shapes(
        """
def mint(gate: str, run_id: str) -> Key:
    return compose_key(t"mixed:{Name(gate)},{Run(run_id)}")
"""
    )
    assert problems == []
    (shape,) = shapes
    assert shape.roles == ("name", "run")


def test_the_registry_records_the_field_behind_the_role():
    """A shape is keyed by the author's field name, so the source map records what the author
    wrote and `explain` answers in the author's vocabulary rather than in markers."""
    shapes, problems = _shapes(
        """
def sub_scope(name: str) -> Key:
    return compose_key(t"sub:{Name(name)}")


def task_scope(task_id: str) -> Key:
    return compose_key(t"task:{Run(task_id)}")


def setpoint_id(request_id: str) -> Key:
    return compose_key(t"setpoint:{Subject(request_id)}")
"""
    )
    assert problems == []
    assert {shape.fields for shape in shapes} == {("name",), ("task_id",), ("request_id",)}


def test_the_grammar_erases_the_role_and_the_key_still_round_trips():
    """What a role is NOT: a coordinate on the wire.

    `Key.parse` reads the bytes, and the bytes of a `Run` are the bytes of a `Segment`, so the
    parsed key compares equal and carries no role. Reading one back is a registry lookup by
    `(tag, position)`, never an inspection of the key."""
    key = compose_key(t"task:{Run('r1')}")
    assert Key.parse(key.stored()) == key
    assert key == compose_key(t"task:{Subject('r1')}"), "differently-roled keys are one key"


def _shapes(*sources):
    """`registry_of` over one named source per site: the two-site cases need two names."""
    return registry_of({f"minters{index}.py": text for index, text in enumerate(sources)})


def _undeclared(source):
    """The coordinate names `--coordinate-roles` reports, in order."""
    return [v.text for v in check_coordinate_roles_source(source)]


ROLED = """
def mint(run_id: str, digest: str) -> Key:
    return compose_key(t"paid:{Run(run_id)},{Subject(digest)}")
"""

BOUND_A_LINE_EARLY = """
def mint(self_gate: str, run: str) -> Key:
    gate, run_id = Name(self_gate), Run(run)
    return compose_key(t"paid:{gate},{run_id}")
"""

REWRAPPED = """
def mint(lane: str) -> Key:
    return compose_key(t"probe:{Index(Name(lane))}")
"""


def test_the_registry_records_the_role_beside_the_field():
    """`paid:r1,d9` says on the wire where its coordinates end; nothing in those bytes says the
    first identifies an execution. The shape says it, which is why a fold over STORED keys can
    drop by role without the minting process being in the room."""
    shapes, problems = _shapes(ROLED)
    assert problems == []
    (shape,) = shapes
    assert shape.fields == ("run_id", "digest")
    assert shape.roles == ("run", "subject")


def test_an_undeclared_site_inherits_the_declared_role():
    """One variant, two sites, one of them converted. This is what lets the tree convert a file
    at a time instead of in a single 148-coordinate commit."""
    shapes, problems = _shapes(
        """
def a(run_id: str) -> Key:
    return compose_key(t"paid:{Run(run_id)}")
""",
        """
def b(run_id: str) -> Key:
    return compose_key(t"paid:{Segment(run_id)}")
""",
    )
    assert problems == []
    (shape,) = shapes
    assert shape.roles == ("run",)
    assert len(shape.sites) == 2


def test_two_sites_may_not_declare_different_roles():
    """A decode dispatches on the shape, so a coordinate meaning `run` at one site and `name` at
    another would have a fold drop it for one producer's keys and keep it for the other's, from
    stored text that cannot tell them apart."""
    _shapes_out, problems = _shapes(
        """
def a(run_id: str) -> Key:
    return compose_key(t"paid:{Run(run_id)}")
""",
        """
def b(gate: str) -> Key:
    return compose_key(t"paid:{Name(gate)}")
""",
    )
    (problem,) = problems
    assert problem.rule == "coordinate-role-conflict"
    assert "'run'" in problem.text
    assert "'name'" in problem.text
    assert "{gate}" in problem.text, "the coordinate, named as the reported site spells it"


def test_a_role_survives_the_persisted_registry(tmp_path):
    """Roles ride the channel `fields` rides, not the one `domain=` rides. `_shape_to_json` drops
    a splice's `domain=`, since that constrains a producer and one variant may be minted at sites
    declaring different domains; a role describes the language, so it persists beside the field."""
    shapes, _problems = _shapes(ROLED)
    artifact = tmp_path / "registry.json"
    artifact.write_text(KeyMap.from_shapes(shapes).to_json(), encoding="utf-8")
    (restored,) = KeyMap.load(artifact).variants["paid"]
    assert restored.roles == ("run", "subject")
    assert restored.fields == ("run_id", "digest")


def test_a_role_bound_one_line_early_is_resolved():
    """A hole's source expression becomes the registry's FIELD NAME, so `govern` binds its markers
    a line early to keep the field reading `gate` rather than `self.gate`. Reading the assignment
    is what lets a template have both."""
    shapes, problems = _shapes(BOUND_A_LINE_EARLY)
    assert problems == []
    (shape,) = shapes
    assert shape.fields == ("gate", "run_id"), "the field names the author chose"
    assert shape.roles == ("name", "run")


def test_a_name_assigned_two_roles_is_left_undeclared():
    """The file offers two readings and the ground to choose between them is outside this scan, so
    it declines, exactly as `_shape_of` declines a tag it cannot settle."""
    shapes, problems = _shapes(
        """
def a(x: str) -> Key:
    v = Name(x)
    return compose_key(t"one:{v}")


def b(x: str) -> Key:
    v = Run(x)
    return compose_key(t"two:{v}")
"""
    )
    assert problems == []
    assert {shape.roles for shape in shapes} == {("",)}


def test_a_role_behind_a_helper_call_is_left_undeclared():
    """One level is resolved and a helper's return value is not. Recording that hole as roleless
    would have the registry answer a question it could not see."""
    shapes, problems = _shapes(
        """
def role_for(x: str) -> Run:
    return Run(x)


def mint(run_id: str) -> Key:
    return compose_key(t"paid:{role_for(run_id)}")
"""
    )
    assert problems == []
    (shape,) = shapes
    assert shape.roles == ("",), "the depth limit, stated by the value it produces"


@pytest.mark.parametrize(
    ("source", "undeclared"),
    [
        (
            """
def mint(subject: str) -> Key:
    return compose_key(t"probe:{Segment(subject)}")
""",
            ["subject"],
        ),
        (
            """
def mint(n: int) -> Key:
    return compose_key(t"probe:{n}")
""",
            ["n"],
        ),
    ],
    ids=["segment-wrapped", "bare"],
)
def test_the_gate_reports_an_undeclared_coordinate(source, undeclared):
    assert _undeclared(source) == undeclared


@pytest.mark.parametrize(
    ("source", "nothing_to_declare"),
    [
        (
            """
def mint(subject: str, n: int) -> Key:
    return compose_key(t"probe:{Name(subject)},{Index(n)}")
""",
            "every coordinate already declares",
        ),
        (
            """
def mint(subject: str) -> Key:
    return compose_key(t"probe:{Name(subject)},{0}")
""",
            "an integer literal is its own identity",
        ),
        (
            """
def mint(subject: str, op_key: Key) -> Key:
    return compose_key(t"probe:{Name(subject)};{op_key:domain=identity}")
""",
            "a splice contributes whole TERMS",
        ),
    ],
    ids=["declared", "integer-literal", "splice"],
)
def test_the_gate_asks_only_of_coordinates(source, nothing_to_declare):
    assert _undeclared(source) == [], nothing_to_declare


def test_the_excluded_tree_is_measured_rather_than_described():
    """`NOT_COORDINATE_ROLE_SCANNED`'s docstring says every shipped tree is scanned. That is a
    claim like any other, so the table matching its declaration and the clean scan are both
    asserted here."""
    from effective import lint

    shipped = ("src", "examples")
    excluded = lint.NOT_COORDINATE_ROLE_SCANNED
    assert [k for k in excluded if k in shipped or k.startswith("src")] == []
    scanned = [f for root in ("src", "examples") for f in sorted(Path(root).rglob("*.py"))]
    assert {p.parts[0] for p in scanned} == {"src", "examples"}, "every root was reached"
    assert check_coordinate_roles(["src", "examples"]) == []


def test_a_role_wrapping_a_role_is_typed_by_the_outer_one():
    assert type(Index(Name("amber"))) is Index
    assert Index(Name("amber")) == "amber"


def test_a_rewrapped_coordinate_declares_by_its_OUTERMOST_marker():
    # `Subject(epoch_atom(until))` at `handlers/absurd.py` is the live case: the helper returns a
    # `Segment` and the mint says what it means.
    shapes, problems = _shapes(REWRAPPED)
    assert problems == []
    (shape,) = shapes
    assert shape.roles == ("index",)
    assert shape.fields == ("lane",), "the inner markers peel off the name the registry records"


def test_a_drop_set_is_typed_over_the_ROLE_protocol():
    """`ty` sees `Any` inside a `Template`, since `Template` is not generic over its
    interpolations, so a role in a key is invisible to a checker however it is annotated. Where a
    role travels as a VALUE it is checkable, and the drop set is where that matters:
    `project(drop=(Segment,))` is an error `ty` reports."""
    assert all(isinstance(role, Role) for role in (*ROLES, Index))
    assert not isinstance(Segment, Role), "the building block declares no meaning"


def test_the_four_roles_name_four_things():
    assert len({cls.role for cls in (*ROLES, Index)}) == 4


def test_the_codegen_rule_reports_an_assembled_module(tmp_path):
    module = tmp_path / "assembles.py"
    module.write_text(
        'NAME = "swallow"\nSRC = f"def {NAME}(op):\\n    return op\\n"\n', encoding="utf-8"
    )
    assert [v.rule for v in check_python_codegen([module])] == ["python-codegen"]


@pytest.mark.parametrize(
    ("line", "not_source"),
    [
        ('MSG = f"import of {name!r} inside a function body"', "a keyword opening a message"),
        ('MSG = f"class pattern with no resolvable class name: {p!r}"', "a keyword mid-sentence"),
        ('MSG = f"no `def {name}` under tests/"', "a message ABOUT source carries no lines"),
        ('SRC = "def mint(x):\\n    return x\\n"', "a literal, which is the shape we want"),
    ],
    ids=["import-in-prose", "class-in-prose", "quoted-def", "source-literal"],
)
def test_the_codegen_rule_asks_only_of_assembled_strings(tmp_path, line, not_source):
    module = tmp_path / "m.py"
    module.write_text(line + "\n", encoding="utf-8")
    assert check_python_codegen([module]) == [], not_source


def test_the_files_the_codegen_rule_skips_are_counted():
    """*Say "10 sites unscanned", never "tests are clean".* The exclusion is per FILE, so a new
    assembly in one of the four is not reported; the figure is what says how much that costs."""
    from effective import lint

    kept = dict(lint.NOT_PYTHON_CODEGEN_SCANNED)
    try:
        lint.NOT_PYTHON_CODEGEN_SCANNED.clear()
        roots = {"src", "tests", "examples", "scripts"} | {path.split("/")[0] for path in kept}
        unscanned = lint.check_python_codegen(sorted(roots))
    finally:
        lint.NOT_PYTHON_CODEGEN_SCANNED.update(kept)
    assert len([v for v in unscanned if v.filename.startswith("tests/")]) == 10
    assert {v.filename for v in unscanned} == set(kept)
