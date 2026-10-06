"""Tests for the determinism-boundary lint and the layer-authority lint."""

import ast
from pathlib import Path
from typing import Literal, get_args, get_origin

import pytest

from effective.domain import AsksModel, CallTool, ModelCall
from effective.lint import (
    DOMAIN_OP_CTORS,
    WORKFLOW_OP_CTORS,
    _import_root,
    _members,
    check_channel_source,
    check_deps_file,
    check_deps_source,
    check_file,
    check_layer_file,
    check_layer_source,
    check_source,
    seam_forbidden_for,
)


def test_clean_workflow_has_no_violations():
    # the test exemplar obeys the rules (dogfooding)
    assert check_file(Path(__file__).parent / "_approval_domain.py") == []


def test_bare_yield_is_flagged():
    src = """
def wf():
    x = yield SomeOp()
    return x
"""
    rules = {v.rule for v in check_source(src)}
    assert "require-yield-from" in rules


def test_an_unreachable_generator_marker_is_not_a_bare_yield():
    """A pure function typed as an `Effect`/`Judge` has to BE a generator, and the way you say so
    is a `yield` after the `return` — dead code whose only job is to change the function's type.
    Flagging it is a false positive with teeth: it is what kept `coding/verdicts.py` out of the
    determinism gate, and a nondeterministic judge really does break replay.

    The exemption is structural, not a comment or a filename: **a statement no execution can reach
    cannot perform an op.** A `# pragma`-keyed exemption would be the curated-list defect again."""
    src = """
def judge(evidence):
    return verdict_for(evidence)
    yield  # a `Judge` is a generator
"""
    assert check_source(src) == []


def test_the_mechanical_judges_are_inside_the_determinism_gate():
    """They are re-executed on replay and they PICK THE NEXT STATE, so a nondeterministic judge
    diverges the walk and replay raises `ReplayMismatch`."""
    from effective.lint import WORKFLOW_ROLE_SRCS, check_file

    path = "src/effective/coding/verdicts.py"
    assert any(entry.endswith("coding/verdicts.py") for entry in WORKFLOW_ROLE_SRCS), (
        "the judges must be role-set, or exempt with a reason"
    )
    assert check_file(path) == []


@pytest.mark.parametrize(
    ("case", "src", "flagged"),
    [
        ("marker after a return", "def j(e):\n    return v(e)\n    yield\n", False),
        ("marker after a raise", "def j(e):\n    raise X()\n    yield\n", False),
        ("marker past a comment", "def j(e):\n    return v(e)\n    # why\n    yield\n", False),
        ("bare yield, then return", "def w():\n    x = yield Op()\n    return x\n", True),
        ("bare yield ahead of one", "def w():\n    yield Op()\n    return 1\n", True),
        ("an OUTER block returns", "def w():\n    if c:\n        return 1\n    yield O()\n", True),
        ("a SIBLING def returned", "def a():\n    return 1\n\ndef w():\n    yield Op()\n", True),
    ],
)
def test_the_unreachable_exemption_discriminates(case: str, src: str, flagged: bool):
    """The exemption's DOMAIN, pinned as a matrix rather than as the one file that motivated it.

    An exemption is a hole in a gate, so what matters is not that the motivating case passes but
    that everything either side of it still fires — a return in an OUTER block does not make the
    next statement dead, and a yield AHEAD of a return is as live as any other."""
    rules = {v.rule for v in check_source(src)}
    assert ("require-yield-from" in rules) is flagged, case


def test_yield_from_is_allowed():
    src = """
def wf():
    x = yield from ask_llm("n", [], int)
    return x
"""
    assert check_source(src) == []


def test_forbidden_imports_are_flagged():
    src = """
import httpx
from anthropic import Anthropic
import random as r
from datetime import datetime
def wf():
    yield from step()
"""
    io_violations = [v for v in check_source(src) if v.rule == "no-io-import"]
    flagged_text = " ".join(v.text for v in io_violations)
    assert "httpx" in flagged_text
    assert "anthropic" in flagged_text
    assert "random" in flagged_text
    # `from datetime import datetime` is allowed — not flagged as an import
    assert "datetime" not in flagged_text


def test_nondeterministic_calls_are_flagged():
    src = """
from datetime import datetime
def wf():
    t = datetime.now()
    yield from step()
"""
    rules = [v for v in check_source(src) if v.rule == "no-nondeterminism"]
    assert any("datetime.now" in v.message for v in rules)


def test_datetime_import_alone_is_ok():
    # importing datetime is fine; only calling .now() between yields is not
    src = """
from datetime import datetime
def wf():
    yield from sleep_until(some_passed_in_time)
"""
    assert check_source(src) == []


# --- the layer-authority lint ---------------------------------------------


def test_domain_layer_may_forward_and_yield_domain_ops():
    src = """
def metered(accrue):
    @domain_layer
    def run(op):
        out = yield op                 # forwarding is allowed
        return out
    return run
"""
    assert check_layer_source(src) == []


DOMAIN_ARMS = {"AskLLM", "CallTool", "Judge"}
WORKFLOW_ARMS = {
    "AppendLedgerRow",
    "AwaitEvent",
    "Gather",
    "Race",
    "Respawn",
    "Scoped",
    "SleepUntil",
    "Step",
    "StoreArtifact",
}


def test_the_layer_alphabets_are_every_arm_of_the_op_unions():
    assert DOMAIN_OP_CTORS == DOMAIN_ARMS
    assert WORKFLOW_OP_CTORS == WORKFLOW_ARMS


SRC = Path(__file__).parent.parent / "src"


def _carriers(tree: ast.AST) -> set[str]:
    """Classes in `tree` that name the model-call marker as a base."""
    found = set()
    for node in ast.walk(tree):
        match node:
            case ast.ClassDef(name=name, bases=bases) if any(map(_is_marker, bases)):
                found.add(name)
    return found


def _is_marker(base: ast.expr) -> bool:
    match base:
        case ast.Name(id="AsksModel") | ast.Attribute(attr="AsksModel"):
            return True
        case _:
            return False


def test_the_model_call_marker_is_carried_by_exactly_the_model_calls():
    """Read from the source tree, so a carrier in a module no test imports is still counted."""
    arms = {get_origin(a) for a in get_args(ModelCall.__value__)}
    tree = set().union(*(_carriers(ast.parse(p.read_text())) for p in SRC.rglob("*.py")))
    assert (tree, set(AsksModel.__subclasses__())) == ({a.__name__ for a in arms}, arms)


type _Nested[T] = CallTool[T] | ModelCall[T]
type _Single = CallTool[int]
type _NotAnOp = CallTool[int] | Literal["x"]


def test_an_op_union_is_read_through_the_aliases_it_nests_and_refuses_anything_else():
    assert (_members(_Nested), _members(_Single)) == (DOMAIN_ARMS, {"CallTool"})
    with pytest.raises(TypeError, match="Literal"):
        _members(_NotAnOp)


def test_an_op_layer_yielding_a_bare_judgment_is_flagged():
    src = """
@op_layer
def bad(op):
    return (yield Judge({}, {}, dict))     # bypasses Step + the durable grain
"""
    assert "op-layer-alphabet" in {v.rule for v in check_layer_source(src)}


def test_a_domain_layer_yielding_a_gather_is_flagged():
    src = """
@domain_layer
def sneaky(op):
    yield Gather([])                        # forbidden: a domain layer cannot fan out
    return (yield op)
"""
    assert "domain-layer-no-control-flow" in {v.rule for v in check_layer_source(src)}


def test_domain_layer_yielding_a_control_flow_op_is_flagged():
    src = """
@domain_layer
def sneaky(op):
    ok = yield AwaitEvent("approve", Approval)   # forbidden: domain layers cannot suspend
    return (yield op)
"""
    violations = check_layer_source(src)
    rules = {v.rule for v in violations}
    assert "domain-layer-no-control-flow" in rules
    assert any("AwaitEvent" in v.message for v in violations)


def test_op_layer_yielding_a_bare_domain_op_is_flagged():
    src = """
@op_layer
def bad(op):
    return (yield AskLLM([], int))     # bypasses Step + the durable grain
"""
    rules = {v.rule for v in check_layer_source(src)}
    assert "op-layer-alphabet" in rules


def test_op_layer_forwarding_is_clean():
    src = """
@op_layer
def retry(op):
    return (yield op)
"""
    assert check_layer_source(src) == []


def test_op_layer_broad_except_is_flagged():
    # a broad catch would swallow the durable suspend signal (C3)
    for catch in ("except Exception:", "except BaseException as e:", "except:"):
        src = f"""
@op_layer
def swallow(op):
    try:
        return (yield op)
    {catch}
        return None
"""
        rules = {v.rule for v in check_layer_source(src)}
        assert "op-layer-no-broad-except" in rules, catch


def test_op_layer_specific_except_is_clean():
    # catching a specific error (as retry does) is fine
    src = """
@op_layer
def retry(op):
    try:
        return (yield op)
    except TransientError:
        raise
"""
    assert check_layer_source(src) == []


def test_channel_lint_flags_nocache_before_cache_in_one_literal():
    src = 'p = t"{volatile:role=system;nocache}{preamble:role=system;cache}"'
    rules = [v.rule for v in check_channel_source(src)]
    assert rules == ["volatile-last-cache"]


def test_channel_lint_allows_cache_before_nocache():
    # cached prefix then volatile tail is the CORRECT order — no violation
    src = 'p = t"{preamble:role=system;cache}{body:role=system;nocache}"'
    assert check_channel_source(src) == []


def test_channel_lint_is_role_scoped():
    # a volatile user segment does not poison a cached SYSTEM segment
    src = 'p = t"{body:nocache}{preamble:role=system;cache}"'
    assert check_channel_source(src) == []


def test_channel_lint_ignores_plain_format_specs():
    src = 'p = t"{amount:.2f} and {note:cache}"'
    assert check_channel_source(src) == []


def test_independence_lint_flags_an_inline_resolution_constructor():
    src = 'p = t"continue from {Done(answer)}"'
    rules = [v.rule for v in check_channel_source(src)]
    assert rules == ["independence"]


def test_independence_lint_flags_an_inline_resolve_call():
    src = 'p = t"fix using {prompt.resolve(response)}"'
    rules = [v.rule for v in check_channel_source(src)]
    assert rules == ["independence"]


def test_independence_lint_allows_pathlib_resolve():
    # a bare .resolve() (pathlib) has no argument — conservative, no false positive
    src = 'p = t"read {path.resolve()} for details"'
    assert check_channel_source(src) == []


def test_independence_lint_allows_keyword_only_resolve():
    # pathlib's .resolve(strict=True) is not a response resolution
    src = 'p = t"read {path.resolve(strict=True)} for details"'
    assert check_channel_source(src) == []


def test_channel_lints_ignore_f_strings():
    # f-strings are not channel templates: neither the independence rule nor
    # the cache rule applies to them.
    src = 'msg = f"debug {Done(x)} {a:nocache}{b:cache}"'
    assert check_channel_source(src) == []


def test_independence_lint_allows_plain_inputs():
    src = 'p = t"continue from {prior_answer}"'
    assert check_channel_source(src) == []


def test_shipping_channel_templates_pass_the_channel_lint():
    import agent.bracket as bracket_mod
    from effective.lint import check_channel_file

    assert check_channel_file(bracket_mod.__file__) == []


def test_shipping_layers_pass_the_authority_lint():
    # dogfood: the real layer module + the cost + permission layers obey the rules
    import effective.cost as cost_mod
    import effective.layers as layers_mod
    import effective.permission as permission_mod

    assert check_layer_file(layers_mod.__file__) == []
    assert check_layer_file(cost_mod.__file__) == []
    assert check_layer_file(permission_mod.__file__) == []


# --- seam dependency-direction (--deps; audit finding C-063) ------------------


def test_effective_may_not_import_a_consumer_package():
    from effective.lint import seam_forbidden_for

    src = "import tui.app\nfrom examples import demo\nimport agent.bracket\n"
    forbidden = seam_forbidden_for(Path("src/effective/foo.py"))
    violations = check_deps_source(src, "src/effective/foo.py", forbidden)
    flagged = {_import_root(v.text) for v in violations}
    assert flagged == {"tui", "examples", "agent"}
    assert all(v.rule == "seam-dep-direction" for v in violations)


def test_agent_may_import_effective_but_not_a_package_outside_the_substrate():
    from effective.lint import seam_forbidden_for

    forbidden = seam_forbidden_for(Path("src/agent/bar.py"))
    # importing the substrate is allowed
    allowed = "import effective.api\nfrom effective import gather\n"
    assert check_deps_source(allowed, "src/agent/bar.py", forbidden) == []
    # importing a package outside the substrate is not
    src = "import tui.app\nimport examples.demo\n"
    denied = check_deps_source(src, "src/agent/bar.py", forbidden)
    assert {_import_root(v.text) for v in denied} == {"tui", "examples"}


def test_packages_outside_the_substrate_are_unpoliced():
    from effective.lint import seam_forbidden_for

    # tui/ may import anything: no forbidden set, no violations
    forbidden = seam_forbidden_for(Path("src/tui/x.py"))
    assert forbidden == frozenset()
    src = "import effective.api\nimport agent.bracket\n"
    assert check_deps_source(src, "src/tui/x.py", forbidden) == []


def test_shipping_substrate_obeys_the_seam():
    # dogfood: no module in the substrate trees imports a package outside them
    from pathlib import Path as _P

    root = _P(__file__).parent.parent / "src"
    offenders = [
        str(v)
        for tree in ("effective", "agent")
        for f in (root / tree).rglob("*.py")
        for v in check_deps_file(f)
    ]
    assert offenders == [], offenders


# --- the intra-package half of the same rule -------------------------------------
#
# A rule keyed on the top-level package sees no edge inside `effective`. These pin the rule by
# MOVING the thing: each case is a back-edge that must fire, paired with the legitimate edge next
# door that must not. A rule that fires on everything is not a gate either.


@pytest.mark.parametrize(
    ("case", "path", "src", "fires"),
    [
        # `effective.machine` is generic over the state type; `effective.coding` is one of three
        # embodiments riding it. An edge this way makes "generic" unfalsifiable.
        (
            "machine reaching an embodiment",
            "src/effective/machine/trampoline.py",
            "from effective.coding.states import State\n",
            True,
        ),
        (
            "machine reaching its own sibling",
            "src/effective/machine/trampoline.py",
            "from effective.machine.outcomes import Advance\n",
            False,
        ),
        # Coding's yielding half must stay importable without ast-grep, jedi and a ruff config.
        (
            "yielding half reaching the gate",
            "src/effective/coding/verdicts.py",
            "from effective.coding.gate import static_merge_gate\n",
            True,
        ),
        (
            "yielding half reaching the runners",
            "src/effective/coding/specs.py",
            "from effective.coding.runners import CODING_TOOLS\n",
            True,
        ),
        (
            "yielding half reaching the editors",
            "src/effective/coding/transition.py",
            "from effective.coding.edits.semantic import rename\n",
            True,
        ),
        # ...but the handler-side half is allowed to reach itself, which is why the rule is keyed
        # on the protected half rather than on a list of the modules it protects.
        (
            "runners reaching the gate",
            "src/effective/coding/runners.py",
            "from effective.coding.gate import static_merge_gate\n",
            False,
        ),
        (
            "editors reaching each other",
            "src/effective/coding/edits/__init__.py",
            "from effective.coding.edits.semantic import rename\n",
            False,
        ),
        # The direction that IS the design: an embodiment imports the generic machine.
        (
            "embodiment reaching the machine",
            "src/effective/coding/verdicts.py",
            "from effective.machine.spec import Evidence\n",
            False,
        ),
    ],
)
def test_the_intra_package_seam_discriminates(case: str, path: str, src: str, fires: bool):
    violations = check_deps_source(src, path, seam_forbidden_for(Path(path)))
    assert bool(violations) is fires, f"{case}: {[str(v) for v in violations]}"
    if fires:
        assert violations[0].rule == "seam-dep-direction"
        # The reason travels with the rule. A gate that says "forbidden" without saying why
        # teaches people to reach for the escape.
        assert "(" in violations[0].message, violations[0].message


def test_a_new_module_in_the_yielding_half_is_covered_without_an_edit():
    """The point of naming the PROTECTED half rather than listing what it protects.

    A list of the light modules goes stale in both directions: it keeps naming a module that
    moved, and it cannot name a module nobody has written."""
    invented = "src/effective/coding/prompts.py"
    repo_root = Path(__file__).parent.parent
    assert not (repo_root / invented).exists(), "pick a name that is still hypothetical"
    violations = check_deps_source(
        "from effective.coding.runners import CODING_TOOLS\n",
        invented,
        seam_forbidden_for(Path(invented)),
    )
    assert [v.rule for v in violations] == ["seam-dep-direction"]


def test_a_relative_import_is_policed():
    """A relative import must resolve to its package: `""` is in no forbidden set, so a root of
    `""` for `from .gate import x` would pass every rule. `src/effective` has no relative import,
    so only this test exercises the path."""
    path = "src/effective/coding/verdicts.py"
    for spelling in ("from .gate import static_merge_gate\n", "from .runners import TOOLS\n"):
        violations = check_deps_source(spelling, path, seam_forbidden_for(Path(path)))
        assert [v.rule for v in violations] == ["seam-dep-direction"], spelling
    # A climb past the top-level package is invalid Python, so it cannot appear in a module that
    # imports; resolving it to something would invent a module name and report against it.
    assert (
        check_deps_source(
            "from ...tui import app\n",
            "src/effective/api.py",
            seam_forbidden_for(Path("src/effective/api.py")),
        )
        == []
    )


def test_every_name_in_a_multi_name_import_is_inspected():
    """A multi-name import yields one root per name, each checked against the forbidden set."""
    path = "src/effective/api.py"
    violations = check_deps_source(
        "import effective.ops, tui.app, examples.demo\n",
        path,
        seam_forbidden_for(Path(path)),
    )
    assert len(violations) == 2, [str(v) for v in violations]


def test_a_narrow_row_does_not_delete_a_wide_one():
    """`seam_forbidden_for` unions every applicable row instead of taking the longest match.

    Under a longest match, `effective.machine`'s intra-package row would silently drop the row
    forbidding packages outside the substrate for every file under `machine/`: a narrow rule
    quietly deleting a wide one."""
    forbidden = seam_forbidden_for(Path("src/effective/machine/trampoline.py"))
    assert "effective.coding" in forbidden  # the narrow row
    assert {"tui", "examples", "agent"} <= forbidden  # and the wide one survives it


def test_the_prefix_match_is_segment_wise():
    """`startswith` alone would flag `agentic.py` under a rule about `agent`."""
    path = "src/effective/machine/y.py"
    assert (
        check_deps_source(
            "import agentx.thing\nimport effective.codingx\n", path, seam_forbidden_for(Path(path))
        )
        == []
    )


# --- the canonical ledger-read rule --------------------------------------------------------
#
# "Every projection reads WHERE NOT hypothetical" held as a convention fails the way conventions
# do: one unfenced query lets a fork's row, later in `seq`, win a last-event-wins fold. These pin
# the GATE: the rule must bite on ANY unfenced read.


def test_an_unfenced_ledger_read_is_flagged():
    from effective.lint import check_ledger_reads_source

    src = (
        "rows = conn.execute(\n"
        '    "SELECT payload, recorded_at FROM ledger "\n'
        "    \"WHERE kind='request_processed' ORDER BY seq\"\n"
        ").fetchall()\n"
    )
    assert [v.rule for v in check_ledger_reads_source(src, "x.py")] == [
        "ledger-read-not-canonical"
    ]


def test_the_predicate_may_live_in_a_different_concatenated_literal():
    """A multi-line query splits `FROM ledger` and the `WHERE` across literals — the rule's
    unit is the whole concatenation group, which is what a reader reasons about too."""
    from effective.lint import check_ledger_reads_source

    src = (
        "rows = conn.execute(\n"
        '    "SELECT payload FROM ledger "\n'
        '    "WHERE NOT hypothetical ORDER BY seq"\n'
        ").fetchall()\n"
    )
    assert check_ledger_reads_source(src, "x.py") == []


def test_a_tstring_ledger_read_is_scanned_too():
    """psycopg t-string SQL is the repo's parameterized form — it must not be a blind spot."""
    from effective.lint import check_ledger_reads_source

    src = 'rows = conn.execute(t"SELECT payload FROM ledger WHERE workflow_run_id={rid}")\n'
    assert [v.rule for v in check_ledger_reads_source(src, "x.py")] == [
        "ledger-read-not-canonical"
    ]


def test_a_deliberate_hypothetical_read_can_opt_out_with_a_reason():
    """A fork's own marginal legitimately reads hypothetical rows. The escape hatch makes that
    a visible, reviewable decision rather than a silent omission."""
    from effective.lint import check_ledger_reads_source

    src = (
        "rows = conn.execute(  # lint: ledger-read-not-canonical: the fork's own marginal\n"
        '    "SELECT payload FROM ledger ORDER BY seq"\n'
        ").fetchall()\n"
    )
    assert check_ledger_reads_source(src, "x.py") == []


def test_shipping_projections_read_only_canonical_rows():
    """Dogfood: no unfenced ledger read exists in the shipping trees today.

    `scripts/` is in the sweep because a script that reads the ledger is a projection too, and a
    gate is bounded by what it scans: widen this sweep and the gate together or neither.
    """
    from pathlib import Path as _P

    from effective.lint import check_ledger_reads_file

    repo = _P(__file__).parent.parent
    trees = [repo / "src" / t for t in ("effective", "agent")]
    trees.append(repo / "scripts")
    offenders = [
        str(v) for tree in trees for f in tree.rglob("*.py") for v in check_ledger_reads_file(f)
    ]
    assert offenders == [], offenders


# --- the SQL-boundary rule (`--sql-templates`) -------------------------------------------
#
# Teeth for "question every f-string outside the render-backend position", scoped
# to the SQL seam BY POSITION: the first argument of an `.execute(...)`.


def test_sql_and_parameters_split_across_two_arguments_is_flagged():
    from effective.lint import check_sql_templates_source

    src = 'conn.execute("SELECT state FROM checkpoints WHERE task_id=?", (task_id,))\n'
    assert [v.rule for v in check_sql_templates_source(src, "x.py")] == ["sql-params-split"]


def test_sql_built_by_an_fstring_is_flagged_as_the_bobby_tables_shape():
    from effective.lint import check_sql_templates_source

    src = 'conn.execute(f"SELECT * FROM {table} WHERE id = {ident}")\n'
    assert [v.rule for v in check_sql_templates_source(src, "x.py")] == ["sql-built-by-formatting"]


def test_an_fstring_hidden_in_a_concatenation_is_still_flagged():
    from effective.lint import check_sql_templates_source

    src = 'conn.execute(\n    "SELECT * FROM t "\n    f"WHERE id = {ident}"\n)\n'
    assert [v.rule for v in check_sql_templates_source(src, "x.py")] == ["sql-built-by-formatting"]


def test_percent_and_format_built_sql_are_flagged():
    from effective.lint import check_sql_templates_source

    percent = 'conn.execute("SELECT * FROM t WHERE id = %s" % ident)\n'
    formatted = 'conn.execute("SELECT * FROM {} ".format(table))\n'
    assert [v.rule for v in check_sql_templates_source(percent, "x.py")] == [
        "sql-built-by-formatting"
    ]
    assert [v.rule for v in check_sql_templates_source(formatted, "x.py")] == [
        "sql-built-by-formatting"
    ]


def test_a_composed_template_is_the_form_the_rule_wants():
    from effective.lint import check_sql_templates_source

    src = 'conn.execute(*bind(t"SELECT state FROM checkpoints WHERE task_id={task_id}"))\n'
    assert check_sql_templates_source(src, "x.py") == []


def test_a_psycopg_tstring_call_passes_unchanged():
    from effective.lint import check_sql_templates_source

    src = 'conn.execute(t"SELECT 1 FROM absurd.{c_tbl:i} WHERE task_id = {task_id}::uuid")\n'
    assert check_sql_templates_source(src, "x.py") == []


def test_a_static_query_with_nothing_to_separate_is_not_flagged():
    from effective.lint import check_sql_templates_source

    src = 'conn.execute("PRAGMA busy_timeout=5000")\n'
    assert check_sql_templates_source(src, "x.py") == []


def test_executemany_is_out_of_scope_because_bind_composes_one_parameter_set():
    from effective.lint import check_sql_templates_source

    src = 'conn.executemany("INSERT INTO t VALUES (?,?)", rows)\n'
    assert check_sql_templates_source(src, "x.py") == []


def test_a_deliberate_non_template_call_can_opt_out_with_a_reason():
    from effective.lint import check_sql_templates_source

    src = (
        "conn.execute(  # lint: sql-not-templated: the driver's own DDL form\n"
        '    "SELECT ?", (1,)\n)\n'
    )
    assert check_sql_templates_source(src, "x.py") == []


def test_the_shipping_sqlite_engine_and_bridges_compose_their_sql():
    """The dogfood sweep — the gate `just lint` runs, as a test."""
    from pathlib import Path

    from effective.lint import check_sql_templates_file

    found = [
        str(v)
        for base in ("src/effective", "src/agent")
        for f in sorted(Path(base).rglob("*.py"))
        for v in check_sql_templates_file(f)
    ]
    assert found == []


# --- the working-note rule ---------------------------------------------------------------

TAG = "TO" + "DO"  # split so this file is not its own violation


def test_a_working_note_is_refused_in_a_comment_dated_or_not():
    from effective.lint import check_working_notes_source as check

    dated = check(f"x = 1  # {TAG}(2026-08-03): finish the arm\n", "m.py")
    assert [v.rule for v in dated] == ["working-note"]
    assert "Discharge it before committing" in dated[0].message

    bare = check(f"x = 1  # {TAG}: finish the arm\n", "m.py")
    assert "Date it" in bare[0].message, "an undated note must say how to make it triageable"


def test_a_working_note_in_a_docstring_is_refused_at_the_holders_line():
    """Reported at the `def`, not the string: that is where a reader looks, and it is the only
    line an opt-out can sit on — a docstring cannot carry a trailing comment."""
    from effective.lint import check_working_notes_source as check

    found = check(f'def f():\n    """Does a thing.\n\n    {TAG}: widen it.\n    """\n', "m.py")
    assert len(found) == 1
    assert found[0].line == 1, "the def line, not the docstring's"
    assert "DOCSTRING" in found[0].message


def test_the_rule_does_not_fire_on_prose_that_merely_contains_the_letters():
    """The negatives are where a tag-scanner earns its keep. A string LITERAL is subject matter,
    not a note; and `METHODOLOGY`-shaped words contain the tag without a word boundary."""
    from effective.lint import check_working_notes_source as check

    assert check(f'MSG = "the {TAG} list feature"\n', "m.py") == []
    assert check(f"x = 1  # ME{TAG[2:]}LOGY and XX{'X'}Y are not tags\n", "m.py") == []
    assert check("x = 1  # an ordinary explanatory comment\n", "m.py") == []


def test_the_opt_out_covers_its_own_line_and_the_one_below():
    from effective.lint import check_working_notes_source as check

    assert check(f"x = 1  # {TAG}: real  # lint: working-note — subject matter\n", "m.py") == []
    above = f'# lint: working-note — subject matter\ndef f():\n    """A {TAG} rubric."""\n'
    assert check(above, "m.py") == []


def test_the_live_tree_carries_no_working_notes():
    """The gate's own anti-vacuity: assert the SCAN reached files, not that a count is large."""
    from effective.lint import check_working_notes_file

    scanned = [
        f
        for base in ("src/effective", "src/agent", "tests", "scripts")
        for f in sorted(Path(base).rglob("*.py"))
    ]
    assert {p.parts[0] for p in scanned} == {"src", "tests", "scripts"}, "every root was reached"
    assert [str(v) for f in scanned for v in check_working_notes_file(f)] == []


# --- the role-coverage rule --------------------------------------------------------------


def test_a_workflow_outside_the_role_set_is_flagged(tmp_path):
    from effective.lint import check_role_coverage

    wf = tmp_path / "rogue.py"
    wf.write_text("from effective.api import step\n\ndef w():\n    yield from step('a', None)\n")
    found = check_role_coverage([wf])
    assert [v.rule for v in found] == ["role-coverage"]
    assert "WORKFLOW_ROLE_SRCS" in found[0].message


def test_the_wrapper_layer_is_not_mistaken_for_a_workflow(tmp_path):
    """The discriminator is IMPORTED, not merely named. `effective.api` DEFINES the wrappers and
    bare-`yield`s the op constructors, the one place that is correct, and a module with wrappers of
    its own would look the same. Both would drown this rule if it keyed on the name alone; both are
    silent here."""
    from effective.lint import check_role_coverage

    defines = tmp_path / "api_like.py"
    defines.write_text("from effective.ops import Step\n\ndef step(n):\n    yield Step(n)\n")
    twin = tmp_path / "twin.py"
    twin.write_text("from mylite import AskLLM\n\ndef w():\n    yield from ask(AskLLM('x'))\n")
    assert check_role_coverage([defines, twin]) == []


# The detector deciding WHO gets scanned was blind to two shapes. Both are pinned here.
def test_a_bare_yield_of_an_effect_op_is_seen(tmp_path):
    """The exact footgun `require-yield-from` exists to catch is the one shape the gate deciding
    who gets scanned could not see — so a file whose only defect is that defect never joins the
    role set, and the determinism lint never runs on it. A gate blind to its own failure mode."""
    from effective.lint import check_role_coverage

    wf = tmp_path / "rogue.py"
    wf.write_text("from effective import step\n\n\ndef w():\n    yield step('a', None)\n")
    assert [v.rule for v in check_role_coverage([wf])] == ["role-coverage"]


def test_a_parenthesized_multiline_import_is_read(tmp_path):
    """`"(\\n    step".strip("()")` leaves the newline and indent attached, so the FIRST name in a
    parenthesized import never matches a callee. The middle names survive (`.strip()` reaches
    them), which is why this hid: the rule works right up until the wrapper you used is listed
    first."""
    from effective.lint import check_role_coverage

    wf = tmp_path / "rogue.py"
    wf.write_text(
        "from effective import (\n    step,\n    ask_llm,\n)\n\n\ndef w():\n"
        "    yield from step('a', None)\n"
    )
    assert [v.rule for v in check_role_coverage([wf])] == ["role-coverage"]


def test_widening_to_a_bare_yield_still_excludes_the_wrapper_layer(tmp_path):
    """MEASURED, because the opposite prediction is plausible and would buy a needless carve-out.
    The prediction: the `yield from`-only test is what keeps `effective/api.py` out, so widening
    to a bare `yield` needs an explicit exemption. It does not: `api.py` imports from
    `effective.ops`/`.domain`/`.keys`, none of which is an `_EFFECT_MODULES` spelling, so the
    IMPORT test excludes it and the widening reaches no new file in `src/`. Pinned so the
    exemption is not added later on the strength of the prose."""
    from effective.lint import _first_effect_yield

    assert _first_effect_yield(Path("src/effective/api.py").read_text()) == 0


def test_the_live_role_set_is_complete():
    """The gate's own claim: no file in the tree yields an effect op unseen. Asserts COVERAGE
    (every scanned root reached) rather than a count, which would fail as the tree grows."""
    from effective.lint import check_role_coverage

    roots = ("src", "examples", "scripts")
    scanned = [f for r in roots if Path(r).exists() for f in sorted(Path(r).rglob("*.py"))]
    assert {p.parts[0] for p in scanned} >= {"src", "examples"}, "the roots were reached"
    assert [str(v) for v in check_role_coverage(scanned)] == []


def test_the_coverage_gate_sees_a_workflow_reached_through_another_module(tmp_path):
    """`_EFFECT_MODULES` enumerates SPELLINGS of one referent, and a domain shaped like that is
    always one spelling short — twice already, both fixed by adding a string.

    The live miss is not a spelling at all. `src/agent/bench_sweep.py` writes
    `yield from improve(...)` where `improve` comes from `effective.improve`, which yields ops of
    its own. It authors a workflow through a module that is not in the list and never could be by
    enumeration, because the property is transitive: **a module that yields an effect op is itself
    an effect module.** The domain is a fixpoint, not a list."""
    from effective.lint import check_role_coverage

    mid = tmp_path / "mid.py"
    mid.write_text(
        "from effective.api import step\n\n\ndef helper():\n    yield from step('a', None)\n"
    )
    top = tmp_path / "top.py"
    top.write_text("from mid import helper\n\n\ndef w():\n    yield from helper()\n")
    assert {v.filename for v in check_role_coverage([mid, top])} == {str(mid), str(top)}


def test_the_live_bench_sweep_is_seen_by_the_coverage_gate():
    """The in-tree instance of the same hole, kept beside the synthetic one because a rule that
    passes on a fixture and misses the real file is a recurring failure.

    Asked of the CLOSURE, not of the seed. Calling `_first_effect_yield` with its default would
    pin the implementation detail (which strings are seeded) instead of the property (bench_sweep
    authors a workflow, through however many hops)."""
    from effective.lint import _effect_module_closure, _first_effect_yield, _module_path

    sources = {_module_path(p): p.read_text() for p in Path("src").rglob("*.py")}
    closure = _effect_module_closure(sources)
    assert "effective.improve" in closure, "the intermediary is what makes bench_sweep reachable"
    assert _first_effect_yield(Path("src/agent/bench_sweep.py").read_text(), closure) != 0
    # …and the seed alone does NOT see it, which is the whole finding.
    assert _first_effect_yield(Path("src/agent/bench_sweep.py").read_text()) == 0


def test_an_unresolvable_role_entry_raises_instead_of_scanning_nothing():
    """It contributed zero files in silence — the shape where a gate reports success for a file
    it never opened. Rename the target and the determinism boundary stops being checked there,
    with `just lint` still green."""
    from effective.lint import _resolve_role_default

    with pytest.raises(FileNotFoundError, match="resolves to no file"):
        _resolve_role_default(("src/effective/does_not_exist.py",))


def test_every_role_set_entry_resolves_today():
    from effective.lint import WORKFLOW_ROLE_SRCS, _resolve_role_default

    resolved = _resolve_role_default(WORKFLOW_ROLE_SRCS)
    assert len(resolved) >= len(WORKFLOW_ROLE_SRCS), resolved


# --- the key-composition rule --------------------------------------------------------------
#
# The negatives are pinned HARDEST here, and deliberately so: the rule is scoped BY POSITION,
# never by f-string-ness, because a rule that flags the correct render backend gets disabled.
# One of the negatives is the false positive a leading-namespace sweep produced.


def test_an_identity_built_by_formatting_in_a_step_slot_is_flagged():
    from effective.lint import check_key_composition_source

    src = 'step(f"tool:{name}", op)\n'
    (v,) = check_key_composition_source(src, "x.py")
    assert v.rule == "key-composition"
    assert "compose_key" in v.message


def test_the_rule_reaches_ONE_hop_through_a_binding():
    """A name like `code.py`'s `action_key` is assigned on one line and passed on the next, so
    a slot-only rule cannot see it."""
    from effective.lint import check_key_composition_source

    src = 'action_key = f"{base}:action:{j}"\nstep(action_key, op)\n'
    (v,) = check_key_composition_source(src, "x.py")
    assert v.line == 1, "report at the line that BUILT the name — that is where the fix goes"


def test_a_composed_idempotency_key_is_flagged_wherever_it_appears():
    from effective.lint import check_key_composition_source

    src = 'spawn("t", params, idempotency_key=f"{mid}:{rev}")\n'
    assert len(check_key_composition_source(src, "x.py")) == 1


# --- the negatives ------------------------------------------------------------------------


def test_a_composed_key_in_the_slot_is_the_point_and_passes():
    from effective.lint import check_key_composition_source

    src = 'step(compose_key(t"tool:{name}").stored(), op)\n'
    assert check_key_composition_source(src, "x.py") == []


def test_an_error_message_that_merely_opens_with_a_namespace_is_not_an_identity():
    """The false positive a leading-namespace sweep really produced: `combinators.py`'s
    `f"respawn: the step returned Again…"`. It is not in an identity slot, so it never arises —
    which is the POINT of scoping by position rather than by f-string-ness."""
    from effective.lint import check_key_composition_source

    src = 'raise RuntimeError(f"respawn: the step returned Again at generation {n}")\n'
    assert check_key_composition_source(src, "x.py") == []


def test_a_sql_like_pattern_over_the_key_grammar_is_not_an_identity():
    """The bridges' `like = f"budget-grant:{run_id},%"` — a QUERY over names, not a name."""
    from effective.lint import check_key_composition_source

    src = 'like = f"budget-grant:{run_id},%"\nconn.execute(t"... LIKE {like}")\n'
    assert check_key_composition_source(src, "x.py") == []


def test_the_render_backend_named_by_function_passes():
    """`scope_prefix`, named by function because line numbers move. It passes BY CONSTRUCTION:
    a processor's internals are not an identity argument slot, so no exemption is involved.

    **By function all the way down.** A pin on the literal text `f"{item.value}"` breaks when a
    match arm renames its binding. A pin on a backend with no callers guards a render path nobody
    renders through: naming a thing by function does not make the thing reachable."""
    import inspect

    from effective.keys import scope_prefix
    from effective.lint import check_key_composition_file

    body = inspect.getsource(scope_prefix)
    assert 'f"' in body, "scope_prefix's render backend moved — re-pin it"
    # Ask the module that DEFINES it, so a rename or a split does not silently point this
    # at a path that no longer exists: `check_key_composition_file` would then scan nothing.
    source = inspect.getsourcefile(scope_prefix)
    assert source is not None, "scope_prefix has no source file"
    assert check_key_composition_file(source) == []


def test_a_foreign_grammars_name_can_opt_out_with_a_reason():
    from effective.lint import check_key_composition_source

    src = 'step(f"$awaitEvent:{n}", op)  # lint: key-composition: the SDK\'s own spelling\n'
    assert check_key_composition_source(src, "x.py") == []


def test_the_opt_out_on_a_binding_line_silences_its_use():
    from effective.lint import check_key_composition_source

    src = 'k = f"a:{b}"  # lint: key-composition: a foreign grammar\nstep(k, op)\n'
    assert check_key_composition_source(src, "x.py") == []


# --- the terminal-hole ratchet (`--terminal-holes`) ----------------------------------------


def test_an_unwrapped_terminal_hole_is_flagged():
    from effective.lint import check_terminal_holes_source

    src = 'k = compose_key(t"extracted:{message_id}")\n'
    [violation] = check_terminal_holes_source(src, "x.py")
    assert violation.rule == "terminal-hole"
    assert (
        "Segment(message_id)" in violation.message
    )  # the message names the fix, not just the sin


def test_a_wrapped_terminal_hole_passes():
    from effective.lint import check_terminal_holes_source

    src = 'k = compose_key(t"extracted:{Segment(message_id)}")\n'
    assert check_terminal_holes_source(src, "x.py") == []


def test_a_template_ending_in_a_STATIC_is_not_scanned_at_all():
    """The distinction the whole rule turns on, and it is not "the last hole".

    The rule scans a hole only when NOTHING follows it, so `t"skill:{name},activate"` is not
    scanned at all: `name` is interior. The composer requires every hole to be a well-formed atom
    wherever it sits, so what this pins is narrow: a rule scanning "the last interpolation" would
    demand a wrapper the composer already demands, and would report the wrong line as the
    reason."""
    from effective.lint import check_terminal_holes_source

    assert check_terminal_holes_source('compose_key(t"skill:{name},activate")\n', "x.py") == []
    assert len(check_terminal_holes_source('compose_key(t"skill:{name}")\n', "x.py")) == 1


def test_an_int_literal_needs_no_wrapper_but_a_string_literal_does():
    from effective.lint import check_terminal_holes_source

    assert check_terminal_holes_source('compose_key(t"rec:{0}")\n', "x.py") == []
    assert check_terminal_holes_source('compose_key(t"rec:{-1}")\n', "x.py") == []
    assert len(check_terminal_holes_source("compose_key(t\"rec:{'0'}\")\n", "x.py")) == 1


def test_an_already_legal_terminal_can_opt_out_with_a_reason():
    from effective.lint import check_terminal_holes_source

    src = "# lint: terminal-hole — `op_key` returns a `Key`, spliced by induction\n"
    src += 'k = compose_key(t"approve;{op_key(op):domain=identity}")\n'
    assert check_terminal_holes_source(src, "x.py") == []


def test_the_one_excluded_file_is_excluded_with_a_reason_and_is_not_clean():
    """The exclusion's blind spot, pinned so it cannot be quietly widened or forgotten.

    `check_terminal_holes_file` returns `[]` for the grammar's own spec file — but the SOURCE
    check still reports its 13 bare terminals, and that gap is the cost. Asserting both halves is
    what makes this a record of a decision rather than a green light."""
    from pathlib import Path

    from effective.lint import (
        NOT_TERMINAL_HOLE_SCANNED,
        check_terminal_holes_file,
        check_terminal_holes_source,
    )

    spec = "tests/test_op_key_injectivity.py"
    assert NOT_TERMINAL_HOLE_SCANNED[spec]  # an entry, and a reason with it
    assert check_terminal_holes_file(spec) == []
    unscanned = check_terminal_holes_source(Path(spec).read_text(), spec)
    assert len(unscanned) >= 10, "the exclusion hides fewer sites than recorded — re-state it"


# --- borrowed production namespaces (`--key-borrowing`) ------------------------------------
#
# The rule exists because `--key-registry` scans `KEY_REGISTRY_SRCS` and nothing else, so a
# second variant minted in `tests/` is invisible to the rule that would refuse it in `src/`.
# These tests pin the four behaviours it has to get right — the three NEGATIVES matter as much
# as the positive, because the naive version of this rule (widen `--key-registry` to scan
# `tests/`) reports 21 violations of which most are the negatives below.


def _borrowing(source: str):
    """Run the rule over one throwaway file, against the REAL production registry."""
    import tempfile
    from pathlib import Path

    from effective.lint import check_key_borrowing

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.py"
        path.write_text(source)
        return check_key_borrowing([path])


def test_a_production_namespace_at_a_shape_it_cannot_mint_is_flagged():
    """`govern:` is minted at five coordinates; a one-coordinate `govern:` key is not in that
    namespace's language, so nothing in production could have produced it."""
    [violation] = _borrowing("compose_key(t\"govern:{Segment('g')}\")\n")
    assert violation.rule == "key-borrowing"
    assert "govern" in violation.message
    assert "src/effective/govern.py" in violation.message  # it names the owner, not just the sin


def test_a_tag_production_does_not_own_is_the_files_own_business():
    """The first negative, and the one that keeps the rule usable. `tests/` invents namespaces
    constantly — `ns:`, `q:`, `x:` — and `test_op_key_injectivity.py` uses several shapes under
    one invented tag ON PURPOSE, to specify what separation means. A rule that registered test
    templates would flag the file that defines the concept."""
    assert _borrowing("compose_key(t\"mytesttag:{Segment('q')}\")\n") == []
    assert _borrowing('compose_key(t"ns:{a}:{b}")\ncompose_key(t"ns:{c}")\n') == []


def test_a_literal_specialization_of_a_production_shape_is_not_a_second_variant():
    """The second negative, and the reason the rule compares LANGUAGES rather than skeletons.

    `budget-grant:{run_id},{trip}` records a trailing HOLE, which is a tail; `budget-grant:{r},0`
    records a trailing LITERAL, which is a slot. Different skeletons — `separated` calls them a
    collision — but the second mints keys the first accepts, so it is the same namespace written
    narrower. Nine of the twenty-one violations the naive rule reports are this."""
    assert _borrowing("compose_key(t\"budget-grant:{Segment('r')},0\")\n") == []


def test_the_registered_shape_itself_passes():
    """The third negative. Stated separately from the one above because a rule that flagged
    this would be refusing production's own template, which no message-writing would excuse."""
    assert _borrowing('compose_key(t"fold:{lv},{k}")\n') == []


def test_scanning_nothing_raises_rather_than_reporting_green():
    """A whole-set rule over an empty path list returns no violations and reads as a pass. This
    one is especially exposed: its subject is `tests`, which no other mode defaults to, so a
    dropped argument would silence it entirely."""
    import pytest

    from effective.lint import check_key_borrowing

    with pytest.raises(ValueError, match="no paths to scan"):
        check_key_borrowing([])


def test_the_tree_is_clean_and_the_rule_is_what_says_so():
    """The ratchet: this holds `tests/` at zero borrowings, and the assertions above prove the
    zero is not a rule that cannot fire."""
    from effective.lint import check_key_borrowing

    assert check_key_borrowing(["tests"]) == []


# --- the totality census (`--totality`) --------------------------------------------------------
#
# The census was FITTED to `keys.py`, so that file is not an independent test of it. These are: a
# synthetic corpus where each population appears in isolation, so a regression names which
# discrimination broke instead of moving one number.

_CENSUS_CORPUS = '''
type Shape = Circle | Square

class Term:
    tag: str | Hole

def dispatch(shape: Shape) -> int:
    """A declared union, tested by a name the signature types."""
    if isinstance(shape, Circle):
        return 1
    return 2

def guard(value: object) -> None:
    """A boundary check: negated, over a raise."""
    if not isinstance(value, Circle):
        raise TypeError("no")

def select(shapes: list[Shape]) -> list[Shape]:
    """One arm SELECTED, not decided."""
    return [s for s in shapes if isinstance(s, Circle)]

def foreign(node: object) -> bool:
    """A third-party hierarchy — nothing declares a union over it."""
    return isinstance(node, SomethingNobodyDeclared)

def attribute_declared(term: Term) -> int:
    """`Term.tag` IS declared `str | Hole`, so the attribute is knowable."""
    return 1 if isinstance(term.tag, Hole) else 0

def attribute_undeclared(interpolation: object) -> int:
    """`.value` is declared nowhere here — `Any`, so not a totality site."""
    return 1 if isinstance(interpolation.value, Circle) else 0

def assigned_from_anything(box: object) -> int:
    """A bare name bound only by assignment carries its right-hand side's type."""
    circle = box.contents[0]
    return 1 if isinstance(circle, Circle) else 0
'''

_DUPLICATED_CORPUS = """
type Item = str | Hole

def first(items: list[Item]) -> list[Item]:
    return [i for i in items if not isinstance(i, str)]

def second(items: list[Item]) -> list[Item]:
    return [i for i in items if not isinstance(i, str)]
"""


def _census_of(source: str, tmp_path: Path) -> dict[str, str]:
    """Classify `source` and return {function name: population}, one site per function."""
    import ast

    from effective.lint import totality_census

    module = tmp_path / "subject.py"
    module.write_text(source)
    tree = ast.parse(source)
    owner = {
        call.lineno: fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        for call in ast.walk(fn)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "isinstance"
    }
    return {owner[s.line]: s.population for s in totality_census([module])}


@pytest.mark.parametrize(
    ("function", "population"),
    [
        ("dispatch", "dispatch"),
        ("guard", "guard"),
        ("select", "filter"),
        ("foreign", "foreign"),
        ("attribute_declared", "dispatch"),
        ("attribute_undeclared", "guard"),
        ("assigned_from_anything", "guard"),
    ],
)
def test_each_population_is_told_apart_from_the_others(function, population, tmp_path):
    """One row per discrimination the rule makes, each in isolation.

    The last three are the ones that took three tries on the real file, and they are the rule's
    whole precision story: a site over a value the checker calls `Any` is not a totality site,
    because `assert_never` on an `Any` fires unconditionally. `attribute_declared` is the control
    — it proves the rule refuses undeclared attributes rather than all attributes."""
    assert _census_of(_CENSUS_CORPUS, tmp_path)[function] == population


def test_one_filter_written_twice_is_a_DUPLICATED_ENUMERATION(tmp_path):
    """The population no `isinstance` count reveals, and the one the pilot actually found.

    A single occurrence of this predicate is a `filter` and stays out of scope; two occurrences of
    the SAME predicate in one file are one enumeration derived twice, where the fix is to build it
    once rather than to `match` harder. Promotion is on textual identity, which is the rule's
    stated limit — the same defect with different variable names reads as two filters."""
    census = _census_of(_DUPLICATED_CORPUS, tmp_path)
    assert census == {"first": "duplicated-enumeration", "second": "duplicated-enumeration"}

    single = _census_of(_DUPLICATED_CORPUS.split("def second")[0], tmp_path)
    assert single == {"first": "filter"}, "one occurrence is not a duplication"


def test_the_gate_reports_only_the_two_IN_SCOPE_populations(tmp_path):
    """`check_totality` is the gate half; the census is the other half and reports everything.

    A gate that fired on all five populations would be a blanket no-`isinstance` rule: ~100
    pragmas, and a gate that teaches people to add pragmas."""
    from effective.lint import check_totality

    module = tmp_path / "subject.py"
    module.write_text(_CENSUS_CORPUS)
    flagged = check_totality([module])
    assert {v.rule for v in flagged} == {"totality"}
    assert len(flagged) == 2, "dispatch + attribute_declared, not the guards/filter/foreign"


def test_a_union_with_only_None_beside_it_is_an_OPTIONAL_not_a_dispatch(tmp_path):
    """`Key | None` is answered by `is None`, not by a `match` closed with `assert_never`.

    Without this the rule would call every optional-typed name a union member, and the census
    would be dominated by a question nobody is asking."""
    source = (
        "def f(shape: Circle | None) -> int:\n    return 1 if isinstance(shape, Circle) else 0\n"
    )
    assert _census_of(source, tmp_path)["f"] == "foreign"


# --- the census's second version: arity, and the population it could not see -------------------

_ARITY_CORPUS = '''
type Pair = Left | Right
type Nine = A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 | A9

def two_arms(pair: Pair) -> int:
    """One test plus an `else` EXHAUSTS a two-arm union — the else is worth one arm."""
    if isinstance(pair, Left):
        return 1
    return 2

def one_of_nine(op: Nine) -> int:
    """One arm named, seven unaccounted for. Converting means writing arms nobody wants."""
    if isinstance(op, A1):
        return 1
    return 0

def chained(op: Nine) -> int:
    """Eight arms across an elif chain plus the else — total, and the chain must read as ONE
    decision rather than as eight selections."""
    if isinstance(op, A1):
        return 1
    elif isinstance(op, A2):
        return 2
    elif isinstance(op, A3):
        return 3
    elif isinstance(op, A4):
        return 4
    elif isinstance(op, A5):
        return 5
    elif isinstance(op, A6):
        return 6
    elif isinstance(op, A7):
        return 7
    elif isinstance(op, A8):
        return 8
    return 9
'''

_MATCH_CORPUS = '''
type Pair = Left | Right

def falls_through(pair: Pair) -> int:
    match pair:
        case Left():
            return 1
        case _:
            return 0

def refuses(pair: Pair) -> int:
    match pair:
        case Left():
            return 1
        case _:
            raise ValueError("no")

def closed(pair: Pair) -> int:
    match pair:
        case Left():
            return 1
        case Right():
            return 2
        case unreachable:
            assert_never(unreachable)

def destructures_a_sequence(atoms: list[Left]) -> int:
    """A SEQUENCE match that merely names a class on the way through — not a union dispatch."""
    match atoms:
        case [Left() as first]:
            return 1
        case _:
            return 0

def destructures_a_mapping(answer: object) -> str:
    """The mapping twin. Both read as union dispatches if patterns are walked rather than
    inspected at their top level."""
    match answer:
        case {"key": Left() as found}:
            return "yes"
        case _:
            return "no"
'''


@pytest.mark.parametrize(
    ("function", "population"),
    [
        ("two_arms", "dispatch"),
        ("one_of_nine", "selection"),
        ("chained", "dispatch"),
    ],
)
def test_arity_tells_a_two_arm_decision_from_one_arm_of_nine(function, population, tmp_path):
    """The discriminator the first census was missing, and the reason `cost.py` read as four
    dispatch sites when it has one.

    `two_arms` and `one_of_nine` name exactly ONE arm each and differ only in the size of the
    union behind them, so a rule counting tests cannot separate them. `chained` is the control in
    the other direction: eight tests across an elif chain are one decision, not eight."""
    assert _census_of(_ARITY_CORPUS, tmp_path)[function] == population


@pytest.mark.parametrize(
    ("function", "population"),
    [
        ("falls_through", "open-match"),
        ("refuses", "unclosed-match"),
        ("closed", None),
        ("destructures_a_sequence", None),
        ("destructures_a_mapping", None),
    ],
)
def test_a_match_that_falls_through_is_the_population_isinstance_cannot_see(
    function, population, tmp_path
):
    """`ReplayHandler` mis-dispatched `Respawn` because an unmatched arm fell to the leaf path.

    There is no `isinstance` in that shape, so an `isinstance` census scores it zero — which is
    what the first version of this one did, over 23 sites in `src/`. The last two rows are the
    false positives the top-level rule removed: a sequence and a mapping pattern name a class
    while dispatching on something else entirely.

    `closed` is `None` because a `match` already ended by `assert_never` is done, not a finding."""
    import ast

    from effective.lint import totality_census

    module = tmp_path / "subject.py"
    module.write_text(_MATCH_CORPUS)
    tree = ast.parse(_MATCH_CORPUS)
    owner = {
        m.lineno: fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        for m in ast.walk(fn)
        if isinstance(m, ast.Match)
    }
    found = {owner[s.line]: s.population for s in totality_census([module]) if s.line in owner}
    if population is None:
        assert found.get(function) in (None, "closed-match"), (
            f"{function} should not be reported as work"
        )
    else:
        assert found[function] == population


def test_a_bound_wildcard_over_a_Never_function_reads_as_CLOSED(tmp_path):
    """A `-> Never` helper proves totality exactly as `assert_never` does, and the census has to
    see both.

    This repo spells it `assert_never`, and a census keyed on the callee's name would report
    any other spelling as outstanding work. The property is about the ARGUMENT: any `-> Never`
    callable handed a bound wildcard closes a match, and a reader who writes one should not
    have to discover that the meter cannot see it."""
    source = '''
from typing import Never

type Pair = Left | Right

def refuse(op: Never) -> Never:
    raise TypeError("no")

def proves(pair: Pair) -> int:
    match pair:
        case Left(): return 1
        case Right(): return 2
        case unreachable: refuse(unreachable)

def passes_the_subject_instead(pair: Pair) -> int:
    """Re-widens the type — `ty` accepts it and the proof is gone."""
    match pair:
        case Left(): return 1
        case Right(): return 2
        case unreachable: refuse(pair)

def discards_the_binding(pair: Pair) -> int:
    match pair:
        case Left(): return 1
        case Right(): return 2
        case _: refuse(pair)
'''
    import ast

    from effective.lint import totality_census

    module = tmp_path / "subject.py"
    module.write_text(source)
    tree = ast.parse(source)
    owner = {
        m.lineno: fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        for m in ast.walk(fn)
        if isinstance(m, ast.Match)
    }
    found = {owner[s.line]: s.population for s in totality_census([module]) if s.line in owner}
    assert found["proves"] == "closed-match"
    assert found["passes_the_subject_instead"] != "closed-match", (
        "passing the subject re-widens the type; only the BINDING carries Never"
    )
    assert found["discards_the_binding"] != "closed-match"


def test_both_handler_tables_are_CLOSED_in_the_tree_as_it_stands():
    """The sweep's first target, asserted against the real files rather than a corpus.

    A regression here is not cosmetic: it means one of the two interpreters of `WorkflowOp` has
    stopped proving it handles every arm, which is the defect the tables were written to fix.
    Both conditions that make the proof live are load-bearing and neither is visible from
    reading — the union annotation on `op`, and the bound wildcard reaching `refuse_unknown_op`.
    `replay._drive` had the second without the first and was green while proving nothing."""
    from effective.lint import totality_census

    # Scans `src`, not `src/effective/handlers`: union membership is a repo-wide fact and the
    # census says so — `WorkflowOp` is declared in `ops.py`, so a handlers-only scan cannot see
    # the union these tables dispatch and drops them from the census entirely. Narrowing the path
    # made this test pass for a while by asking a question the instrument does not answer.
    closed = {
        (s.filename.rsplit("/", 1)[-1], s.population)
        for s in totality_census(["src"])
        if s.population.endswith("-match") and s.tested == "op" and "/handlers/" in s.filename
    }
    assert ("recording.py", "closed-match") in closed
    assert ("replay.py", "closed-match") in closed


def test_no_union_match_in_src_falls_through_silently():
    """`open-match` is empty, and this is the population the whole plan was for.

    `ReplayHandler` mis-dispatched `Respawn` because an unmatched arm fell to the leaf path. This
    asserts no `match` over a declared union in `src/` still ends that way — a wildcard that
    catches everything, does not raise, and does not prove anything.

    Kept separate from the baseline gate because the baseline tolerates its own entries by
    design, and this one population should stay at zero rather than be tolerated at a number."""
    from effective.lint import totality_census

    open_matches = sorted(
        f"{s.filename}:{s.line} over {'|'.join(s.types)}"
        for s in totality_census(["src"])
        if s.population == "open-match"
    )
    assert not open_matches, "a union match fell through silently:\n" + "\n".join(open_matches)


def test_an_escape_needs_a_CATEGORY_the_rule_knows(tmp_path):
    """`# lint: totality(selection) — …` opts a row out; prose and a bare token do not.

    Nine of the first twenty-three escape reasons did not describe the live decision, and they
    were not nine slips: the escape took free prose and validated nothing, so `"churn"` became a
    category and a SCHEDULE passed as a classification. Naming a category from a fixed set makes
    the escape a claim the rule can check — and it forces the honest outcome when a site is a real
    dispatch, because there is no category for *I would rather not*."""
    from effective.lint import check_totality

    source = (
        "type Pair = Left | Right\n\n"
        "def f(pair: Pair) -> int:\n"
        "    {comment}\n"
        "    if isinstance(pair, Left):\n"
        "        return 1\n"
        "    return 2\n"
    )
    module = tmp_path / "subject.py"
    for comment, escapes in [
        ("# an ordinary remark about the code", False),
        ("# lint: totality — a selection, read and declined", False),
        ("# lint: totality(churn) — I would rather not", False),
        ("# lint: totality(selection) — one arm chosen, the other does nothing", True),
    ]:
        module.write_text(source.format(comment=comment))
        assert bool(not check_totality([module])) is escapes, comment


def test_every_escape_in_src_names_a_category_and_a_reason():
    """The gate `just lint` runs, asserted against the real tree.

    Two failure modes it keeps out, both of which the free-prose version admitted: a category the
    rule does not know, and a `deferred` that never says what unblocks it — the second is how a
    temporary exclusion becomes permanent by being forgotten."""
    from effective.lint import check_totality_escapes

    bad = [f"{v.filename}:{v.line} {v.message}" for v in check_totality_escapes(["src"])]
    assert not bad, "\n".join(bad)


_ORDERED_ARMS_CORPUS = '''
type Pair = Left | Right

def shadowed(pair: Pair) -> int:
    """Broad first: the refined arm can never match."""
    match pair:
        case Left():
            return 0
        case Left(tag="x"):
            return 1
        case Right():
            return 2

def absorbed(pair: Pair) -> int:
    """Refined first: correct, and deleting the refined arm stays exhaustive."""
    match pair:
        case Left(tag="x"):
            return 1
        case Left():
            return 0
        case Right():
            return 2

def guarded(pair: Pair) -> int:
    """A guard already declares the arm partial — neither hazard."""
    match pair:
        case Left(tag="x") if pair.ready:
            return 1
        case Left():
            return 0
        case Right():
            return 2

def alternation(pair: Pair) -> int:
    """The broad arm hides inside `A() | B()`, which a bare-MatchClass reader misses."""
    match pair:
        case Left(tag="x"):
            return 1
        case Right() | Left():
            return 0

def qualified(node: mod.Thing) -> int:
    """A QUALIFIED class pattern parses as an `Attribute`, not a `Name`."""
    match node:
        case mod.Thing():
            return 0
        case mod.Thing(id="x"):
            return 1
'''


def test_ordered_arms_tells_shadowed_from_absorbed_from_guarded(tmp_path):
    """A `match` is an ORDERED table, and a refined arm beside a broad one has two failure modes.

    `shadowed` is dead code and is gated. `absorbed` is correct — and is the one that matters,
    because deleting its refined arm leaves the match exhaustive, so `ty` says nothing and only a
    named test pin guards it. `guarded` is neither: a guard already declares its arm partial.
    `alternation` pins a real under-report: the broad arm inside `A() | B()` is invisible to a
    reader that only looks at bare class patterns, scoring 3 sites as 2."""
    from effective.lint import ordered_arm_report

    module = tmp_path / "subject.py"
    module.write_text(_ORDERED_ARMS_CORPUS)
    import ast

    tree = ast.parse(_ORDERED_ARMS_CORPUS)
    owner = {
        c.pattern.lineno: fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        for m in ast.walk(fn)
        if isinstance(m, ast.Match)
        for c in m.cases
    }
    found = {(owner[v.line], kind) for kind, v in ordered_arm_report([module])}
    assert found == {
        ("shadowed", "shadowed"),
        ("absorbed", "absorbed"),
        ("alternation", "absorbed"),
        ("qualified", "shadowed"),
    }, f"got {sorted(found)}"


def test_the_gate_catches_only_the_ALWAYS_WRONG_direction(tmp_path):
    """`--ordered-arms` fires on shadowed and not on absorbed, and that asymmetry is the design.

    Absorbed is correct code. Gating it would fire on three live sites the day it landed and teach
    people to silence it, which is the denylist-by-pragma shape. It is reported instead, and the
    remedy is a pin per site."""
    from effective.lint import check_ordered_arms

    module = tmp_path / "subject.py"
    module.write_text(_ORDERED_ARMS_CORPUS)
    flagged = check_ordered_arms([module])
    assert [v.rule for v in flagged] == ["ordered-arms-shadowed"] * 2, (
        "both dead arms — the plain `Left` one and the QUALIFIED `mod.Thing` one"
    )


def test_no_shadowed_arm_in_src():
    """The gate's live claim, asserted against the real tree rather than a corpus.

    A shadowed arm is unreachable code inside a decision table, which is the one thing a totality
    sweep must never leave behind."""
    from effective.lint import check_ordered_arms

    dead = [f"{v.filename}:{v.line} {v.text}" for v in check_ordered_arms(["src"])]
    assert not dead, "a refined arm that can never match:\n" + "\n".join(dead)


def test_every_file_that_declares_a_layer_is_reachable_by_the_layer_gate():
    """The coverage question for layers, which `--role-coverage` already answers for workflows.

    `--layers` scans a CURATED list of five names. `effective/tape.py:417` declares an `@op_layer`
    and is not on it, so a broad `except` there, which would swallow the durable suspend signal
    on the REPLAY path, is caught when the file is named explicitly and missed by `just lint`.
    Measured both ways: `EXIT=1` handed the path, `EXIT=0` as invoked.

    This is the general failure mode *a domain enumerated as spellings of one referent is always
    one spelling short*. `lint.declares_layer`'s own docstring names it in particular terms,
    calling itself "the content-side twin of `is_layer_role(name)`" because "a file the coding
    machine writes has no name in `LAYER_ROLE_SRCS`"; this test composes the two into a gate.

    Asserted over the tree with `declares_layer` and `LAYER_ROLE_SRCS` rather than by calling
    `check_layer_coverage`, so it asserts the PROPERTY and holds if the gate is ever rewritten. A
    pin written against a gate's own name raises while the gate is absent, and a strict xfail
    treats a raise as an ordinary xfail."""
    from effective.lint import LAYER_ROLE_SRCS, declares_layer

    unreachable = sorted(
        str(path)
        for path in Path("src").rglob("*.py")
        if declares_layer(path.read_text())
        and not any(str(path).endswith(name.removeprefix("src/")) for name in LAYER_ROLE_SRCS)
    )

    assert not unreachable, (
        f"these files declare a layer the layer-authority gate cannot reach: {unreachable}"
    )


def _forged(source: str):
    """Run `--forged-join` over one throwaway file."""
    import tempfile
    from pathlib import Path

    from effective.lint import check_forged_joins

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.py"
        path.write_text(source)
        return check_forged_joins([path])


def test_joining_rendered_terms_is_flagged():
    """The shape that had four instances, only one of which was found by running anything.

    A foreign term joins its payload with `:`, so `;`.join over rendered terms emits
    `$awaitEvent;review:m1` where the engine wrote `$awaitEvent:review:m1` -- a different key
    that still parses, which is why no round-trip assertion reddens on it.
    """
    [violation] = _forged("x = TERM_SEPARATOR.join(t.render() for t in terms)\n")
    assert violation.rule == "forged-join"
    assert "ParsedKey" in violation.message  # it names the fix, not just the sin

    # every real spelling the five sites used
    assert _forged("x = TERM_SEPARATOR.join(term.render() for term in terms)\n")
    assert _forged("x = TERM_SEPARATOR.join(t.render() for t in terms[i + 1 :])\n")
    assert _forged("x = TERM_SEPARATOR.join([t.render() for t in terms])\n")

    # ... and the one the first version was BLIND to, which is the shape that actually shipped:
    # `_projected` appended in a loop and joined a bare name, so nothing was in the argument for a
    # spelling-based scan to find.
    assert _forged(
        "out = []\nfor t in terms:\n    out.append(t.render())\nx = TERM_SEPARATOR.join(out)\n"
    )


def test_the_legitimate_joins_are_designated_BY_POSITION():
    """The negative that keeps the rule from being a ban on a function.

    Three live sites join with the term separator and are correct: `_past_frames_text` rejoins
    pieces a TEXT split produced, `_render_skeleton` joins template parts, and `_project_walk`
    rejoins its own split after projecting each piece, where a dropped coordinate leaves the key
    language entirely. None holds a `Term`, so none has a field to lose. They are exempt by
    (file, function) rather than by what their argument looks like -- a spelling-based exemption
    is what let the shipped defect through, so the exemption is spelled the same way the rule is.
    """
    from effective.lint import DESIGNATED_JOINS

    assert {
        ("src/effective/keys/frame.py", "_past_frames_text"),
        ("src/effective/keys/registry.py", "_render_skeleton"),
        ("src/effective/keys/registry.py", "_project_walk"),
    } == DESIGNATED_JOINS
    # A join in an UNDESIGNATED position is a violation whatever it holds -- including the two
    # spellings above, moved elsewhere.
    assert _forged("x = TERM_SEPARATOR.join([*kept, rest])\n")
    assert _forged("x = TERM_SEPARATOR.join(parts)\n")


# --- configuration a larger tree adds -------------------------------------------------------


def _tree(tmp_path, pyproject: str):
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text(pyproject)
    (tmp_path / "priv").mkdir()
    return tmp_path


def test_a_configured_value_of_the_wrong_shape_raises(tmp_path, monkeypatch):
    from effective.lint import configured

    monkeypatch.chdir(_tree(tmp_path, '[tool.effective.lint]\nextra_paths = "priv"\n'))
    with pytest.raises(TypeError, match="extra_paths"):
        configured("extra_paths")


def test_configured_roots_resolve_against_the_pyproject_that_names_them(tmp_path, monkeypatch):
    from effective.lint import configured_roots

    root = _tree(tmp_path, '[tool.effective.lint]\nextra_paths = ["priv"]\n')
    (root / "sub").mkdir()
    monkeypatch.chdir(root / "sub")
    assert configured_roots("--deps", ["."]) == []  # `priv` is under neither root given
    assert configured_roots("--deps", ["x"]) == ["../priv"]


def test_a_configured_root_that_does_not_exist_raises(tmp_path, monkeypatch):
    from effective.lint import configured_roots

    monkeypatch.chdir(_tree(tmp_path, '[tool.effective.lint]\nextra_paths = ["gone"]\n'))
    with pytest.raises(FileNotFoundError, match="gone"):
        configured_roots("--deps", ["src"])


def test_a_nearer_pyproject_without_the_table_does_not_hide_the_roots(tmp_path, monkeypatch):
    from effective.lint import configured

    root = _tree(tmp_path, '[tool.effective.lint]\nextra_paths = ["priv"]\n')
    (root / "member").mkdir()
    (root / "member" / "pyproject.toml").write_text('[project]\nname = "member"\n')
    monkeypatch.chdir(root / "member")
    assert configured("extra_paths") == ["priv"]
