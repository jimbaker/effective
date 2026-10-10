"""The seam finder's enumerator, pinned on the five defects it shipped with.

ROLE: adversarial. Every test here is a measured defect, so a pass means only that one
particular way of lying has been closed. The tool enumerates candidate sites syntactically and
resolves them with `ty`, and reports and skills quote its output as evidence.

Most of it starts no LSP session. The defects are in which sites get enumerated and where the
cursor lands, which is decidable from source, and the homonym pin would otherwise need a server to
state. The last two tests do need one, because what they pin is a fact about `ty` rather than about
this script; `just check` already requires the same toolchain for `uvx ty check`, so they fail
rather than skip when it is absent.
"""

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from effective.budget import as_policy

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts import whouses as wh  # noqa: E402

pytestmark = pytest.mark.adversarial


def _sites(name: str, roots: list[str], *, member: bool) -> list[tuple]:
    """Both halves of `member_sites`, which is what a caller comparing shapes wants."""
    reads, keywords = wh.member_sites(name, roots, member=member)
    return reads + keywords


GRAMMAR = "src/effective/keys/grammar.py"
FORK = "src/effective/fork.py"
API = "src/effective/api.py"
BUDGET = "src/effective/budget.py"


def _line_of(path: str, needle: str) -> int:
    """The 1-based line in `path` holding `needle`, which must occur exactly once.

    Every pin below names a construct in another module, and a literal line number goes stale the
    moment anything above it moves; four of them did, on a docstring edit. Locating the construct
    keeps the pin about the construct.
    """
    hits = [n for n, text in enumerate(wh._lines(Path(path)), 1) if text.strip() == needle]
    assert len(hits) == 1, f"{needle!r} in {path}: {len(hits)} lines, want exactly 1"
    return hits[0]


KIND_OF = "def kind_of(text: str) -> Kind | None:"
TERM_SEPARATOR = 'TERM_SEPARATOR = ";"'
SCOPE_PROPERTY = "def scope(self) -> Scope | None:"
GRANTED_FIELD = "granted: float"
LINT = "src/effective/lint.py"
NAME_PATTERN = 'case ast.Name(id="None"):'
GATHER_FIELD = "gather: int"
CLEARED_PATTERN = "case Cleared(granted=advanced, trips=advanced_trips):"
GATHER_PATTERN = "case GatherBranch(gather=g, index=i):"
SCOPE_PATTERN = "case Key(scope=None) if _leading_tag(name.stored()) in RESERVED_AUTHORITY_TAGS:"


def test_a_module_level_def_is_not_swept_as_an_attribute() -> None:
    """`kind_of` has six callers and the tool reported zero.

    `_named_at` set `member = not isinstance(node, ast.ClassDef)`, so every module-level function
    was searched as `$X.kind_of`, a shape no call site has. 819 module-level defs under
    `src/effective` and `src/agent` answered that way, and the answer was `USES (0)`.

    The name is a good one to pin because it is ambiguous: `graphview.py:178` declares a second
    `kind_of`, and four of the eleven candidates belong to it. The resolver separates them, which
    is the whole reason the enumerator is allowed to over-generate."""
    name, _source, member = wh._named_at(Path(GRAMMAR), _line_of(GRAMMAR, KIND_OF))
    assert name == "kind_of"
    assert not member, "a module-level def is reached by name, never through a dot"

    at = {(str(p), line + 1) for p, line, _col, _text in _sites(name, ["src"], member=member)}
    # Located rather than numbered, for the reason `_line_of` gives: the marker pin was a literal
    # and went stale twice on docstring edits in the file it names.
    marker = "src/effective/keys/marker.py"
    calls = (
        (
            GRAMMAR,
            _line_of(GRAMMAR, "return bool(TAG.match(text)) and kind_of(text) is Kind.NAME"),
        ),
        (GRAMMAR, _line_of(GRAMMAR, "if (kind := kind_of(text)) is None:")),
        (marker, _line_of(marker, "if (kind := kind_of(value)) in (Kind.UUID, Kind.DIGEST):")),
    )
    for call in calls:
        assert call in at, f"{call} calls it"
    # Located, not numbered, for the same reason as the three above: a literal went stale the
    # first time anything was added to the file's imports.
    graphview = "src/effective/graphview.py"
    homonym = (graphview, _line_of(graphview, "kind=kind_of(key),"))
    assert homonym in at, "the homonym is enumerated, for ty to reject"


def test_a_module_level_constant_is_a_target() -> None:
    """`_named_at` walked only `FunctionDef | AsyncFunctionDef | ClassDef`, so a constant was
    refused outright, and the refusal named the wrong reason: it told the caller to point at a
    `def` or `class` line when they already were pointing at the definition."""
    name, _source, member = wh._named_at(Path(GRAMMAR), _line_of(GRAMMAR, TERM_SEPARATOR))
    assert name == "TERM_SEPARATOR"
    assert not member

    sites = _sites(name, ["src"], member=member)
    assert len(sites) >= 40, f"41 sites when measured; enumerated {len(sites)}"


def test_a_line_that_really_defines_nothing_is_still_refused() -> None:
    """Widening the target kinds must not turn the tool back into a text tool. A docstring line
    defines nothing, and deriving a name from the words on it is the defect this tool was built
    to fix, one rung up."""
    with pytest.raises(SystemExit):
        wh._named_at(Path(GRAMMAR), _line_of(GRAMMAR, KIND_OF) + 1)


def test_a_multi_line_receiver_does_not_crash() -> None:
    """`column = text.index(f".{name}") + 1` assumed the attribute sits on the match's first line.

    `DurableHandler(\\n    ...\\n).run(...)` in `fork.py` puts `.run` below its opening line. Two
    of 169 member names under `src/effective` raised `ValueError: substring not found`, and this
    is one of them."""
    sites = _sites("run", ["src"], member=True)
    assert sites, "`.run` has call sites"

    at = {(str(p), line + 1) for p, line, _col, _text in sites}
    assert (FORK, _line_of(FORK, ").run(workflow)")) in at, (
        "the `).run(` line, below the receiver's first"
    )

    for path, line, column, _text in sites:
        source = path.read_text().splitlines()[line]
        assert source[column : column + 3] == "run", (
            f"{path}:{line + 1} column {column} points at {source[column : column + 3]!r}"
        )


def test_a_keyword_pattern_is_enumerated_and_aimed_at_its_class() -> None:
    """The enumerator saw none of the 159 keyword patterns under `src/effective`, and the obvious
    repair is the wrong one.

    `ty` does not decline to answer at a keyword position: it answers with a HOMONYM. At
    `fork.py`'s `case Cleared(granted=advanced)`, the `granted` keyword resolves to `trip_at`'s
    parameter of that name, while `Cleared.granted` is the field in `budget.py`. Wrong file, wrong
    kind of symbol, and nothing announces it.

    So the cursor goes on the CLASS, which `ty` resolves reliably, and PEP 634 supplies the rest:
    a keyword in a class pattern MUST name an attribute of the matched class, so the attribution
    is a language guarantee rather than an inference."""
    cleared = _line_of(FORK, CLEARED_PATTERN)
    sites = _sites("granted", ["src"], member=True)
    at = {(str(p), line + 1): col for p, line, col, _text in sites}
    assert (FORK, cleared) in at, "the `granted=` keyword pattern is a use of `Cleared.granted`"

    line = wh._lines(Path(FORK))[cleared - 1]
    column = at[(FORK, cleared)]
    assert line[column:].startswith("Cleared"), (
        f"the cursor must land on the class, not the keyword; it landed on {line[column:][:12]!r}"
    )


def test_every_keyword_pattern_is_reachable() -> None:
    """The enumerator's grammar is its domain, so the count is the claim worth pinning. `ast` and
    ast-grep agreed on 348 class patterns under `src/effective` once dotted class names were
    counted — 294 bare and 54 like `case ast.Name(id=name)`, which the fix must also aim."""
    reached = 0
    for path in sorted(Path("src/effective").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.MatchClass):
                reached += len(node.kwd_attrs)
    # The integer moves when the corpus does, which is the pin working: `lint._bound_roles`,
    # `_assigned_pairs`, `_identifier` and `check_python_codegen` match on assignment and
    # expression shapes, `react.default_decide` dispatches on which role its turn carries,
    # `KeyMap.project` matches on whether it was handed a key or a projection of one,
    # `DurableHandler._run_step` picks the spawn tool out of its `CallTool` arm,
    # `spawning.join_answer` reads a child's value, its refusals or what a cancel left,
    # `DurableHandler.run` and `spawning.run_child` read a walk that finished or continued, and
    # `react.default_act` reads the tool and diagnostic of a refusal the deployment answered with.
    # `trampoline.machine_frames` reads a `d:` term and the `state:` term after it.
    # `budget.refuse_two_drivers` reads the name of a budget policy it refuses.
    # `choice.stored_endings` reads a refusal's reason and an error's text, and
    # `handlers.base.settled` and `ending_of` read the error a race branch's slot holds.
    # `DurableHandler._await_bounded` reads the payload an `Arrived` carries, and
    # `sandbox.inspect_only` destructures a wait's deadline in both of its arms so a wait a
    # counterfactual cannot answer is a missing arm rather than a dropped field.
    # `telemetry._op_span` reads each op's own fields, and
    # `judgment.battery` reads a hole's value, expression and spec and each question's criteria,
    # and `judgment.checked` reads a choice question's criteria and the choice it got.
    # `interpreters.jev` reads each question's criteria and each answer Jev returns.
    # `cache.op_digest` reads each op's content and schema, `_encoded` a message's role and
    # content, and `Cache.chooses` a tool's name.
    # `react._guarded` reads a refusal's reason and an ask's question.
    # `graphview._statement` reads a sequence statement's hole.
    # `markdown._render` reads a hole's value to compose a nested template.
    # `Fetched.transient` reads an unreadable answer's code, `research._readable` a page's text.
    # `telemetry._guarded` reads a hole's value, expression, conversion and spec to rebuild it.
    # `lint.check_sdk_private_source` reads an attribute's receiver and name, and an import's
    # module and names.
    # `DurableHandler._keyed` reads a tool call's name and args.
    assert reached == 255, f"the corpus this fix was measured on held 255; found {reached}"

    seen = {
        (str(p), line + 1)
        for name in {"granted", "trips", "scope"}
        for p, line, _col, _text in _sites(name, ["src/effective"], member=True)
    }
    assert (FORK, _line_of(FORK, CLEARED_PATTERN)) in seen


def test_the_pinned_ast_grep_is_the_one_that_runs() -> None:
    """The CLI was pinned because this script shells to it, and the script still said `ast-grep`.

    A bare name resolves against PATH, so the enumerator ran on whatever the machine had —
    linuxbrew's 0.45.1 here against the project's 0.42.3 — while the pin's own comment named this
    file as the reason it exists. The two agree on every rule this tool sends today (6,711
    attribute identifiers under `src/`, both versions), which is why it was invisible."""
    binary = getattr(wh, "ast_grep", lambda: "ast-grep")()
    assert Path(binary).is_absolute(), f"resolve the pinned CLI, not a PATH name: {binary!r}"
    assert Path(binary).parent == Path(sys.executable).parent, (
        f"{binary} is not the interpreter's sibling"
    )

    expected = subprocess.run(
        [binary, "--version"], capture_output=True, text=True, check=True
    ).stdout.split()[-1]
    assert expected.startswith("0.42."), f"co-versioned with ast-grep-py 0.42.x; got {expected}"


def test_reference_search_is_blind_to_keyword_patterns() -> None:
    """`ty`'s own reference search cannot replace the candidate set, which is why it does not.

    `textDocument/references` reaches files nobody opened: 30 locations for `Key.scope` across
    `src/` and `tests/`, against a candidate set bounded by the roots you pass. It is also blind to
    exactly the construct the enumerator was just taught to see, so each reaches what the other
    cannot. Measured on three attributes with a keyword-pattern use; all three are missing, and
    `GatherBranch.gather` answers zero against a live use in the same file."""
    ops = "src/effective/ops.py"
    cases = [
        (BUDGET, GRANTED_FIELD, "granted", (FORK, _line_of(FORK, CLEARED_PATTERN)), 1),
        (GRAMMAR, SCOPE_PROPERTY, "scope", (ops, _line_of(ops, SCOPE_PATTERN)), 30),
        (API, GATHER_FIELD, "gather", (API, _line_of(API, GATHER_PATTERN)), 0),
    ]
    server = wh.Server(Path.cwd())
    try:
        for file, construct, attribute, keyword_site, floor in cases:
            line = _line_of(file, construct)
            column = wh._lines(Path(file))[line - 1].index(attribute)
            found = server.references(Path(file).resolve(), line - 1, column)
            # EXACT, not a floor. "The keyword site is not in nothing" holds of every empty set,
            # so a floor of zero asserts nothing at all and two of these three had one. The count
            # is also the finding: `GatherBranch.gather` answers ZERO against a live use.
            assert len(found) == floor, f"{attribute}: reference search returned {len(found)}"
            assert keyword_site not in found, (
                f"{keyword_site} is a keyword pattern; if `ty` has learned to see it, the "
                f"cross-check in `--references` can stop reporting it as a disagreement"
            )
            enumerated = {
                (str(p), site_line + 1)
                for p, site_line, _col, _text in _sites(attribute, ["src"], member=True)
            }
            assert keyword_site in enumerated, "the rule enumerator is what sees it"
    finally:
        server.close()


def test_a_use_under_ty_but_not_under_the_rules_is_still_a_use() -> None:
    """The other side of the same coin, so neither direction of the disagreement goes unpinned.

    `Key.scope` is read at `handlers/base.py:395`, a plain attribute read both enumerators see,
    and referenced 29 more times under `tests/`, which a `src`-rooted sweep cannot reach."""
    server = wh.Server(Path.cwd())
    try:
        row = _line_of(GRAMMAR, SCOPE_PROPERTY) - 1
        column = wh._lines(Path(GRAMMAR))[row].index("scope")
        found = server.references(Path(GRAMMAR).resolve(), row, column)
        base = "src/effective/handlers/base.py"
        assert (
            base,
            _line_of(base, "case AwaitEvent(name=name) if name.scope is Scope.SETTLEMENT:"),
        ) in found
        assert sum(1 for where, _line in found if where.startswith("tests/")) == 29
    finally:
        server.close()


def test_call_hierarchy_names_the_calling_function() -> None:
    """`callHierarchy/incomingCalls` answers a question neither rung can: not *which lines mention
    this* but *which functions call it*. `kind_of` comes back with `is_tag`, `__post_init__` and
    `axes`: the callers by name, workspace-wide, with no candidate set to bound them."""
    server = wh.Server(Path.cwd())
    try:
        row = _line_of(GRAMMAR, KIND_OF) - 1
        column = wh._lines(Path(GRAMMAR))[row].index("kind_of")
        callers = server.callers(Path(GRAMMAR).resolve(), row, column)
        assert {name for name, _path, _line in callers} >= {"is_tag", "__post_init__"}
    finally:
        server.close()


def test_a_class_head_the_cursor_cannot_land_on_is_refused() -> None:
    """The dotted-class offset was measured off the text before the paren, so whitespace moved it.

    `case mod . Name(...)` put the cursor on a space and a head split across lines put it on the
    module — filing a genuine use as a non-use, silently, which is the answer this tool exists to
    turn into a refusal. There are no instances in the tree; the point is that the shipped version
    of this defect was also latent until it was not."""
    assert wh._class_offset("Cleared(granted=x)") == 0
    assert wh._class_offset("ast.Name(id=x)") == 4
    assert wh._class_offset("mod . Name(id=x)") == 6
    with pytest.raises(SystemExit):
        wh._class_offset("mod.\n    Name(id=x)")


def test_a_scan_that_fails_refuses_instead_of_returning_nothing() -> None:
    """`check=False` plus `json.loads(stdout or "[]")` turned a missing root into an empty domain —
    this file's own named antipattern, inside the file that names it."""
    with pytest.raises(SystemExit):
        _sites("Key", ["no/such/directory"], member=False)


def test_a_nested_keyword_pattern_aims_at_the_inner_class(tmp_path: Path) -> None:
    """Containment forms a chain, so innermost is the greatest start — but `min` also type-checks.

    Under `min` the cursor lands on the OUTER class and a real use is filed elsewhere: measured on
    `Atom.text`, `USES (10)` becomes 8, with `registry.py:357,358` reattributed. Fifteen nested
    class patterns with an inner keyword live under `src/effective`, so the domain is not empty."""
    source = tmp_path / "nested.py"
    source.write_text(
        "class Inner:\n    id: int = 0\n\n\nclass Outer:\n    inner: Inner = Inner()\n\n\n"
        "def go(x: object) -> None:\n    match x:\n        case Outer(inner=Inner(id=k)):\n"
        "            print(k)\n"
    )
    sites = _sites("id", [str(tmp_path)], member=True)
    aimed = [(line, column) for _p, line, column, _t in sites if line == 10]
    assert aimed, "the keyword pattern is enumerated"
    text = source.read_text().splitlines()[10]
    assert text[aimed[0][1] :].startswith("Inner"), f"aimed at {text[aimed[0][1] :][:10]!r}"


def test_a_dotted_class_pattern_aims_at_its_last_segment() -> None:
    """`case ast.Name(id=x)` has to resolve `Name` and not the `ast` module: one is a class in
    typeshed and the other is a module, and both answer."""
    line = _line_of(LINT, NAME_PATTERN)
    sites = _sites("id", [LINT], member=True)
    at = {row: column for _p, row, column, _t in sites}
    assert line - 1 in at, f"{LINT}:{line}'s `case ast.Name(id=...)` is enumerated"
    assert wh._lines(Path(LINT))[line - 1][at[line - 1] :].startswith("Name")


def test_a_class_attribute_is_reached_through_a_dot() -> None:
    """`member` decides which sweep runs, so getting it wrong for a class attribute costs the
    keyword-pattern half AND the class target that makes those sites resolve."""
    name, _source, member = wh._named_at(Path(BUDGET), _line_of(BUDGET, GRANTED_FIELD))
    assert (name, member) == ("granted", True)
    free, _source, not_member = wh._named_at(Path(GRAMMAR), _line_of(GRAMMAR, TERM_SEPARATOR))
    assert (free, not_member) == ("TERM_SEPARATOR", False)


def test_the_attribute_rule_is_what_finds_a_plain_read(tmp_path: Path) -> None:
    """Dropping `inside: {kind: attribute, field: attribute}` leaves the bare-identifier sweep,
    which is a strict superset and so reddens nothing — the member path would silently become a
    slower way to ask a looser question. Dropping only `field` also takes the object side."""
    corpus = ["stored = 1", "print(stored)", "stored.other", "key.stored()", "nested.key.stored"]
    (tmp_path / "m.py").write_text("\n".join(corpus) + "\n")

    reads, keywords = wh.member_sites("stored", [str(tmp_path)], member=True)

    assert [text for _path, _line, _column, text in reads] == corpus[3:]
    assert keywords == []


def test_a_renamed_import_is_swept_under_its_local_name() -> None:
    """The single-symbol path never learned the rule the export sweep learned, and said so.

    `budget.as_policy` reported `USES (0)` directly beneath three `IMPORTED` lines, because every
    call site spells `budget_policy`. The export sweep answered 3 consumers for the same name in
    the same run — the tool contradicting itself, which is the tell that a rule landed in one place
    and not the other. `kind_of` was the same: 4 uses, with a fifth at `graphview.py:652` spelled
    `atom_kind`. 41 names under `src` and `tests` are imported under a different local name."""
    assert wh._aliases_of("as_policy", ["src", "tests"]) == {"budget_policy"}

    reads, _keywords = wh.member_sites("as_policy", ["src", "tests"], member=False)
    at = {(str(path), line + 1) for path, line, _col, _text in reads}
    govern = "tests/test_govern.py"
    call = (govern, _line_of(govern, "at_spend(0.010, budget_policy(budget)),"))
    assert call in at, "a call spelled under the local name"
    budget = "src/effective/budget.py"
    assert (budget, inspect.getsourcelines(as_policy)[1]) in at, (
        "and the definition, which the report drops"
    )


def test_the_enclosing_class_walk_agrees_with_the_definition_walk(tmp_path: Path) -> None:
    """The scope fix half-landed: `_named_at` walked the chain and `_enclosing_class` still walked
    one level, so a keyword site resolved to its class, was compared against `None`, and was filed
    as resolving elsewhere. The tool computed the right answer and discarded it."""
    module = tmp_path / "guarded.py"
    module.write_text(
        "from typing import TYPE_CHECKING\n\n\nclass Cleared:\n"
        "    if TYPE_CHECKING:\n        granted: int = 0\n"
    )
    _name, _source, member = wh._named_at(module, 6)
    assert member, "the N7 fix"
    assert wh._enclosing_class(module, 6) == 4, "and the walk beside it"


def test_a_hidden_directory_is_still_scanned(tmp_path: Path) -> None:
    """The stderr rule holds forward and not backward. A dot-directory, a `.gitignore`d file and a
    symlink are skipped writing neither stderr nor a non-zero exit, so the guard cannot see them —
    a silent under-count in the enumerator, which is the class this file exists for."""
    hidden = tmp_path / ".vendored"
    hidden.mkdir()
    (hidden / "mod.py").write_text("def only_here() -> int:\n    return 1\n")
    found = _sites("only_here", [str(tmp_path)], member=False)
    assert found, "a dot-directory under a scanned root is still source"


def test_a_def_ends_the_class_body_for_this_purpose(tmp_path: Path) -> None:
    """`_scoped` resets `in_class` at a `def`, and nothing pinned it: the tree holds no class
    attribute nested under a non-def statement, so both the fix and its absence look identical
    against `src/`. Constructed, they do not — dropping the reset makes a method local a member,
    which sweeps `$X.total` for a name no call site spells that way."""
    module = tmp_path / "reset.py"
    module.write_text(
        "class Holder:\n    attribute: int = 0\n\n"
        "    def method(self) -> int:\n        total = 1\n        return total\n"
    )
    assert wh._named_at(module, 2)[2] is True, "a class-body annotation is reached through a dot"
    assert wh._named_at(module, 5)[2] is False, "a method local is not, however deep the class is"


def test_the_default_root_is_stated_once() -> None:
    """A symbol query and an export sweep want different roots, and the defect was one default
    written twice with two values. Two scripts now hold one default each, so the shape cannot
    recur here; `tests/test_dead_exports.py` holds the other half."""
    assert wh.SYMBOL_ROOTS == ("src",)
    assert not hasattr(wh, "EXPORT_ROOTS"), "the export default left with the export sweep"
    assert wh._parser().parse_args(["m.py:1"]).roots == [], "argparse supplies no default"


def test_a_pep_695_type_alias_is_a_target() -> None:
    """`_defined_at` grew an `ast.TypeAlias` arm for the export sweep and `_named_at` did not, so
    the same construct resolved on one path and was refused on the other. `src/` holds 67 of them.
    """
    line = _line_of(API, "type Effect[T] = Generator[WorkflowOp, Any, T]")
    name, _source, member = wh._named_at(Path(API), line)
    assert (name, member) == ("Effect", False)


def test_the_reporters_run(capsys: pytest.CaptureFixture[str]) -> None:
    """Four reporters are reached by no other test, and two skills plus four reports quote their
    output. A smoke test is not a claim about the wording; it is a claim that the paths execute and
    that each prints the heading its callers cite.
    """
    wh._report_references({("a.py", 1), ("b.py", 2)}, {("a.py", 1), ("c.py", 3)})
    out = capsys.readouterr().out
    assert "REFERENCES, per ty (2)" in out
    assert "1 the rules found alone, 1 ty found alone" in out

    wh._report_timing([10.0, 20.0, 30.0], [0.1, 0.2, 0.3])
    out = capsys.readouterr().out
    assert "median ratio" in out
    assert "100x" in out

    assert wh._importers(Path(GRAMMAR), ["src"]) == 0
    assert "importers, found SYNTACTICALLY" in capsys.readouterr().out


def test_the_symbol_report_splits_uses_from_imports(capsys: pytest.CaptureFixture[str]) -> None:
    """`USES` counts uses. A bare-name sweep matches the definition's own line and every import of
    it, and folding those into one total is what made `kind_of` read seven against four calls."""
    args = wh._parser().parse_args([f"{GRAMMAR}:1"])
    home = (str(Path(GRAMMAR).resolve()), 130)
    used: list[tuple[Path, int, str, tuple[str, int] | None]] = [
        (Path(GRAMMAR), 200, "is_tag(text)", home),
        (Path(GRAMMAR), 12, "from .grammar import kind_of", home),
    ]
    wh._report(args, used, [], set(), [], {(GRAMMAR, 12)}, home)
    out = capsys.readouterr().out
    # The import line is a MEMBER of `used`, so the two headings partition it rather than
    # double-counting: one use, one import, and never "2 uses".
    assert "USES (1)" in out
    assert "IMPORTED (1), which is reach and not use" in out
