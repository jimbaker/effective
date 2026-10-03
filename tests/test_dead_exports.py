"""The export sweep, pinned on the defects it shipped with.

ROLE: adversarial. The enumerator's own pins live in `tests/test_whouses.py`.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts import dead_exports as de  # noqa: E402
from scripts import whouses as wh  # noqa: E402

pytestmark = pytest.mark.adversarial

GRAMMAR = "src/effective/keys/grammar.py"


def test_a_computed_all_is_not_readable_from_source() -> None:
    """`_declared_all` reads the tree, so a module that COMPUTES its `__all__` declares nothing it
    can see — and that is the honest answer rather than a zero.

    It is also why the export census reconciles the way it does. 421 of the 444 entries under
    `src/` are spelled; the other 23 come from two modules whose `__all__` is
    `[n for n in dir() if not n.startswith("_")]`, evaluated at import, and both of them leak
    `StrEnum` into the public surface (the dynamic `__all__` is not yet spelled out).

    **The integer moves when the public surface does, which is the pin working**, so a diff that
    moves it says which name and why. What the count covers: the coordinate roles are spelled
    by `keys/__init__` and again by `effective` itself, because a call site names one instead of
    `Segment` and an example imports from the package;
    `Role`, the protocol a drop set is typed over, is spelled beside them, and `Ordinal` twice
    over, since the fold keeps what it numbers. `coding.tier` exports `UnsafeTreePath`, which its
    exported `run_tree_command` raises, and the two checks that raise it. `PROJECTION_SIGIL` is
    NOT among them, and that is the shape to expect from a name that belongs to the grammar:
    `grammar.py` computes no `__all__` and spells none, so a constant living there is invisible
    to this census however public it is. `coding.specs` exports `coding_bind` beside `coding_read`,
    the two halves of what a worker over the coding table is handed and hands back."""
    spelled = {
        module: de._declared_all(module)
        for module in sorted(Path("src").rglob("*.py"))
        if de._declared_all(module)
    }
    assert sum(len(v) for v in spelled.values()) == 421, "the AST-readable half of the surface"
    for computed in ("src/effective/coding/states.py", "src/effective/prose/states.py"):
        assert de._declared_all(Path(computed)) == [], "computed at import, invisible to the tree"


def test_the_export_sweep_finds_a_name_with_no_outside_consumer() -> None:
    """`graphview.has_cycle` is exported and used only inside its own file.

    It is a good pin because it is known independently of the tool: two hand sweeps agree on it,
    so a disagreement here is about the tool, not the code."""
    index = wh._name_index(["src", "tests"])
    home = Path("src/effective/graphview.py").resolve()
    assert "has_cycle" in de._declared_all(Path("src/effective/graphview.py"))
    assert de._defined_at(Path("src/effective/graphview.py"), "has_cycle") is not None
    outside = {site[0].resolve() for site in index.get("has_cycle", [])} - {home}
    assert not outside, f"syntactically reachable only from its own file; saw {outside}"


def test_a_module_qualified_use_is_a_consumer() -> None:
    """The sweep contradicted a single-symbol query in one screen, and the query was right.

    `elkjs.available` is called at `dashboard.py:355` as `elkjs.available()`, which is an
    `ast.Attribute` and not an `ast.Name`, so an index built from `Name` alone reported it unused.
    Nine of the thirty-nine published zeros were this."""
    index = wh._name_index(["src", "tests"])
    module = Path("src/effective/graphlayout/elkjs.py")
    line = de._defined_at(module, "available")
    assert line is not None, "elkjs declares it and defines it"
    server = wh.Server(Path.cwd())
    try:
        consumers = de._consumers("available", module.resolve(), line, index, server)
    finally:
        server.close()
    assert Path("src/effective/dashboard.py") in consumers


def test_an_import_alias_is_not_consumption_but_a_renamed_import_is() -> None:
    """Two opposite errors in one place, and a rule that fixes one can cause the other.

    A package `__init__` that re-exports a name resolves to it and does nothing with it, so
    counting the alias line made six pure re-exports look consumed. But a file that renames its
    import — `from effective.budget import as_policy as budget_policy` — spells every real use
    under the local name, so discounting aliases alone would drop it. `budget.as_policy` has three
    consumers and every one of them is that shape."""
    index = wh._name_index(["src", "tests"])
    module = Path("src/effective/budget.py")
    line = de._defined_at(module, "as_policy")
    assert line is not None, "budget declares it and defines it"
    server = wh.Server(Path.cwd())
    try:
        consumers = de._consumers("as_policy", module.resolve(), line, index, server)
    finally:
        server.close()
    assert Path("tests/test_govern.py") in consumers, "renamed on import, used under the new name"

    # The other half, and it is the one a mutation survived: `graphlayout/__init__.py:36` imports
    # `classify` and re-exports it at `:64`, doing nothing else with it. Counting that line made
    # six pure re-exports look consumed.
    prepare = Path("src/effective/graphlayout/prepare.py")
    line = de._defined_at(prepare, "classify")
    assert line is not None
    server = wh.Server(Path.cwd())
    try:
        assert not de._consumers("classify", prepare.resolve(), line, index, server)
    finally:
        server.close()
    facade = wh._name_index(["src/effective/graphlayout"])["classify"]
    assert {site.kind for site in facade if site.path.name == "__init__.py"} == {"alias"}


def test_the_two_paths_agree_on_who_consumes_a_name() -> None:
    """The single-symbol answer and the sweep's answer are one question asked twice, so a
    disagreement is a defect in one of them.

    The comparison has to RESOLVE, because two weaker versions of this assertion are wrong. File
    containment alone is satisfied by the import line, which carries the original spelling;
    excluding imports is still satisfied by a homonym, because `test_govern.py` also
    calls `permission.as_policy`. Only asking `ty` what each site means separates them."""
    module = Path("src/effective/budget.py")
    line = de._defined_at(module, "as_policy")
    assert line is not None
    index = wh._name_index(["src", "tests"])
    reads, _keywords = wh.member_sites("as_policy", ["src", "tests"], member=False)
    server = wh.Server(Path.cwd())
    try:
        by_sweep = {
            path.resolve()
            for path in de._consumers("as_policy", module.resolve(), line, index, server)
        }
        by_symbol = {
            path.resolve()
            for path, site_line, column, _text in reads
            if server.definition(path.resolve(), site_line, column)
            == (str(module.resolve()), line)
            and (path.resolve(), site_line + 1)
            not in {
                (site.path.resolve(), site.line)
                for site in index["as_policy"]
                if site.kind == "alias"
            }
        }
    finally:
        server.close()
    assert by_sweep, "the sweep finds consumers, so there is something to agree about"
    assert by_sweep <= by_symbol, (
        f"consumers the symbol path never resolves to the target in: {by_sweep - by_symbol}"
    )


def test_a_pep_695_type_alias_is_defined_where_it_is_written(tmp_path: Path) -> None:
    """`type X = …` needs its own arm: without one, an `__all__` entry naming an alias is labelled
    *re-exported, defined elsewhere*, falsely, and drops out of the consumer analysis. `src/`
    holds 67 top-level type aliases, a construct this repo showcases."""
    module = tmp_path / "aliases.py"
    module.write_text('__all__ = ["Dollars"]\n\ntype Dollars = float\n')
    assert de._declared_all(module) == ["Dollars"]
    assert de._defined_at(module, "Dollars") == 3


def test_the_documented_command_uses_the_documented_roots(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The usage documents `--exports <module>`, and the tool answers over `src` and `tests`.

    A `roots` declared `nargs="*", default=["src"]` is never empty, so an `or ["src", "tests"]`
    beside it never fires, and `graphview.py` gives 11 zeros the documented way against 4 with
    explicit roots. Only running `main` can show it, so this runs `main`."""
    assert de.main(["src/effective/graphview.py"]) == 0
    assert "over src tests" in capsys.readouterr().out


def test_the_default_root_is_stated_once() -> None:
    """The other half of `tests/test_whouses.py`'s pin: this tool owns `src tests`, and owns it
    once. `src` alone answers 109 where `src tests` answers 39."""
    assert de.EXPORT_ROOTS == ("src", "tests")
    assert not hasattr(de, "SYMBOL_ROOTS")
