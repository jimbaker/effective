"""The altitude guard: the card IR is framework-free.

Enforces the firewall the card IR rests on: ``effective/cards/`` may import
``htmltools`` (the tag library) *only* in ``render_shiny``, and nothing there
imports ``shiny`` (the app framework) at all; the card spec and the domain
projectors import neither. If ``{Action}`` or a card ever has to know an input id,
the levels have leaked, and this test fails first. Infra-free (it parses source).
"""

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"


def _top_imports(path: Path) -> set[str]:
    """The top-level package of every import in a module."""
    tree = ast.parse(path.read_text(), filename=str(path))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def test_effective_cards_imports_no_shiny_and_htmltools_only_in_render_shiny() -> None:
    for f in sorted((_SRC / "effective" / "cards").glob("*.py")):
        imports = _top_imports(f)
        assert "shiny" not in imports, f"{f.name} imports shiny (the framework)"
        if f.name != "render_shiny.py":
            assert "htmltools" not in imports, (
                f"{f.name} imports htmltools — only render_shiny (the tag renderer) may"
            )


def test_card_spec_and_projectors_are_framework_free() -> None:
    framework = {"shiny", "htmltools"}
    for rel in ("effective/cards/spec.py",):
        leaked = _top_imports(_SRC / rel) & framework
        assert not leaked, f"{rel} imports a UI framework: {sorted(leaked)}"
