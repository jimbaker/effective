"""The minter view's fixture: the pair it exists to surface must stay adjacent.

ROLE: adversarial. A normal form is an editorial choice, and a wrong one hides what it was
built to show. Measured: a shared-fragment census whose readability filter
required a leading alphanumeric excluded its own exemplar, because the fragment it was built for
starts with a separator.

So the view ships with its exemplar pinned: the `tool` namespace, whose two templates are minted in
two modules and read as one family only when they sort together. If a future sort or filter drops
them out of each other's neighborhood, this reddens.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


from effective.keys.registry import KeyMap  # noqa: E402
from effective.lint import build_key_registry  # noqa: E402
from scripts.key_view import rows  # noqa: E402

pytestmark = pytest.mark.adversarial


@pytest.fixture(scope="module")
def table() -> list[tuple[str, tuple[str, ...]]]:
    """The view, built from source rather than from `build/key-registry.json`.

    The artifact is gitignored, so a test reading it would pass or fail on whether someone had run
    `just key-registry`. Scanning `src/` is a superset of the roots the gate scans, which keeps the
    domain here from becoming a second copy of the justfile's list.
    """
    shapes, _problems = build_key_registry(sorted(Path("src").rglob("*.py")))
    return rows(KeyMap.from_shapes(shapes))


def test_the_tool_minters_render_as_one_block(table: list[tuple[str, tuple[str, ...]]]) -> None:
    templates = [template for template, _site in table]
    first = templates.index("tool:{}")
    kinded = templates.index("tool;{}")
    assert kinded - first == 1, (
        f"the `tool` minters span {kinded - first + 1} rows; the view exists to put them together"
    )
    # The module, not the line: a line number here would rot on the next docstring edit, which is
    # the defect `tests/test_whouses.py::_line_of` was just built to stop repeating.
    assert [
        sorted(s.rsplit(":", 1)[0] for s in sites) for _t, sites in table[first : kinded + 1]
    ] == [
        ["src/effective/api.py"],
        ["src/effective/handlers/base.py"],
    ]
