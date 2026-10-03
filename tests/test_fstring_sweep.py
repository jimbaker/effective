"""The f-string gate: what counts as a designated render backend."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts import fstring_sweep  # noqa: E402

SOURCE = """
class Problem:
    def render(self, x):
        return f"at {x}"


class Other:
    def render(self, x):
        return f"other {x}"


def _render_name(op, target):
    def inner():
        return f"{op} {target}"
    return f"{op} {target}", inner
"""


def test_a_designation_names_one_function_by_its_qualified_name(tmp_path, monkeypatch):
    """Designating `Problem.render` exempts no other `render` in the file, and designating a
    function exempts no function nested inside it."""
    path = tmp_path / "telemetry.py"
    path.write_text(SOURCE)
    monkeypatch.setattr(
        fstring_sweep,
        "DESIGNATED",
        frozenset({(str(path), "Problem.render"), (str(path), "_render_name")}),
    )

    flagged = sorted(line for _, line, _, _ in fstring_sweep.findings([str(path)]))

    assert flagged == [9, 14]  # `Other.render`, and the function nested in `_render_name`
