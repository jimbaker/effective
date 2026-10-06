"""Which paths a commit row calls changed.

ROLE: unit. The coder's tools only add and replace files, so no run yet delivers a removed path
to the row; these rows state every arm.
"""

import pytest

from effective.machine.trampoline import changed_paths

ROWS = {
    "an edit": ({"a.py": "1", "b.py": "2"}, {"a.py": "1", "b.py": "3"}, ("b.py",)),
    "an addition": ({"a.py": "1"}, {"a.py": "1", "c.py": "4"}, ("c.py",)),
    "a removal": ({"a.py": "1", "b.py": "2"}, {"a.py": "1"}, ("b.py",)),
    "nothing": ({"a.py": "1"}, {"a.py": "1"}, ()),
    "an empty seed": ({}, {"z.py": "", "a.py": ""}, ("a.py", "z.py")),
    "an emptied file": ({"a.py": "1"}, {"a.py": ""}, ("a.py",)),
}


@pytest.mark.parametrize(("seed", "tree", "changed"), ROWS.values(), ids=ROWS.keys())
def test_changed_paths(seed: dict[str, str], tree: dict[str, str], changed: tuple[str, ...]):
    assert changed_paths(seed, tree) == changed
