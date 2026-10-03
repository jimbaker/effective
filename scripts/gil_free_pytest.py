"""Run pytest on a free-threaded CPython and fail unless the GIL stayed off throughout.

A C extension can re-enable the GIL when it is imported, which would turn a free-threaded run into
an ordinary one while every test still passed. So the check runs before pytest imports anything,
and again after every test has run.

    PYTHON_GIL=0 build/venv-ft/bin/python scripts/gil_free_pytest.py tests/test_race.py
"""

import sys


def main() -> int:
    if not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled():
        print("the GIL is enabled: run this on a free-threaded build with PYTHON_GIL=0")
        return 2
    import pytest

    code = pytest.main(sys.argv[1:])
    if sys._is_gil_enabled():
        print("an import re-enabled the GIL during the run")
        return 3
    print("the GIL stayed off")
    return int(code)


if __name__ == "__main__":
    sys.exit(main())
