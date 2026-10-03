"""The failing test the coder's goal is to make pass."""

from mod import add


def test_add() -> None:
    assert add(2, 3) == 5
