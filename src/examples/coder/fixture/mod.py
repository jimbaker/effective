"""A module with a one-character bug that `test_mod.py` pins."""


def add(a: int, b: int) -> int:
    return a - b
