"""The agent task suite: tools, checkers, and task sanity (red-green spec).

A small, deterministic, tool-using suite so `quality` becomes a pass-rate over
varied difficulty — not one binary sample. The tasks themselves (multiply, etc.)
are throwaway canaries chosen to be cheap; what we care about is the *scorer as a
seam* for end-to-end optimization against model choices later.
"""

import pytest

from agent.tasks import TASKS, TOOLS, all_of, has_number, has_text, task_scope


def test_tools_compute_with_coerced_args():
    assert TOOLS["multiply"]({"a": 19, "b": 23}) == 437
    assert TOOLS["add"]({"a": 7, "b": 5}) == 12
    assert TOOLS["subtract"]({"a": 715000, "b": 108000}) == 607000
    assert TOOLS["population"]({"city": "Boulder"}) == 108000
    assert TOOLS["multiply"]({"a": "107", "b": "12"}) == 1284  # string args coerced


def test_has_number_is_boundary_and_comma_aware():
    assert has_number(36)("the result is 36.")
    assert not has_number(36)("the result is 360")  # boundary: not a substring of 360
    assert not has_number(36)("value 136")  # not a suffix of 136
    assert has_number(108000)("Boulder has 108,000 people")  # comma-tolerant


def test_text_and_all_of_combinators():
    chk = all_of(has_text("denver"), has_number(607000))
    assert chk("Denver is larger by 607,000")
    assert not chk("Boulder is larger by 607000")  # missing 'denver'


def test_every_task_checker_accepts_its_canonical_answer():
    canonical = {
        "multiply_basic": "437",
        "two_step": "36",
        "word_args": "1284",
        "lookup": "Boulder has 108,000",
        "compare": "Denver is larger by 607000",
        "direct": "10",
    }
    assert {t.name for t in TASKS} == set(canonical)
    for t in TASKS:
        assert t.check(canonical[t.name]), t.name


# --- `task_scope`: a borrowed identity kept RAW, and enforced -----------------------------


def test_a_real_bench_id_stays_READABLE_in_the_key():
    """The reason `task_scope` does not digest: these ids are already `NAME` atoms.

    Digesting `astropy__astropy-12907` would trade the legibility that makes a bench run
    readable for a guarantee the value already satisfies. Both shapes the two datasets
    actually produce: SWE-bench style and the generated contrast tasks.
    """
    assert task_scope("astropy__astropy-12907").stored() == "task:astropy__astropy-12907"
    assert task_scope("sem-count-3-64").stored() == "task:sem-count-3-64"


@pytest.mark.parametrize(
    ("bad", "why"),
    [
        ("a:b", "a delimiter — refused by `Segment` before the atom rule is reached"),
        ("7zip-task", "digit-led: no delimiter, and still not an atom"),
        ("skills:astropy__astropy-12907:1", "the composite an f-string would build"),
        ("", "the empty id names nothing"),
    ],
)
def test_an_out_of_language_bench_id_RAISES_rather_than_digesting_quietly(bad: str, why: str):
    """The enforcement half of "keep it raw *where you can state and enforce the guarantee*".

    A quiet digest here would be the worse failure: the run continues, the frame becomes
    unreadable, and nobody learns that a dataset grew an id out of language. The raise happens
    at the boundary, where the id is still in hand.
    """
    with pytest.raises(ValueError, match="not a well-formed atom"):
        task_scope(bad)
