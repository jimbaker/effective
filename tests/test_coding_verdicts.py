"""Deterministic verdicts, and the predicate that feeds them.

**Roles.** The verdict tables are `unit`: pure functions over a `CommandRun`, so they cost no
model call. `test_run_suite_*` is `spine`: it drives the real
predicate against a real pytest, because a predicate that has only ever been faked is a predicate
nobody has run.

The tables are asserted as a PRODUCT over `(suite shape x state)` rather than as a handful of
examples, because the interesting content is where two states read the SAME measurement
differently — a collection error is `BROKEN_ENV` in TEST and `STILL_RED` in DRAFT, and that
divergence is the entire argument for a verdict enum per state rather than one shared enum.
"""

import pytest

from effective.coding.runners import run_suite
from effective.coding.states import DraftVerdict, FinalizeVerdict, TestVerdict
from effective.coding.verdicts import (
    StructuralJudgementRequired,
    verdict_for_draft,
    verdict_for_finalize,
    verdict_for_test,
)
from effective.machine.evidence import CommandRun, Commitment

TARGET = "test_off_by_one"
"""The target as a BARE name, not a full node id — which is both what a caller naturally has and
what `CommandRun.failed` is built for. A caller knows the test it asked for; the file, class and
parametrization are the runner's business, so matching is by substring."""

GREEN = CommandRun(exit_code=0)
TARGET_RED = CommandRun(exit_code=1, failures=("tests/t_pager.py::" + TARGET,))
OTHER_RED = CommandRun(exit_code=1, failures=("tests/t_other.py::t_elsewhere",))
BOTH_RED = CommandRun(
    exit_code=1, failures=("tests/t_pager.py::" + TARGET, "tests/t_other.py::t_elsewhere")
)
BROKEN = CommandRun(exit_code=2, collection_error="ImportError: no module named pager")
NOTHING = CommandRun(exit_code=5)


# --- CommandRun itself ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("run", "green"),
    [(GREEN, True), (TARGET_RED, False), (BROKEN, False), (NOTHING, False)],
)
def test_green_means_passed_AND_actually_ran(run: CommandRun, green: bool):
    """Exit 5 is the interesting row: the command SUCCEEDED at running nothing. A boolean derived
    from `exit_code == 0` alone would call that green and ship untested code."""
    assert run.green is green


def test_a_target_is_matched_by_substring_not_equality():
    """A pytest node id carries file, class and parameters; a caller naming the test knows the
    test, not the parametrization."""
    parametrized = CommandRun(exit_code=1, failures=(f"tests/t_pager.py::{TARGET}[case-3]",))
    assert parametrized.failed(TARGET)
    assert parametrized.failures_besides(TARGET) == ()


# --- the same measurement, read differently per state ------------------------------------


@pytest.mark.parametrize(
    ("run", "expected"),
    [
        (TARGET_RED, TestVerdict.RED),  # the test discriminates — this is TEST's PASS condition
        (GREEN, TestVerdict.GREEN_ALREADY),  # proves nothing about the test just written
        (OTHER_RED, TestVerdict.RED_ELSEWHERE),  # not this test's verdict
        (BROKEN, TestVerdict.BROKEN_ENV),  # a fact about the environment, not the test
        (NOTHING, TestVerdict.GREEN_ALREADY),
    ],
)
def test_TEST_reads_the_suite(run: CommandRun, expected: TestVerdict):
    assert verdict_for_test(run, TARGET) == expected


@pytest.mark.parametrize(
    ("run", "expected"),
    [
        (GREEN, DraftVerdict.GREEN),
        (TARGET_RED, DraftVerdict.STILL_RED),
        (OTHER_RED, DraftVerdict.REGRESSED),  # broke something else
        (BOTH_RED, DraftVerdict.STILL_RED),  # the target is still the story
        (BROKEN, DraftVerdict.STILL_RED),  # does not import, so certainly not passing
        (NOTHING, DraftVerdict.STILL_RED),
    ],
)
def test_DRAFT_reads_the_suite(run: CommandRun, expected: DraftVerdict):
    assert verdict_for_draft(run, TARGET) == expected


def test_the_two_states_disagree_about_a_broken_environment():
    """**The argument for a verdict enum per state, in one assertion.** One `CommandRun`, two
    states, two different correct answers: in TEST the environment IS the finding; in DRAFT the
    honest report to a state whose job is to keep editing is that the target is not passing. A
    single shared verdict enum would have to pick one and be wrong in the other state."""
    assert verdict_for_test(BROKEN, TARGET) == TestVerdict.BROKEN_ENV
    assert verdict_for_draft(BROKEN, TARGET) == DraftVerdict.STILL_RED


# --- FINALIZE's product, and its named impossible cell -----------------------------------


@pytest.mark.parametrize("run", [TARGET_RED, BROKEN, NOTHING])
def test_FINALIZE_does_not_reach_the_structural_question_when_the_suite_is_red(run: CommandRun):
    """`unfold`'s refusal of a node that descends at its final level, in another domain. A
    refactor that changed behaviour is a fact about the code, and no opinion about its structure
    is worth having yet, so the judged axis is not consulted, and passing it changes nothing."""
    assert verdict_for_finalize(run) == FinalizeVerdict.BROKE_IT
    assert verdict_for_finalize(run, debt_remains=True) == FinalizeVerdict.BROKE_IT
    assert verdict_for_finalize(run, debt_remains=False) == FinalizeVerdict.BROKE_IT


def test_FINALIZE_refuses_to_default_the_structural_judgement():
    """Both defaults are wrong in a way that hides: "tidy" ships unpaid debt, "debt remains" loops
    FINALIZE forever. So the green case demands an answer rather than inventing one."""
    with pytest.raises(StructuralJudgementRequired):
        verdict_for_finalize(GREEN)
    assert verdict_for_finalize(GREEN, debt_remains=False) == FinalizeVerdict.STILL_GREEN_TIDY
    assert (
        verdict_for_finalize(GREEN, debt_remains=True) == FinalizeVerdict.STILL_GREEN_DEBT_REMAINS
    )


# --- the predicate, run for real ----------------------------------------------------------


def test_run_suite_measures_a_real_tree():
    """**The predicate, exercised for real — once.**

    One subprocess, three claims: a mixed tree exits 1, NAMES which test failed (the field
    `failed()` reads, without which STILL_RED and REGRESSED collapse into one verdict), and the
    tree it measured is untouched, because a predicate that could write to what it measures would
    make its own verdict unreproducible.

    Deliberately ONE run rather than four. `python -m pytest` costs ~1.1s of process startup, and
    everything else about the verdict tables is a pure function of a `CommandRun` — so the tables
    are driven by fixtures above and only the parser needs the real thing. Fast tests are a better
    oracle precisely because you can afford to run all of them."""
    tree = {
        "test_two.py": (
            "def test_good():\n    assert True\n\n\ndef test_bad():\n    assert False\n"
        )
    }
    before = dict(tree)
    run = run_suite(tree)
    assert run.exit_code == 1
    assert not run.green
    assert any("test_bad" in f for f in run.failures)
    assert not any("test_good" in f for f in run.failures)
    assert tree == before, "the predicate wrote to the tree it was measuring"


def test_run_suite_separates_a_broken_environment_from_a_red_test():
    """The second real run, and it needs its own tree: a collection error takes the whole suite
    down, so it cannot share a process with the passing/failing case above.

    This is the distinction the entire `collection_error` field exists for — an import that fails
    is not a test that failed, and reading it as one sends the machine to fix code that is fine."""
    run = run_suite(
        {"test_broken.py": "import no_such_module_anywhere\n\n\ndef test_x():\n    pass\n"}
    )
    assert not run.green
    assert run.collection_error is not None
    assert verdict_for_test(run, "test_x") == TestVerdict.BROKEN_ENV


def test_a_commitment_carries_the_verdict_even_when_it_failed():
    """An honest failure on the canonical record is the property the unconditional tail buys."""
    c = Commitment(artifact_id="sha-1", measured=TARGET_RED, files=("m.py",))
    assert not c.passed
    assert c.measured == TARGET_RED
    assert TARGET in TARGET_RED.failures[0]
