"""GSM8K loader + answer checker — the standard quality axis for the cache bench.

Pins the answer extraction (the trailing number, comma/`$`-tolerant) and the
gold parsing (`####` marker), so a model's free-text final answer scores
correctly without an LLM in the loop.
"""

from agent.gsm8k import extract_final_number, gold_answer, gsm8k_check, load_gsm8k


def test_gold_answer_reads_the_marker():
    assert gold_answer("She makes 9 * 2 = $18 every day.\n#### 18") == 18.0
    assert gold_answer("... total\n#### 1,800") == 1800.0


def test_extract_final_number_is_the_trailing_value():
    assert extract_final_number("First 9, then 9 * 2 = 18") == 18.0
    assert extract_final_number("The answer is $1,800.") == 1800.0
    assert extract_final_number("18.0 dollars") == 18.0
    assert extract_final_number("no number here") is None


def test_checker_compares_numerically():
    check = gsm8k_check(18.0)
    assert check("the result is 18")
    assert check("$18.00")
    assert not check("the result is 19")
    assert not check("")


def test_load_gsm8k_yields_tasks_in_order():
    tasks = load_gsm8k(limit=5)
    assert len(tasks) == 5
    assert [t.name for t in tasks] == [f"gsm8k_{i:03d}" for i in range(5)]
    # The first vendored problem's gold is 18 (Janet's ducks).
    assert tasks[0].check("18")
    assert not tasks[0].check("17")
    assert tasks[0].prompt  # the question rode through
