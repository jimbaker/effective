"""Contrast-bench pre-flight (offline, no API): generator determinism, the
registered scoring rule, and — the money-saving one — the structured arm plus
its H3 replay driven end-to-end by a scripted fake client.

The live arms are exercised by `scripts/contrast_run.py`; these tests pin what
must not drift underneath the registered protocol."""

import json
from types import SimpleNamespace

import pytest

from agent.contrastbench import (
    SIZES,
    ContrastTask,
    _price_subcalls,
    make_log,
    make_tasks,
    normalize,
    replay_structured,
    run_repl,
    run_structured,
    run_trial,
)
from effective.interpreters.openai import GPT5_NANO, usage_from_openai

# ------------------------------------------------------------------- generator


def test_log_and_tasks_are_seed_deterministic():
    rows1, text1 = make_log(7, 150)
    rows2, text2 = make_log(7, 150)
    assert rows1 == rows2
    assert text1 == text2
    t1 = make_tasks(7)
    t2 = make_tasks(7)
    assert t1 == t2
    assert len(t1) == 16  # 4 types x 2 instances x 2 sizes
    assert {t.size for t in t1} == set(SIZES)
    # ground truth is computable and self-consistent: spot-check one sum task
    sums = [t for t in t1 if t.qtype == "sum" and t.size == "S"]
    assert sums
    assert all(t.truth == normalize(t.truth) for t in t1)


def test_normalization_rule_as_registered():
    assert normalize(" 1,234 ") == "1234"
    assert normalize(1234) == "1234"
    assert normalize("Uma ") == "uma"
    assert normalize("07") == "7"


# ------------------------------------- offline end-to-end (scripted fake client)


class _FakeClient:
    """chat.completions.create returning scripted JSON turns, OpenAI-shaped."""

    def __init__(self, turns: list[dict | str]):
        self._turns = list(turns)
        self.calls = 0
        self.seen: list[list[dict]] = []  # the messages each call received
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        self.seen.append(kwargs["messages"])
        item = self._turns.pop(0)
        content = item if isinstance(item, str) else json.dumps(item)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")
            ],
            usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=20, prompt_tokens_details=None
            ),
        )


def _task(size="S"):
    return ContrastTask(
        task_id=f"sum-0-{size}",
        qtype="sum",
        size=size,
        question="What is the total amount over all records with user=uma and action=refund?",
        truth="0",  # not used by the arm runners; scoring happens in run_trial
        seed=7,
    )


def _sum_code(var: str) -> str:
    if var == "records":
        return "sum(r['amount'] for r in records if r['user']=='uma' and r['action']=='refund')"
    return (
        "total = 0\n"
        "for line in ctx.splitlines():\n"
        "    if 'user=uma' in line and 'action=refund' in line:\n"
        "        total = total + int(line.split('amount=')[1])\n"
        "total"
    )


def test_structured_arm_runs_and_replays_with_zero_model_calls(tmp_path):
    from agent.contrastbench import _STRUCTURED_SYSTEM, GPT5_NANO, WireCaller

    task = _task()
    rows, text = make_log(task.seed, SIZES[task.size])
    truth = sum(r["amount"] for r in rows if r["user"] == "uma" and r["action"] == "refund")
    fake = _FakeClient(
        [
            {"thought": "compute", "code": _sum_code("records"), "answer": None},
            {"thought": "done", "code": None, "answer": str(truth)},
        ]
    )
    caller = WireCaller(fake, _STRUCTURED_SYSTEM, price=GPT5_NANO)
    db = tmp_path / "task.db"
    outcome = run_structured(caller, task, rows, text, typed=True, db_path=db, budget_usd=0.10)
    assert outcome.stop_reason == "finish"
    assert outcome.answer == str(truth)
    assert outcome.cells == 1
    assert caller.calls == 2
    # H3: the DB replays to the same answer; the fake client is NOT consulted
    replayed = replay_structured(task, rows, text, typed=True, db_path=db)
    assert replayed == str(truth)
    assert fake.calls == 2  # zero additional model calls during replay


def test_repl_arm_truncates_observations_to_their_500(tmp_path):
    from agent.contrastbench import _REPL_SYSTEM, GPT5_NANO, WireCaller

    task = _task()
    _, text = make_log(task.seed, SIZES[task.size])
    fake = _FakeClient(
        [
            {"thought": "peek", "code": "ctx", "answer": None},  # huge value -> truncated
            {"thought": "give up precision", "code": None, "answer": "42"},
        ]
    )
    caller = WireCaller(fake, _REPL_SYSTEM, price=GPT5_NANO)
    outcome = run_repl(caller, task, text)
    assert outcome.answer == "42"
    assert outcome.cells == 1
    # the observation the model saw was truncated to their 500-char preview
    outputs = [
        m["content"]
        for m in fake.seen[-1]
        if m["role"] == "user" and m["content"].startswith("OUTPUT:")
    ]
    assert len(outputs) == 1
    assert len(outputs[0]) <= len("OUTPUT: ") + 500
    assert len(text) > 500  # the value really was bigger than the preview


# --------------------------------------------------- contrast-2 (semantic family)


def test_sem_generator_is_deterministic_and_leak_free():
    from agent.contrastbench import make_sem_log, make_sem_tasks

    t1, t2 = make_sem_tasks(7), make_sem_tasks(7)
    assert t1 == t2
    assert len(t1) == 8  # 2 types x 2 sentiments x 2 sizes
    assert all(t.semantic for t in t1)
    rows, text, labels = make_sem_log(7, 150)
    assert len(rows) == len(labels) == 150
    # the hidden label NEVER rides a row or the rendered text
    assert set(rows[0]) == {"ts", "user", "action", "amount", "note"}
    assert 'note="' in text
    for token in ("complaint", "praise", "neutral"):
        assert all(lab in ("complaint", "praise", "neutral") for lab in labels)
        assert f"label={token}" not in text


def test_structured_semantic_subcalls_seal_into_fn_log_and_replay(tmp_path):
    from agent.contrastbench import (
        GPT5_NANO,
        SubcallMeter,
        WireCaller,
        contrast_system_prompt,
        make_sem_log,
        make_sem_tasks,
    )

    task = next(t for t in make_sem_tasks(7) if t.task_id == "sem-count-0-S")
    rows, text, _ = make_sem_log(7, task.n_records)
    fake = _FakeClient(
        [
            {
                "thought": "ask the sub-model",
                "code": "llm_query('classify this note: ' + records[0]['note'])",
                "answer": None,
            },
            "complaint",  # the llm_query completion (plain text)
            {"thought": "done", "code": None, "answer": "3"},
        ]
    )
    caller = WireCaller(fake, contrast_system_prompt("structured", semantic=True))
    sub = SubcallMeter(fake, model="gpt-5-nano", price=GPT5_NANO)
    db = tmp_path / "task.db"
    outcome = run_structured(
        caller, task, rows, text, typed=True, db_path=db, budget_usd=0.10, subcalls=sub
    )
    assert outcome.answer == "3"
    assert sub.calls == 1
    assert outcome.sub == {"calls": 1, "ptok": 100, "ctok": 20}
    # H2': the replay re-binds EVERYTHING from the record — turns and the
    # llm_query result alike; the fake client is never consulted again
    replayed = replay_structured(task, rows, text, typed=True, db_path=db, declared=("llm_query",))
    assert replayed == "3"
    assert fake.calls == 3


def test_subcall_truncation_is_loud_not_silent(tmp_path):
    """A2: an empty or length-truncated llm_query reply RAISES — the silent-""
    contract is the registered failure class, three sightings deep."""
    from agent.contrastbench import GPT5_NANO, SubcallMeter

    fake = _FakeClient([""])  # empty content
    sub = SubcallMeter(fake, model="gpt-5-nano", price=GPT5_NANO)
    with pytest.raises(RuntimeError, match="truncated"):
        sub.fn("classify these")
    # a complete (stop-terminated, non-empty) reply passes through unchanged
    sub2 = SubcallMeter(_FakeClient(["all labels here"]), model="gpt-5-nano", price=GPT5_NANO)
    assert sub2.fn("small ask") == "all labels here"


def test_semantic_prompts_declare_llm_query_and_note_format():
    from agent.contrastbench import contrast_system_prompt

    for arm in ("repl", "structured", "structured-text"):
        sem = contrast_system_prompt(arm, semantic=True)
        assert "llm_query" in sem
        assert "note" in sem
        assert "llm_query" not in contrast_system_prompt(arm, semantic=False)
    assert "llm_query" not in contrast_system_prompt("stuff", semantic=True)


def test_run_trial_scores_exact_match_and_writes_artifacts(tmp_path):
    task = _task()
    rows, _ = make_log(task.seed, SIZES[task.size])
    truth = sum(r["amount"] for r in rows if r["user"] == "uma" and r["action"] == "refund")
    from dataclasses import replace

    task = replace(task, truth=str(truth))
    scripted = [
        {"thought": "compute", "code": _sum_code("records"), "answer": None},
        {"thought": "done", "code": None, "answer": f" {truth:,} "},  # normalization case
    ]
    trial = run_trial(
        task,
        "structured",
        1,
        make_client=lambda: _FakeClient(list(scripted)),
        out_dir=tmp_path,
        budget_usd=0.10,
    )
    assert trial.reward == 1.0
    assert trial.replay_ok is True
    assert trial.llm_calls == 2
    assert trial.cost_usd > 0
    trial_json = tmp_path / task.task_id / "structured" / "attempt-1" / "trial.json"
    saved = json.loads(trial_json.read_text())
    assert saved["reward"] == 1.0


# --- subcall pricing symmetry (audit item 5: the cost-comparison-fairness fix) ---


def test_subcall_pricing_is_cache_aware_and_symmetric_with_the_structured_arm():
    """The repl arm's sub-calls must be priced the SAME way usage_from_openai
    prices the structured arm's — cached input at the cheap rate. Before the fix
    the repl arm paid full input rate on cached tokens, inflating the competitor
    arm's cost in a cost-comparison bench."""
    sub = {"ptok": 1000, "ctok": 200, "cached": 800}
    repl_cost = _price_subcalls(sub, GPT5_NANO)

    # the same tokens through the structured arm's pricing path
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=200,
            prompt_tokens_details=SimpleNamespace(cached_tokens=800),
        )
    )
    assert repl_cost == usage_from_openai(resp, GPT5_NANO).cost  # symmetric

    # and the cache discount is real: cached tokens cost less than uncached
    no_cache = _price_subcalls({"ptok": 1000, "ctok": 200, "cached": 0}, GPT5_NANO)
    assert repl_cost < no_cache


def test_price_subcalls_tolerates_missing_cached_key():
    # a repl arm with no cache detail bills every prompt token at the input rate
    assert (
        _price_subcalls({"ptok": 100, "ctok": 10}, GPT5_NANO)
        == (100 * GPT5_NANO.input_per_1m + 10 * GPT5_NANO.output_per_1m) / 1_000_000
    )
