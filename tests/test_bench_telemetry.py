"""The benchmark-agnostic analysis + the two benchmarks' specs — without a live model
(the analysis half is pure; the HotpotQA retriever/tools run offline)."""

import json
from collections import Counter
from types import SimpleNamespace

from agent.bench_telemetry import (
    BenchSpec,
    SelectionStats,
    StrategyResult,
    ToolContext,
    format_table,
    frontier_points,
    replay_free,
    rescore,
    run_strategy,
    selection_stats,
    validate_sidecar,
)
from agent.gsm8k import GSM8K_SPEC
from agent.hotpotqa import HOTPOTQA_SPEC, em_check, load_hotpotqa, normalize_answer
from agent.tasks import Task
from effective.cost import CostBudget, Usage
from effective.interpreters.openai import GPT5_NANO
from effective.telemetry import Span


def _bracket(chose: str, allowed: tuple[str, ...]) -> Span:
    """A per-turn bracketing-decision span (the caller's event)."""
    return Span(
        name="bracket",
        kind="CHAIN",
        session_id="t",
        fields={"chose": chose, "allowed_tools": allowed, "strategy_name": "mask"},
    )


def _retrieval(label: str) -> Span:
    """A retrieval-outcome span (a paragraph tool's event), carrying the ground-truth label."""
    return Span(name="retrieval", kind="CHAIN", session_id="t", fields={"retrieved_label": label})


# --- the agnostic analysis ----------------------------------------------------


def test_selection_stats_tool_choice_proxy_no_labels() -> None:
    """GSM8K regime: no retrieval labels -> distract is the tool-choice proxy."""
    keep = {"add", "subtract"}
    is_distractor = lambda f: f["chose"] not in keep  # noqa: E731
    spans = [
        _bracket("add", ("add", "subtract", "weather")),  # kept tool, in allowlist
        _bracket("weather", ("add", "subtract", "weather")),  # distractor, in allowlist
        _bracket("sql_query", ("add", "subtract")),  # NOT in allowlist -> violation + distractor
    ]
    stats = selection_stats(spans, is_distractor)
    assert stats.masked_violations == 1
    assert stats.distractor_calls == 2  # weather + sql_query (proxy)
    assert stats.chosen == {"add": 1, "weather": 1, "sql_query": 1}
    assert stats.trap_rate is None  # no labeled retrievals


def test_selection_stats_trap_rate_from_labels() -> None:
    """HotpotQA regime: retrieval labels present -> distract is ground truth, trap_rate set.
    masked! still comes from the bracketing events (the mask guarantee)."""
    is_distractor = HOTPOTQA_SPEC.resolved_is_distractor()
    spans = [
        _bracket("p2", ("p2", "p4")),  # decode within allowlist -> no violation
        _retrieval("gold"),
        _bracket("p4", ("p2", "p4")),
        _retrieval("gold"),
        _bracket("p7", ("p2", "p7")),
        _retrieval("distractor"),
    ]
    stats = selection_stats(spans, is_distractor)
    assert stats.masked_violations == 0
    assert (stats.gold_reads, stats.distractor_reads) == (2, 1)
    assert stats.distractor_calls == 1  # ground truth = distractor reads
    assert stats.trap_rate == round(1 / 3, 3)


def test_selection_stats_no_reads_trap_rate_none() -> None:
    assert SelectionStats(0, 0, Counter()).trap_rate is None


def test_strategy_result_quality() -> None:
    assert StrategyResult("mask", 3, 4, Usage()).quality == 0.75
    assert StrategyResult("x", 0, 0, Usage()).quality == 0.0


def test_format_table_headers_and_rows() -> None:
    rows = [
        StrategyResult("full", 2, 4, Usage(prompt_tokens=100, cache_read_input_tokens=80)),
        StrategyResult("mask", 2, 4, Usage(prompt_tokens=90, cache_read_input_tokens=82)),
    ]
    table = format_table(rows)
    for token in ("strategy", "n", "cost_usd", "cache_hit", "masked!", "trap", "full", "mask"):
        assert token in table


def test_validate_sidecar_clean(tmp_path) -> None:
    from effective.telemetry import otlp_jsonl_sink

    path = tmp_path / "spans.jsonl"
    sink = otlp_jsonl_sink(path)
    sink(Span(name="LM.0", kind="LLM", session_id="s", iteration=0))
    assert validate_sidecar(path) == []


# --- the two specs ------------------------------------------------------------


def test_gsm8k_spec_is_static_binding() -> None:
    """GSM8K uses the static catalog path: every task shares one binding; the default
    is_distractor is 'a non-keep (non-arithmetic) tool'."""
    is_distractor = GSM8K_SPEC.resolved_is_distractor()
    assert is_distractor({"chose": "weather"}) is True
    assert is_distractor({"chose": "add"}) is False
    binding = GSM8K_SPEC.binding(GSM8K_SPEC.load(1)[0], ToolContext(None, "s"))
    assert set(GSM8K_SPEC.keep_tools) <= set(binding.catalog)
    for turn in range(5):
        assert set(GSM8K_SPEC.keep_tools) <= binding.select(turn)


def test_specs_well_formed() -> None:
    # static benchmark: catalog tools all documented
    assert set(GSM8K_SPEC.catalog) <= set(GSM8K_SPEC.doc)
    # dynamic benchmark: a bind function, no static catalog
    assert HOTPOTQA_SPEC.bind is not None
    assert HOTPOTQA_SPEC.catalog == []


# --- HotpotQA specifics -------------------------------------------------------


def test_hotpotqa_loads_with_gold_labels() -> None:
    tasks = load_hotpotqa()
    assert len(tasks) == 6
    t0 = tasks[0]
    paras = t0.payload["paras"]
    assert len(paras) == 10
    golds = [p["title"] for p in paras if p["gold"]]
    assert set(golds) == {"Crimson Harbor", "Lena Cole"}  # the 2 supporting paragraphs


def test_hotpotqa_em_check_and_normalize() -> None:
    assert normalize_answer("The Astoria!") == "astoria"
    check = em_check("Astoria")
    assert check("Astoria") is True
    assert check("She was born in Astoria, Oregon.") is True  # containment
    assert check("Sefton") is False


def test_hotpotqa_relevance_select_is_multi_hop() -> None:
    """The content-aware retriever: paragraph A (Crimson Harbor) is surfaced by the
    question; reading it boosts the bridge paragraph B (Lena Cole) into the next top-k."""
    task = load_hotpotqa(limit=1)[0]
    binding = HOTPOTQA_SPEC.binding(task, ToolContext(None, "s"))
    # p1 = Crimson Harbor (gold A), p4 = Lena Cole (gold B)
    turn0 = binding.select(0)
    assert "p1" in turn0  # the film paragraph is relevant to the question
    binding.tools["p1"]({})  # "read" A -> its mention of Lena Cole enters the query
    assert "p4" in binding.select(1)  # the bridge paragraph is now surfaced


class _AlwaysFinish:
    """A fake OpenAI-shaped client whose every completion finishes with a fixed answer
    and a fixed (large) prompt-token usage — so a priced budget is exhausted predictably."""

    def __init__(self, prompt_tokens: int = 100_000) -> None:
        self.calls = 0
        self.prompt_tokens = prompt_tokens

    def create(self, **_kw: object) -> object:
        self.calls += 1
        usage = SimpleNamespace(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=10,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        turn = {"thought": "t", "tool_name": "", "tool_args_json": "", "answer": "42"}
        message = SimpleNamespace(content=json.dumps(turn))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def test_run_strategy_stops_at_budget(tmp_path) -> None:
    """Budget-first: the run stops launching tasks once the per-strategy budget is spent,
    and reports quality over *attempted* tasks (n < the full list)."""
    client = SimpleNamespace(chat=SimpleNamespace(completions=_AlwaysFinish()))
    spec = BenchSpec(
        name="t",
        load=lambda lim: [Task(f"t{i}", "q", lambda _a: False) for i in range(5)],
        catalog_tools={"noop": lambda _a: "x"},
        keep_tools=("noop",),
        doc={"noop": "noop"},
    )
    budget = CostBudget(0.011)  # ~2 calls at GPT5_NANO pricing for 100k prompt tokens
    res = run_strategy(
        client,
        "m",
        spec,
        "full",
        spec.load(5),
        sidecar=tmp_path / "s.jsonl",
        max_iters=2,
        caller_kw={"cloud": True, "price": GPT5_NANO, "max_tokens": 50},
        budget=budget,
    )
    assert budget.exceeded()
    assert 0 < res.total < 5  # the cap stopped it before all five tasks


def _recorded_run(tmp_path, n=3):
    """A small recorded strategy run (fake always-finish client) -> StrategyResult."""
    client = SimpleNamespace(chat=SimpleNamespace(completions=_AlwaysFinish(prompt_tokens=10)))
    spec = BenchSpec(
        name="t",
        load=lambda lim: [Task(f"t{i}", "q", lambda _a: False) for i in range(n)],
        catalog_tools={"noop": lambda _a: "x"},
        keep_tools=("noop",),
        doc={"noop": "noop"},
    )
    return run_strategy(
        client,
        "m",
        spec,
        "full",
        spec.load(n),
        sidecar=tmp_path / "s.jsonl",
        max_iters=2,
        caller_kw={"cloud": True, "price": GPT5_NANO, "max_tokens": 50},
    )


def test_records_trajectories_and_rescores_free(tmp_path) -> None:
    """(a) as replay: the run records trajectories; a *different* scorer re-measures them
    with no model calls."""
    res = _recorded_run(tmp_path)
    assert len(res.recordings) == 3
    # live scorer was the default boolean check (False) -> quality 0
    assert res.quality == 0.0
    # re-score the SAME recordings with a new metric, free
    assert rescore(res.recordings, lambda ans, _t: float(ans == "42")) == 1.0


def test_replay_is_free_and_matches(tmp_path) -> None:
    """The proof: every recorded trajectory replays model-free and re-derives its answer."""
    res = _recorded_run(tmp_path)
    assert replay_free(res.recordings) is True


def test_frontier_points_labels_nondominated() -> None:
    rows = [
        StrategyResult("a", 8, 10, Usage(cost=0.01, latency_s=1.0)),  # q.8 cheap+fast -> frontier
        StrategyResult("b", 9, 10, Usage(cost=0.05, latency_s=2.0)),  # q.9 pricier -> frontier
        StrategyResult("c", 5, 10, Usage(cost=0.09, latency_s=3.0)),  # dominated by a
    ]
    pts = {p["strategy"]: p["on_frontier"] for p in frontier_points(rows)}
    assert pts["a"] is True
    assert pts["b"] is True
    assert pts["c"] is False


def test_hotpotqa_tool_emits_retrieved_label() -> None:
    """Reading a paragraph emits a retrieval event with the ground-truth gold/distractor
    label — the one emit-side addition that makes trap rate ground truth."""
    task = load_hotpotqa(limit=1)[0]
    spans: list[Span] = []
    binding = HOTPOTQA_SPEC.binding(task, ToolContext(spans.append, "s"))
    binding.tools["p1"]({})  # gold (Crimson Harbor)
    binding.tools["p0"]({})  # distractor (Marlowe Drift)
    labels = [s.fields["retrieved_label"] for s in spans if "retrieved_label" in s.fields]
    assert labels == ["gold", "distractor"]
