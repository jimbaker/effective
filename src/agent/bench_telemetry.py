"""Benchmark telemetry analysis: the config-sweep half, factored out of any one task.

There are two halves to "telemetry analysis." The *emitting* half lives in
`effective.telemetry` (the `traced` domain layer, the t-string `event()`, the `Span`, the
OTLP span file and its validator `check_otlp_line`). This
module is the *other* half: turning the span stream a bracketing sweep produces into a
**cross-strategy comparison**:

  - the per-strategy table: quality, prompt/cached tokens, **`cache_hit_ratio`**,
    latency, tok/s;
  - the selection lenses: `masked!` (turns that chose a tool outside that turn's
    allowlist — the GBNF/`Gated` guarantee, must be 0), `distract` (turns that chose a
    distractor), and `trap` (the **trap rate** — fraction of retrievals that returned a
    *distractor* item, when the benchmark labels them: HotpotQA's headline metric);
  - the offline OTLP gate over the span file (`check_otlp_line`).

Everything here is **benchmark-agnostic**. The task-specific axes are injected via
`BenchSpec`: a *static* catalog (GSM8K — the same tools every task) or a per-task
`bind` (HotpotQA — each question's own paragraphs as the tool universe, plus a
content-aware `select`). Adding a benchmark is "write a `BenchSpec`", not "re-derive
the analysis". Imports only `agent`/`effective`.
"""

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.bench import RecordingCtx, ReplayCtx
from agent.bracket import STRATEGIES, Select, make_bracketed_caller, rotating_select
from agent.runtime import make_tool_runner
from agent.scoring import Scorer, boolean_scorer
from agent.tasks import Task
from effective.cost import BudgetExceeded, CostBudget, MeteredInterpreter, Usage
from effective.handlers.absurd import DurableHandler
from effective.keys import Key
from effective.pareto import Objective, label_frontier
from effective.react import Trajectory, run_agent
from effective.telemetry import (
    Sink,
    Span,
    multi,
    otlp_jsonl_sink,
    problems_in,
    traced,
)

# Span `chose` values that are not tool selections (so they never count as a
# violation or a distractor): finishing the task, or hitting the repair guardrail.
NON_TOOL_CHOICES: frozenset[str] = frozenset({"(finish)", "(guardrail)"})

# A distractor predicate reads a telemetry event's `fields` dict and decides whether the
# choice was a distractor — from a bracketing event (`chose`, `allowed_tools`) for a
# tool-choice proxy, or from a retrieval event (`retrieved_label`) for ground truth.
DistractorFn = Callable[[Mapping[str, Any]], bool]

# A tool: a pure `dict -> value` callable (the `make_tool_runner` shape).
ToolFn = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class ToolContext:
    """Telemetry handed to per-task tools so a retrieval can emit its outcome event
    (e.g. HotpotQA's gold/distractor `retrieved_label`) onto the same sidecar."""

    sink: Sink | None
    session_id: str


@dataclass(frozen=True)
class TaskBinding:
    """The per-task action space the bracketed caller runs over: the tool universe
    (names + docs + implementations), the per-turn allowlist `select`, the always-kept
    tools, and the benchmark's task framing. For a static benchmark (GSM8K) every task
    shares one binding; for a dynamic one (HotpotQA) each task gets its own."""

    catalog: list[str]
    doc: Mapping[str, str]
    tools: Mapping[str, ToolFn]
    select: Select
    keep: tuple[str, ...] = ()
    instructions: str | None = None


@dataclass(frozen=True)
class BenchSpec:
    """The benchmark-specific axes the agnostic analysis is parameterized over. Provide
    one per benchmark; the lenses and the strategy table are fixed.

    Two ways to supply the action space:
    - **static** (GSM8K) — `catalog_tools` / `keep_tools` / `doc` (+ optional `select`):
      the same tool catalog for every task.
    - **dynamic** (HotpotQA) — `bind(task, ctx) -> TaskBinding`: each task's own tools +
      a content-aware `select`. Takes precedence when set.

    `is_distractor` overrides the default tool-choice proxy ("chose a tool not in
    `keep_tools`"); set it to read a labeled signal (e.g. `retrieved_label`).
    `instructions` is the task framing for the turn prompt (defaults to GSM8K's).
    """

    name: str
    load: Callable[[int | None], list[Task]]
    catalog_tools: Mapping[str, ToolFn] = field(default_factory=dict)
    keep_tools: tuple[str, ...] = ()
    doc: Mapping[str, str] = field(default_factory=dict)
    select: Select | None = None
    is_distractor: DistractorFn | None = None
    instructions: str | None = None
    bind: Callable[[Task, ToolContext], TaskBinding] | None = None
    scorer: Scorer | None = None  # quality metric; default = the task's own `check` (0/1)

    @property
    def catalog(self) -> list[str]:
        return list(self.catalog_tools)

    def resolved_select(self) -> Select:
        return self.select or rotating_select(self.catalog, keep=self.keep_tools)

    def resolved_is_distractor(self) -> DistractorFn:
        if self.is_distractor is not None:
            return self.is_distractor
        keep = set(self.keep_tools)
        return lambda fields: fields["chose"] not in keep

    def binding(self, task: Task, ctx: ToolContext) -> TaskBinding:
        """The per-task action space — `bind` if dynamic, else the static catalog."""
        if self.bind is not None:
            return self.bind(task, ctx)
        return TaskBinding(
            catalog=self.catalog,
            doc=self.doc,
            tools=self.catalog_tools,
            select=self.resolved_select(),
            keep=self.keep_tools,
            instructions=self.instructions,
        )


@dataclass
class SelectionStats:
    """The selection lenses over one strategy's span stream."""

    masked_violations: int  # chose a tool outside the turn's allowlist (must be 0)
    distractor_calls: int  # distractor selections (proxy, or ground truth if labeled)
    chosen: Counter[str]  # histogram of chosen tools
    gold_reads: int = 0  # retrievals that returned a gold item (labeled benchmarks)
    distractor_reads: int = 0  # retrievals that returned a distractor item

    @property
    def has_retrieval_labels(self) -> bool:
        return (self.gold_reads + self.distractor_reads) > 0

    @property
    def trap_rate(self) -> float | None:
        """Fraction of labeled retrievals that hit a distractor — None when the
        benchmark does not label retrievals (e.g. GSM8K)."""
        total = self.gold_reads + self.distractor_reads
        return round(self.distractor_reads / total, 3) if total else None


@dataclass
class RecordedTrajectory:
    """A paid trajectory, kept so scoring can run at *replay* time for free: the recorded
    answer + its `RecordingCtx` op-log (re-derives the run with the model exploding)."""

    task: Task
    answer: str
    usage: Usage
    log: dict[Key, Any]
    max_iters: int


@dataclass
class StrategyResult:
    """One bracketing strategy's aggregate over a benchmark sample. `passed` is the sum of
    per-task scores (a float — boolean scorers make it the pass count); `recordings` are the
    paid trajectories, so `rescore`/`replay_free` can re-measure for free."""

    name: str
    passed: float
    total: int
    usage: Usage
    selection: SelectionStats = field(default_factory=lambda: SelectionStats(0, 0, Counter()))
    recordings: list[RecordedTrajectory] = field(default_factory=list)

    @property
    def quality(self) -> float:
        return round(self.passed / self.total, 3) if self.total else 0.0


def selection_stats(spans: list[Span], is_distractor: DistractorFn) -> SelectionStats:
    """The selection lenses from the telemetry stream. `masked!` and the chosen-tool
    histogram come from the per-turn **bracketing** events (`chose` + `allowed_tools`).
    Distractors come from **retrieval** events (`retrieved_label`) when the benchmark
    emits them — ground truth, the trap rate — and otherwise fall back to the tool-choice
    proxy on the bracketing events (GSM8K)."""
    has_labels = any("retrieved_label" in s.fields for s in spans)
    violations = proxy_distractors = gold_reads = distractor_reads = 0
    chosen: Counter[str] = Counter()
    for span in spans:
        f = span.fields
        if "retrieved_label" in f:  # a retrieval outcome event (labeled benchmark)
            if is_distractor(f):
                distractor_reads += 1
            else:
                gold_reads += 1
            continue
        if "chose" not in f:
            continue
        chose = f["chose"]
        if chose in NON_TOOL_CHOICES:
            continue
        chosen[chose] += 1
        if chose not in set(f.get("allowed_tools", ())):
            violations += 1
        if not has_labels and is_distractor(f):
            proxy_distractors += 1
    distractors = distractor_reads if has_labels else proxy_distractors
    return SelectionStats(violations, distractors, chosen, gold_reads, distractor_reads)


def run_strategy(
    client: Any,
    model: str,
    spec: BenchSpec,
    strategy_name: str,
    tasks: list[Task],
    *,
    sidecar: Path,
    max_iters: int,
    caller_kw: dict[str, Any],
    budget: CostBudget | None = None,
    scorer: Scorer | None = None,
) -> StrategyResult:
    """Run tasks under one bracketing strategy, **recording each trajectory** (so scoring
    is a free replay-time step), accumulating quality + usage and capturing the span stream
    for the selection lenses. The caller is a seam (local GBNF-mask or cloud `Gated`-repair,
    per `caller_kw`); the analysis is identical. Each task's action space comes from
    `spec.binding(task, ctx)` — static or per-task. `scorer` defaults to `spec.scorer`, then
    the task's own boolean `check`; quality = mean score over attempted tasks.

    **Budget-first:** with a `budget`, stop launching tasks once it is spent and a task
    truncated mid-run by `BudgetExceeded` is not counted — so `total` is the number of
    tasks *actually attempted* (the budget is a hard ceiling, not a way to tank the score)."""
    strategy = STRATEGIES[strategy_name]
    score = scorer or spec.scorer or boolean_scorer
    res = StrategyResult(name=strategy_name, passed=0.0, total=0, usage=Usage())
    spans: list[Span] = []
    sink = multi(otlp_jsonl_sink(sidecar), spans.append)

    for task in tasks:
        if budget is not None and budget.exceeded():
            break
        sid = f"{strategy_name}:{task.name}"
        binding = spec.binding(task, ToolContext(sink=sink, session_id=sid))
        tools = make_tool_runner(dict(binding.tools))
        caller = make_bracketed_caller(
            client,
            model,
            strategy=strategy,
            catalog=binding.catalog,
            doc=dict(binding.doc),
            select=binding.select,
            instructions=binding.instructions,
            sink=sink,
            session_id=sid,
            **caller_kw,
        )
        interp = MeteredInterpreter(
            llm=caller,
            tools=tools,
            budget=budget,
            domain_layers=[traced(sink, session_id=sid, agent_name=strategy_name)],
        )
        rec_ctx = RecordingCtx()  # record the paid trajectory so scoring can replay free
        truncated = False
        try:
            traj = Trajectory.model_validate(
                DurableHandler(rec_ctx, interp).run(
                    lambda t=task: run_agent(t.prompt, max_iters=max_iters)
                )
            )
            res.passed += score(traj.answer, task)
            res.total += 1
            res.recordings.append(
                RecordedTrajectory(task, traj.answer, interp.meter, dict(rec_ctx.log), max_iters)
            )
        except BudgetExceeded:  # ran out mid-task: don't count it, stop the strategy
            truncated = True
        except Exception as exc:  # a genuine failure is an attempted miss; the bench continues
            print(f"  ! {task.name}: {type(exc).__name__}: {exc}")
            res.total += 1
        res.usage = res.usage + interp.meter
        if truncated:
            break

    res.selection = selection_stats(spans, spec.resolved_is_distractor())
    return res


def rescore(recordings: list[RecordedTrajectory], scorer: Scorer) -> float:
    """Re-apply *any* scorer to already-recorded trajectories — **free** (no model call).
    The cost thesis for evaluation: pay for the run once, re-measure with EM, F1, the
    official scorer, … at zero marginal cost. Returns the mean score."""
    if not recordings:
        return 0.0
    return sum(scorer(r.answer, r.task) for r in recordings) / len(recordings)


def replay_free(recordings: list[RecordedTrajectory]) -> bool:
    """Replay each recorded trajectory model-free (`ReplayCtx`; any model/tool call
    explodes) and confirm it re-derives the same answer — the proof that replay (and thus
    re-scoring) costs nothing. Returns True iff every replay was free and matched."""

    def boom(_op: Any) -> Any:
        raise AssertionError("model/tool called on replay — replay must be free")

    for r in recordings:
        interp = MeteredInterpreter(llm=boom, tools=boom)
        rep = Trajectory.model_validate(
            DurableHandler(ReplayCtx(r.log), interp).run(
                lambda rr=r: run_agent(rr.task.prompt, max_iters=rr.max_iters)
            )
        )
        if rep.answer != r.answer:
            return False
    return True


# Pareto: quality (max) vs cost + latency (min) — the cost-on-the-axis frontier.
_FRONTIER_OBJECTIVES = (
    Objective("quality", "max"),
    Objective("cost", "min"),
    Objective("latency", "min"),
)


def frontier_points(results: list[StrategyResult]) -> list[dict[str, Any]]:
    """Each strategy as a (quality, cost, latency) point, labeled `on_frontier` — the
    non-dominated set is the real menu (accuracy-only comparisons hide it)."""
    points = [
        {
            "strategy": r.name,
            "quality": r.quality,
            "cost": round(r.usage.cost, 4),
            "latency": round(r.usage.latency_s, 1),
        }
        for r in results
    ]
    return label_frontier(points, _FRONTIER_OBJECTIVES)


_COLS = (
    "strategy",
    "n",
    "quality",
    "cost_usd",
    "prompt_tok",
    "cached_tok",
    "cache_hit",
    "lat_s",
    "tok/s",
    "masked!",
    "distract",
    "trap",
)
_WIDTHS = (9, 5, 9, 10, 12, 12, 11, 9, 8, 9, 9, 7)


def format_table(rows: list[StrategyResult]) -> str:
    """The per-strategy comparison table as a string (so it is testable, not just
    printed). `n` = tasks attempted (under a budget, what the spend bought); `cost_usd`
    is the spend; the cache columns are the headline (MASK holds the cache near FULL while
    MUTATE busts it); `trap` is the headline for a labeled-retrieval benchmark."""
    header = "".join(f"{c:<{w}}" for c, w in zip(_COLS, _WIDTHS, strict=True))
    lines = [header, "-" * len(header)]
    for r in rows:
        u, sel = r.usage, r.selection
        trap = "-" if sel.trap_rate is None else str(sel.trap_rate)
        lines.append(
            f"{r.name:<9}{r.total:<5}{r.quality:<9}{round(u.cost, 4):<10}"
            f"{u.prompt_tokens:<12}{u.cache_read_input_tokens:<12}"
            f"{u.cache_hit_ratio:<11}{round(u.latency_s, 1):<9}{u.tokens_per_second:<8}"
            f"{sel.masked_violations:<9}{sel.distractor_calls:<9}{trap:<7}"
        )
    lines.append(
        "\nn = tasks attempted (a budget stops launching once spent — cost is a hard ceiling)."
        "\nmasked! = turns that chose a tool outside the allowlist (the mask must keep this 0)."
        "\ndistract = distractor selections (tool-choice proxy, or labeled retrievals if any)."
        "\ntrap = fraction of labeled retrievals that returned a distractor item (or - if none)."
    )
    return "\n".join(lines)


def validate_sidecar(path: Path) -> list[tuple[int, str]]:
    """Gate the span file offline: `(lineno, problem)` for each violation, empty when clean."""
    problems: list[tuple[int, str]] = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        problems.extend((i, problem.render()) for problem in problems_in(line))
    return problems
