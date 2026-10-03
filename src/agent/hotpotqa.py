"""HotpotQA-distractor as an agent task suite — multi-hop QA, the content axis.

HotpotQA (Yang et al. 2018) is the canonical *multi-hop* QA benchmark: a question
whose answer requires chaining facts across two supporting paragraphs. The
**distractor** setting ships each question's 2 gold paragraphs plus 8 adversarial
distractors, so retrieving the wrong one is a real, **labeled** failure — unlike
GSM8K, where distractor tools had to be synthesized.

This is where tool-bracketing's central claim earns its keep. We model retrieval as
**paragraphs-as-tools**: each of a question's paragraphs is a `p<i>` tool that returns
its text. The bracketing `select` is then a **content-aware retriever** — a per-turn
top-k over the paragraphs by relevance to (question + what's been read so far), so the
allowlist *varies by content*, not by turn index (the GSM8K stand-in). Across the three
strategies this is a genuine experiment:

- **FULL** — all paragraphs decodable (no retrieval bracketing): the model picks freely
  among gold + distractors. Expect a higher **trap rate**.
- **MASK** — the retriever brackets the decode to its top-k (stable full prefix): the
  model can only read what the retriever surfaced. Trap rate tracks retriever quality.
- **MUTATE** — same top-k, but the prefix is rewritten to it each turn (busts the cache).

Multi-hop falls out of the shared read-state: reading gold paragraph A adds A's text to
the query, which boosts the bridge paragraph B for the next turn. A "trap" is reading a
distractor — surfaced via a `retrieved_label` telemetry event the analysis turns into the
trap-rate lens (ground truth, not the GSM8K tool-choice proxy).

The lexical retriever (`relevance_select`) is a BM25-lite stand-in — exactly the role
`rotating_select` played for GSM8K — and is a drop-in seam for a real embedding retriever.
The sample is public-style fixture data.
"""

import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agent.bench_telemetry import BenchSpec, TaskBinding, ToolContext, ToolFn
from agent.bracket import Select
from agent.scoring import exact_match, f1_score, normalize_answer
from agent.tasks import Task
from effective.telemetry import event

_DATA = Path(__file__).parent / "data" / "hotpotqa_distractor_sample.jsonl"

HOTPOTQA_TURN_INSTRUCTIONS = (
    "You are a ReAct agent answering a multi-hop question. Each turn, read ONE paragraph "
    "tool to gather a fact, chaining across paragraphs — a clue in one paragraph tells you "
    "which to read next. When you have the supporting facts, FINISH with the answer as a "
    "short phrase (a name, place, year, or yes/no)."
)

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def em_check(gold: str) -> Any:
    """The task's boolean `check`: normalized exact match, or the normalized gold as a
    phrase in the answer (the agent often answers in a sentence) — lenient, for the
    pass/fail gate. The strict, comparable metrics are the official EM/F1 *scorers* below
    (`agent.scoring`), applied at replay time."""
    g = normalize_answer(gold)

    def check(answer: str) -> bool:
        a = normalize_answer(answer)
        return bool(g) and (a == g or f" {g} " in f" {a} ")

    return check


def hotpot_em_scorer(answer: str, task: Task) -> float:
    """Official HotpotQA answer **EM** over the recorded answer (replay-time, free)."""
    return exact_match(answer, task.payload["answer"])


def hotpot_f1_scorer(answer: str, task: Task) -> float:
    """Official HotpotQA answer **token-F1** — the headline metric (replay-time, free)."""
    return f1_score(answer, task.payload["answer"])


def relevance_select(
    paras: list[dict[str, Any]], question: str, read_so_far: list[str], k: int
) -> Select:
    """A content-aware allowlist: each turn, the top-`k` paragraphs by BM25-lite relevance
    to (question + everything read so far). Reading a paragraph mutates `read_so_far`, so
    the next turn's query — and thus the top-k — shifts toward the bridge paragraph (the
    multi-hop dynamic). A lexical stand-in for a real retriever; same seam role as
    `rotating_select` for GSM8K."""
    docs = [_tokenize(p["text"]) for p in paras]
    n = len(docs)
    df: Counter[str] = Counter()
    for doc in docs:
        for term in set(doc):
            df[term] += 1
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in df}
    tfs = [Counter(doc) for doc in docs]

    def select(turn: int) -> set[str]:
        query = set(_tokenize(question)) | set(_tokenize(" ".join(read_so_far)))
        scored = sorted(
            (
                (sum(idf.get(t, 0.0) * tf[t] for t in query), p["id"])
                for p, tf in zip(paras, tfs, strict=True)
            ),
            reverse=True,
        )
        return {pid for _, pid in scored[:k]}

    return select


def _bind(task: Task, ctx: ToolContext, *, k: int = 4) -> TaskBinding:
    """Each question's own paragraphs become the tool universe, with a content-aware
    `select` and tools that emit a `retrieved_label` event (gold/distractor) per read."""
    paras: list[dict[str, Any]] = task.payload["paras"]
    read_so_far: list[str] = []  # shared: tools append, select reads (the multi-hop state)

    def make_para_tool(p: dict[str, Any]) -> ToolFn:
        def tool(_args: dict[str, Any]) -> str:
            read_so_far.append(p["text"])
            if ctx.sink is not None:
                pid = p["id"]
                title = p["title"]
                retrieved_label = "gold" if p["gold"] else "distractor"
                event(
                    t"retrieval {pid=} {title=} {retrieved_label=}",
                    sink=ctx.sink,
                    session_id=ctx.session_id,
                    kind="CHAIN",
                )
            return p["text"]

        return tool

    catalog = [p["id"] for p in paras]
    doc = {p["id"]: f"{p['id']}() — read the paragraph titled '{p['title']}'" for p in paras}
    tools = {p["id"]: make_para_tool(p) for p in paras}
    select = relevance_select(paras, task.prompt, read_so_far, k=min(k, len(paras)))
    return TaskBinding(
        catalog=catalog,
        doc=doc,
        tools=tools,
        select=select,
        keep=(),
        instructions=HOTPOTQA_TURN_INSTRUCTIONS,
    )


def hotpotqa_is_distractor(fields: Mapping[str, Any]) -> bool:
    """Lens 1 as ground truth: a retrieval is a distractor when it returned a distractor
    paragraph (`retrieved_label`). Falls back to "not a paragraph tool" if the label is
    absent (defensive — the emit site always sets it)."""
    label = fields.get("retrieved_label")
    if label is not None:
        return label == "distractor"
    return not str(fields.get("chose", "")).startswith("p")


def _resolve_path(path: Path | None) -> Path:
    """Where to load from: an explicit `path`, else `$HOTPOTQA_DATA` (the real dev split,
    pinned at `infra/hotpotqa/PIN.txt`), else the vendored hand-authored fixture."""
    if path is not None:
        return path
    env = os.environ.get("HOTPOTQA_DATA")
    return Path(env) if env else _DATA


def load_hotpotqa(limit: int | None = None, *, path: Path | None = None) -> list[Task]:
    """Load HotpotQA-distractor records (the public schema) into `Task`s. Each `payload`
    carries the question's paragraphs tagged gold/distractor (gold = a title in
    `supporting_facts`); `check` scores by normalized EM/containment.

    Source resolution (see `_resolve_path`): explicit `path` > `$HOTPOTQA_DATA` (the real
    dev split, fetched + SHA256-verified out of the repo) > the vendored hand-authored
    fixture (fully fictional, self-contained — so the bench runs offline and reproducibly).
    The real and fixture data share one schema, so the swap is data-only, no code change."""
    path = _resolve_path(path)
    tasks: list[Task] = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        if limit is not None and i >= limit:
            break
        record = json.loads(line)
        gold_titles = {title for title, _ in record["supporting_facts"]}
        paras = [
            {"id": f"p{j}", "title": title, "text": " ".join(sents), "gold": title in gold_titles}
            for j, (title, sents) in enumerate(record["context"])
        ]
        tasks.append(
            Task(
                name=record.get("_id", f"hotpot_{i:03d}"),
                prompt=record["question"],
                check=em_check(record["answer"]),
                payload={"paras": paras, "answer": record["answer"]},
            )
        )
    return tasks


# The HotpotQA-distractor `BenchSpec` — dynamic (per-task) binding: each question's
# paragraphs are its tool universe and the content-aware retriever its `select`.
HOTPOTQA_SPEC = BenchSpec(
    name="hotpotqa-distractor",
    load=load_hotpotqa,
    bind=_bind,
    is_distractor=hotpotqa_is_distractor,
    instructions=HOTPOTQA_TURN_INSTRUCTIONS,
    scorer=hotpot_f1_scorer,  # headline quality = official answer F1 (EM available via rescore)
)
