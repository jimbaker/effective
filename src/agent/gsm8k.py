"""GSM8K as an agent task suite: a standard, non-trivial quality axis.

GSM8K (Cobbe et al. 2021) is the canonical grade-school-math word-problem
benchmark: multi-step arithmetic that a model must *reason* through.
Run as a ReAct task (a calculator-shaped toolbox, one arithmetic op per turn) it
is **multi-turn**, which a KV-cache measurement needs: cache reuse exists only
across turns.

The sample is *vendored* (`data/gsm8k_sample.jsonl`, the first N of the public test
split) so a bench is reproducible and fully offline.

`load_gsm8k` yields `Task`s (the same dataclass the scorer already consumes); the
checker extracts the final number from the agent's answer and compares it to the
gold value after GSM8K's `####` marker.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path

from agent.bench_telemetry import BenchSpec
from agent.bracket import ARITHMETIC, CATALOG_TOOLS, TOOL_DOC
from agent.tasks import Task

_DATA = Path(__file__).parent / "data" / "gsm8k_sample.jsonl"

# A signed integer or decimal, with optional thousands separators ("1,800", "18.0").
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _to_number(token: str) -> float:
    return float(token.replace(",", ""))


def gold_answer(record_answer: str) -> float:
    """The reference value: GSM8K puts it after a `####` marker on the last line."""
    return _to_number(record_answer.split("####")[-1].strip())


def extract_final_number(answer: str) -> float | None:
    """The last number in the agent's free-text answer (its final result), or None.

    GSM8K answers are a single value; agents tend to restate it last ("... = 18"),
    so the trailing number is the answer-bearing token."""
    matches = _NUMBER.findall(answer)
    return _to_number(matches[-1]) if matches else None


def gsm8k_check(gold: float) -> Callable[[str], bool]:
    """A checker: the agent's final number must equal `gold` (gold values are integral)."""

    def check(answer: str) -> bool:
        got = extract_final_number(answer)
        return got is not None and abs(got - gold) < 1e-6

    return check


def load_gsm8k(limit: int | None = None, *, path: Path = _DATA) -> list[Task]:
    """Load the vendored GSM8K sample into `Task`s (deterministic order)."""
    tasks: list[Task] = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        if limit is not None and i >= limit:
            break
        record = json.loads(line)
        gold = gold_answer(record["answer"])
        tasks.append(
            Task(
                name=f"gsm8k_{i:03d}",
                prompt=record["question"],
                check=gsm8k_check(gold),
            )
        )
    return tasks


# GSM8K's `BenchSpec` — the task-specific axes the agnostic bench analysis needs
# (`agent.bench_telemetry`). The four arithmetic tools are the `keep_tools`; the rest of
# the 12-tool catalog are distractors. The default `is_distractor` ("chose a tool not in
# `keep_tools`") is exactly the prior bench's non-arithmetic-selection signal — GSM8K
# *synthesizes* its distractors (HotpotQA-distractor ships them labeled), so the proxy
# lives here.
GSM8K_SPEC = BenchSpec(
    name="gsm8k",
    load=load_gsm8k,
    catalog_tools=CATALOG_TOOLS,
    keep_tools=ARITHMETIC,
    doc=TOOL_DOC,
)
