"""A small, cheap, deterministic agent task suite (the quality-axis canary).

These tasks (multiply, lookups, a two-step chain, word-number parsing) are
*throwaway* — chosen to run for fractions of a cent and to vary in difficulty so
a pass-rate over them differentiates models. They are NOT what we ultimately care
about; they exist to exercise the scorer as a seam. Real workflows replace them.

`TOOLS` are pure local callables (no I/O); `SYSTEM_PROMPT` describes them + the
ReAct protocol with a few-shot example (small models need the exact JSON shape);
`TASKS` pairs a prompt with a checker over the final answer.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from effective.keys import Key, Subject, compose_key
from effective.keys.grammar import KeySyntaxError


def task_scope(task_id: str) -> Key:
    """The frame every bench trial runs inside — `task:{task_id}`, minted in ONE place.

    **A borrowed identity kept RAW, deliberately.** A foreign value must be normalized into an
    atom kind; a digest is the total function that always works, but you may keep the id raw
    *where you can state the guarantee that it is in-language and enforce it at the boundary*.
    A bench task id
    (`astropy__astropy-12907`, `sem-count-3-64`) is already a well-formed `NAME`, so digesting
    it would destroy the legibility that makes a bench run readable and buy nothing.

    So this is the enforcement, and it RAISES rather than digesting quietly: a dataset that grows
    a task id out of language should stop the run at the boundary, where the id is still in hand
    and the fix is obvious, instead of silently becoming a hash nobody can read back."""
    try:
        return compose_key(t"task:{Subject(task_id)}")
    except (ValueError, KeySyntaxError) as refused:
        raise ValueError(
            f"bench task id {task_id!r} is not a well-formed atom, so it cannot name a trial's "
            f"scope. Bench ids are kept RAW for legibility on the promise that they are in "
            f"language; this one breaks it. Either normalize the id in the dataset loader, or "
            f"digest it here (`handlers.base.digest_atom`) and accept an unreadable frame: "
            f"{refused}"
        ) from refused


def _num(x: Any) -> float | int:
    f = float(x)
    return int(f) if f.is_integer() else f


_POPULATION = {"denver": 715000, "boulder": 108000, "aspen": 7400, "silverthorne": 4800}


def _multiply(args: dict[str, Any]) -> Any:
    return _num(args["a"]) * _num(args["b"])


def _add(args: dict[str, Any]) -> Any:
    return _num(args["a"]) + _num(args["b"])


def _subtract(args: dict[str, Any]) -> Any:
    return _num(args["a"]) - _num(args["b"])


def _population(args: dict[str, Any]) -> Any:
    return _POPULATION.get(str(args["city"]).strip().lower(), "unknown")


TOOLS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "multiply": _multiply,
    "add": _add,
    "subtract": _subtract,
    "population": _population,
}


# --- answer checkers -------------------------------------------------------


def has_number(n: int | float) -> Callable[[str], bool]:
    """Match the number as a standalone token (boundary-aware, comma-tolerant)."""
    pattern = re.compile(rf"(?<!\d){n}(?!\d)")

    def check(answer: str) -> bool:
        return bool(pattern.search(answer.replace(",", "")))

    return check


def has_text(s: str) -> Callable[[str], bool]:
    return lambda answer: s.lower() in answer.lower()


def all_of(*checks: Callable[[str], bool]) -> Callable[[str], bool]:
    return lambda answer: all(c(answer) for c in checks)


@dataclass(frozen=True)
class Task:
    name: str
    prompt: str
    check: Callable[[str], bool]
    payload: Any = None  # opaque per-benchmark data a BenchSpec.bind reads (e.g. HotpotQA paras)


SYSTEM_PROMPT = r"""You are a ReAct agent that answers by calling a tool or giving a final answer.

Tools available:
  multiply(a, b)   -> product of two numbers
  add(a, b)        -> sum of two numbers
  subtract(a, b)   -> a minus b
  population(city) -> the city's population

Fill `thought` with brief reasoning each turn. To ACT, copy this shape exactly:
  {"thought":"x","tool":{"name":"add","arguments_json":"{\"a\":1,\"b\":2}"},"answer":null}
To FINISH:
  {"thought":"...","tool":null,"answer":"the result"}
Use tools for arithmetic and lookups rather than doing it from memory. If you have
not gathered what you need yet, you MUST set `tool` this turn — never leave both
`tool` and `answer` null."""


TASKS: list[Task] = [
    Task("multiply_basic", "Compute 19 times 23 and give the product.", has_number(437)),
    Task(
        "two_step",
        "Add 7 and 5, then multiply that result by 3. Give the final number.",
        has_number(36),
    ),
    Task(
        "word_args", "Multiply one hundred seven by twelve and give the product.", has_number(1284)
    ),
    Task("lookup", "Report Boulder's population.", has_number(108000)),
    Task(
        "compare",
        "How much larger is Denver's population than Boulder's? Give the difference.",
        all_of(has_text("denver"), has_number(607000)),
    ),
    Task("direct", "What is 5 plus 5? Answer with just the number.", has_number(10)),
]
