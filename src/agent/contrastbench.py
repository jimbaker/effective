"""RLM-vs-structured contrast bench: a REPL-style RLM against the durable, typed run_code loop.

Four arms over seeded synthetic ops-log aggregation QA:

| arm               | shape                                                                    |
|-------------------|--------------------------------------------------------------------------|
| `stuff`           | the long-context baseline: one completion, full log in the prompt        |
| `repl`            | the Zhang/dspy shape: a client-side loop OUTSIDE the substrate (no ops,  |
|                   | no checkpoints), freeform code over a `ctx` string in a direct Monty     |
|                   | exec, observations truncated to 500 chars (``REPLVariable`` previews)    |
| `structured`      | ``run_agent(act=code_act(...))`` over the embedded SQLite engine, typed  |
|                   | ``records`` rows, schema-validated whole observations, every turn        |
|                   | checkpointed                                                             |
| `structured-text` | `structured` with the raw ``ctx`` string as input (the ablation)         |

A completed structured task's DB replays to the same answer with **zero model calls**
(`replay_structured`).

The generator computes ground truth; scoring never consults a model. This is bench driver
code: the workflows inside obey the determinism boundary, and the driver around them is
ordinary I/O code, as in `skillsbench`.
"""

import json
import random
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from agent.tasks import task_scope
from effective.api import Effect, ask_llm, scoped
from effective.code import EXECUTE_TOOL, CodeOutcome, run_code
from effective.combinators import route
from effective.compose import code_act
from effective.cost import CostBudget, MeteredInterpreter, Usage
from effective.domain import AskLLM, CallTool
from effective.engines.sqlite import SqliteApp, SqliteTaskContext
from effective.envelope import Envelope, JsonEnvelope
from effective.handlers.durable import DurableHandler
from effective.interpreters.openai import (
    GPT5_NANO,  # noqa: F401 — re-exported; `tests/test_contrastbench.py` imports it from here
    Price,
    resolve_price,
    structured_extract,
    to_openai_messages,
    usage_from_openai,
)
from effective.monty import MontyEngine, execute_tool
from effective.react import AssistantTurn, ToolRequest, Trajectory, run_agent

ARMS = ("stuff", "repl", "structured", "structured-text", "combinator")
SIZES: Mapping[str, int] = {"S": 150, "L": 4000}
SEM_SIZES: Mapping[str, int] = {"S": 150, "L": 800}  # per-record judgment economics

USERS = ("uma", "raj", "lena", "kai", "noor", "ivan", "sara", "tom")
ACTIONS = ("refund", "order", "login", "upload", "delete", "export")

MAX_TURNS = 8
REPL_TRUNCATE = 500  # their REPLVariable preview cap — the measured sis bug, kept faithfully
SANDBOX_SECONDS = 5.0
# Monty's max_duration_secs counts wall time ACROSS host calls,
# so a semantic segment's budget must cover llm_query network time; the sub-call
# cap is what actually bounds the trial.
SEM_SANDBOX_SECONDS = 300.0
MAX_SUBCALLS = 128  # loud at the cap: a measured behavior, not a crash
# Enough output budget that reasoning + ~150 labels fit; truncation RAISES.
SUBCALL_MAX_TOKENS = 8000

# The note pool: ~12 fixed templates per hidden label, paraphrase
# variety, no single giveaway token per label, a few deliberately tricky items.
# The pre-flight probe gates on >=90% pool agreement before any trial.
NOTE_TEMPLATES: Mapping[str, tuple[str, ...]] = {
    "complaint": (
        "took three weeks and nobody answered my emails",
        "the box arrived crushed and half the items were missing",
        "second time this has gone wrong, I want my money back",
        "support kept transferring me in circles",
        "charged twice for the same order",
        "not what I hoped for, honestly",
        "the app crashed every time I tried to check out",
        "promised by Friday, still nothing a week later",
        "quality has really gone downhill lately",
        "I had to fix the paperwork myself, again",
        "the replacement was worse than the original",
        "waited forty minutes on hold and then got cut off",
    ),
    "praise": (
        "arrived a day early, perfectly packed",
        "support sorted it in five minutes, brilliant",
        "exactly as described, would order again",
        "the new dashboard is a huge improvement",
        "smoothest checkout I've had anywhere",
        "can't complain, everything just worked",
        "the team went out of their way to help",
        "best value I've found this year",
        "refund processed same day, no questions asked",
        "setup took two minutes flat",
        "documentation answered every question I had",
        "the follow-up call was a nice touch",
    ),
    "neutral": (
        "reordered the usual monthly supplies",
        "invoice number updated per accounting",
        "shipping address changed to the new office",
        "duplicate of an earlier ticket, merged",
        "scheduled the renewal for next quarter",
        "standard onboarding, nothing to report",
        "exported the report for the audit file",
        "payment method switched to the corporate card",
        "account transferred to the new manager",
        "quantity adjusted from three to four",
        "confirmed the delivery window by phone",
        "invoice copy forwarded to accounting",
    ),
}


# --- task generation (seeded; ground truth computed, never modeled) ------------


@dataclass(frozen=True)
class ContrastTask:
    task_id: str
    qtype: str
    size: str
    question: str
    truth: str  # normalized ground truth
    seed: int
    n_records: int = 0  # 0 -> legacy contrast-1 lookup via SIZES
    semantic: bool = False

    @property
    def records(self) -> int:
        return self.n_records or SIZES[self.size]


def make_log(seed: int, n_records: int) -> tuple[list[dict[str, Any]], str]:
    """The ops log for (seed, size): typed rows and their text rendering."""
    rng = random.Random(f"log:{seed}:{n_records}")
    t = datetime(2026, 3, 1, 0, 0)
    rows: list[dict[str, Any]] = []
    for _ in range(n_records):
        t += timedelta(minutes=rng.randint(1, 30))
        rows.append(
            {
                "ts": t.strftime("%Y-%m-%dT%H:%M"),
                "user": rng.choice(USERS),
                "action": rng.choice(ACTIONS),
                "amount": rng.randint(1, 500),
            }
        )
    text = "\n".join(
        f"{r['ts']} user={r['user']} action={r['action']} amount={r['amount']}" for r in rows
    )
    return rows, text


def make_sem_log(seed: int, n_records: int) -> tuple[list[dict[str, Any]], str, list[str]]:
    """The semantic log: rows carry a free-text ``note``; the HIDDEN labels
    come back separately (ground truth only — never in a row, never rendered)."""
    rows, _ = make_log(seed, n_records)
    rng = random.Random(f"notes:{seed}:{n_records}")
    labels: list[str] = []
    noted: list[dict[str, Any]] = []
    kinds = tuple(NOTE_TEMPLATES)
    for r in rows:
        label = rng.choice(kinds)
        labels.append(label)
        noted.append({**r, "note": rng.choice(NOTE_TEMPLATES[label])})
    text = "\n".join(
        f'{r["ts"]} user={r["user"]} action={r["action"]} amount={r["amount"]} note="{r["note"]}"'
        for r in noted
    )
    return noted, text, labels


def _sem_questions_for(seed: int, size: str, n: int) -> list[ContrastTask]:
    rows, _, labels = make_sem_log(seed, n)
    qrng = random.Random(f"semq:{seed}:{size}")
    actions = list(ACTIONS)
    qrng.shuffle(actions)
    tasks: list[ContrastTask] = []
    for i, sentiment in enumerate(("complaint", "praise")):
        # sem-filter-count: action chosen so the count is nonzero
        for action in actions:
            count = sum(
                1
                for r, lab in zip(rows, labels, strict=True)
                if r["action"] == action and lab == sentiment
            )
            if count:
                break
        else:  # pragma: no cover — statistically impossible at these sizes
            raise ValueError(f"no {sentiment} records under any action at {size}")
        tasks.append(
            ContrastTask(
                task_id=f"sem-count-{i}-{size}",
                qtype="sem-filter-count",
                size=size,
                question=(
                    f"How many records with action={action} have a note whose text "
                    f"expresses a {sentiment}? Notes are free text, not labels — "
                    f"judge what each note MEANS (every note expresses a complaint, "
                    f"praise, or neutral routine content)."
                ),
                truth=normalize(count),
                seed=seed,
                n_records=n,
                semantic=True,
            )
        )
        # sem-argmax: scoped to an action whose top user is unique — the same
        # unique-by-construction move as contrast-1's group-argmax
        for action in reversed(actions):
            counts: dict[str, int] = {}
            for r, lab in zip(rows, labels, strict=True):
                if r["action"] == action and lab == sentiment:
                    counts[r["user"]] = counts.get(r["user"], 0) + 1
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
                break
        else:  # pragma: no cover — statistically implausible at these sizes
            raise ValueError(f"no unique {sentiment} argmax under any action at {size}")
        tasks.append(
            ContrastTask(
                task_id=f"sem-argmax-{i}-{size}",
                qtype="sem-argmax",
                size=size,
                question=(
                    f"Which user has the most records with action={action} whose "
                    f"note text expresses a {sentiment}? Notes are free text, not "
                    f"labels — judge what each note MEANS (every note expresses a "
                    f"complaint, praise, or neutral routine content)."
                ),
                truth=normalize(ranked[0][0]),
                seed=seed,
                n_records=n,
                semantic=True,
            )
        )
    return tasks


def make_sem_tasks(seed: int, sizes: Mapping[str, int] = SEM_SIZES) -> list[ContrastTask]:
    out: list[ContrastTask] = []
    for size, n in sizes.items():
        out.extend(_sem_questions_for(seed, size, n))
    return out


def normalize(x: Any) -> str:
    """The registered scoring normalization."""
    s = str(x).strip().lower()
    t = s.replace(",", "").replace(" ", "")
    try:
        return str(int(t))
    except ValueError:
        return s


_Rows = list[dict[str, Any]]
_Built = tuple[str, Any] | None  # (question, truth) or None = not scorable here


def _q_filter_count(sub: _Rows, action: str, qrng: random.Random) -> _Built:
    return (
        f"How many records have action={action} and amount > 250?",
        sum(1 for r in sub if r["amount"] > 250),
    )


def _q_count_distinct(sub: _Rows, action: str, qrng: random.Random) -> _Built:
    return (
        f"How many distinct users have at least one record with action={action}?",
        len({r["user"] for r in sub}),
    )


def _q_group_argmax(sub: _Rows, action: str, qrng: random.Random) -> _Built:
    counts: dict[str, int] = {}
    for r in sub:
        counts[r["user"]] = counts.get(r["user"], 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None  # tie -> not exact-match scorable with this action
    return (f"Which user has the most records with action={action}?", ranked[0][0])


def _q_sum(sub: _Rows, action: str, qrng: random.Random) -> _Built:
    user = qrng.choice(sorted({r["user"] for r in sub}))
    return (
        f"What is the total amount over all records with user={user} and action={action}?",
        sum(r["amount"] for r in sub if r["user"] == user),
    )


_QBUILDERS: Mapping[str, Callable[[_Rows, str, random.Random], _Built]] = {
    "filter-count": _q_filter_count,
    "count-distinct": _q_count_distinct,
    "group-argmax": _q_group_argmax,
    "sum": _q_sum,
}


def _questions_for(rows: _Rows, seed: int, size: str) -> list[ContrastTask]:
    tasks: list[ContrastTask] = []
    for qtype, build in _QBUILDERS.items():
        qrng = random.Random(f"q:{seed}:{size}:{qtype}")
        actions = list(ACTIONS)
        qrng.shuffle(actions)
        made = 0
        for action in actions:
            if made == 2:
                break
            sub = [r for r in rows if r["action"] == action]
            if not sub:
                continue
            built = build(sub, action, qrng)
            if built is None:
                continue
            question, truth = built
            tasks.append(
                ContrastTask(
                    task_id=f"{qtype}-{made}-{size}",
                    qtype=qtype,
                    size=size,
                    question=question,
                    truth=normalize(truth),
                    seed=seed,
                )
            )
            made += 1
        if made < 2:
            raise ValueError(f"generator could not make 2 {qtype} questions at {size}")
    return tasks


def make_tasks(seed: int, sizes: Mapping[str, int] = SIZES) -> list[ContrastTask]:
    out: list[ContrastTask] = []
    for size, n in sizes.items():
        rows, _ = make_log(seed, n)
        out.extend(_questions_for(rows, seed, size))
    return out


# --- the wire (one JSON alphabet for every arm) --------------------------------

_CHANNELS = ("thought", "code", "answer")


class _Turn(BaseModel):
    thought: str | None = None
    code: str | None = None
    answer: str | None = None


_NULLISH = frozenset({"", "null", "none", "n/a", "-"})


def _to_turn(raw: Mapping[str, Any]) -> _Turn:
    filled: dict[str, Any] = {}
    for name in _CHANNELS:
        value = raw.get(name)
        if isinstance(value, str) and value.strip().lower() in _NULLISH:
            value = None
        if value is not None and name != "code":
            value = str(value)
        filled[name] = value
    return _Turn.model_validate(filled)


class WireCaller:
    """One JSON-mode completion returning a `_Turn` + priced Usage; one retry
    (`wire_failures` counts first-parse losses — the skillsbench wire lesson)."""

    def __init__(
        self,
        client: Any,
        system_prompt: str,
        *,
        model: str = "gpt-5-nano",
        price: Price | None = None,
        max_completion_tokens: int = 2000,
        extra: Mapping[str, Any] | None = None,
        envelope: Envelope | None = None,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.model = model
        self.price = resolve_price(model, price)
        self.max_completion_tokens = max_completion_tokens
        self.extra = dict(extra or {})
        self.envelope = envelope if envelope is not None else JsonEnvelope()
        self.wire_failures = 0
        self.calls = 0
        self.meter = Usage()

    def turn(self, messages: list[dict[str, Any]]) -> _Turn:
        turn, _ = self.turn_usage(messages)
        return turn

    def turn_usage(self, messages: list[dict[str, Any]]) -> tuple[_Turn, Usage]:
        try:
            turn, usage = self._once(messages)
        except Exception:
            self.wire_failures += 1
            turn, usage = self._once(messages)
        self.calls += 1
        self.meter = self.meter + usage
        return turn, usage

    def _once(self, messages: list[dict[str, Any]]) -> tuple[_Turn, Usage]:

        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.system_prompt}, *messages],
            max_completion_tokens=self.max_completion_tokens,
            response_format={"type": "json_object"},
            **self.extra,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("empty completion (refusal or truncation)")
        turn = _to_turn(self.envelope.parse(content))
        usage = replace(
            usage_from_openai(response, self.price), latency_s=time.perf_counter() - start
        )
        return turn, usage

    # LLMCall for MeteredInterpreter (the structured arms): AskLLM -> AssistantTurn.
    # Real usage flows to the metered layer so CostBudget enforces the trial cap;
    # cost is REPORTED off self.meter (one accrual point, no double count).
    def __call__(self, op: AskLLM[Any]) -> tuple[AssistantTurn, Usage]:
        turn, usage = self.turn_usage(to_openai_messages(op.messages))
        tool = ToolRequest(name="run_code", args={"code": turn.code}) if turn.code else None
        assistant = AssistantTurn(thought=turn.thought or "", tool=tool, answer=turn.answer)
        return assistant, usage


class SubcallMeter:
    """The `llm_query` host function for the structured arms:
    one plain-text completion per call, capped loudly at MAX_SUBCALLS, usage
    metered separately from the turn path so batching reads on its own."""

    SYSTEM = (
        "Answer the sub-question concisely and follow the requested output "
        "format exactly. When classifying, use only the labels the prompt names."
    )

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        price: Price,
        max_calls: int = MAX_SUBCALLS,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.price = price
        self.max_calls = max_calls
        self.extra = dict(extra or {})
        self.calls = 0
        self.meter = Usage()

    def fn(self, prompt: Any) -> str:
        if self.calls >= self.max_calls:
            raise RuntimeError(f"llm_query cap ({self.max_calls}) exceeded")
        self.calls += 1
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": str(prompt)},
            ],
            max_completion_tokens=SUBCALL_MAX_TOKENS,
            **self.extra,
        )
        self.meter = self.meter + usage_from_openai(response, self.price)
        content = response.choices[0].message.content or ""
        # A silently truncated/empty reply is the registered failure class:
        # be LOUD so the code can adapt (split the batch, ask for less).
        if not content or response.choices[0].finish_reason == "length":
            raise RuntimeError(
                "llm_query reply truncated — batch fewer items or request shorter output"
            )
        return content


def probe_templates(
    client: Any, *, model: str = "gpt-5-nano", price: Price | None = None
) -> tuple[float, list[tuple[str, str, str]], Usage]:
    """The pre-flight gate: ONE batched call classifies every
    template; returns (pool agreement, misses as (truth, note, got), usage).
    The campaign must not start below 0.90 — revise templates pre-data instead."""
    price = resolve_price(model, price)
    items = [(label, t) for label, ts in NOTE_TEMPLATES.items() for t in ts]
    listing = "\n".join(f"{i + 1}. {t}" for i, (_, t) in enumerate(items))
    prompt = (
        "For each numbered note below, judge what it expresses. Reply with exactly "
        f"{len(items)} lines, one label per line in order, each chosen from: "
        "complaint, praise, neutral. No numbering, nothing else.\n\n" + listing
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SubcallMeter.SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=2000,
    )
    lines = [ln.strip() for ln in (response.choices[0].message.content or "").splitlines()]
    got = [ln.split()[-1].strip(".").lower() for ln in lines if ln]
    pairs = list(zip(items, got, strict=False))
    hits = sum(1 for (label, _t), g in pairs if g == label)
    agreement = hits / len(items)
    misses = [(label, t, g) for (label, t), g in pairs if g != label]
    return agreement, misses, usage_from_openai(response, price)


# --- system prompts -------------------------------------------------------------

_FORMAT = JsonEnvelope().instructions(dict.fromkeys(_CHANNELS))

_STUFF_SYSTEM = f"""You answer one question about an operations log, exactly.
The log and the question are in the user message. Reply in ONE turn: `thought`
(brief), `code` always null, and `answer` — a plain number or username, nothing else.

{_FORMAT}"""

_REPL_SYSTEM = f"""You answer one question about an operations log by writing Python.
The full log text is ALREADY in the sandbox variable `ctx` (a string, one record
per line, format: 2026-03-14T09:41 user=uma action=refund amount=142). Do NOT ask
for the log; write code against `ctx`.

Each turn reply with `thought` (brief) and EITHER `code` OR a final `answer`
(a plain number or username, nothing else — give it as soon as you know it).
Anything your code prints, plus the final expression's value, is shown to you,
TRUNCATED to 500 characters. Each turn's code runs fresh — no variables persist
between turns. Plain Python only: str/list/dict/set methods, comprehensions,
functions; no imports, classes, yield, or match.

{_FORMAT}"""

_STRUCTURED_SYSTEM = f"""You answer one question about an operations log by writing Python.
The log is ALREADY in the sandbox as `records`: a list of dicts with keys
ts (str), user (str), action (str), amount (int). Do NOT ask for the data;
write code against `records`.

Each turn reply with `thought` (brief) and EITHER `code` OR a final `answer`
(a plain number or username, nothing else — give it as soon as you know it).
Your code's final expression value is validated and returned to you IN FULL as
JSON. Each turn's code runs fresh — no variables persist between turns. Plain
Python only: str/list/dict/set methods, comprehensions, functions; no imports,
classes, yield, or match.

{_FORMAT}"""

_STRUCTURED_TEXT_SYSTEM = _STRUCTURED_SYSTEM.replace(
    """The log is ALREADY in the sandbox as `records`: a list of dicts with keys
ts (str), user (str), action (str), amount (int). Do NOT ask for the data;
write code against `records`.""",
    """The full log text is ALREADY in the sandbox variable `ctx` (a string, one
record per line, format: 2026-03-14T09:41 user=uma action=refund amount=142).
Do NOT ask for the data; write code against `ctx`.""",
)

_SEM_TOOL = """
You ALSO have llm_query(prompt) -> str inside the sandbox: ask a small language
model a sub-question (e.g. judge what a note means). At most 128 calls per code
run. A truncated reply raises an error, and OVERSIZED batches can come back
misaligned — keep each batch small enough to verify, demand a strictly
formatted reply, and check you got as many answers as items. Notes express
exactly one of: complaint, praise, neutral. Judge meaning — keyword matching
is unreliable."""


def contrast_system_prompt(arm: str, *, semantic: bool = False) -> str:
    base = SYSTEM_PROMPTS[arm]
    if not semantic or arm == "stuff":
        return base
    # patch the record-format examples to include the note field, loudly
    if arm == "structured":
        patched = base.replace(
            "ts (str), user (str), action (str), amount (int).",
            "ts (str), user (str), action (str), amount (int), note (str).",
        )
    else:
        patched = base.replace("amount=142)", 'amount=142 note="took three weeks..." )')
    if patched == base:
        raise ValueError(f"format example not found in {arm} prompt — template drift")
    return patched + _SEM_TOOL


# --- arm outcomes ---------------------------------------------------------------


@dataclass
class ArmOutcome:
    answer: str | None
    turns: int
    cells: int  # code executions
    sandbox_errors: int
    stop_reason: str
    replay_ok: bool | None = None  # structured arms only (the replay check)
    transcript: list[dict[str, Any]] | None = None  # repl arm: the raw loop messages
    sub: dict[str, int] = field(default_factory=lambda: {"calls": 0, "ptok": 0, "ctok": 0})


def run_stuff(caller: WireCaller, task: ContrastTask, text: str) -> ArmOutcome:
    turn = caller.turn([{"role": "user", "content": f"LOG:\n{text}\n\nQUESTION: {task.question}"}])
    return ArmOutcome(
        answer=turn.answer,
        turns=1,
        cells=0,
        sandbox_errors=0,
        stop_reason="finish" if turn.answer is not None else "no-answer",
    )


_REPL_DRIVER = """
import json, sys
sys.path.insert(0, sys.argv[1])
from effective.monty import MontyEngine
args = json.loads(sys.stdin.read())
llm = args.pop("_llm", None)
seconds = args.pop("_seconds")
sub = {"calls": 0, "ptok": 0, "ctok": 0, "cached": 0}
functions = {}
if llm is not None:
    from openai import OpenAI
    client = OpenAI(api_key=llm["api_key"])
    def llm_query(prompt):
        if sub["calls"] >= llm["max_calls"]:
            raise RuntimeError("llm_query cap (%d) exceeded" % llm["max_calls"])
        sub["calls"] += 1
        r = client.chat.completions.create(
            model=llm["model"],
            messages=[{"role": "system", "content": llm["system"]},
                      {"role": "user", "content": str(prompt)}],
            max_completion_tokens=llm["max_tokens"],
            **llm.get("extra", {}),
        )
        sub["ptok"] += int(r.usage.prompt_tokens or 0)
        sub["ctok"] += int(r.usage.completion_tokens or 0)
        _det = getattr(r.usage, "prompt_tokens_details", None)
        sub["cached"] += int(getattr(_det, "cached_tokens", 0) or 0) if _det is not None else 0
        content = r.choices[0].message.content or ""
        if not content or r.choices[0].finish_reason == "length":
            raise RuntimeError(
                "llm_query reply truncated — batch fewer items or request shorter output"
            )
        return content
    functions["llm_query"] = llm_query
    args["functions"] = ["llm_query"]
engine = MontyEngine(functions=functions, limits={"max_duration_secs": seconds})
try:
    outcome = engine.execute(args)
    result = {"output_repr": None if outcome.output is None else repr(outcome.output)}
except Exception as exc:
    result = {"error": str(exc)}
result["sub"] = sub
with open(sys.argv[2], "w") as f:
    json.dump(result, f)
"""


def _repl_exec(
    code: str, text: str, llm: dict[str, Any] | None = None
) -> tuple[str, bool, dict[str, int]]:
    """One repl cell in a subprocess: (observation, errored, subcall counters).
    The observation is the cell's stdout plus the final expression's repr
    (omitted when None — a real REPL doesn't echo None), the dspy-interpreter
    semantics the arm replicates. Monty's Rust `print`
    writes to OS fd 1, so capture requires the subprocess boundary. ``llm``
    (semantic tasks) registers `llm_query` in the subprocess, which makes its own
    metered calls and reports usage back."""
    src_root = str(Path(__file__).resolve().parent.parent)
    seconds = SEM_SANDBOX_SECONDS if llm is not None else SANDBOX_SECONDS
    args: dict[str, Any] = {
        "code": code,
        "inputs": {"ctx": text},
        "functions": [],
        "actions": [],
        "fn_log": [],
        "action_results": [],
        "_seconds": seconds,
        "_llm": llm,
    }
    with tempfile.NamedTemporaryFile(mode="r", suffix=".json") as out:
        proc = subprocess.run(
            [sys.executable, "-c", _REPL_DRIVER, src_root, out.name],
            input=json.dumps(args),
            capture_output=True,
            text=True,
            timeout=seconds + 25,
        )
        payload = out.read()
    result = json.loads(payload) if payload else {"error": proc.stderr[-300:] or "driver died"}
    raw_sub = result.get("sub")
    if not isinstance(raw_sub, dict):
        raw_sub = {}
    sub = {k: int(raw_sub.get(k, 0)) for k in ("calls", "ptok", "ctok")}
    if "error" in result:
        return f"error: {result['error']}", True, sub
    parts = [p for p in (proc.stdout.rstrip(), result["output_repr"]) if p]
    return ("\n".join(parts) if parts else "(no output)"), False, sub


def run_repl(
    caller: WireCaller, task: ContrastTask, text: str, llm: dict[str, Any] | None = None
) -> ArmOutcome:
    """The Zhang/dspy shape: client loop + subprocess Monty exec, outside the
    substrate — nothing recorded, stdout shown, observations truncated to their
    500-char preview. With ``llm`` set (semantic tasks), the sandbox has
    `llm_query`; the subprocess meters its own sub-calls."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": f"QUESTION: {task.question}"}]
    cells = errors = 0
    sub = {"calls": 0, "ptok": 0, "ctok": 0, "cached": 0}
    for turn_no in range(1, MAX_TURNS + 1):
        turn = caller.turn(messages)
        if turn.answer is not None and turn.code is None:
            return ArmOutcome(
                turn.answer, turn_no, cells, errors, "finish", transcript=messages, sub=sub
            )
        if turn.code is None:
            messages.append({"role": "user", "content": "Reply with code or an answer."})
            continue
        cells += 1
        obs, errored, cell_sub = _repl_exec(turn.code, text, llm)
        errors += errored
        for k in sub:
            sub[k] += int(cell_sub.get(k, 0))
        messages.append({"role": "assistant", "content": turn.model_dump_json()})
        messages.append({"role": "user", "content": f"OUTPUT: {obs[:REPL_TRUNCATE]}"})
    return ArmOutcome(None, MAX_TURNS, cells, errors, "max_iters", transcript=messages, sub=sub)


class _StructuredInterp:
    """MeteredInterpreter base wiring for the structured arms: AskLLM through the
    wire caller; the reserved execute tool through a guarded Monty engine —
    sandbox errors become checkpointed observation VALUES (the model routes
    around them), never task crashes. ``functions`` (semantic tasks) registers host
    fns like `llm_query`; their results seal into fn_log and re-bind on replay."""

    def __init__(
        self,
        caller: WireCaller,
        budget_usd: float,
        functions: Mapping[str, Any] | None = None,
        sandbox_seconds: float = SANDBOX_SECONDS,
    ) -> None:
        engine = MontyEngine(
            functions=dict(functions or {}), limits={"max_duration_secs": sandbox_seconds}
        )
        raw = execute_tool(engine)
        self.sandbox_errors = 0

        def guarded(op: CallTool[Any]) -> CodeOutcome:
            try:
                return raw(op)
            except Exception as exc:
                self.sandbox_errors += 1
                return CodeOutcome(status="complete", output={"sandbox_error": str(exc)[:500]})

        def tools(op: CallTool[Any]) -> Any:
            if op.name == EXECUTE_TOOL:
                return guarded(op)
            raise ValueError(f"unexpected tool in contrast trial: {op.name}")

        self.metered = MeteredInterpreter(llm=caller, tools=tools, budget=CostBudget(budget_usd))

    def run(self, op: Any) -> Any:
        return self.metered.run(op)


def run_structured(
    caller: WireCaller,
    task: ContrastTask,
    rows: list[dict[str, Any]],
    text: str,
    *,
    typed: bool,
    db_path: Path,
    budget_usd: float,
    subcalls: SubcallMeter | None = None,
) -> ArmOutcome:
    """The structured arm on the embedded durable engine: every turn and code segment
    checkpointed in `db_path`. With ``subcalls``, `llm_query` is a declared observational
    function whose results seal into fn_log, so they replay like everything else."""
    functions = {"llm_query": subcalls.fn} if subcalls is not None else {}
    declared = tuple(functions)
    seconds = SEM_SANDBOX_SECONDS if subcalls is not None else SANDBOX_SECONDS
    interp = _StructuredInterp(caller, budget_usd, functions=functions, sandbox_seconds=seconds)
    inputs: dict[str, Any] = {"records": rows} if typed else {"ctx": text}
    app = SqliteApp(str(db_path))
    try:

        @app.register_task("trial")
        def _trial(params: dict[str, Any], ctx: Any) -> Any:
            handler = DurableHandler(ctx, interp)
            traj = handler.run(
                lambda: scoped(
                    task_scope(task.task_id),
                    lambda: run_agent(
                        f"QUESTION: {task.question}",
                        max_iters=MAX_TURNS,
                        act=code_act(inputs=inputs, functions=declared),
                    ),
                )
            )
            return to_jsonable_python(traj)

        task_id = app.spawn("trial", {}, max_attempts=1)
        snap = app.run_until_result(task_id)
        if snap is None or snap.state != "completed":
            failure = snap.failure if snap is not None else "no snapshot"
            return ArmOutcome(None, 0, 0, interp.sandbox_errors, f"error: {failure}")
        traj = Trajectory.model_validate(snap.result)
        cells = sum(1 for s in traj.steps if s.tool is not None)
        sub = (
            {
                "calls": subcalls.calls,
                "ptok": subcalls.meter.prompt_tokens,
                "ctok": subcalls.meter.completion_tokens,
            }
            if subcalls is not None
            else {"calls": 0, "ptok": 0, "ctok": 0}
        )
        return ArmOutcome(
            answer=traj.answer or None,
            turns=len(traj.steps),
            cells=cells,
            sandbox_errors=interp.sandbox_errors,
            stop_reason=traj.stop_reason,
            sub=sub,
        )
    finally:
        app.close()


def replay_structured(
    task: ContrastTask,
    rows: list[dict[str, Any]],
    text: str,
    *,
    typed: bool,
    db_path: Path,
    declared: tuple[str, ...] = (),
) -> str | None:
    """The replay check: re-drive the SAME workflow over the completed task's checkpoints with
    a model that RAISES if called and a sandbox that RAISES if executed — every
    op must re-bind from the record. Returns the replayed answer (None = replay
    itself failed).

    **PRECONDITION — a replay must not outrun the tape's frontier.** This drives a real
    `DurableHandler` over a real task ctx, so an op the record lacks is not skipped: its thunk runs
    and the checkpoint COMMITS into the bound run. Safe here because it is only ever pointed at a
    task the campaign ran to completion; against a crashed or mid-flight one it would manufacture
    checkpoints into somebody else's run. `effective.viewing.ViewingCtx` is the enforcement — it
    turns the miss into `OutranTheTape` instead of a write.

    Note the `except Exception: return None` around the drive below: a violation here is silent by
    construction, which is why the precondition is stated rather than left to a failure."""

    def explode_llm(op: Any) -> Any:
        raise AssertionError("model called during replay")

    def explode_tool(op: Any) -> Any:
        raise AssertionError("sandbox executed during replay")

    inputs: dict[str, Any] = {"records": rows} if typed else {"ctx": text}
    app = SqliteApp(str(db_path))
    try:
        # Derive the task id from the db rather than hardcode "trial-1" — that name
        # only holds for the first spawn on a fresh file, so a db with any prior
        # spawn (resume, retry, a second trial) would replay the WRONG task's
        # checkpoints and read as a generic replay failure. A malformed db (not
        # exactly one task) is a bench bug and raises loudly, distinct from a
        # genuine replay failure (which returns None).
        task_ids = [r[0] for r in app.conn.execute("SELECT task_id FROM tasks").fetchall()]
        if len(task_ids) != 1:
            raise RuntimeError(
                f"replay_structured expects one task in {db_path}, found {task_ids}"
            )
        try:
            ctx = SqliteTaskContext(app.conn, task_ids[0], app.write_lock)
            interp = MeteredInterpreter(llm=explode_llm, tools=explode_tool)
            traj = Trajectory.model_validate(
                to_jsonable_python(
                    DurableHandler(ctx, interp).run(
                        # The SAME wrap the record path used — replay re-derives the keys by
                        # re-execution, so a missing frame here reads as a replay failure.
                        lambda: scoped(
                            task_scope(task.task_id),
                            lambda: run_agent(
                                f"QUESTION: {task.question}",
                                max_iters=MAX_TURNS,
                                act=code_act(inputs=inputs, functions=declared),
                            ),
                        )
                    )
                )
            )
            return traj.answer or None
        except Exception:
            return None
    finally:
        app.close()


# --- the combinator arm (amendment C1) -------------------------------------------
#
# The opposite architecture from `run_structured`'s model-driven loop: an
# author-FIXED decomposition on the same engine, wire, and caps. `route`'s
# recorded classifier picks the qtype (Literal-constrained one-shot extraction);
# the handler extracts that qtype's params against a typed schema and runs the
# qtype's PINNED script — zero model-written code. Registered before the data.

_PINNED_SCRIPTS: Mapping[str, str] = {
    "filter-count": (
        "count = 0\n"
        "for r in records:\n"
        '    if r["action"] == params["action"] and r["amount"] > params["min_amount"]:\n'
        "        count = count + 1\n"
        "count\n"
    ),
    "count-distinct": (
        "seen = {}\n"
        "for r in records:\n"
        '    if r["action"] == params["action"]:\n'
        '        seen[r["user"]] = True\n'
        "len(seen)\n"
    ),
    "group-argmax": (
        "counts = {}\n"
        "for r in records:\n"
        '    if r["action"] == params["action"]:\n'
        '        if r["user"] in counts:\n'
        '            counts[r["user"]] = counts[r["user"]] + 1\n'
        "        else:\n"
        '            counts[r["user"]] = 1\n'
        'best = ""\n'
        "best_n = -1\n"
        "for u in counts:\n"
        "    if counts[u] > best_n or (counts[u] == best_n and u < best):\n"
        "        best = u\n"
        "        best_n = counts[u]\n"
        "best\n"
    ),
    "sum": (
        "total = 0\n"
        "for r in records:\n"
        '    if r["user"] == params["user"] and r["action"] == params["action"]:\n'
        '        total = total + r["amount"]\n'
        "total\n"
    ),
}

_PINNED_OUT: Mapping[str, type] = {
    "filter-count": int,
    "count-distinct": int,
    "group-argmax": str,
    "sum": int,
}


class _QtypeLabel(BaseModel):
    qtype: Literal["filter-count", "count-distinct", "group-argmax", "sum"]


class _ActionParams(BaseModel):
    action: str


class _ThresholdParams(BaseModel):
    action: str
    min_amount: int


class _UserActionParams(BaseModel):
    user: str
    action: str


_PARAM_SCHEMAS: Mapping[str, type[BaseModel]] = {
    "filter-count": _ThresholdParams,
    "count-distinct": _ActionParams,
    "group-argmax": _ActionParams,
    "sum": _UserActionParams,
}

_CLASSIFY_SYSTEM = (
    "Classify the QUESTION into exactly one qtype: filter-count (count records matching "
    "a filter with a numeric threshold), count-distinct (count distinct users), "
    "group-argmax (which user has the most records), sum (total amount for a user)."
)
_EXTRACT_SYSTEM = "Extract the parameters the QUESTION names. Copy values verbatim."


def combinator_wf(question: str, records: list[dict[str, Any]]) -> Effect[str]:
    """The fixed-decomposition workflow: classify -> extract -> pinned script.

    Both model calls are sealed ops (recorded, replayed by name); the script is
    an author constant. The only thing a model can get wrong is extraction —
    the architectural claim in executable form."""

    def classify(q: str) -> Effect[str]:
        label = yield from ask_llm(
            "classify",
            [
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": f"QUESTION: {q}"},
            ],
            _QtypeLabel,
        )
        return label.qtype

    def handler_for(qtype: str) -> Callable[[str], Effect[str]]:
        def handler(q: str) -> Effect[str]:
            params = yield from ask_llm(
                "extract",
                [
                    {"role": "system", "content": _EXTRACT_SYSTEM},
                    {"role": "user", "content": f"QUESTION: {q}"},
                ],
                _PARAM_SCHEMAS[qtype],
            )
            result = yield from run_code(
                "answer",
                _PINNED_SCRIPTS[qtype],
                schema=_PINNED_OUT[qtype],
                inputs={"records": records, "params": params.model_dump()},
            )
            return str(result)

        return handler

    handlers = {qtype: handler_for(qtype) for qtype in _PINNED_SCRIPTS}
    answer = yield from route(question, classify, handlers)
    return answer


class SchemaCaller:
    """LLMCall for the combinator arm: one-shot structured extraction honoring each
    op's ``response_schema`` (``chat.completions.parse``). Meter-compatible with
    ``WireCaller`` (``.calls`` / ``.meter`` / ``.wire_failures``) so ``run_trial``'s
    accounting is arm-uniform; a parse refusal raises (-> the trial's error path),
    it is never a wire retry."""

    def __init__(
        self,
        client: Any,
        *,
        model: str = "gpt-5-nano",
        price: Price | None = None,
        extra: Mapping[str, Any] | None = None,
        max_completion_tokens: int = 2000,
    ) -> None:
        self._client = client
        self._model = model
        self._price = resolve_price(model, price)
        self._extra = dict(extra or {})
        self._max_completion_tokens = max_completion_tokens
        self.calls = 0
        self.meter = Usage()
        self.wire_failures = 0

    def __call__(self, op: AskLLM[Any]) -> tuple[Any, Usage]:
        self.calls += 1
        parsed, usage = structured_extract(
            self._client,
            model=self._model,
            messages=list(op.messages),
            response_format=op.response_schema,
            price=self._price,
            max_completion_tokens=self._max_completion_tokens,
            extra=self._extra,
        )
        self.meter = self.meter + usage
        return parsed, usage


def run_combinator(
    llm: Any,
    task: ContrastTask,
    rows: list[dict[str, Any]],
    *,
    db_path: Path,
    budget_usd: float,
) -> ArmOutcome:
    """One combinator trial on the durable engine — same lifecycle as
    ``run_structured``, a different workflow inside."""
    interp = _StructuredInterp(llm, budget_usd)
    app = SqliteApp(str(db_path))
    try:

        @app.register_task("trial")
        def _trial(params: dict[str, Any], ctx: Any) -> Any:
            handler = DurableHandler(ctx, interp)
            return handler.run(
                lambda: scoped(
                    task_scope(task.task_id),
                    lambda: combinator_wf(task.question, rows),
                )
            )

        task_id = app.spawn("trial", {}, max_attempts=1)
        snap = app.run_until_result(task_id)
        if snap is None or snap.state != "completed":
            failure = snap.failure if snap is not None else "no snapshot"
            return ArmOutcome(None, 0, 0, interp.sandbox_errors, f"error: {failure}")
        return ArmOutcome(
            answer=str(snap.result),
            turns=llm.calls,
            cells=1,
            sandbox_errors=interp.sandbox_errors,
            stop_reason="finish",
        )
    finally:
        app.close()


def replay_combinator(
    task: ContrastTask, rows: list[dict[str, Any]], *, db_path: Path
) -> str | None:
    """HC3: re-drive ``combinator_wf`` over the completed task's checkpoints with a
    model that RAISES if called and a sandbox that RAISES if executed — every op
    must re-bind from the record (the ``replay_structured`` pattern verbatim).

    **PRECONDITION — a replay must not outrun the tape's frontier.** This drives a real
    `DurableHandler` over a real task ctx, so an op the record lacks is not skipped: its thunk runs
    and the checkpoint COMMITS into the bound run. Safe here because it is only ever pointed at a
    task the campaign ran to completion; against a crashed or mid-flight one it would manufacture
    checkpoints into somebody else's run. `effective.viewing.ViewingCtx` is the enforcement — it
    turns the miss into `OutranTheTape` instead of a write."""

    def explode_llm(op: Any) -> Any:
        raise AssertionError("model called during replay")

    def explode_tool(op: Any) -> Any:
        raise AssertionError("sandbox executed during replay")

    app = SqliteApp(str(db_path))
    try:
        task_ids = [r[0] for r in app.conn.execute("SELECT task_id FROM tasks").fetchall()]
        if len(task_ids) != 1:
            raise RuntimeError(
                f"replay_combinator expects one task in {db_path}, found {task_ids}"
            )
        try:
            ctx = SqliteTaskContext(app.conn, task_ids[0], app.write_lock)
            interp = MeteredInterpreter(llm=explode_llm, tools=explode_tool)
            answer = DurableHandler(ctx, interp).run(
                lambda: scoped(
                    task_scope(task.task_id),
                    lambda: combinator_wf(task.question, rows),
                )
            )
            return str(answer)
        except Exception:
            return None
    finally:
        app.close()


def _combinator_trial(
    caller: SchemaCaller,
    task: ContrastTask,
    rows: list[dict[str, Any]],
    trial_dir: Path,
    *,
    budget_usd: float,
) -> ArmOutcome:
    """One combinator cell: fresh db, the trial, then the HC3 poison replay."""
    db_path = trial_dir / "task.db"
    if db_path.exists():
        db_path.unlink()
    outcome = run_combinator(caller, task, rows, db_path=db_path, budget_usd=budget_usd)
    if outcome.answer is not None:
        outcome.replay_ok = replay_combinator(task, rows, db_path=db_path) == outcome.answer
    return outcome


def _structured_trial(
    caller: WireCaller,
    task: ContrastTask,
    rows: list[dict[str, Any]],
    text: str,
    trial_dir: Path,
    *,
    typed: bool,
    budget_usd: float,
    subcalls: SubcallMeter | None,
    declared: tuple[str, ...],
) -> ArmOutcome:
    """One structured/structured-text cell: fresh db, the trial, the replay
    check."""
    db_path = trial_dir / "task.db"
    if db_path.exists():
        db_path.unlink()
    outcome = run_structured(
        caller,
        task,
        rows,
        text,
        typed=typed,
        db_path=db_path,
        budget_usd=budget_usd,
        subcalls=subcalls,
    )
    if outcome.answer is not None:
        replayed = replay_structured(
            task, rows, text, typed=typed, db_path=db_path, declared=declared
        )
        outcome.replay_ok = replayed == outcome.answer
    return outcome


# --- one trial ------------------------------------------------------------------


@dataclass
class TrialResult:
    task_id: str
    qtype: str
    size: str
    arm: str
    attempt: int
    reward: float
    answer: str | None
    truth: str
    stop_reason: str
    turns: int
    cells: int
    sandbox_errors: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    elapsed_sec: float
    wire_failures: int
    replay_ok: bool | None = None
    error: str | None = None
    llm_subcalls: int = 0
    subcall_ptok: int = 0
    subcall_ctok: int = 0
    subcall_cached: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def _price_subcalls(sub: Mapping[str, int], price: Price) -> float:
    """Price repl-arm sub-call tokens the SAME way ``usage_from_openai`` prices the
    structured arm's — cached input at the cheap rate. Without this the repl arm
    paid full input rate on cached prompt tokens while the structured arm got the
    discount, inflating the competitor arm's cost in a cost-comparison bench."""
    cached = sub.get("cached", 0)
    uncached = max(sub["ptok"] - cached, 0)
    return (
        uncached * price.input_per_1m
        + cached * price.cached_per_1m
        + sub["ctok"] * price.output_per_1m
    ) / 1_000_000


SYSTEM_PROMPTS: Mapping[str, str] = {
    "stuff": _STUFF_SYSTEM,
    "repl": _REPL_SYSTEM,
    "structured": _STRUCTURED_SYSTEM,
    "structured-text": _STRUCTURED_TEXT_SYSTEM,
    # no wire loop in the combinator arm — its prompts live in combinator_wf;
    # the entry keeps run_trial's unconditional WireCaller construction total
    "combinator": "",
}


def run_trial(
    task: ContrastTask,
    arm: str,
    attempt: int,
    *,
    make_client: Callable[[], Any],
    out_dir: Path,
    model: str = "gpt-5-nano",
    price: Price | None = None,
    extra: Mapping[str, Any] | None = None,
    budget_usd: float = 0.10,
) -> TrialResult:
    price = resolve_price(model, price)
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r} (use {ARMS})")
    if arm == "combinator" and task.semantic:
        raise ValueError("the combinator arm is registered for the aggregation family only (C1)")
    trial_dir = out_dir / task.task_id / arm / f"attempt-{attempt}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    if task.semantic:
        rows, text, _ = make_sem_log(task.seed, task.records)
    else:
        rows, text = make_log(task.seed, task.records)
    client = make_client()
    caller: WireCaller | SchemaCaller = WireCaller(
        client,
        contrast_system_prompt(arm, semantic=task.semantic),
        model=model,
        price=price,
        extra=extra,
    )
    subcalls = (
        SubcallMeter(client, model=model, price=price, extra=extra)
        if task.semantic and arm in ("structured", "structured-text")
        else None
    )
    declared = ("llm_query",) if subcalls is not None else ()
    started = time.monotonic()
    error: str | None = None
    outcome = ArmOutcome(None, 0, 0, 0, "error")
    try:
        if arm == "stuff":
            outcome = run_stuff(caller, task, text)
        elif arm == "repl":
            llm = None
            if task.semantic:
                llm = {
                    "model": model,
                    "api_key": client.api_key,
                    "max_calls": MAX_SUBCALLS,
                    "max_tokens": SUBCALL_MAX_TOKENS,
                    "extra": dict(extra or {}),
                    "system": SubcallMeter.SYSTEM,
                }
            outcome = run_repl(caller, task, text, llm)
        elif arm == "combinator":
            # no wire loop for this arm: its model calls are one-shot structured
            # extractions, so the meter source is the SchemaCaller (rebound for
            # the arm-uniform accounting below)
            caller = SchemaCaller(client, model=model, price=price, extra=extra)
            outcome = _combinator_trial(caller, task, rows, trial_dir, budget_usd=budget_usd)
        else:
            # lint: totality(guard) — this assertion rejects a violated flow invariant and narrows
            # `caller`; only the combinator branch can rebind it to `SchemaCaller`.
            assert isinstance(caller, WireCaller)
            outcome = _structured_trial(
                caller,
                task,
                rows,
                text,
                trial_dir,
                typed=arm == "structured",
                budget_usd=budget_usd,
                subcalls=subcalls,
                declared=declared,
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    reward = (
        1.0 if (outcome.answer is not None and normalize(outcome.answer) == task.truth) else 0.0
    )
    trial = TrialResult(
        task_id=task.task_id,
        qtype=task.qtype,
        size=task.size,
        arm=arm,
        attempt=attempt,
        reward=reward,
        answer=outcome.answer,
        truth=task.truth,
        stop_reason=outcome.stop_reason if error is None else "error",
        turns=outcome.turns,
        cells=outcome.cells,
        sandbox_errors=outcome.sandbox_errors,
        llm_calls=caller.calls,
        prompt_tokens=caller.meter.prompt_tokens,
        completion_tokens=caller.meter.completion_tokens,
        cost_usd=caller.meter.cost
        + (subcalls.meter.cost if subcalls is not None else _price_subcalls(outcome.sub, price)),
        elapsed_sec=time.monotonic() - started,
        wire_failures=caller.wire_failures,
        replay_ok=outcome.replay_ok,
        error=error,
        llm_subcalls=outcome.sub["calls"],
        subcall_ptok=outcome.sub["ptok"],
        subcall_ctok=outcome.sub["ctok"],
        subcall_cached=outcome.sub.get("cached", 0),
    )
    (trial_dir / "trial.json").write_text(json.dumps(asdict(trial), indent=1))
    if outcome.transcript is not None:
        (trial_dir / "transcript.json").write_text(json.dumps(outcome.transcript, indent=1))
    return trial


def summarize(trials: list[TrialResult]) -> list[dict[str, Any]]:
    """Per (size, arm) aggregates — accuracy first, then the CFO columns."""
    by: dict[tuple[str, str], list[TrialResult]] = {}
    for t in trials:
        by.setdefault((t.size, t.arm), []).append(t)
    rows = []
    for (size, arm), ts in sorted(by.items()):
        n = len(ts)
        rows.append(
            {
                "size": size,
                "arm": arm,
                "n": n,
                "accuracy": sum(t.reward for t in ts) / n,
                "mean_prompt_tokens": sum(t.prompt_tokens for t in ts) / n,
                "mean_completion_tokens": sum(t.completion_tokens for t in ts) / n,
                "mean_cost_usd": sum(t.cost_usd for t in ts) / n,
                "mean_turns": sum(t.turns for t in ts) / n,
                "mean_elapsed_sec": sum(t.elapsed_sec for t in ts) / n,
                "sandbox_errors": sum(t.sandbox_errors for t in ts),
                "wire_failures": sum(t.wire_failures for t in ts),
                "replays_ok": sum(1 for t in ts if t.replay_ok),
                "replays_run": sum(1 for t in ts if t.replay_ok is not None),
                "errors": sum(1 for t in ts if t.error),
                "mean_subcalls": sum(t.llm_subcalls for t in ts) / n,
                "subcall_tokens": sum(t.subcall_ptok + t.subcall_ctok for t in ts),
            }
        )
    return rows
