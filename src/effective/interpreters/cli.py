"""Serve `AskLLM` through a subscription CLI, `claude -p` or `codex exec`, for prototyping.

Both answer the same op as the API callers, so moving a workflow onto an API is a change of
interpreter. Each call runs from an empty directory with no session, takes its prompt on stdin,
and returns the answer the CLI validated against the op's schema.

| CLI | tools | input tokens, a one-line question, 2026-09-22 |
|---|---|---|
| `claude -p --safe-mode` | `tools`, none by default | 1,159 (Haiku 4.5) |
| `codex exec -s read-only` | none: `CODEX_OFF` disabled | 13,747 (GPT-6 Luna), shell on |

Recompute a figure with one call through the interpreter and its `Usage.prompt_tokens`.
"""

import json
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_never

from pydantic import TypeAdapter

from effective.channels import Message
from effective.cost import Usage
from effective.domain import AskLLM
from effective.interpreters.openai import GPT6_LUNA, GPT6_SOL, Price
from effective.spend import TokenBudget

type Runner = Callable[[Sequence[str], Path, str], str]
"""Run `argv` in a directory with the prompt on stdin and return its stdout; a test passes a
fake."""

SYSTEM = "Answer the request in the structure its schema gives."
NO_TOOLS = "Do not use tools."
"""Said only to a call given no tools: a model told this obeys it over the tools it is handed."""


class CLIFailed(RuntimeError):
    """The CLI ran and reported an error in place of an answer."""


def run(argv: Sequence[str], cwd: Path, prompt: str) -> str:
    """Both CLIs report a failed call on stdout, so a non-zero exit that printed is parsed."""
    done = subprocess.run(argv, cwd=cwd, input=prompt, capture_output=True, text=True, timeout=600)
    if done.returncode and not done.stdout.strip():
        raise CLIFailed(f"{argv[0]} exited {done.returncode}: {done.stderr[-500:]}")
    return done.stdout


def _split(messages: str | list[Message], tools: Sequence[str]) -> tuple[str, str]:
    """The system text and the prompt: a channel's rendered messages join by role."""
    head = SYSTEM if tools else " ".join((SYSTEM, NO_TOOLS))
    match messages:
        case str():
            return head, messages
        case list():
            system = [m.content for m in messages if m.role == "system"]
            rest = [m.content for m in messages if m.role != "system"]
            return "\n\n".join([head, *system]), "\n\n".join(rest)
        case unreachable:
            assert_never(unreachable)


def _parsed(cli: str, stdout: str) -> Any:
    try:
        return json.loads(stdout)
    except ValueError as err:
        raise CLIFailed(f"{cli}: output is not JSON: {stdout[:200]!r}") from err


def _schema(op: AskLLM[Any]) -> dict[str, Any]:
    return TypeAdapter(op.response_schema).json_schema()


def _strict(schema: Any) -> Any:
    """OpenAI's strict form: every object closed, every property required."""
    match schema:
        case {"properties": dict(properties), **rest}:
            closed = {**rest, "additionalProperties": False, "required": list(properties)}
            return {**closed, "properties": {k: _strict(v) for k, v in properties.items()}}
        case dict():
            return {k: _strict(v) for k, v in schema.items()}
        case list():
            return [_strict(v) for v in schema]
        case _:
            return schema


@dataclass(frozen=True)
class ClaudePrint:
    """`claude -p` on the subscription; `cost` is the CLI's figure at list price."""

    model: str
    budget: TokenBudget
    run: Runner = run
    tools: tuple[str, ...] = ()
    """Built-in tools the call may use, such as `WebSearch` and `WebFetch`."""

    def __call__(self, op: AskLLM[Any]) -> tuple[Any, Usage]:
        system, prompt = _split(op.messages, self.tools)
        self.budget.check()
        argv = [
            "claude", "-p",
            "--model", self.model,
            "--output-format", "json",
            "--json-schema", json.dumps(_schema(op)),
            "--system-prompt", system,
            "--tools", ",".join(self.tools),
            "--allowedTools", ",".join(self.tools),
            "--safe-mode",
            "--no-session-persistence",
        ]  # fmt: skip
        with tempfile.TemporaryDirectory() as empty:
            out = _parsed("claude -p", self.run(argv, Path(empty), prompt))
        if out.get("is_error") or "structured_output" not in out:
            raise CLIFailed(f"claude -p: {out.get('subtype')}: {str(out.get('result'))[:200]}")
        used = out["usage"]
        read = used.get("cache_read_input_tokens", 0)
        written = used.get("cache_creation_input_tokens", 0)
        # Claude reports `input_tokens` with the cached share left out; `prompt_tokens` counts all
        # of the input, so the cache reads and writes are added back.
        usage = Usage(
            prompt_tokens=used["input_tokens"] + read + written,
            completion_tokens=used["output_tokens"],
            cache_read_input_tokens=read,
            cache_creation_input_tokens=written,
            cost=out.get("total_cost_usd", 0.0),
        )
        self.budget.add(usage.prompt_tokens + usage.completion_tokens)
        return TypeAdapter(op.response_schema).validate_python(out["structured_output"]), usage


CODEX_PRICES: dict[str, Price] = {"gpt-6-luna": GPT6_LUNA, "gpt-6-sol": GPT6_SOL}

CODEX_OFF = (
    "shell_tool", "unified_exec", "apps", "plugins", "browser_use", "computer_use",
    "in_app_browser", "multi_agent", "image_generation", "view_image", "hooks", "skill_search",
    "tool_suggest", "sleep_tool", "goals", "skill_mcp_dependency_install", "code_mode_host",
)  # fmt: skip
"""Codex features a call runs without. With the shell on, a read-only sandbox still reads any
file the user can; with these off, the same request reports it could not read the file."""


@dataclass(frozen=True)
class CodexExec:
    """`codex exec` on the subscription, with no tools; `cost` is the same tokens at list price."""

    model: str
    budget: TokenBudget
    run: Runner = run

    def __call__(self, op: AskLLM[Any]) -> tuple[Any, Usage]:
        price = CODEX_PRICES[self.model]
        system, prompt = _split(op.messages, ())
        self.budget.check()
        with tempfile.TemporaryDirectory() as empty:
            schema = Path(empty) / "schema.json"
            schema.write_text(json.dumps(_strict(_schema(op))))
            argv = [
                "codex", "exec",
                "-m", self.model,
                "-s", "read-only",
                "--ephemeral",
                "--skip-git-repo-check",
                "--output-schema", str(schema),
                "--json",
                *(flag for feature in CODEX_OFF for flag in ("--disable", feature)),
                "-",
            ]  # fmt: skip
            stdout = self.run(argv, Path(empty), "\n\n".join([system, prompt]))
            events = [_parsed("codex exec", line) for line in stdout.splitlines()]
        answer, used = _codex_turn(events)
        cached = used.get("cached_input_tokens", 0)
        self.budget.add(used["input_tokens"] + used["output_tokens"])
        usage = Usage(
            prompt_tokens=used["input_tokens"],
            completion_tokens=used["output_tokens"],
            cache_read_input_tokens=cached,
            cost=price.cost(used["input_tokens"] - cached, cached, used["output_tokens"]),
        )
        return TypeAdapter(op.response_schema).validate_json(answer), usage


def _codex_turn(events: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
    """The last agent message and the turn's usage, from one `codex exec --json` run."""
    messages = [
        e["item"]["text"]
        for e in events
        if e.get("type") == "item.completed" and e["item"].get("type") == "agent_message"
    ]
    turns = [e["usage"] for e in events if e.get("type") == "turn.completed"]
    failures = [e["error"]["message"] for e in events if e.get("type") == "turn.failed"]
    match messages, turns, failures:
        case [*_, answer], [used], []:
            return answer, used
        case _, _, [*_, failure]:
            raise CLIFailed(f"codex exec: {failure[:500]}")
        case _:
            raise CLIFailed(f"codex exec: no single completed turn with an answer in {events!r}")
