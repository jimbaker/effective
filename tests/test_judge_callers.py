"""The CLI callers and the token budget: argv, stdin, strict schemas, and exact spend."""

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from effective.channels import Message
from effective.domain import (
    AskLLM,
)
from effective.interpreters.cli import ClaudePrint, CLIFailed, CodexExec
from effective.interpreters.openai import GPT6_LUNA
from effective.spend import TokenBudget, TokenBudgetExhausted


class Prime(BaseModel):
    prime: bool


@dataclass
class FakeRun:
    stdout: str
    argv: list[Sequence[str]] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], cwd: Path, prompt: str) -> str:
        self.argv.append(argv)
        self.prompts.append(prompt)
        return self.stdout


CLAUDE_OUT = {
    "is_error": False,
    "subtype": "success",
    "structured_output": {"prime": True},
    "usage": {"input_tokens": 1159, "output_tokens": 195, "cache_read_input_tokens": 0},
    "total_cost_usd": 0.002134,
}


def _ask(messages: Any = "Is 7 prime?") -> AskLLM[Prime]:
    return AskLLM(messages=messages, response_schema=Prime)


def test_claude_print_returns_the_validated_answer_and_its_usage(tmp_path):
    budget = TokenBudget(tmp_path / "claude.json", 100_000)
    fake = FakeRun(json.dumps(CLAUDE_OUT))
    answer, usage = ClaudePrint("haiku", budget, run=fake)(_ask())
    assert answer == Prime(prime=True)
    assert (usage.prompt_tokens, usage.completion_tokens, usage.cost) == (1159, 195, 0.002134)
    assert budget.spent == 1159 + 195
    [argv] = fake.argv
    assert argv[argv.index("--tools") + 1] == ""


def test_claude_print_counts_cached_input_in_prompt_tokens(tmp_path):
    """Claude leaves the cached share out of `input_tokens`; `prompt_tokens` is all the input."""
    used = {
        "input_tokens": 100,
        "output_tokens": 10,
        "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 50,
    }
    budget = TokenBudget(tmp_path / "claude.json", 100_000)
    fake = FakeRun(json.dumps({**CLAUDE_OUT, "usage": used}))
    _, usage = ClaudePrint("haiku", budget, run=fake)(_ask())
    assert usage.prompt_tokens == 1050
    assert usage.cache_hit_ratio == round(900 / 1050, 4)
    assert budget.spent == 1060


def test_claude_print_refuses_an_error_result(tmp_path):
    fake = FakeRun(json.dumps({"is_error": True, "subtype": "error_max_turns", "result": "x"}))
    with pytest.raises(CLIFailed, match="error_max_turns"):
        ClaudePrint("haiku", TokenBudget(tmp_path / "c.json", 100), run=fake)(_ask())


CODEX_EVENTS = [
    {"type": "thread.started", "thread_id": "t"},
    {"type": "item.completed", "item": {"type": "agent_message", "text": '{"prime": true}'}},
    {
        "type": "turn.completed",
        "usage": {"input_tokens": 13747, "cached_input_tokens": 747, "output_tokens": 15},
    },
]


def test_codex_exec_prices_the_turn_at_list_price(tmp_path):
    budget = TokenBudget(tmp_path / "codex.json", 100_000)
    fake = FakeRun("\n".join(json.dumps(e) for e in CODEX_EVENTS))
    answer, usage = CodexExec("gpt-6-luna", budget, run=fake)(_ask())
    assert answer == Prime(prime=True)
    assert usage.cost == GPT6_LUNA.cost(13000, 747, 15)
    assert budget.spent == 13747 + 15


def test_codex_exec_refuses_a_run_with_no_answer(tmp_path):
    fake = FakeRun(json.dumps(CODEX_EVENTS[0]))
    with pytest.raises(CLIFailed):
        CodexExec("gpt-6-luna", TokenBudget(tmp_path / "x.json", 100_000), run=fake)(_ask())


def test_a_cli_budget_refuses_before_the_cli_runs(tmp_path):
    ledger = tmp_path / "cli.json"
    TokenBudget(ledger, 10).add(10)
    fake = FakeRun("")
    with pytest.raises(TokenBudgetExhausted):
        ClaudePrint("haiku", TokenBudget(ledger, 10), run=fake)(_ask())
    assert fake.argv == []


def test_rendered_messages_split_the_system_text_from_the_prompt(tmp_path):
    fake = FakeRun(json.dumps(CLAUDE_OUT))
    rendered = [Message(role="system", content="Be terse."), Message(role="user", content="7?")]
    ClaudePrint("haiku", TokenBudget(tmp_path / "c.json", 100_000), run=fake)(_ask(rendered))
    [argv] = fake.argv
    assert fake.prompts == ["7?"]
    assert argv[argv.index("--system-prompt") + 1].endswith("Be terse.")


def test_codex_gets_a_strict_schema_and_its_failed_turn_is_reported(tmp_path):
    failed = {"type": "turn.failed", "error": {"message": "invalid_json_schema"}}
    fake = FakeRun("\n".join(json.dumps(e) for e in [CODEX_EVENTS[0], failed]))
    written: list[dict[str, Any]] = []

    def reading(argv: Sequence[str], cwd: Path, prompt: str) -> str:
        written.append(json.loads(Path(argv[argv.index("--output-schema") + 1]).read_text()))
        return fake(argv, cwd, prompt)

    with pytest.raises(CLIFailed, match="invalid_json_schema"):
        CodexExec("gpt-6-luna", TokenBudget(tmp_path / "x.json", 100_000), run=reading)(_ask())
    [schema] = written
    assert (schema["additionalProperties"], schema["required"]) == (False, ["prime"])


def test_concurrent_callers_each_add_exactly_what_they_spent(tmp_path):
    budget = TokenBudget(tmp_path / "spend", 10**9)
    seen: list[int] = []

    def spend() -> None:
        for _ in range(250):
            budget.add(1)
            seen.append(budget.spent)

    workers = [threading.Thread(target=spend) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert budget.spent == 1000
    assert len(seen) == 1000


def test_a_prompt_that_looks_like_a_flag_travels_on_stdin(tmp_path):
    fake = FakeRun(json.dumps(CLAUDE_OUT))
    ClaudePrint("haiku", TokenBudget(tmp_path / "c", 100_000), run=fake)(_ask("--version"))
    [argv] = fake.argv
    assert "--version" not in argv
    assert fake.prompts == ["--version"]


def test_output_that_is_not_json_is_a_cli_failure(tmp_path):
    fake = FakeRun("claude 2.1.280")
    with pytest.raises(CLIFailed, match="not JSON"):
        ClaudePrint("haiku", TokenBudget(tmp_path / "c", 100_000), run=fake)(_ask())


def test_codex_runs_without_its_shell(tmp_path):
    fake = FakeRun("\n".join(json.dumps(e) for e in CODEX_EVENTS))
    CodexExec("gpt-6-luna", TokenBudget(tmp_path / "x", 100_000), run=fake)(_ask())
    [argv] = fake.argv
    disabled = {argv[i + 1] for i, flag in enumerate(argv) if flag == "--disable"}
    assert {"shell_tool", "unified_exec"} <= disabled


@pytest.mark.parametrize(
    ("tools", "forbidden"), [((), True), (("WebSearch",), False)], ids=["none", "web"]
)
def test_claude_print_forbids_tools_only_when_it_gives_none(tmp_path, tools, forbidden):
    fake = FakeRun(json.dumps(CLAUDE_OUT))
    ClaudePrint("haiku", TokenBudget(tmp_path / "claude.json", 100_000), run=fake, tools=tools)(
        _ask()
    )
    [argv] = fake.argv
    assert ("Do not use tools" in argv[argv.index("--system-prompt") + 1]) is forbidden
