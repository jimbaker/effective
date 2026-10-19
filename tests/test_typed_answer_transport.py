"""A typed run's answer across each transport: the provider is asked for the run's schema, and
the answer comes back as raw JSON for the run to resolve.

The Responses caller runs the real SDK against an HTTP transport that answers in the provider's
wire format, so what the SDK parses is what a provider would send. A model's validator runs only
when the run resolves the answer, never inside the SDK: one that rejects leaves the turn readable
and one that normalizes leaves the recorded answer as the provider wrote it."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, field_validator

from effective.channels import Field, Gated, Prompt, render
from effective.contexts import RecordingCtx
from effective.cost import MeteredInterpreter
from effective.domain import AskLLM, CallTool
from effective.handlers.durable import DurableHandler
from effective.interpreters.cli import _schema
from effective.interpreters.openai import (
    GPT56_LUNA,
    OpenAITurnCaller,
    ResponsesTurnCaller,
    _WireTool,
    _WireTurn,
)
from effective.interpreters.tools import make_tool_runner
from effective.react import AssistantTurn, Tool, Trajectory, load_ending, run_agent

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")


class Shouted(BaseModel):
    answer: str

    @field_validator("answer")
    @classmethod
    def louder(cls, value: str) -> str:
        return value.upper()


class Refusing(BaseModel):
    answer: str

    @field_validator("answer")
    @classmethod
    def never(cls, value: str) -> str:
        raise ValueError("this validator refuses every answer")


def shouted() -> Prompt[Shouted]:
    answer = Field(str)
    return render(t"Answer: {answer}", output=Shouted)


def _reply(text: str) -> dict[str, Any]:
    """A Responses reply carrying one message whose text is `text`."""
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-5.6-luna",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


class _Provider:
    """An HTTP transport answering each request with the next turn, and keeping the bodies."""

    def __init__(self, *turns: dict[str, Any]) -> None:
        self.turns = list(turns)
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: Any) -> Any:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_reply(json.dumps(self.turns.pop(0))))

    def caller(self) -> ResponsesTurnCaller:
        client = openai.OpenAI(
            api_key="test",
            base_url="http://provider.test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(self)),
        )
        return ResponsesTurnCaller(client, "system", "gpt-5.6-luna", GPT56_LUNA)


def _asked(schema: dict[str, Any] | None) -> AskLLM[AssistantTurn]:
    messages = [{"role": "user", "content": "answer"}]
    return AskLLM(messages=messages, response_schema=AssistantTurn, answer_schema=schema)


@pytest.mark.parametrize("model", [Shouted, Refusing], ids=["normalizing", "rejecting"])
def test_the_sdk_hands_back_the_raw_answer_whatever_the_models_validator_does(model):
    provider = _Provider({"thought": "t", "tool": None, "answer": {"answer": "quiet"}})
    turn, usage = provider.caller()(_asked(model.model_json_schema()))
    assert isinstance(turn, AssistantTurn)
    assert turn.answer == {"answer": "quiet"}
    assert (usage.prompt_tokens, usage.completion_tokens) == (10, 5)
    assert usage.cost > 0


def test_the_provider_is_asked_for_the_answers_schema_and_told_nothing_of_encoding():
    provider = _Provider({"thought": "t", "tool": None, "answer": {"answer": "quiet"}})
    provider.caller()(_asked(Shouted.model_json_schema()))
    [body] = provider.bodies
    shape = body["text"]["format"]
    assert (shape["type"], shape["strict"]) == ("json_schema", False)
    answer = shape["schema"]["properties"]["answer"]
    assert answer == {"anyOf": [Shouted.model_json_schema(), {"type": "null"}]}
    assert all("JSON object" not in str(message["content"]) for message in body["input"])


def test_an_untyped_turn_asks_for_the_text_envelope():
    provider = _Provider({"thought": "t", "tool": None, "answer": "plain"})
    turn, _ = provider.caller()(_asked(None))
    assert isinstance(turn, AssistantTurn)
    assert turn.answer == "plain"
    [body] = provider.bodies
    answer = body["text"]["format"]["schema"]["properties"]["answer"]
    assert answer == _WireTurn.model_json_schema()["properties"]["answer"]


def test_the_run_records_the_raw_answer_and_resolves_it_through_the_model():
    provider = _Provider({"thought": "t", "tool": None, "answer": {"answer": "quiet"}})
    ctx = RecordingCtx()
    interpreter = MeteredInterpreter(llm=provider.caller(), tools=make_tool_runner({}))
    stored = DurableHandler(ctx, interpreter).run(lambda: run_agent("answer", answer=shouted))
    ending = load_ending(stored, answer=Shouted)
    assert isinstance(ending, Trajectory)
    assert ending.answer == Shouted(answer="QUIET")
    [recorded] = [value for key, value in ctx.log.items() if key.stored().endswith("react:turn")]
    assert json.dumps(recorded).count('"quiet"') == 1
    assert interpreter.meter.prompt_tokens == 10


def test_a_string_transport_is_told_to_write_the_answers_json_in_its_answer():
    sent: list[list[dict[str, Any]]] = []

    def parse(**kwargs: Any) -> Any:
        sent.append(kwargs["messages"])
        parsed = _WireTurn(thought="t", tool=None, answer='{"answer": "quiet"}')
        usage = SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=2,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        choice = SimpleNamespace(message=SimpleNamespace(parsed=parsed))
        return SimpleNamespace(choices=[choice], usage=usage)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    turn, _ = OpenAITurnCaller(client, "system")(_asked(Shouted.model_json_schema()))
    assert isinstance(turn, AssistantTurn)
    assert turn.answer == '{"answer": "quiet"}'
    instruction = sent[0][-1]["content"]
    assert "JSON object, inside the answer string" in instruction
    assert json.dumps(Shouted.model_json_schema(), sort_keys=True) in instruction


@pytest.mark.parametrize("schema", [None, Shouted.model_json_schema()], ids=["text", "typed"])
def test_each_cli_is_handed_the_answers_shape(schema):
    answer = _schema(_asked(schema))["properties"]["answer"]
    expected = {"type": "string"} if schema is None else Shouted.model_json_schema()
    assert answer == {"anyOf": [expected, {"type": "null"}]}


class Cited(BaseModel):
    answer: str
    citation: int


def cited(passages: list[str]) -> Prompt[Cited]:
    answer = Field(str)
    citation = Gated(int, lambda i: 0 <= i < len(passages), "cite a passage by its number")
    return render(t"Answer: {answer}\nCitation: {citation}", output=Cited)


class Query(BaseModel):
    query: str


SEARCH = Tool("search", Query, list[str], observe=lambda passages: "\n".join(passages))
SEARCHED: list[str] = []


def _search(op: CallTool[Any]) -> list[str]:
    SEARCHED.append(op.args["query"])
    return ["first", "second"]


def _rag(llm: Any) -> Any:
    SEARCHED.clear()
    interpreter = MeteredInterpreter(llm=llm, tools=_search)
    program = lambda: run_agent("cite", answer=cited, tools=[SEARCH])  # noqa: E731
    return load_ending(DurableHandler(RecordingCtx(), interpreter).run(program), answer=Cited)


def _called(text: dict[str, Any]) -> dict[str, Any]:
    call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "search",
        "arguments": json.dumps({"query": "q"}),
        "status": "completed",
    }
    return {**text, "output": [call]}


def test_a_miscited_answer_costs_one_turn_on_the_native_transport():
    """Prediction 1: a model-chosen search, a bad citation and a good one make three model
    requests and one search."""
    replies = [
        _called(_reply("")),
        _reply(json.dumps({"thought": "a", "answer": {"answer": "a", "citation": 7}})),
        _reply(json.dumps({"thought": "a", "answer": {"answer": "a", "citation": 1}})),
    ]
    requests: list[Any] = []

    def respond(request: Any) -> Any:
        requests.append(request)
        return httpx.Response(200, json=replies[len(requests) - 1])

    client = openai.OpenAI(
        api_key="test",
        base_url="http://provider.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    ending = _rag(ResponsesTurnCaller(client, "system", "gpt-5.6-luna", GPT56_LUNA))
    assert isinstance(ending, Trajectory)
    assert ending.answer == Cited(answer="a", citation=1)
    assert (len(requests), SEARCHED) == (3, ["q"])


def test_a_miscited_answer_costs_one_turn_on_the_string_fallback():
    """Prediction 1 on a transport whose answer is a string holding the answer's JSON."""
    turns = [
        _WireTurn(
            thought="s",
            tool=_WireTool(name="search", arguments_json='{"query": "q"}'),
            answer=None,
        ),
        _WireTurn(thought="a", tool=None, answer='{"answer": "a", "citation": 7}'),
        _WireTurn(thought="a", tool=None, answer='{"answer": "a", "citation": 1}'),
    ]
    asked: list[Any] = []

    def parse(**kwargs: Any) -> Any:
        asked.append(kwargs)
        usage = SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=2,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        choice = SimpleNamespace(message=SimpleNamespace(parsed=turns[len(asked) - 1]))
        return SimpleNamespace(choices=[choice], usage=usage)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    ending = _rag(OpenAITurnCaller(client, "system"))
    assert isinstance(ending, Trajectory)
    assert ending.answer == Cited(answer="a", citation=1)
    assert (len(asked), SEARCHED) == (3, ["q"])
