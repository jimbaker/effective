"""`run_agent(tools=...)`: a run dispatching through declared tools.

Each tool is declared once under its name; a second declaration under that name is refused when
the run is built. `tools` and `act` each dispatch every action, so a run takes one."""

import pytest
from _scripted import Sequenced
from pydantic import BaseModel

from effective.handlers.recording import RecordingHandler
from effective.react import (
    AssistantTurn,
    DuplicateToolName,
    Tool,
    ToolRequest,
    Trajectory,
    declared,
    default_act,
    run_agent,
)


class Query(BaseModel):
    query: str


SEARCH = Tool("search", Query, list[str], observe=lambda passages: "\n".join(passages))
LOOKUP = Tool("search", Query, list[str], observe=lambda passages: passages[0])


def searched(*tools: Tool) -> tuple[Trajectory, RecordingHandler]:
    table = Sequenced(
        {
            "react:turn": [
                AssistantTurn(
                    thought="look", tool=ToolRequest(name="search", args={"query": "q"})
                ),
                AssistantTurn(thought="done", answer="found"),
            ],
            "tool:search": [["first", "second"]],
        }
    )
    handler = RecordingHandler(table)
    ending = handler.run(lambda: run_agent("find it", tools=tools))
    assert isinstance(ending, Trajectory)
    assert table.unasked == {}
    return ending, handler


def test_a_declared_tool_answers_its_result_type_and_the_model_reads_its_observation():
    ending, _ = searched(SEARCH)
    assert ending.steps[0].observation == "first\nsecond"


def test_two_tools_declared_under_one_name_are_refused_when_the_run_is_built():
    with pytest.raises(DuplicateToolName) as refused:
        searched(SEARCH, LOOKUP)
    assert refused.value.name == "search"


def test_a_table_holds_each_tool_under_its_name():
    assert declared([SEARCH]) == {"search": SEARCH}


def test_a_run_takes_tools_or_an_act():
    handler = RecordingHandler({})
    with pytest.raises(ValueError, match="pass one"):
        handler.run(lambda: run_agent("find it", tools=[SEARCH], act=default_act))
