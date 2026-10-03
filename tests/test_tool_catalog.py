"""The derived tool catalog: strict in the shape a native function-calling endpoint requires, and
refusing a tool it cannot describe."""

from typing import Any

import pytest
from pydantic import BaseModel, Field

from effective.interpreters.tool_catalog import UndescribedTool, strict_function_tools
from effective.react import Tool
from examples.coder.tools import TOOLS


class Nested(BaseModel):
    """A nested object."""

    inner: str = Field(description="inner")


class Args(BaseModel):
    """Do the thing."""

    path: str = Field(description="where")
    many: list[Nested] = Field(description="several")
    count: int | None = Field(default=None, description="how many, or null")


DESCRIBED = Tool("thing", Args, str, observe=str)


def objects(node: Any):
    """Every object schema in a catalog entry, the top level included."""
    match node:
        case {"type": "object", "properties": properties}:
            yield node
            for field in properties.values():
                yield from objects(field)
        case dict():
            for value in node.values():
                yield from objects(value)
        case list():
            for item in node:
                yield from objects(item)


CATALOG = strict_function_tools({**TOOLS, "thing": DESCRIBED})


@pytest.mark.parametrize("entry", CATALOG, ids=[entry["name"] for entry in CATALOG])
def test_every_object_requires_every_property_and_takes_no_others(entry):
    schemas = list(objects(entry["parameters"]))
    assert schemas, entry["name"]
    for schema in schemas:
        assert schema["required"] == list(schema["properties"])
        assert schema["additionalProperties"] is False


@pytest.mark.parametrize("entry", CATALOG, ids=[entry["name"] for entry in CATALOG])
def test_an_entry_carries_no_keyword_strict_mode_rejects(entry):
    rendered = repr(entry)
    for rejected in ("$ref", "$defs", "'default'", "'title'"):
        assert rejected not in rendered, rejected


@pytest.mark.parametrize("entry", CATALOG, ids=[entry["name"] for entry in CATALOG])
def test_an_entry_describes_itself_and_every_field(entry):
    assert entry["type"] == "function"
    assert entry["strict"] is True
    assert entry["description"]
    for schema in objects(entry["parameters"]):
        for field in schema["properties"].values():
            assert field["description"]


def test_an_optional_field_is_required_and_nullable():
    [entry] = strict_function_tools({"thing": DESCRIBED})
    assert entry["parameters"]["properties"]["count"]["anyOf"] == [
        {"type": "integer"},
        {"type": "null"},
    ]
    assert "count" in entry["parameters"]["required"]


def test_the_catalog_names_what_the_loop_dispatches():
    assert [entry["name"] for entry in strict_function_tools(TOOLS)] == list(TOOLS)


class Undocumented(BaseModel):
    field: str = Field(description="d")


class Unlabelled(BaseModel):
    """Has a docstring."""

    field: str


def test_a_tool_with_no_docstring_is_refused():
    with pytest.raises(UndescribedTool, match="no docstring"):
        strict_function_tools({"x": Tool("x", Undocumented, str, observe=str)})


def test_a_field_with_no_description_is_refused():
    with pytest.raises(UndescribedTool, match="field 'field' has no description"):
        strict_function_tools({"x": Tool("x", Unlabelled, str, observe=str)})


class OpenMap(BaseModel):
    """Take an open map."""

    values: dict[str, int] = Field(description="named values")


class AnyMap(BaseModel):
    """Take anything."""

    values: dict[str, Any] = Field(description="anything")


class Node(BaseModel):
    """One node."""

    name: str = Field(description="its name")
    child: Node | None = Field(description="the next node, or null")


class Recursive(BaseModel):
    """Take a node."""

    node: Node = Field(description="the first node")


REJECTED = [
    pytest.param(OpenMap, "an open mapping", id="open-mapping"),
    pytest.param(AnyMap, "no declared fields", id="undeclared-object"),
    pytest.param(Recursive, "contains itself", id="recursive"),
]


@pytest.mark.parametrize(("args", "message"), REJECTED)
def test_a_shape_strict_mode_cannot_take_is_refused(args, message):
    """Each was measured by a review as emitted and accepted here, and rejected by the API."""
    with pytest.raises(UndescribedTool, match=message):
        strict_function_tools({"x": Tool("x", args, str, observe=str)})


@pytest.mark.parametrize("name", ["bad.name", "x" * 65, "", "spaced name"])
def test_a_name_the_api_refuses_is_refused_here(name):
    with pytest.raises(UndescribedTool, match="function name"):
        strict_function_tools({name: Tool(name, Args, str, observe=str)})
