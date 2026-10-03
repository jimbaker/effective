"""A strict function-tool catalog, derived from the argument model of each `Tool`.

The model reads a tool's docstring and field descriptions, and `typed_act` validates its call
against the same model, so what a tool is described as taking and what it accepts come from one
declaration. Strict mode takes a narrower schema than pydantic emits: every property is required
(an optional one is typed `T | null`), no object admits extra properties, and there are no
`default`, `title` or `$ref` keywords.
"""

import re
from collections.abc import Mapping
from typing import Any

from pydantic import TypeAdapter

from effective.react import Tool


class UndescribedTool(ValueError):
    """A tool the catalog will not describe: an argument model with no docstring, a field with no
    description, a name the API refuses, or a shape strict mode cannot take."""


NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
"""What a function name may be: letters, digits, underscores and dashes, up to 64 characters."""


def strict_function_tools(tools: Mapping[str, Tool[Any, Any]]) -> list[dict[str, Any]]:
    """One strict function entry per tool, in the mapping's order."""
    return [_function(tool) for tool in tools.values()]


def _function(tool: Tool[Any, Any]) -> dict[str, Any]:
    if not NAME.match(tool.name):
        raise UndescribedTool(f"tool {tool.name!r}: a function name is up to 64 of [A-Za-z0-9_-]")
    schema = TypeAdapter(tool.args).json_schema()
    definitions = schema.pop("$defs", {})
    description = schema.pop("description", None)
    if not description:
        raise UndescribedTool(f"tool {tool.name!r}: its argument model has no docstring")
    return {
        "type": "function",
        "name": tool.name,
        "description": description,
        "strict": True,
        "parameters": _strict(schema, definitions, tool.name, ()),
    }


def _strict(
    node: Any, definitions: Mapping[str, Any], tool: str, open_refs: tuple[str, ...]
) -> Any:
    match node:
        case {"$ref": ref}:
            name = ref.rpartition("/")[2]
            if name in open_refs:
                # A model that contains itself has no finite inlining, and this is the one shape
                # that recursed until the interpreter stopped rather than refusing.
                raise UndescribedTool(
                    f"tool {tool!r}: {name} contains itself, so it cannot inline"
                )
            return _strict(definitions[name], definitions, tool, (*open_refs, name))
        case {"type": "object", "additionalProperties": dict()}:
            # An open mapping (`dict[str, int]`): strict mode requires `additionalProperties:
            # false` on every object, so the API refuses this schema outright.
            raise UndescribedTool(
                f"tool {tool!r}: an open mapping has no strict schema; declare the fields"
            )
        case {"type": "object"} if "properties" not in node:
            # `dict[str, Any]`: no declared fields at all, which strict mode has no schema for.
            raise UndescribedTool(
                f"tool {tool!r}: an object with no declared fields has no strict schema"
            )
        case {"type": "object", "properties": properties}:
            for name, field in properties.items():
                if "description" not in field:
                    raise UndescribedTool(f"tool {tool!r}: field {name!r} has no description")
            return {
                **({"description": node["description"]} if "description" in node else {}),
                "type": "object",
                "properties": {
                    name: _strict(field, definitions, tool, open_refs)
                    for name, field in properties.items()
                },
                "required": list(properties),
                "additionalProperties": False,
            }
        case dict():
            return {
                key: _strict(value, definitions, tool, open_refs)
                for key, value in node.items()
                if key not in ("default", "title")
            }
        case list():
            return [_strict(item, definitions, tool, open_refs) for item in node]
        case _:
            return node
