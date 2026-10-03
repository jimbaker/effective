"""Judgments over a state: a t-string whose holes are the state and the questions.

| part of the template | becomes |
|---|---|
| a hole holding `Choice`, `Noul` or `Score` | a question named by its expression |
| a question hole `{q:each path}` | `q` asked once per element of state list `path`, as `q[i]` |
| literal text before a question hole | that question's instructions |
| any other hole | a state field named by its dotted expression, nested |
| literal text before a state hole | context, each state hole written as its backticked path |

A value never crosses into a question's instructions, which is what makes mixing safe. A judgment
answers the same state with the same decision, so there is no `Repair`: policy reads the recorded
answer, and a low answer escalates to another interpreter. `effective.api.judge` and
`effective.api.select` yield the op this processor builds.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from string.templatelib import Interpolation, Template
from typing import Any, Literal, assert_never

from pydantic_core import to_jsonable_python

from effective.domain import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    WireChoice,
    WireNoul,
    WireQuestion,
    WireScore,
)

NO_MATCH = "none of these"


@dataclass(frozen=True)
class Choice:
    """One of a set; a label maps to its description, or to `None` when the label says it all."""

    criteria: Mapping[str, str | None]


@dataclass(frozen=True)
class Noul:
    """The probability that a condition holds."""

    true: str | None = None
    false: str | None = None


@dataclass(frozen=True)
class Score:
    """A position on ordered levels, lowest first; each level describes a concrete situation."""

    levels: Sequence[str]


type Question = Choice | Noul | Score


class TemplateError(ValueError):
    pass


class Unasked(ValueError):
    """An answer that does not fit what was asked: a missing or extra question, a shape of
    another question type, or a choice outside the criteria."""


@dataclass(frozen=True)
class Battery:
    """What one call asks: the state and every question over it."""

    state: dict[str, Any]
    questions: dict[str, WireQuestion]


def _wire(question: Question, instructions: str) -> WireQuestion:
    match question:
        case Choice(criteria=criteria):
            return WireChoice(instructions=instructions, criteria=dict(criteria))
        case Noul(true=true, false=false):
            sides: dict[Literal["true", "false"], str | None] = {"true": true, "false": false}
            return WireNoul(
                instructions=instructions,
                criteria={k: v for k, v in sides.items() if v is not None},
            )
        case Score(levels=levels):
            return WireScore(instructions=instructions, criteria=list(levels))
        case unreachable:
            assert_never(unreachable)


def _place(state: dict[str, Any], path: str, value: Any) -> None:
    *parents, leaf = path.split(".")
    if not all(part.isidentifier() for part in (*parents, leaf)):
        raise TemplateError(f"a state hole must be a dotted name: {path!r}")
    node = state
    for part in parents:
        node = node.setdefault(part, {})
    if leaf in node:
        raise TemplateError(f"state field {path!r} appears twice")
    node[leaf] = to_jsonable_python(value)  # the wire carries JSON, whatever the hole held


def _lookup(state: dict[str, Any], path: str) -> Any:
    node: Any = state
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise TemplateError(f"`each {path}` names no state field placed before it")
        node = node[part]
    return node


def _questions(
    name: str, spec: str, question: Question, text: str, state: dict[str, Any]
) -> list[tuple[str, WireQuestion]]:
    match spec.split():
        case []:
            return [(name, _wire(question, text))]
        case ["each", path]:
            items = _lookup(state, path)
            if not isinstance(items, list) or not items:
                raise TemplateError(f"`each {path}` needs a non-empty list")
            return [
                (_element(name, i), _wire(question, _element_instructions(text, path, i)))
                for i in range(len(items))
            ]
        case _:
            raise TemplateError(f"question spec {spec!r} is not implemented")


def _element(name: str, i: int) -> str:
    """Render the question asking element `i`; `gathered` parses it back."""
    return f"{name}[{i}]"


def _element_instructions(text: str, path: str, i: int) -> str:
    """Render one element's instructions: the shared text, then which element it judges."""
    return f"{text} Judge `{path}[{i}]`."


def _cited(path: str) -> str:
    """Render a state hole in the context prose as its path, never its value."""
    return f"`{path}`"


def _set_context(state: dict[str, Any], text: str) -> None:
    if not text:
        return
    if "context" in state:
        raise TemplateError("a state field named 'context' would shadow the prose")
    state["context"] = text


def battery(template: Template) -> Battery:
    """The processor: split `template` into state and questions by the table above."""
    state: dict[str, Any] = {}
    questions: dict[str, WireQuestion] = {}
    context: list[str] = []
    pending = ""
    for item in template:
        match item:
            case str() as text:
                pending += text
            case Interpolation(
                value=Choice() | Noul() | Score() as question, expression=name, format_spec=spec
            ):
                for asked, wire in _questions(name, spec, question, pending.strip(), state):
                    if asked in questions:
                        raise TemplateError(f"question {asked!r} appears twice")
                    questions[asked] = wire
                pending = ""
            case Interpolation(value=value, expression=path, format_spec=spec):
                if spec not in ("", "data"):
                    raise TemplateError(f"format spec {spec!r} is not implemented")
                _place(state, path, value)
                context.extend((pending, _cited(path)))
                pending = ""
            case unreachable:
                assert_never(unreachable)
    context.append(pending)
    _set_context(state, "".join(context).strip())
    if not questions:
        raise TemplateError("a judgment asks at least one question")
    return Battery(state, questions)


_ELEMENT = re.compile(r"\[(\d+)\]$")


def field_names(questions: Mapping[str, WireQuestion]) -> set[str]:
    """The output fields `questions` fill: `q[i]` fills the list field `q`."""
    return {_ELEMENT.sub("", asked) for asked in questions}


def gathered(
    answers: Mapping[str, ChoiceAnswer | NoulAnswer | ScoreAnswer],
) -> dict[str, Any]:
    """Collect `name[i]` answers into a list under `name`, in element order."""
    out: dict[str, Any] = {}
    elements: dict[str, dict[int, Any]] = {}
    for asked, answer in answers.items():
        if match := _ELEMENT.search(asked):
            elements.setdefault(asked[: match.start()], {})[int(match[1])] = answer
        else:
            out[asked] = answer
    for name, by_index in elements.items():
        out[name] = [by_index[i] for i in sorted(by_index)]
    return out


def checked(
    questions: Mapping[str, WireQuestion],
    answers: Mapping[str, ChoiceAnswer | NoulAnswer | ScoreAnswer],
) -> None:
    """Refuse answers that do not fit `questions`, one arm per question type."""
    if set(answers) != set(questions):
        raise Unasked(f"answered {sorted(answers)}, asked {sorted(questions)}")
    for name, question in questions.items():
        match question, answers[name]:
            case WireChoice(criteria=criteria), ChoiceAnswer(choice=choice):
                if choice not in criteria:
                    raise Unasked(f"{name}: {choice!r} is not one of {sorted(criteria)}")
            case (WireNoul(), NoulAnswer()) | (WireScore(), ScoreAnswer()):
                pass
            case ((WireChoice() | WireNoul() | WireScore()), answer):
                raise Unasked(f"{name}: a {question.type} question answered by {answer!r}")
            case unreachable:
                assert_never(unreachable)
