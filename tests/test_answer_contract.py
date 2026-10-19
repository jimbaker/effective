"""A prompt carries its form gates, and an answer contract declares the model it resolves to.

A carried gate runs before a caller's, and a caller cannot remove one. A contract is refused when
its annotation does not name a model the channels can fill by field name, or when the prompt it
returns declares another output."""

from typing import Annotated

import pytest
from pydantic import AliasPath, BaseModel, ConfigDict, Field, RootModel
from pydantic.alias_generators import to_camel

from effective.channels import (
    ContractRefused,
    FormGate,
    Gated,
    Prompt,
    Repair,
    answer_model,
    answering,
    render,
)
from effective.channels import Field as Channel

type Passages = list[str]


class GenerateAnswer(BaseModel):
    answer: str
    citation: int


PASSAGES = ["first", "second"]
CITED = FormGate(lambda values: values["citation"] < len(PASSAGES), "cite a passage")
ANSWERED = FormGate(lambda values: values["answer"] != "", "give an answer")


def cited_answer(passages: Passages) -> Prompt[GenerateAnswer]:
    answer = Channel(str)
    citation = Gated(int, lambda i: i >= 0, "a citation is a passage number")
    return render(t"{answer}{citation}", output=GenerateAnswer, form_gates=[CITED])


def test_a_carried_gate_runs_on_every_resolve():
    prompt = cited_answer(PASSAGES)
    assert prompt.resolve({"answer": "x", "citation": 5}) == Repair("cite a passage")
    assert prompt.resolve({"answer": "x", "citation": 1}) == GenerateAnswer(answer="x", citation=1)


def test_an_empty_call_time_tuple_keeps_the_carried_gate():
    prompt = cited_answer(PASSAGES)
    assert prompt.resolve({"answer": "x", "citation": 5}, form_gates=()) == Repair(
        "cite a passage"
    )


def test_the_carried_gates_run_before_the_callers():
    prompt = cited_answer(PASSAGES)
    both_fail = {"answer": "", "citation": 5}
    assert prompt.resolve(both_fail, form_gates=[ANSWERED]) == Repair("cite a passage")
    assert prompt.resolve({"answer": "", "citation": 1}, form_gates=[ANSWERED]) == Repair(
        "give an answer"
    )


def test_a_gate_list_changed_after_render_leaves_the_prompts_gates():
    gates = [CITED]
    answer, citation = Channel(str), Channel(int)
    prompt = render(t"{answer}{citation}", output=GenerateAnswer, form_gates=gates)
    gates.clear()
    assert prompt.form_gates == (CITED,)


def test_a_contract_declares_its_model():
    assert answer_model(cited_answer) is GenerateAnswer


def test_metadata_on_the_return_annotation_leaves_its_model():
    assert answer_model(returns_annotated) is GenerateAnswer


def test_the_returned_prompt_must_declare_the_contracts_model():
    prompt = cited_answer(PASSAGES)
    assert answering(cited_answer, prompt, GenerateAnswer) is prompt

    class Other(BaseModel):
        answer: str
        citation: int

    with pytest.raises(ContractRefused) as refused:
        answering(cited_answer, prompt, Other)
    assert (refused.value.fault, refused.value.declared) == ("returned", repr(GenerateAnswer))


class Rooted(RootModel[list[str]]):
    pass


class Aliased(BaseModel):
    answer: str = Field(alias="Answer")


class ValidationAliased(BaseModel):
    answer: str = Field(validation_alias=AliasPath("reply", 0))


class Generated(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)
    cited_by: int


def unannotated(passages):
    return cited_answer(passages)


def returns_a_str_prompt(passages: Passages) -> Prompt[str]:
    raise AssertionError("never called")


def returns_a_model(passages: Passages) -> GenerateAnswer:
    raise AssertionError("never called")


def returns_unresolvable(passages: Passages) -> Prompt[Missing]:  # noqa: F821  # ty: ignore[unresolved-reference]  -- the name the contract check finds missing
    raise AssertionError("never called")


def returns_rooted(passages: Passages) -> Prompt[Rooted]:
    raise AssertionError("never called")


def returns_aliased(passages: Passages) -> Prompt[Aliased]:
    raise AssertionError("never called")


def returns_validation_aliased(passages: Passages) -> Prompt[ValidationAliased]:
    raise AssertionError("never called")


def returns_generated(passages: Passages) -> Prompt[Generated]:
    raise AssertionError("never called")


def returns_annotated(passages: Passages) -> Annotated[Prompt[GenerateAnswer], "note"]:
    raise AssertionError("never called")


@pytest.mark.parametrize(
    ("contract", "fault", "declared"),
    [
        (unannotated, "not-a-prompt", "None"),
        (returns_a_str_prompt, "not-a-prompt", repr(Prompt[str])),
        (returns_a_model, "not-a-prompt", repr(GenerateAnswer)),
        (returns_unresolvable, "unevaluable", "name 'Missing' is not defined"),
        (returns_rooted, "root-model", "Rooted"),
        (returns_aliased, "aliased", "answer"),
        (returns_validation_aliased, "aliased", "answer"),
        (returns_generated, "aliased", "cited_by"),
    ],
    ids=lambda case: case.__name__ if callable(case) else None,
)
def test_a_contract_whose_model_channels_cannot_fill_is_refused(contract, fault, declared):
    with pytest.raises(ContractRefused) as refused:
        answer_model(contract)
    assert (refused.value.contract, refused.value.fault, refused.value.declared) == (
        contract.__qualname__,
        fault,
        declared,
    )
