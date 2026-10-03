"""Tests for the typed-channel spike (§12) — t-string interpolations as I/O."""

from decimal import Decimal
from string.templatelib import Template
from typing import Annotated, assert_type

import pytest
from annotated_types import Ge, Le
from pydantic import AfterValidator, BaseModel
from pydantic.errors import PydanticSchemaGenerationError

from effective import RecordingHandler, Suspended, ask_llm
from effective.channels import (
    CacheOrderError,
    ChannelCollisionError,
    ChannelMismatchError,
    Done,
    Field,
    FormGate,
    Gated,
    Output,
    Prompt,
    Repair,
    check_channels,
    render,
)

# The typed-vocabulary form: the schema, the constraint, and any canonicalisation
# all ride the type (intrinsic) — no per-site lambda, repair reason is Pydantic's.
Amount = Annotated[float, Ge(0), Output]
Conf = Annotated[float, Ge(0), Le(1), Output]
Category = Annotated[str, AfterValidator(lambda s: s.strip().title()), Output]


class Decision(BaseModel):
    approve: bool
    amount: Decimal


class Outcome(BaseModel):
    """A declared output signature whose field names match the channel names."""

    decision: Decision


class Outcome2(BaseModel):
    answer: str


class Empty(BaseModel):
    """A declared output with no fields — for templates that exercise message
    structure (roles/cache) without output channels."""


class Composed(BaseModel):
    """The output signature for a composed prompt (a nested sub-template +
    a sibling channel)."""

    decision: Decision
    confidence: float


# Small single-/two-field signatures for the channel-vocabulary tests below. Each
# output model's fields match its template's channel names (the signature validator).
class AmountDecimal(BaseModel):
    amount: Decimal


class AmountFloat(BaseModel):
    amount: float


class ConfOut(BaseModel):
    confidence: float


class ToolAnswer(BaseModel):
    tool: str
    answer: str


class TotalConf(BaseModel):
    total_amount: float
    confidence: float


class TotalOut(BaseModel):
    total_amount: float


class CategoryOut(BaseModel):
    item_category: str


def test_input_renders_and_outputs_are_collected():
    history = "3 prior orders"
    decision = Field(Decision)
    confidence = Field(float)
    prompt = render(
        t"History: {history}\nDecision: {decision}\nConfidence: {confidence}", output=Composed
    )

    assert "History: 3 prior orders" in prompt.messages[0].content  # input rendered in place
    assert set(prompt.channels) == {"decision", "confidence"}  # output slots by field name


def test_output_channels_deserialize_via_pydantic():
    decision = Field(Decision)
    confidence = Field(float)
    prompt = render(t"Decision: {decision}\nConfidence: {confidence}", output=Composed)

    values = prompt.resolve({"decision": {"approve": True, "amount": "42.00"}, "confidence": 0.9})
    assert isinstance(values, Composed)
    assert values.decision.amount == Decimal("42.00")
    assert values.confidence == 0.9


def test_gated_channel_repairs_on_constraint_breach():
    amount = Gated(Decimal, lambda a: a <= Decimal("100"), reason="over the cap")
    prompt = render(t"Amount: {amount}", output=AmountDecimal)

    out = prompt.resolve({"amount": "500"})
    assert isinstance(out, Repair)
    assert "cap" in out.reason


def test_gated_channel_passes_when_satisfied():
    amount = Gated(Decimal, lambda a: a <= Decimal("100"))
    prompt = render(t"Amount: {amount}", output=AmountDecimal)

    assert prompt.resolve({"amount": "42"}) == AmountDecimal(amount=Decimal("42"))


def test_missing_field_becomes_a_repair_not_a_crash():
    decision = Field(Decision)
    confidence = Field(float)
    prompt = render(t"Decision: {decision}\nConfidence: {confidence}", output=Composed)

    out = prompt.resolve({"decision": {"approve": True, "amount": "1"}})  # confidence omitted
    assert isinstance(out, Repair)
    assert "confidence" in out.reason


def test_bad_type_becomes_a_repair():
    confidence = Field(float)
    prompt = render(t"Confidence: {confidence}", output=ConfOut)

    out = prompt.resolve({"confidence": "not-a-number"})
    assert isinstance(out, Repair)
    assert "confidence" in out.reason


def test_form_gate_checks_across_fields():
    # "not both empty" — a cross-field invariant no single channel can express
    tool = Field(str)
    answer = Field(str)
    prompt = render(t"tool: {tool}\nanswer: {answer}", output=ToolAnswer)
    either = FormGate(lambda v: bool(v["tool"] or v["answer"]), "set a tool or an answer")

    ok = prompt.resolve({"tool": "", "answer": "42"}, form_gates=[either])
    assert ok == ToolAnswer(tool="", answer="42")

    bad = prompt.resolve({"tool": "", "answer": ""}, form_gates=[either])
    assert isinstance(bad, Repair)
    assert "tool or an answer" in bad.reason


def test_field_repair_short_circuits_before_form_gates():
    # a field that can't be read never reaches the form gate
    amount = Field(float)
    prompt = render(t"amount: {amount}", output=AmountFloat)
    always_fail = FormGate(lambda v: False, "form gate should not run")

    out = prompt.resolve({}, form_gates=[always_fail])  # amount missing -> field Repair first
    assert isinstance(out, Repair)
    assert "amount" in out.reason  # the field reason, not the form gate's


def test_output_annotated_type_is_a_channel():
    # interpolating an Output-annotated TYPE collects an output slot, same as a Field
    total_amount = Amount
    confidence = Conf
    prompt = render(t"total: {total_amount}\nconfidence: {confidence}", output=TotalConf)

    assert set(prompt.channels) == {"total_amount", "confidence"}

    values = prompt.resolve({"total_amount": 14.16, "confidence": 0.7})
    assert values == TotalConf(total_amount=14.16, confidence=0.7)


def test_output_type_constraint_breach_is_a_repair():
    total_amount = Amount  # Annotated[float, Ge(0), Output]
    prompt = render(t"total: {total_amount}", output=TotalOut)

    out = prompt.resolve({"total_amount": -5})
    assert isinstance(out, Repair)
    assert "total_amount" in out.reason  # the field name; pydantic carries the why


def test_output_type_canonicalizes_on_parse():
    # an AfterValidator on the type folds normalization into the channel
    item_category = Category
    prompt = render(t"category: {item_category}", output=CategoryOut)

    values = prompt.resolve({"item_category": "  meals  "})
    assert values == CategoryOut(item_category="Meals")


def test_output_type_missing_field_is_a_repair():
    confidence = Conf
    prompt = render(t"confidence: {confidence}", output=ConfOut)

    out = prompt.resolve({})
    assert isinstance(out, Repair)
    assert "confidence" in out.reason


def test_render_returns_a_typed_prompt_and_resolves_to_the_declared_model():
    # render(..., output=Outcome) -> Prompt[Outcome]; .resolve() -> Outcome | Repair.
    # The signature rides the explicit `output` arg, so it stays ty-legible even though
    # the underlying Template is untyped (the A2 answer to C1).
    decision = Field(Decision)
    prompt = render(t"Decide: {decision}", output=Outcome)

    # The decisive A2 feasibility check, locked statically: the output signature
    # rides `output=`, so ty sees Prompt[Outcome] and resolve() -> Outcome | Repair —
    # even though the Template's interpolations are erased. (Channel<->field
    # correspondence is NOT static; the §7 validator enforces that at render time.)
    assert_type(prompt, Prompt[Outcome])
    assert_type(prompt.resolve({}), Outcome | Repair)

    assert prompt.messages[0].role == "user"
    assert "Decide:" in prompt.messages[0].content
    assert set(prompt.channels) == {"decision"}
    assert prompt.seams["decision"] == "<<decision>>"  # the placeholder is the seam
    assert prompt.output is Outcome

    out = prompt.resolve({"decision": {"approve": True, "amount": "42.00"}})
    assert isinstance(out, Outcome)
    assert out.decision.approve is True
    assert out.decision.amount == Decimal("42.00")


def test_render_resolve_repairs_a_missing_field():
    decision = Field(Decision)
    prompt = render(t"Decide: {decision}", output=Outcome)

    out = prompt.resolve({})  # decision omitted
    assert isinstance(out, Repair)
    assert "decision" in out.reason


def test_render_seams_capture_inputs_and_outputs():
    history = "3 prior orders"
    decision = Field(Decision)
    prompt = render(t"History: {history}\nDecision: {decision}", output=Outcome)

    assert prompt.seams["history"] == "3 prior orders"  # input rendered in place
    assert prompt.seams["decision"] == "<<decision>>"  # output placeholder


def test_render_directives_split_messages_by_role_and_cache():
    # a role/cache directive scopes to its interpolation; consecutive same-(role,cache)
    # segments coalesce. Here: a cached system preamble, then the volatile user body.
    preamble = "You are precise."
    body = "the email text"
    answer = Field(str)
    prompt = render(t"{preamble:role=system;cache}{body}answer: {answer}", output=Outcome2)

    assert [(m.role, m.cache) for m in prompt.messages] == [
        ("system", True),
        ("user", False),
    ]
    assert prompt.messages[0].content == "You are precise."
    assert "the email text" in prompt.messages[1].content
    assert "<<answer>>" in prompt.messages[1].content


def test_render_raises_on_volatile_before_cached_in_a_role():
    # nocache then cache in the same role poisons the cache point -> CacheOrderError
    a = "volatile"
    b = "preamble"
    with pytest.raises(CacheOrderError):
        render(t"{a:role=system;nocache}{b:role=system;cache}", output=Empty)


def test_render_allows_cached_then_volatile():
    a = "preamble"
    b = "body"
    prompt = render(t"{a:role=system;cache}{b:role=system}", output=Empty)
    # cached system first, then a default (volatile) system segment — correct order
    assert [(m.role, m.cache) for m in prompt.messages] == [
        ("system", True),
        ("system", False),
    ]


def _decision_block() -> Template:
    # a composable sub-prompt contributing the `decision` channel
    decision = Field(Decision)
    return t"  decide: {decision}"


def test_render_composes_a_nested_template_and_merges_channels():
    # the tdom win: a sub-Template splices in; render recurses and merges channels
    confidence = Field(float)
    prompt = render(t"{_decision_block()}\n  confidence: {confidence}", output=Composed)

    assert set(prompt.channels) == {"decision", "confidence"}
    out = prompt.resolve({"decision": {"approve": True, "amount": "5"}, "confidence": 0.9})
    assert isinstance(out, Composed)
    assert out.decision.approve is True
    assert out.confidence == 0.9


def test_composition_channel_name_collision_is_a_render_error():
    confidence = Field(float)
    # the sub-block already contributes `decision`; clashing on it is a located error
    decision = Field(Decision)
    with pytest.raises(ChannelCollisionError):
        render(t"{_decision_block()}{decision}{confidence}", output=Composed)


def test_signature_validator_rejects_field_channel_mismatch():
    # a channel with no matching output field (and a field with no channel) is caught
    confidence = Field(float)
    with pytest.raises(ChannelMismatchError):
        render(t"only: {confidence}", output=Composed)  # missing the `decision` channel


def test_duplicate_channel_name_in_one_template_is_a_render_error():
    confidence = Field(float)
    with pytest.raises(ChannelCollisionError, match="confidence"):
        render(t"one: {confidence}\ntwo: {confidence}", output=ConfOut)


def test_collision_error_names_the_colliding_channel_under_composition():
    decision = Field(Decision)
    with pytest.raises(ChannelCollisionError, match="decision"):
        render(t"{_decision_block()}{decision}", output=Outcome)


def test_collision_under_composition_preempts_later_subtree_errors():
    # the reduction drives the walk lazily, so a colliding name raises as its
    # declaration streams out — BEFORE a later breach (an independence re-entry
    # here) in the not-yet-walked tail is reached. Earlier than the pre-streaming
    # form, which finished the sub-tree before its _merge collision check.
    decision = Field(Decision)
    poison = Done(1)
    with pytest.raises(ChannelCollisionError, match="decision"):
        render(t"{_decision_block()}{decision}{poison}", output=Outcome)


def test_duplicate_channel_render_side_effect_runs_once():
    # the duplicate is collision-checked at its declaration, so its render()
    # never runs — a raising render() cannot preempt the ChannelCollisionError
    calls: list[int] = []

    class Counting:
        def render(self) -> str:
            calls.append(1)
            return ""

        def write(self, name: str, response: object) -> object:  # pragma: no cover
            raise NotImplementedError

    x = Counting()
    with pytest.raises(ChannelCollisionError, match="x"):
        render(t"{x}{x}", output=ConfOut)
    assert calls == [1]


def test_collision_raises_before_cache_order():
    # one template trips both breaches; the duplicate channel name wins
    answer = Field(str)
    volatile = "v"
    cached = "c"
    with pytest.raises(ChannelCollisionError):
        render(t"{answer}{answer}{volatile:nocache}{cached:cache}", output=Outcome2)


def test_cache_order_raises_before_signature_mismatch():
    # trips both the volatile-then-cached breach and a channel/field mismatch;
    # the cache iron-rule is checked first (render's documented order)
    confidence = Field(float)
    volatile = "v"
    cached = "c"
    with pytest.raises(CacheOrderError):
        render(t"{volatile:nocache}{cached:cache}{confidence}", output=Composed)


def test_seams_capture_each_composite_boundary_exactly():
    # the whole seams surface: every interpolation's rendered text, plus — for a
    # composite node — the joined text of exactly that sub-tree, keyed by its
    # expression. Sub-tree boundaries must survive any walk restructuring.
    history = "3 prior orders"
    confidence = Field(float)
    prompt = render(
        t"History: {history}\n{_decision_block()}\nconf: {confidence}", output=Composed
    )
    assert prompt.seams == {
        "history": "3 prior orders",
        "decision": "<<decision>>",
        "_decision_block()": "  decide: <<decision>>",
        "confidence": "<<confidence>>",
    }


def test_check_channels_helper_passes_on_a_match():
    channels = {"decision": Field(Decision), "confidence": Field(float)}
    check_channels(channels, Composed)  # no raise


def test_channels_compose_with_recording_handler():
    # the template flows as the AskLLM payload; the handler returns the raw dict;
    # the workflow resolves it into typed outputs (resolution-in-caller variant).
    decision = Field(Decision)
    prompt = render(t"Decide: {decision}", output=Outcome)

    def workflow():
        raw = yield from ask_llm("decide", prompt.messages, dict)
        return prompt.resolve(raw)

    handler = RecordingHandler({"decide": {"decision": {"approve": False, "amount": "10"}}})
    result = handler.run(workflow)
    assert not isinstance(result, Suspended | Repair)
    assert isinstance(result, Outcome)
    assert result.decision.approve is False


@pytest.mark.parametrize("answer", ["45", 45, ["decision", 1], None])
def test_an_answer_that_is_not_a_mapping_is_a_repair(answer):
    # A model can answer with a bare string or a list; that is a re-prompt, never a crash.
    decision = Field(Decision)
    prompt = render(t"Decide: {decision}", output=Outcome)
    assert isinstance(prompt.resolve(answer), Repair)


class _Named(BaseModel):
    name: str


class _Listed(BaseModel):
    names: list[_Named]
    label: str


class _Opaque:
    """A class pydantic has no schema for."""


def test_every_channel_schema_is_read_by_one_validator():
    names, label = Field(list[_Named]), Field(str)
    prompt = render(t"{names} {label}", output=_Listed)
    assert prompt.resolve({"names": [{"name": "a"}], "label": "x"}) == _Listed(
        names=[_Named(name="a")], label="x"
    )
    assert isinstance(prompt.resolve({"names": [{"nom": "a"}], "label": "x"}), Repair)
    assert isinstance(prompt.resolve({"names": [], "label": 5}), Repair)


def test_a_schema_pydantic_cannot_describe_fails_where_the_channel_is_written():
    with pytest.raises(PydanticSchemaGenerationError):
        Field(_Opaque)
