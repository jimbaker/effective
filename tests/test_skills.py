"""The `Skill` node + `SkillRegistry` filesystem loader.

The registry loads standard-layout packs (subset frontmatter, blake2b tree
hash); a `Skill` node discloses a body through the same applicative merge as
nested-`Template` composition, with disclosure-is-the-cache-boundary falling
out of the existing volatile-last rule."""

from pathlib import Path
from string.templatelib import Template

import pytest
from pydantic import BaseModel

from effective.channels import (
    CacheOrderError,
    ChannelCollisionError,
    Done,
    Field,
    IndependenceError,
    Repair,
    SkillCycleError,
    SkillResolutionError,
    render,
    skill,
)
from effective.domain import CallTool
from effective.handlers import RecordingHandler, ReplayHandler
from effective.handlers.recording import Suspended
from effective.keys import Segment, compose_key
from effective.skills import (
    DISCLOSE_TOOL,
    Pin,
    SkillPackError,
    SkillRegistry,
    activate_skill,
    parse_frontmatter,
    refresh_skill,
)

PACKS = Path(__file__).parent / "skill_packs"


class Answer(BaseModel):
    answer: str


class Empty(BaseModel):
    pass


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return SkillRegistry.load(PACKS)


# --- loader ---------------------------------------------------------------


def test_load_finds_every_skill_and_orders_the_catalog(registry: SkillRegistry):
    assert registry.names() == ("greeting-cards", "unit-conversion")
    index = registry.index()
    assert index.splitlines() == [
        "- greeting-cards: Compose short greeting-card messages in a requested tone, "
        "using the house tone guide.",
        "- unit-conversion: Convert quantities between metric and imperial units "
        "with rounding rules",
    ]


def test_body_is_a_degenerate_template(registry: SkillRegistry):
    body = registry.body("unit-conversion")
    assert isinstance(body, Template)
    assert not list(body.interpolations)  # plain markdown: zero interpolations
    assert "Always convert via SI base units" in body.strings[0]


def test_flat_extra_fields_survive_and_nested_metadata_is_skipped(registry: SkillRegistry):
    info = registry.info("greeting-cards")
    assert info.fields["license"] == "Apache-2.0"  # quotes stripped
    assert "category" not in info.fields  # nested under metadata:, not consumed
    assert "metadata" not in info.fields  # a bare key heading a block is no field


def test_unknown_skill_is_a_lookup_error(registry: SkillRegistry):
    with pytest.raises(LookupError, match="no-such-skill"):
        registry.body("no-such-skill")


def test_load_rejects_a_name_directory_mismatch(tmp_path: Path):
    d = tmp_path / "some-dir"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: other-name\ndescription: d\n---\nbody\n")
    with pytest.raises(SkillPackError, match="!= directory"):
        SkillRegistry.load(tmp_path)


def test_load_rejects_a_spec_grammar_breach(tmp_path: Path):
    d = tmp_path / "Bad_Name"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: Bad_Name\ndescription: d\n---\nbody\n")
    with pytest.raises(SkillPackError, match="spec grammar"):
        SkillRegistry.load(tmp_path)


def test_load_requires_name_and_description(tmp_path: Path):
    d = tmp_path / "nameless"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: nameless\n---\nbody\n")
    with pytest.raises(SkillPackError, match="name and description"):
        SkillRegistry.load(tmp_path)


def test_load_rejects_an_empty_root(tmp_path: Path):
    with pytest.raises(SkillPackError, match="not a skill pack"):
        SkillRegistry.load(tmp_path)


# --- frontmatter subset ----------------------------------------------------


def test_frontmatter_block_scalars_fold_and_keep_lines():
    fields, body = parse_frontmatter(
        "---\n"
        "name: x\n"
        "description: >\n"
        "  first line\n"
        "  second line\n"
        "notes: |\n"
        "  keep\n"
        "  lines\n"
        "---\n"
        "the body\n"
    )
    assert fields["description"] == "first line second line"
    assert fields["notes"] == "keep\nlines"
    assert body == "the body"


def test_frontmatter_requires_the_fences():
    with pytest.raises(SkillPackError, match="frontmatter fence"):
        parse_frontmatter("name: x\n")
    with pytest.raises(SkillPackError, match="unterminated"):
        parse_frontmatter("---\nname: x\n")


def test_frontmatter_rejects_an_unparseable_line():
    with pytest.raises(SkillPackError, match="unparseable"):
        parse_frontmatter("---\nname x\n---\nbody\n")


def test_frontmatter_parses_a_block_sequence_faithfully():
    # allowed-tools as a YAML block list must survive verbatim for its S3
    # consumer — a silently dropped grant list is the wrong kind of quiet.
    fields, _ = parse_frontmatter(
        "---\nname: x\ndescription: d\nallowed-tools:\n  - Bash(git:*)\n  - Read\n---\nbody\n"
    )
    assert fields["allowed-tools"] == "Bash(git:*)\nRead"


def test_frontmatter_rejects_an_unclosed_flow_list():
    # a multi-line flow list would otherwise be stored TRUNCATED — reject loudly
    with pytest.raises(SkillPackError, match="flow list"):
        parse_frontmatter("---\nname: x\nallowed-tools: [Bash,\n  Read]\n---\nbody\n")


def test_frontmatter_still_skips_a_nested_map():
    fields, _ = parse_frontmatter(
        "---\nname: x\ndescription: d\nmetadata:\n  category: writing\n---\nbody\n"
    )
    assert "metadata" not in fields
    assert "category" not in fields


# --- the pin hash ----------------------------------------------------------


def test_tree_hash_is_stable_across_loads(registry: SkillRegistry):
    again = SkillRegistry.load(PACKS)
    for name in registry.names():
        assert registry.info(name).content_hash == again.info(name).content_hash


def test_tree_hash_moves_when_any_file_changes(tmp_path: Path):
    d = tmp_path / "hashy"
    (d / "references").mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: hashy\ndescription: d\n---\nbody\n")
    (d / "references" / "guide.md").write_text("v1\n")
    before = SkillRegistry.load(tmp_path).info("hashy").content_hash
    (d / "references" / "guide.md").write_text("v2\n")
    after = SkillRegistry.load(tmp_path).info("hashy").content_hash
    assert before != after  # references are part of the pin identity (§3.2)


def test_load_rejects_symlinks_in_a_skill_pack(tmp_path: Path):
    # rglob does not descend a symlinked dir and a symlinked file hashes
    # out-of-tree state — either way the pin hash would lie about its coverage.
    d = tmp_path / "linky"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: linky\ndescription: d\n---\nbody\n")
    (tmp_path / "outside.md").write_text("outside the pack\n")
    (d / "guide.md").symlink_to(tmp_path / "outside.md")
    with pytest.raises(SkillPackError, match="symlink"):
        SkillRegistry.load(tmp_path)


def test_load_rejects_a_symlinked_directory(tmp_path: Path):
    d = tmp_path / "linky"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: linky\ndescription: d\n---\nbody\n")
    real = tmp_path / "real-refs"
    real.mkdir()
    (real / "guide.md").write_text("content the hash would miss\n")
    (d / "references").symlink_to(real, target_is_directory=True)
    with pytest.raises(SkillPackError, match="symlink"):
        SkillRegistry.load(tmp_path)


# --- the Skill node in render ---------------------------------------------


def test_skill_discloses_after_the_cached_catalog(registry: SkillRegistry):
    preamble = "You answer with the pack's conventions."
    answer = Field(str)
    prompt = render(
        t"""{preamble:role=system;cache}
{registry.index():role=system;cache}
{skill("unit-conversion")}
  "answer": {answer}""",
        output=Answer,
        registry=registry,
    )
    catalog = next(m for m in prompt.messages if "- unit-conversion:" in m.content)
    assert (catalog.role, catalog.cache) == ("system", True)
    body = next(m for m in prompt.messages if "Always convert via SI base units" in m.content)
    assert (body.role, body.cache) == ("user", False)  # the volatile disclosed tail
    assert prompt.messages.index(catalog) < prompt.messages.index(body)
    assert "Always convert via SI base units" in prompt.seams["unit-conversion"]


def test_skill_body_scopes_a_role_directive(registry: SkillRegistry):
    prompt = render(
        t"{skill('unit-conversion'):role=system}ask away",
        output=Empty,
        registry=registry,
    )
    assert prompt.messages[0].role == "system"
    assert "SI base units" in prompt.messages[0].content
    tail = prompt.messages[-1]
    assert (tail.role, tail.content, tail.cache) == ("user", "ask away", False)


def test_cached_segment_after_a_disclosed_body_still_breaches_volatile_last(
    registry: SkillRegistry,
):
    late = "late cached prose"
    with pytest.raises(CacheOrderError):
        render(
            t"{skill('unit-conversion')}{late:cache}",
            output=Empty,
            registry=registry,
        )


def test_skill_without_a_registry_is_a_located_render_error():
    with pytest.raises(SkillResolutionError, match="no registry"):
        render(t"{skill('unit-conversion')}", output=Empty)


def test_unknown_skill_is_a_located_render_error(registry: SkillRegistry):
    with pytest.raises(SkillResolutionError, match="no-such-skill"):
        render(t"{skill('no-such-skill')}", output=Empty, registry=registry)


# --- native bodies (in-memory registry): channels merge, cycles rejected ---


def test_native_skill_body_channels_merge_into_the_signature():
    answer = Field(str)
    body = t'respond with  "answer": {answer}'
    reg = SkillRegistry.in_memory({"native-ask": ("a native skill", body)})
    prompt = render(t"{skill('native-ask')}", output=Answer, registry=reg)
    assert set(prompt.channels) == {"answer"}
    assert prompt.resolve({"answer": "42"}) == Answer(answer="42")


def test_skill_seam_is_keyed_by_name_with_the_disclosed_text():
    # the disclosure boundary, exactly: the seam is the skill NAME mapped to the
    # joined text of just that disclosed sub-tree
    answer = Field(str)
    body = t'respond with "answer": {answer}'
    reg = SkillRegistry.in_memory({"native-ask": ("a native skill", body)})
    prompt = render(t"{skill('native-ask')}", output=Answer, registry=reg)
    assert prompt.seams["native-ask"] == 'respond with "answer": <<answer>>'


def test_native_skill_channel_collision_is_a_render_error():
    answer = Field(str)
    body = t"{answer}"
    reg = SkillRegistry.in_memory({"clashing": ("clashes with the parent", body)})
    with pytest.raises(ChannelCollisionError):
        render(t"{answer}{skill('clashing')}", output=Answer, registry=reg)


def test_self_disclosing_skill_is_a_cycle_error():
    body = t"see also {skill('recursive')}"
    reg = SkillRegistry.in_memory({"recursive": ("discloses itself", body)})
    with pytest.raises(SkillCycleError, match="recursive"):
        render(t"{skill('recursive')}", output=Empty, registry=reg)


def test_in_memory_pin_hash_tracks_the_body():
    a = SkillRegistry.in_memory({"s": ("d", Template("one"))}).info("s").content_hash
    b = SkillRegistry.in_memory({"s": ("d", Template("two"))}).info("s").content_hash
    assert a != b


def test_same_plain_skill_can_be_disclosed_twice():
    # Two disclosures of one channel-free skill are legal composition — the
    # body text simply appears twice (no channels, so nothing collides).
    reg = SkillRegistry.in_memory({"plain": ("plain prose", Template("PROSE"))})
    prompt = render(t"{skill('plain')} and {skill('plain')}", output=Empty, registry=reg)
    assert prompt.messages[0].content.count("PROSE") == 2


# --- the independence invariant (step-6 §2.3): dynamic, authoritative -------


def test_a_prior_resolution_as_an_input_breaches_independence():
    prior = Done("last turn's answer")
    with pytest.raises(IndependenceError, match="Done"):
        render(t"continue from {prior}", output=Empty)


def test_a_repair_as_an_input_breaches_independence():
    verdict = Repair("too vague")
    with pytest.raises(IndependenceError, match="Repair"):
        render(t"fix per {verdict}", output=Empty)


def test_independence_holds_inside_a_disclosed_skill_body():
    leaked = Done(41)
    body = t"the answer so far is {leaked}"
    reg = SkillRegistry.in_memory({"leaky": ("reads a resolution", body)})
    with pytest.raises(IndependenceError):
        render(t"{skill('leaky')}", output=Empty, registry=reg)


def test_an_unwrapped_prior_value_is_legal_turn_axis_threading():
    # Value-dependence across turns is the monadic axis and is fine — what the
    # invariant rejects is the *resolution wrapper* re-entering a render.
    prior = "last turn's answer"
    prompt = render(t"continue from {prior}", output=Empty)
    assert prompt.messages[0].content == "continue from last turn's answer"


# --- S2: pins — activation/refresh ops, pinned rendering, span attribution --

V1 = Pin(name="advice", content_hash="aaa", body="ADVICE V1")
V2 = Pin(name="advice", content_hash="bbb", body="ADVICE V2")
_PIN_RESPONSES = {"skill:advice,activate": V1, "skill:advice,refresh,1": V2}


def _pin_workflow():
    p1 = yield from activate_skill("advice")
    p2 = yield from refresh_skill("advice", 1)
    return (p1, p2)


def test_activate_and_refresh_mint_distinct_checkpoint_keys():
    h = RecordingHandler(responses=dict(_PIN_RESPONSES))
    result = h.run(_pin_workflow)
    assert not isinstance(result, Suspended)
    activation, refresh = result
    # injectivity at the key level: refresh never collides with its activation
    assert [e.key.stored() for e in h.trace] == [
        "step;skill:advice,activate",
        "step;skill:advice,refresh,1",
    ]
    # and each pin CARRIES the key of the activation that minted it, so a script taken from
    # it can be scoped to that activation rather than annotated with a provenance string.
    assert (activation.key.stored(), refresh.key.stored()) == (
        "skill:advice,activate",
        "skill:advice,refresh,1",
    )
    keyless = (
        activation.model_copy(update={"key": None}),
        refresh.model_copy(update={"key": None}),
    )
    assert keyless == (V1, V2)


def test_pin_workflow_replays_identically():
    h = RecordingHandler(responses=dict(_PIN_RESPONSES))
    first = h.run(_pin_workflow)
    assert ReplayHandler(h.trace).run(_pin_workflow) == first


def test_pinned_skill_renders_recorded_content_without_a_registry():
    prompt = render(t"{skill('advice', pin=V1)}", output=Empty)
    assert prompt.messages[0].content == "ADVICE V1"
    assert prompt.seams["advice"] == "ADVICE V1"


def test_pinned_render_ignores_registry_drift(registry: SkillRegistry):
    # The pin is the recorded truth — the (moved-on) registry snapshot never wins.
    pin = Pin(name="unit-conversion", content_hash="old", body="OLD BODY")
    prompt = render(t"{skill('unit-conversion', pin=pin)}", output=Empty, registry=registry)
    assert prompt.messages[0].content == "OLD BODY"


def test_pin_name_mismatch_is_a_located_render_error():
    with pytest.raises(SkillResolutionError, match="threaded into"):
        render(t"{skill('other-skill', pin=V1)}", output=Empty)


def test_disclose_returns_the_pin_for_the_snapshot(registry: SkillRegistry):
    pin = registry.disclose("unit-conversion")
    assert pin.content_hash == registry.info("unit-conversion").content_hash
    # pinned and run-start-snapshot renders agree while the tree hasn't moved
    pinned = render(t"{skill('unit-conversion', pin=pin)}", output=Empty)
    snapshot = render(t"{skill('unit-conversion')}", output=Empty, registry=registry)
    assert pinned.messages == snapshot.messages


def test_pin_carries_scripts_and_a_disclosed_pin_has_no_key_yet(registry: SkillRegistry):
    """The pin covers the pack's scripts/ tier: pin.script() hands run_code a replay-exact
    source, never a working-tree read.

    **A registry-disclosed pin has no activation key, and `script()` refuses loudly.** The
    registry is deployment infrastructure and knows nothing about keys; only `activate_skill`
    /`refresh_skill` mint one, because only they are the recorded op. A script whose key is
    unknown cannot be scoped to the pack that pinned it, and silently running it unscoped would
    lose the improvised-vs-pinned split, so the refusal is the design.
    """
    pin = registry.disclose("greeting-cards")
    assert sorted(pin.scripts) == ["make_card.py"]
    assert pin.key is None
    with pytest.raises(ValueError, match="no activation key"):
        pin.script("make_card.py")
    activation = compose_key(t"skill:{Segment('greeting-cards')},activate")
    keyed = pin.model_copy(update={"key": activation})
    s = keyed.script("make_card.py")
    assert "def make_card" in s.source
    assert s.key.stored() == "skill:greeting-cards,activate"
    with pytest.raises(LookupError, match=r"no script 'ghost\.py'"):
        keyed.script("ghost.py")
    # a pack with no scripts/ discloses an empty tier, and stays pin-able
    assert registry.disclose("unit-conversion").scripts == {}


def test_pin_scripts_round_trip_the_checkpoint_serde(registry: SkillRegistry):
    """Scripts ride the activation checkpoint like the body: JSON out and back
    reconstructs a Pin whose script() output is identical."""
    from pydantic import TypeAdapter
    from pydantic_core import to_jsonable_python

    # The RECORDED pin is the one the domain interpreter built, and it is keyless: the
    # activation key is attached by `activate_skill` AFTER the step returns, so it is derived
    # on both the live and the replay path and never rides the checkpoint: the identity belongs
    # in the key, and a copy in the payload beside it would be denormalized.
    pin = registry.disclose("greeting-cards")
    assert pin.key is None
    revived = TypeAdapter(Pin).validate_python(to_jsonable_python(pin))
    assert revived == pin

    # And if a keyed pin ever DID reach the wire, it would ride as its `stored()` text rather
    # than leaking the opaque type's structure — `Key.__get_pydantic_core_schema__`'s job.
    activation = compose_key(t"skill:{Segment('greeting-cards')},activate")
    keyed = pin.model_copy(update={"key": activation})
    assert to_jsonable_python(keyed)["key"] == "skill:greeting-cards,activate"
    assert keyed.script("make_card.py").key == keyed.key


def test_disclose_rejects_a_native_body():
    answer = Field(str)
    reg = SkillRegistry.in_memory({"native-ask": ("a native skill", t"{answer}")})
    with pytest.raises(SkillPackError, match="native"):
        reg.disclose("native-ask")


def test_pin_round_trips_the_checkpoint_serde():
    # the Absurd checkpoint path: to_jsonable_python out, TypeAdapter back
    from pydantic import TypeAdapter
    from pydantic_core import to_jsonable_python

    assert TypeAdapter(Pin).validate_python(to_jsonable_python(V1)) == V1


def test_skill_disclose_span_carries_pin_fields():
    from effective.layers import Interpreter, compose_domain
    from effective.telemetry import Span, genai_attributes, traced

    spans: list[Span] = []

    class _Base:
        def run(self, op):
            return V1

    stack: Interpreter = compose_domain([traced(spans.append, session_id="s1")], base=_Base())
    op = CallTool(
        name=DISCLOSE_TOOL, args={"skill": "advice", "event": "activate"}, result_schema=Pin
    )
    assert stack.run(op) == V1
    attrs = genai_attributes(spans[0])
    assert attrs["gen_ai.skill.name"] == "advice"
    assert attrs["effective.skill.content_hash"] == "aaa"
    assert attrs["effective.skill.pin_event"] == "activate"


def test_a_FAILED_disclose_still_carries_skill_name_from_the_args():
    """The error path is part of the verified-use record: a disclose that raised
    still says WHICH skill was asked for, or the SkillsBench present-vs-consulted split loses
    exactly the runs where consultation went wrong.

    **This is the arm a NAME test breaks, and it is why attribution reads the args instead.** The
    pin-shaped arm below fires on every SUCCESSFUL disclose and would keep every other test in
    this file green while this one silently emitted nothing, which is what a rename of the tool
    does, undetected by a full suite.

    MUTATION: put `op.name.startswith("skill:") and` back in front of `_skill_fields`' first
    arm; this reddens and nothing else does.
    """
    from effective.layers import Interpreter, compose_domain
    from effective.telemetry import Span, genai_attributes, traced

    spans: list[Span] = []

    class _Base:
        def run(self, op):
            raise RuntimeError("registry unreachable")

    stack: Interpreter = compose_domain([traced(spans.append, session_id="s1")], base=_Base())
    with pytest.raises(RuntimeError, match="registry unreachable"):
        stack.run(
            CallTool(
                name=DISCLOSE_TOOL,
                args={"skill": "advice", "event": "activate"},
                result_schema=Pin,
            )
        )
    attrs = genai_attributes(spans[0])
    assert attrs["gen_ai.skill.name"] == "advice"
    assert attrs["effective.skill.pin_event"] == "activate"
    assert "effective.skill.content_hash" not in attrs  # no pin came back


def test_a_pin_shaped_result_carries_skill_fields_without_the_namespace():
    # Model-facing tool names can't contain ':' (provider grammars), so a
    # bench 'activate_skill' tool attributes by its pin-shaped RESULT.
    from effective.layers import Interpreter, compose_domain
    from effective.telemetry import Span, genai_attributes, traced

    spans: list[Span] = []

    class _Base:
        def run(self, op):
            return V1

    stack: Interpreter = compose_domain([traced(spans.append, session_id="s1")], base=_Base())
    stack.run(CallTool(name="activate_skill", args={"skill": "advice"}, result_schema=Pin))
    attrs = genai_attributes(spans[0])
    assert attrs["gen_ai.skill.name"] == "advice"
    assert attrs["effective.skill.content_hash"] == "aaa"


def test_a_plain_tool_span_has_no_skill_fields():
    from effective.layers import Interpreter, compose_domain
    from effective.telemetry import Span, genai_attributes, traced

    spans: list[Span] = []

    class _Base:
        def run(self, op):
            return {"ok": True}

    stack: Interpreter = compose_domain([traced(spans.append, session_id="s1")], base=_Base())
    stack.run(CallTool(name="fetch_email", args={}, result_schema=dict))
    assert "gen_ai.skill.name" not in genai_attributes(spans[0])
