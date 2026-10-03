"""The envelope seam: the wire alphabet as an explicit value, JsonEnvelope
(lenient flat JSON) or SectionEnvelope (sentinel sections, raw bodies), feeding
the same ``Prompt.resolve`` Mapping contract."""

import pytest
from pydantic import BaseModel

from effective.channels import Field, render
from effective.envelope import EnvelopeParseError, JsonEnvelope, SectionEnvelope

HEREDOC = (
    'cat <<\'EOF\' > parser.py\nimport re\npat = re.compile(r"\\d+ \\(x\\)")\nprint("done")\nEOF'
)


class Turn(BaseModel):
    thought: str
    command: str


def _turn_prompt():
    thought = Field(str)
    command = Field(str)
    return render(t"{thought} then run {command}", output=Turn)


# --- SectionEnvelope: the heterogeneous alphabet ----------------------------


def test_sections_round_trip_a_heredoc_with_zero_escaping():
    env = SectionEnvelope()
    completion = (
        "[[ ## thought ## ]]\nwrite the parser\n"
        f"[[ ## command ## ]]\n{HEREDOC}\n"
        "[[ ## completed ## ]]\n"
    )
    parsed = env.parse(completion)
    # the exact bytes survive: newlines, regex backslashes, quotes, EOF sentinel
    assert parsed["command"] == HEREDOC
    turn = _turn_prompt().resolve(parsed)
    assert isinstance(turn, Turn)
    assert turn.command == HEREDOC


def test_sections_tolerate_preamble_reorder_and_repeats():
    env = SectionEnvelope()
    parsed = env.parse(
        "Sure! Here is my answer.\n"  # models preface — ignored
        "[[ ## command ## ]]\nls\n"
        "[[ ## thought ## ]]\nfirst\n"
        "[[ ## thought ## ]]\nsecond\n"  # repeat: last wins
        "[[ ## completed ## ]]"
    )
    assert parsed == {"command": "ls", "thought": "second"}


def test_sections_omitted_section_is_simply_absent():
    parsed = SectionEnvelope().parse("[[ ## thought ## ]]\ndone\n[[ ## completed ## ]]")
    assert parsed == {"thought": "done"}


def test_sections_no_markers_is_a_parse_error():
    with pytest.raises(EnvelopeParseError, match="no section markers"):
        SectionEnvelope().parse('{"thought": "wrong alphabet"}')


def test_sections_instructions_name_every_channel_and_the_terminator():
    prompt = _turn_prompt()
    text = SectionEnvelope().instructions(prompt.channels)
    assert "[[ ## thought ## ]]" in text
    assert "[[ ## command ## ]]" in text
    assert "[[ ## completed ## ]]" in text


# --- JsonEnvelope: the lenient flat-JSON alphabet ----------------------------


def test_json_takes_the_first_object_and_ignores_trailing():
    # the "trailing characters" failure: two objects in one completion
    parsed = JsonEnvelope().parse('{"thought":"a","command":"ls"}\n{"thought":"b"}')
    assert parsed == {"thought": "a", "command": "ls"}


def test_json_tolerates_control_characters_in_strings():
    parsed = JsonEnvelope().parse('{"command":"line1\nline2"}')
    assert parsed["command"] == "line1\nline2"


def test_json_skips_prose_before_the_object():
    parsed = JsonEnvelope().parse('Here you go:\n{"thought":"x","command":null}')
    assert parsed["thought"] == "x"


def test_json_without_an_object_is_a_parse_error():
    with pytest.raises(EnvelopeParseError, match="no JSON object"):
        JsonEnvelope().parse("[[ ## thought ## ]]\nwrong alphabet")


def test_json_instructions_name_every_channel():
    text = JsonEnvelope().instructions(_turn_prompt().channels)
    assert '"thought"' in text
    assert '"command"' in text


# --- the seam property: both alphabets feed the same resolve contract --------


def test_both_envelopes_resolve_through_the_same_typed_prompt():
    prompt = _turn_prompt()
    via_json = prompt.resolve(JsonEnvelope().parse('{"thought":"t","command":"ls"}'))
    via_sections = prompt.resolve(
        SectionEnvelope().parse(
            "[[ ## thought ## ]]\nt\n[[ ## command ## ]]\nls\n[[ ## completed ## ]]"
        )
    )
    assert via_json == via_sections == Turn(thought="t", command="ls")
