"""`to_sequence`: a run's keys drawn as a Mermaid sequence diagram, one participant per branch."""

import re
import sys
from pathlib import Path

import pytest

from effective.graphview import PARKED, from_keys, to_sequence
from effective.keys import Key

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.lint_mermaid import check_block  # noqa: E402

FAN = [
    "step;tool:runbook",
    "gather:0,2;step;tool:errors",
    "gather:0,0;step;tool:deploys",
    "gather:0,1;step;tool:traces",
    "d:0;step:narrow",
    "d:1;step:narrow",
]


def _said(diagram: str) -> dict[str, list[str]]:
    """Each participant's statements, in the order the diagram draws them; the world's reply
    belongs to the participant it answers."""
    said: dict[str, list[str]] = {}
    for line in diagram.splitlines()[1:]:
        statement = line.strip()
        if statement.startswith("participant"):
            continue
        reply = re.match(r"world-->>(\w+)", statement)
        speaker = (
            reply[1] if reply else re.split(r"->>|:", statement.removeprefix("Note over "))[0]
        )
        said.setdefault(speaker, []).append(statement)
    return said


def test_each_branch_is_a_participant_ordered_by_coordinate():
    diagram = to_sequence(from_keys("r", FAN))
    participants = [line.strip() for line in diagram.splitlines() if "participant" in line]
    assert participants == [
        "participant run as workflow",
        "participant b1 as gather:0,0",
        "participant b2 as gather:0,1",
        "participant b3 as gather:0,2",
        "participant world",
    ]
    assert _said(diagram)["b3"] == ["b3->>world: step#59;tool:errors"]


def test_another_interleaving_changes_no_participants_message_list():
    reordered = [FAN[0], FAN[2], FAN[3], FAN[1], *FAN[4:]]
    assert _said(to_sequence(from_keys("r", reordered))) == _said(to_sequence(from_keys("r", FAN)))
    assert to_sequence(from_keys("r", reordered)) != to_sequence(from_keys("r", FAN))


def test_a_scope_frame_stays_in_the_label_on_its_branch():
    said = _said(to_sequence(from_keys("r", FAN)))
    assert said["run"][1:] == [
        "run->>world: d:0#59;step:narrow",
        "run->>world: d:1#59;step:narrow",
    ]


def test_nested_branches_are_one_participant_each():
    keys = ["gather:0,1;gather:0,0;step;tool:a", "gather:0,1;gather:0,1;step;tool:b"]
    diagram = to_sequence(from_keys("r", keys))
    assert "participant b1 as gather:0,1 / gather:0,0" in diagram
    assert "b2->>world: step#59;tool:b" in diagram


def test_a_race_settles_as_notes_on_the_branch_that_ran_it():
    keys = [
        "race:0,0;step;tool:fast",
        "race:0,1;step;tool:slow",
        "gather:0,0;race:0;choice",
        "gather:0,0;race:0;endings",
    ]
    said = _said(to_sequence(from_keys("r", keys)))
    assert said["b1"] == ["Note over b1: race:0 choice", "Note over b1: race:0 endings"]
    assert said["b2"] == ["b2->>world: step#59;tool:fast"]


def test_a_delivered_await_is_answered_and_a_parked_one_is_not():
    delivered = to_sequence(from_keys("r", ["step;tool:a", "event;remedy:INC-7,rollback"]))
    assert delivered.splitlines()[-2:] == [
        "  Note over run: awaits event#59;remedy:INC-7,rollback",
        "  world-->>run: event#59;remedy:INC-7,rollback",
    ]
    pending = Key.parse("event;remedy:INC-7,rollback")
    parked = from_keys("r", ["step;tool:a"], pending=pending)
    assert [node.state for node in parked.nodes][-1] == PARKED
    assert (
        to_sequence(parked).splitlines()[-1]
        == "  Note over run: awaits event#59;remedy:INC-7,rollback"
    )


@pytest.mark.parametrize(
    "awaited",
    ["event;gather:0,0;remedy:x0", "$awaitEvent:gather:0,0;remedy:x0"],
    ids=["embedded", "absurd"],
)
def test_an_await_in_a_branch_is_drawn_on_that_branch(awaited):
    said = _said(to_sequence(from_keys("r", ["gather:0,0;step;tool:a", awaited])))
    assert [s.split(":")[0] for s in said["b1"]] == ["b1->>world", "Note over b1", "world-->>b1"]
    assert "run" not in said


def test_an_await_whose_name_spells_a_branch_frame_stays_where_it_ran():
    keys = ["step:before", "event;notice;gather:9,8;done", "step:after"]
    diagram = to_sequence(from_keys("r", keys))
    assert "participant b1" not in diagram
    assert "  Note over run: awaits event#59;notice#59;gather:9,8#59;done" in diagram


def test_an_await_under_a_scope_and_a_branch_is_drawn_on_that_branch():
    keys = ["d:0;gather:0,0;step:a", "event;d:0;gather:0,0;e"]
    said = _said(to_sequence(from_keys("r", keys)))
    assert len(said["b1"]) == 3


def test_a_parked_await_in_a_branch_is_drawn_on_that_branch():
    parked = from_keys("r", ["gather:0,0;step;tool:a"], pending=Key.parse("event;gather:0,0;e"))
    assert (
        to_sequence(parked).splitlines()[-1] == "  Note over b1: awaits event#59;gather:0,0#59;e"
    )


def test_branches_whose_coordinates_tie_are_ordered_by_name():
    keys = ["task:b;gather:0,0;step:x", "task:a;gather:0,0;step:y"]
    assert "participant b1 as task:a / gather:0,0" in to_sequence(from_keys("r", keys))


@pytest.mark.parametrize(
    ("key", "label"),
    [
        ("step;tool:a#2", "step#59;tool:a#35;2"),
        ("step:say\nhi", "step:say hi"),
        ("step:<b>x</b>", "step:#60;b#62;x#60;/b#62;"),
        ("step;tool:a-->>b", "step#59;tool:a--#62;#62;b"),
    ],
)
def test_key_text_renders_as_itself(key, label):
    diagram = to_sequence(from_keys("r", [key]))
    assert diagram.splitlines()[-1] == "  run->>world: " + label
    assert check_block(Path("generated.md"), 0, diagram) == []
