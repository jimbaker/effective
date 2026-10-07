"""The startup workflows, end to end on embedded SQLite against their scripted worlds.

Expectations are computed from each scenario's own inputs. Each world counts its calls, so one
call per recorded op, however many attempts the run took, is replay seen from the world's side.
"""

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from math import ceil
from pathlib import Path

import pytest
from pydantic import BaseModel

from effective.api import Effect, sleep_until
from effective.channels import DATA_MARK
from effective.judgment import NO_MATCH
from examples.startup import incident, launch, voice
from examples.startup.asking import REPAIRS
from examples.startup.engine import World, run
from examples.startup.scenarios import CANDIDATE, SCENARIOS, TICKETS

pytestmark = pytest.mark.journey


def _choice(label: str) -> dict[str, object]:
    return {"choice": label, "confidence": 0.9, "probabilities": {label: 0.9}}


def _world(name: str, **changes: object) -> World:
    return replace(SCENARIOS[name].world(), **changes)


def _answers(*answers: dict[str, object]) -> Callable[[str], dict[str, object]]:
    """A model that gives `answers` in turn, then repeats the last."""
    given = list(answers)

    def model(prompt: str) -> dict[str, object]:
        return given.pop(0) if len(given) > 1 else given[0]

    return model


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_recorded_op_reaches_the_world_once_across_attempts(name):
    scenario = SCENARIOS[name]
    world = scenario.world()
    ran = run(scenario.program, world, scenario.deliver)
    assert ran.attempts == 1 + len(ran.delivered)
    assert world.calls.total() == len(ran.keys)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_timeline_places_each_await_where_the_run_parked(name):
    scenario = SCENARIOS[name]
    ran = run(scenario.program, scenario.world(), scenario.deliver)
    awaited = [key for key in ran.timeline if key.startswith("event;")]
    assert awaited == ["event;" + event for event in ran.delivered]
    assert tuple(key for key in ran.timeline if key not in awaited) == ran.keys


def test_the_remediation_follows_its_approval_on_the_timeline():
    scenario = SCENARIOS["incident"]
    timeline = run(scenario.program, scenario.world(), scenario.deliver).timeline
    approval = next(i for i, key in enumerate(timeline) if key.startswith("event;remedy:"))
    assert timeline[approval + 1 :] == ("step;tool:remediate",)


# --- incident ------------------------------------------------------------------------------


def _settles(remedy: str) -> dict[str, object]:
    return {"finding": "found", "remedy": remedy, "settled": True, "follow": ""}


def test_the_incident_remediates_only_after_approval():
    scenario = SCENARIOS["incident"]
    approved = scenario.world()
    ran = run(scenario.program, approved, scenario.deliver)
    assert len(ran.delivered) == 1
    assert ran.result["remediation"] is not None
    assert approved.calls["remediate"] == 1

    refused = scenario.world()
    denied = incident.Approval(approved=False, by="on-call")
    ran = run(scenario.program, refused, lambda event: denied)
    assert ran.result["remediation"] is None
    assert refused.calls["remediate"] == 0


def test_an_approval_authorizes_only_the_remedy_it_was_asked_about(tmp_path: Path):
    scenario, store = SCENARIOS["incident"], tmp_path / "shared.db"
    first = run(scenario.program, _world("incident"), scenario.deliver, store=store)
    failover = _world("incident", llm=_answers(_settles("failover")))
    second = run(scenario.program, failover, scenario.deliver, store=store)
    assert second.result["diagnosis"]["remedy"] == "failover"
    assert len(second.delivered) == 1
    assert second.delivered != first.delivered

    again = run(scenario.program, _world("incident"), store=store)
    assert again.delivered == ()
    assert again.result["diagnosis"]["remedy"] == first.result["diagnosis"]["remedy"]


def test_a_refused_remedy_is_asked_again_with_the_reason():
    world = _world(
        "incident", llm=_answers(_settles("restart-the-database"), _settles("rollback"))
    )
    ran = run(SCENARIOS["incident"].program, world, SCENARIOS["incident"].deliver)
    assert ran.result["diagnosis"]["remedy"] == "rollback"
    assert world.calls["llm"] == 2


def test_a_remedy_refused_after_every_reprompt_never_reaches_a_person():
    world = _world("incident", llm=_answers(_settles("restart-the-database")))
    ran = run(SCENARIOS["incident"].program, world)
    assert ran.result["diagnosis"]["remedy"] is None
    assert ran.result["diagnosis"]["refused"] is not None
    assert ran.delivered == ()
    assert world.calls["llm"] == 1 + REPAIRS


def test_a_drill_that_never_settles_proposes_no_remedy():
    unsettled = {"finding": "still looking", "remedy": "rollback", "settled": False}
    world = _world("incident", llm=_answers(unsettled | {"follow": "And then?"}))
    ran = run(SCENARIOS["incident"].program, world)
    assert ran.result["diagnosis"]["remedy"] is None
    assert ran.delivered == ()
    assert world.calls["llm"] == incident.LEVELS + 1


def test_a_drill_that_settles_with_no_remedy_reaches_no_person():
    world = _world("incident", llm=_answers(_settles(incident.UNREMEDIED)))
    ran = run(SCENARIOS["incident"].program, world)
    assert ran.result["diagnosis"]["remedy"] is None
    assert ran.delivered == ()
    assert world.calls["remediate"] == 0


def test_a_reprompt_shows_the_model_why_its_answer_was_refused():
    prompts: list[str] = []
    model = _answers(_settles("restart-the-database"), _settles("rollback"))

    def recording(prompt: str) -> dict[str, object]:
        prompts.append(prompt)
        return model(prompt)

    run(
        SCENARIOS["incident"].program,
        _world("incident", llm=recording),
        SCENARIOS["incident"].deliver,
    )
    first, second = prompts
    assert "a remedy the runbook does not hold" not in first
    assert "a remedy the runbook does not hold" in second


def test_the_drills_next_question_reaches_the_model_as_data():
    prompts: list[str] = []
    base = SCENARIOS["incident"].world()

    def recording(prompt: str) -> object:
        prompts.append(prompt)
        return base.llm(prompt)

    run(SCENARIOS["incident"].program, replace(base, llm=recording), SCENARIOS["incident"].deliver)
    asked = SCENARIOS["incident"].world().llm(incident.START)["follow"]
    assert asked in prompts[1]
    assert DATA_MARK + " question" in prompts[1]


def test_a_runbook_id_that_cannot_name_an_event_is_never_offered():
    base = SCENARIOS["incident"].world()
    runbook = base.tools["runbook"]({}) | {"roll:back": "an id no key can hold"}
    world = replace(base, tools=dict(base.tools) | {"runbook": lambda args: runbook})
    world = replace(world, llm=_answers(_settles("roll:back"), _settles("rollback")))
    ran = run(SCENARIOS["incident"].program, world, SCENARIOS["incident"].deliver)
    assert ran.result["diagnosis"]["remedy"] == "rollback"


def test_an_alert_no_named_cause_fits_ends_at_the_diagnosis():
    world = _world("incident", judge=lambda op: {"pick": _choice(NO_MATCH)})
    ran = run(SCENARIOS["incident"].program, world)
    assert ran.result["diagnosis"]["cause"] == NO_MATCH
    assert ran.result["diagnosis"]["remedy"] is None
    assert ran.delivered == ()
    assert world.calls["llm"] == 0


def test_a_run_waiting_on_no_event_is_refused_rather_than_polled():
    def sleeps() -> Effect[BaseModel]:
        yield from sleep_until(datetime(2099, 1, 1, tzinfo=UTC))
        return incident.Approval(approved=False, by="nobody")

    with pytest.raises(RuntimeError, match="no event to answer"):
        run(sleeps, _world("incident"))


# --- launch --------------------------------------------------------------------------------


def test_a_launch_the_judgment_clears_ships_unattended():
    clear = {"ready": {"p": launch.SHIP_AT}, "blocker": _choice("docs")}
    world = _world("launch", judge=lambda op: clear)
    ran = run(SCENARIOS["launch"].program, world)
    assert ran.result["shipped"] is True
    assert ran.delivered == ()
    checks = {check: world.calls[check] for check in launch.CHECKS}
    assert checks == dict.fromkeys(launch.CHECKS, 1)
    assert world.calls["release"] == 1


def test_a_blocked_launch_waits_on_the_blockers_owner():
    scenario = SCENARIOS["launch"]
    world = scenario.world()
    ran = run(scenario.program, world, scenario.deliver)
    assert ran.result["blocker"] in launch.CHECKS
    assert len(ran.delivered) == 1
    assert ran.result["shipped"] is False
    assert world.calls["release"] == 0


def test_a_ruling_settles_only_the_build_it_was_given_for(tmp_path: Path):
    store = tmp_path / "shared.db"
    ship = launch.Ruling(ship=True, note="ship it")
    first = run(SCENARIOS["launch"].program, _world("launch"), lambda event: ship, store=store)
    rebuilt = partial(launch.launch, replace(CANDIDATE, build="rc4"))
    second = run(rebuilt, _world("launch"), SCENARIOS["launch"].deliver, store=store)
    assert first.result["shipped"] is True
    assert len(second.delivered) == 1
    assert second.delivered != first.delivered
    assert second.result["shipped"] is False


# --- voice ---------------------------------------------------------------------------------


def _reading(world: World) -> tuple[World, list[str]]:
    """`world`, with every page read it is asked recorded by its prompt."""
    reads: list[str] = []

    def model(prompt: str) -> object:
        if "Support tickets:" in prompt:
            reads.append(prompt)
        return world.llm(prompt)

    return replace(world, llm=model), reads


def test_the_radar_reads_each_page_once_and_settles():
    world, reads = _reading(SCENARIOS["voice"].world())
    ran = run(SCENARIOS["voice"].program, world)
    themes = ran.result["themes"]
    assert len(reads) == ceil(len(TICKETS) / voice.PAGE)
    assert ran.result["settled"] is True
    assert sum(themes.values()) == len(TICKETS)
    assert world.calls["judge"] == world.calls["file"] == len(themes)


def test_a_radar_that_never_settles_says_so():
    base = SCENARIOS["voice"].world()
    renamed = iter(range(voice.ROUNDS + 1))

    def restless(prompt: str) -> object:
        if "Merge themes" in prompt:
            return {"themes": {"theme " + str(next(renamed)): len(TICKETS)}}
        return base.llm(prompt)

    world = replace(base, llm=restless, judge=lambda op: {"pick": _choice("bug")})
    ran = run(SCENARIOS["voice"].program, world)
    assert ran.result["settled"] is False
    assert world.calls["file"] == len(ran.result["themes"])


def test_a_theme_of_no_named_kind_goes_to_triage():
    world = _world("voice", judge=lambda op: {"pick": _choice(NO_MATCH)})
    ran = run(SCENARIOS["voice"].program, world)
    assert set(ran.result["routed"].values()) == {"triage"}


@pytest.mark.parametrize("asking", ["Support tickets:", "Merge themes"])
def test_an_answer_refused_after_every_reprompt_fails_the_radar(asking):
    base = SCENARIOS["voice"].world()

    def garbled(prompt: str) -> object:
        return {"themes": "prose"} if asking in prompt else base.llm(prompt)

    with pytest.raises(RuntimeError, match="Unanswered"):
        run(SCENARIOS["voice"].program, replace(base, llm=garbled))


def test_an_empty_corpus_is_an_empty_radar():
    base = SCENARIOS["voice"].world()
    world = replace(base, tools=dict(base.tools) | {"tickets": lambda args: []})
    ran = run(SCENARIOS["voice"].program, world)
    assert ran.result == {"themes": {}, "settled": True, "routed": {}}
    assert world.calls["llm"] == 0
