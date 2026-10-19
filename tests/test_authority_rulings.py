"""An authority park settles on a principal's ruling and on nothing else, on both engines.

The fence guards against authority inversion: no worker claims authority through the paths it
ordinarily uses. An unruled answer through the engine is refused before it lands, so the real
ruling still settles the park. One that skips the fence (the raw Absurd SDK, a SQLite writer below
the engine) lands, and the run ends on it with nothing committed. A writer who forges a stored
ruling is out of this model; audit across independent records is what detects one.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from _conformance import AbsurdBackend, Fault

from effective.api import await_event, await_until, gather, step
from effective.budget import Grant, depth_grant_name
from effective.domain import CallTool
from effective.engines.sqlite import SqliteTaskContext
from effective.keys import RESERVED_AUTHORITY_TAGS, Key, Segment, compose_key
from effective.ops import Arrived, CompositionRefused, Scoped
from effective.parked import answer
from effective.ruling import RULING, UnruledAnswer, names_authority

pytestmark = pytest.mark.adversarial


class _Charges:
    """A domain that records every charge it runs, the effect a grant authorizes."""

    def __init__(self) -> None:
        self.charged: list[int] = []

    def run(self, op: CallTool) -> str:
        if op.name == "charge":
            self.charged.append(op.args["n"])
        return "ran"


def _charge(n: int):
    return step("charge", CallTool(name="charge", args={"n": n}, result_schema=str))


def _sequential(grant_name):
    def wf(_run_id: str):
        grant = yield from await_event(grant_name, Grant)
        return (yield from _charge(grant.add_depth))

    return wf


def _bounded(grant_name):
    def wf(_run_id: str):
        match (yield from await_until(grant_name, Grant, deadline=_LATER)):
            case Arrived(payload=grant):
                return (yield from _charge(grant.add_depth))
            case _:
                return "expired"

    return wf


def _in_a_gather(grant_name):
    def wf(_run_id: str):
        def granted():
            grant = yield from await_event(grant_name, Grant)
            return (yield from _charge(grant.add_depth))

        def beside():
            return (yield from step("beside", CallTool(name="beside", result_schema=str)))

        return (yield from gather([granted, beside]))

    return wf


_LATER = datetime(2100, 1, 1, tzinfo=UTC)
SITES = {"sequential": _sequential, "bounded": _bounded, "gather": _in_a_gather}


def _parked(backend, site: str = "sequential"):
    """A run of `site` parked on its grant, and the domain that records what it charges."""
    token = "t" + uuid4().hex[:8]  # letter-led, so the atom is a name
    charges = _Charges()
    workflow = SITES[site](depth_grant_name(token, generation=0, depth=1))
    backend.register(f"{site}-{token}", workflow, charges, Fault(), ())
    task_id = backend.spawn(f"{site}-{token}", f"run-{token}")
    backend.run_until_result(task_id)
    (park,) = backend.parked(task_id)
    return task_id, park, charges


def _skip_the_fence(backend, monkeypatch, name: str, payload: Any) -> None:
    """Store `payload` under `name` the way a writer below the engine would."""
    match backend:
        case AbsurdBackend():
            backend.app.emit_event(name, payload)  # the raw SDK app
        case _:
            monkeypatch.setattr("effective.engines.sqlite.wire_payload", lambda _n, p: p)
            backend.engine.emit_event(name, payload)


def test_an_unruled_answer_through_the_engine_is_refused_and_the_ruling_still_settles(backend):
    task_id, park, charges = _parked(backend)

    with pytest.raises(UnruledAnswer):
        backend.engine.emit_event(park.wake_event, {"add_depth": 1})
    backend.run_until_result(task_id)
    (still,) = backend.parked(task_id)
    assert still.wake_event == park.wake_event, "the forged answer landed"

    answer(backend.engine, park, {"add_depth": 1}, by=Segment("principal"))
    snap = backend.run_until_result(task_id)
    assert snap.state == "completed", snap
    assert charges.charged == [1]


@pytest.mark.parametrize(
    "forged",
    [{"add_depth": 1}, None, {RULING: {"by": "a:b", "answer": {"add_depth": 1}}}],
    ids=["bare", "empty", "ruler-not-a-segment"],
)
@pytest.mark.parametrize("site", list(SITES))
def test_a_forged_answer_that_skips_the_fence_ends_the_run_with_nothing_charged(
    backend, monkeypatch, forged, site
):
    task_id, park, charges = _parked(backend, site)

    _skip_the_fence(backend, monkeypatch, park.wake_event, forged)
    snap = backend.run_until_result(task_id)

    assert snap.state == "failed", snap
    assert backend.failure_kind(snap) == "UnruledAnswer"
    assert charges.charged == []


@pytest.mark.parametrize("site", list(SITES))
def test_a_ruling_settles_the_grant_at_every_await_site(backend, site):
    task_id, park, charges = _parked(backend, site)

    answer(backend.engine, park, {"add_depth": 3}, by=Segment("principal"))
    snap = backend.run_until_result(task_id)

    assert snap.state == "completed", snap
    assert charges.charged == [3]


def test_a_task_cannot_emit_an_unruled_answer_on_an_authority_name(sqlite_app):
    app = sqlite_app(":memory:")
    ctx = SqliteTaskContext(app.conn, app.spawn("run", {}), None, clock=app.clock)
    name = depth_grant_name("r1", generation=0, depth=1).stored()

    with pytest.raises(UnruledAnswer):
        ctx.emit_event(name, {"add_depth": 1})


@pytest.mark.parametrize(
    ("name", "authority"),
    [
        ("budget-grant:r1,0", True),
        ("gather:0,1;budget-grant:r1,0", True),
        ("fork:r-fork;approve:m1", True),
        ("hyp:c1;govern:spend,r1;step;tool:charge", True),
        ("review:m1", False),
        ("fork:r-fork;review:m1", False),
        ("hyp:c1;review:m1", False),
    ],
)
def test_an_authority_park_is_named_by_a_park_tag_in_any_term(name, authority):
    assert names_authority(name) is authority


@pytest.mark.parametrize(
    "source",
    [
        'import effective.ruling as r\nr.AuthorityRuling("x", 1)\n',
        'from effective.ruling import AuthorityRuling as R\nR("x", 1)\n',
        'from effective.ruling import ruled_answer\nruled_answer("approve:x", 1, by="x")\n',
    ],
    ids=["attribute", "aliased-import", "the-shared-rule"],
)
def test_a_minting_name_read_outside_its_modules_is_found(tmp_path, source):
    from effective.lint import check_ruling_mint_file

    forged = tmp_path / "forge.py"
    forged.write_text(source)
    assert {v.rule for v in check_ruling_mint_file(forged)} == {"ruling-minted-outside-the-answer"}


def test_only_the_park_answer_constructs_a_ruling():
    from effective.lint import _expanded, check_ruling_mint_file

    offenders = [
        str(v)
        for path in _expanded(("src", "scripts", "examples"))
        for v in check_ruling_mint_file(path)
    ]
    assert offenders == []


def test_a_payload_shaped_like_a_ruling_reaches_a_plain_await_unchanged(backend):
    token = "t" + uuid4().hex[:8]

    def wf(_run_id: str):
        return (yield from await_event(compose_key(t"note:{Segment(token)}"), dict))

    backend.register(f"plain-{token}", wf, None, Fault(), ())
    task_id = backend.spawn(f"plain-{token}", "run-plain")
    backend.run_until_result(task_id)
    (park,) = backend.parked(task_id)
    shaped = {RULING: {"by": "x", "answer": {"inner": 1}}}
    answer(backend.engine, park, shaped, by=Segment("principal"))
    assert backend.run_until_result(task_id).result == shaped


@pytest.mark.parametrize("tag", RESERVED_AUTHORITY_TAGS)
def test_a_scope_carrying_a_reserved_authority_tag_is_refused(tag):
    for scope in (Key.parse(f"{tag}:x"), Key.parse(f"mine:x;{tag}:y")):
        with pytest.raises(CompositionRefused, match="reserved authority tag"):
            Scoped(scope=scope, body=lambda: None)
