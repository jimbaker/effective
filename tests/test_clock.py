"""A held clock answers every reader it holds with one instant, and moves only forward."""

import datetime as dt

import psycopg
import pytest
from _clock import HeldClock
from _durable import DSN, absurd, pg_ready

from effective.handlers import base

AN_HOUR = dt.timedelta(hours=1)


def test_a_held_clock_moves_forward_and_refuses_to_step_back(monkeypatch):
    start = dt.datetime(2026, 9, 21, 12, 0, tzinfo=dt.UTC)
    clock = HeldClock(start).on_races(monkeypatch)
    assert base.race_time() == start.timestamp()

    clock.advance(AN_HOUR)
    clock.advance_to(start + AN_HOUR)
    assert (clock.at, base.race_time()) == (start + AN_HOUR, (start + AN_HOUR).timestamp())
    with pytest.raises(AssertionError, match="moves forward"):
        clock.advance_to(start)
    with pytest.raises(AssertionError, match="moves forward"):
        clock.advance(-AN_HOUR)
    assert clock.at == start + AN_HOUR


def _fake_now(session) -> str:
    return session.execute("SELECT current_setting('absurd.fake_now', true)").fetchone()[0]


@pytest.mark.skipif(not pg_ready(), reason="needs Postgres/Absurd (just pgt-up)")
def test_a_held_absurd_clock_moves_the_sdk_and_every_held_session_together():
    import absurd_sdk

    engine, start = absurd(), dt.datetime.now(dt.UTC) + AN_HOUR
    with psycopg.connect(DSN, autocommit=True) as own:
        with HeldClock(start).on_absurd(engine) as clock:
            clock.hold(own)
            later = start + AN_HOUR
            clock.advance_to(later)
            assert absurd_sdk._get_current_time() == later
            for session in (engine.app._conn, own):
                [(server,)] = session.execute("SELECT absurd.current_time()").fetchall()
                assert server == later
        assert {_fake_now(engine.app._conn), _fake_now(own)} == {""}
        assert abs(absurd_sdk._get_current_time() - dt.datetime.now(dt.UTC)) < AN_HOUR


@pytest.mark.skipif(not pg_ready(), reason="needs Postgres/Absurd (just pgt-up)")
def test_a_held_absurd_clock_starts_at_or_after_real_time():
    engine = absurd()
    early = HeldClock(dt.datetime.now(dt.UTC) - AN_HOUR)
    with pytest.raises(AssertionError, match="at or after real time"), early.on_absurd(engine):
        pass


@pytest.mark.skipif(not pg_ready(), reason="needs Postgres/Absurd (just pgt-up)")
def test_a_held_absurd_clock_that_fails_to_hold_leaves_the_sdk_on_real_time():
    import absurd_sdk

    engine = absurd()
    engine.app._conn.close()
    with (
        pytest.raises(psycopg.OperationalError),
        HeldClock(dt.datetime.now(dt.UTC) + AN_HOUR * 24 * 365).on_absurd(engine),
    ):
        pass
    assert abs(absurd_sdk._get_current_time() - dt.datetime.now(dt.UTC)) < AN_HOUR


@pytest.mark.skipif(not pg_ready(), reason="needs Postgres/Absurd (just pgt-up)")
def test_a_held_session_closed_inside_the_hold_lets_the_tests_own_error_through():
    engine, own = absurd(), psycopg.connect(DSN, autocommit=True)

    def fails_holding() -> None:
        with HeldClock(dt.datetime.now(dt.UTC) + AN_HOUR).on_absurd(engine) as clock:
            clock.hold(own)
            own.close()
            raise KeyError("the test's own failure")

    with pytest.raises(KeyError, match="the test's own"):
        fails_holding()


@pytest.mark.skipif(not pg_ready(), reason="needs Postgres/Absurd (just pgt-up)")
def test_an_inner_hold_releases_only_its_own_engine():
    outer, inner = absurd(), absurd()
    with HeldClock(dt.datetime.now(dt.UTC) + AN_HOUR).on_absurd(outer) as clock:
        with clock.on_absurd(inner):
            pass
        clock.advance(AN_HOUR)
        [(server,)] = outer.app._conn.execute("SELECT absurd.current_time()").fetchall()
        assert server == clock.at
    assert _fake_now(inner.app._conn) == _fake_now(outer.app._conn) == ""


@pytest.mark.skipif(not pg_ready(), reason="needs Postgres/Absurd (just pgt-up)")
def test_a_held_session_autocommits_so_no_rollback_drops_its_clock():
    with psycopg.connect(DSN) as session, pytest.raises(AssertionError, match="autocommits"):
        HeldClock(dt.datetime.now(dt.UTC) + AN_HOUR).hold(session)
