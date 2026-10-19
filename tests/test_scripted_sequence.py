"""The scripted-sequence fixtures answer one step twice in one scope, in order.

A plain table cannot: the recorder looks a step up by its authored name and its scope, so both
asks find one answer. The durable handler numbers the second ask, and a replay re-serves both
recorded answers without the script."""

import pytest
from _scripted import Scripted, Sequenced

from effective import Effect, ask_llm, scoped
from effective.contexts import RecordingCtx, ReplayCtx
from effective.handlers.durable import DurableHandler
from effective.handlers.recording import RecordingHandler
from effective.keys import compose_key


def asked_twice() -> Effect[list[str]]:
    def body() -> Effect[list[str]]:
        first = yield from ask_llm("react:turn", [], str)
        second = yield from ask_llm("react:turn", [], str)
        return [first, second]

    return (yield from scoped(compose_key(t"sub:child"), body))


def test_a_plain_table_answers_a_repeated_step_alike():
    handler = RecordingHandler({"react:turn": "bad"})
    assert handler.run(asked_twice) == ["bad", "bad"]


def test_a_sequence_answers_a_repeated_step_in_order_and_is_consumed():
    table = Sequenced({"sub:child;react:turn": ["bad", "good"]})
    assert RecordingHandler(table).run(asked_twice) == ["bad", "good"]
    assert table.unasked == {}


def test_a_step_past_the_sequence_finds_nothing_canned():
    table = Sequenced({"react:turn": ["only"]})
    with pytest.raises(KeyError, match="no canned response"):
        RecordingHandler(table).run(asked_twice)


def test_a_sequence_says_what_no_step_asked_for():
    table = Sequenced({"react:turn": ["bad", "good", "spare"]})
    RecordingHandler(table).run(asked_twice)
    assert table.unasked == {"react:turn": ["spare"]}


def test_the_durable_handler_numbers_the_repeat_and_replays_both_without_the_script():
    ctx, domain = RecordingCtx(), Scripted(["bad", "good"])
    assert DurableHandler(ctx, domain).run(asked_twice) == ["bad", "good"]
    assert domain.answers == []
    assert [key.stored() for key in ctx.log] == [
        "sub:child;step;react:turn",
        "sub:child;step;react:turn#2",
    ]
    replayed = DurableHandler(ReplayCtx(ctx.log), Scripted([])).run(asked_twice)
    assert replayed == ["bad", "good"]
