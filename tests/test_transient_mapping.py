"""The provider→substrate error boundary, on ALL THREE call paths (retry consult, decision 1).

Only `structured_extract` mapped provider failures into `TransientError`. A blip on the ReAct
turn path or the JSON path was therefore a hard failure `retry_domain` never saw, while the same
blip on the extraction path retried cleanly — a boundary that covers one of three doors is not a
boundary. These pin all three, plus the 429 rule.

The 429 rule is a coupling, not a preference: a `RateLimited` is transient (it will pass) but
re-firing at once makes the limit worse, so `retry_domain` retries it ONLY with a `backoff`
configured. The two are one switch so that "retry a 429 immediately" cannot be spelled.

`openai` is an optional import here: `_transient_types()`/`_rate_limit_types()` return `()` when
the SDK is absent, so these tests construct the real exception types when available and skip the
provider-shaped rows when not.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from effective.domain import AskLLM
from effective.interpreters.openai import GPT5_NANO, json_complete, structured_extract
from effective.layers import RateLimited, TransientError, drive_through, retry_domain

openai = pytest.importorskip("openai", reason="the provider mapping needs the real SDK types")


def _connection_error() -> Exception:
    return openai.APIConnectionError(request=SimpleNamespace())


def _rate_limit_error() -> Exception:
    response = SimpleNamespace(status_code=429, headers={}, request=SimpleNamespace())
    return openai.RateLimitError("slow down", response=response, body=None)


class _Raises:
    """A client whose every completion call raises `exc`."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def parse(self, **kwargs):
        self.calls += 1
        raise self.exc

    def create(self, **kwargs):
        self.calls += 1
        raise self.exc


class _Turn(SimpleNamespace):
    pass


# --- door 1: structured_extract (the path that was already mapped) ------------------------


def test_structured_extract_maps_a_connection_error():
    client = _Raises(_connection_error())
    with pytest.raises(TransientError, match="APIConnectionError"):
        structured_extract(client, model="m", messages=[], response_format=_Turn, price=GPT5_NANO)


# --- door 2: json_complete (was UNMAPPED) -------------------------------------------------


def test_json_complete_maps_a_connection_error():
    client = _Raises(_connection_error())
    with pytest.raises(TransientError, match="APIConnectionError"):
        json_complete(client, model="m", messages=[], price=GPT5_NANO)


# --- door 3: the turn caller, covering BOTH subclasses via `__call__` (was UNMAPPED) -------


def test_the_turn_caller_maps_a_connection_error():
    from effective.interpreters.openai import OpenAITurnCaller

    caller = OpenAITurnCaller(_Raises(_connection_error()), system_prompt="s")
    with pytest.raises(TransientError, match="APIConnectionError"):
        caller(AskLLM(messages=[], response_schema=str))


def test_the_llamacpp_caller_maps_through_the_same_door():
    """One `_as_transient` in `__call__` covers every `_complete` subclass — the reason it lives
    there and not in each override."""
    from effective.interpreters.openai import LlamaCppTurnCaller

    caller = LlamaCppTurnCaller(_Raises(_connection_error()), system_prompt="s", model="local")
    with pytest.raises(TransientError, match="APIConnectionError"):
        caller(AskLLM(messages=[], response_schema=str))


# --- the 429 rule -------------------------------------------------------------------------


def test_a_429_maps_to_RateLimited_which_IS_a_TransientError():
    client = _Raises(_rate_limit_error())
    with pytest.raises(RateLimited) as ei:
        json_complete(client, model="m", messages=[], price=GPT5_NANO)
    assert isinstance(ei.value, TransientError)  # it is transient...


def test_retry_domain_does_NOT_re_fire_a_429_without_a_backoff():
    """...but re-firing at once makes the limit worse, so the default re-raises it unretried."""
    attempts = []

    def base(op):
        attempts.append(op)
        raise RateLimited("429")

    with pytest.raises(RateLimited):
        drive_through([retry_domain(attempts=3)], AskLLM(messages=[], response_schema=str), base)
    assert len(attempts) == 1  # tried once, not four times


def test_retry_domain_DOES_re_fire_a_429_with_a_backoff_and_sleeps_between():
    slept: list[float] = []
    attempts: list[Any] = []

    def base(op):
        attempts.append(op)
        raise RateLimited("429")

    with pytest.raises(RateLimited):
        drive_through(
            [retry_domain(attempts=2, backoff=lambda n: 0.5 * 2**n, sleep=slept.append)],
            AskLLM(messages=[], response_schema=str),
            base,
        )
    assert len(attempts) == 3  # 2 retries + the original
    assert slept == [0.5, 1.0]  # and it waited longer each time


def test_a_plain_transient_still_retries_immediately_by_default():
    """The 429 rule is scoped to 429s: a connection blip keeps its immediate retry."""
    calls = {"n": 0}

    def base(op):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientError("blip")
        return "ok"

    result = drive_through(
        [retry_domain(attempts=3)], AskLLM(messages=[], response_schema=str), base
    )
    assert result == "ok"
    assert calls["n"] == 3


def test_no_backoff_sleep_happens_after_the_final_attempt():
    """A backoff between attempts, not a pointless wait before giving up."""
    slept: list[float] = []

    def base(op):
        raise TransientError("blip")

    with pytest.raises(TransientError):
        drive_through(
            [retry_domain(attempts=2, backoff=lambda n: 1.0, sleep=slept.append)],
            AskLLM(messages=[], response_schema=str),
            base,
        )
    assert len(slept) == 2  # between the three attempts, never after the last


# --- the policy, tested directly -----------------------------------------------------------
#
# `wait_before_retry` is module-level precisely so the RULE can be read and checked without
# driving a generator through a trampoline. These rows are the decision table; the tests above
# are the integration.


@pytest.mark.parametrize(
    ("exc", "attempt", "attempts", "backoff", "expected"),
    [
        # exhausted: nothing retries, whatever the exception or config
        (TransientError("x"), 2, 2, None, None),
        (TransientError("x"), 2, 2, lambda n: 1.0, None),
        (RateLimited("x"), 5, 2, lambda n: 1.0, None),
        # no backoff configured: a plain transient re-fires at once...
        (TransientError("x"), 0, 2, None, 0.0),
        # ...but a 429 does not re-fire at all — there is nothing to wait with
        (RateLimited("x"), 0, 2, None, None),
        # backoff configured: both retry, after the configured delay
        (TransientError("x"), 0, 2, lambda n: 0.5 * 2**n, 0.5),
        (RateLimited("x"), 1, 2, lambda n: 0.5 * 2**n, 1.0),
    ],
)
def test_the_retry_policy_decision_table(exc, attempt, attempts, backoff, expected):
    from effective.layers import wait_before_retry

    assert wait_before_retry(exc, attempt, attempts, backoff) == expected


def test_the_policy_gives_ONE_answer_for_every_reason_to_stop():
    """Exhaustion and un-retryability collapse to the same `None`, which is what lets the layer
    stay mechanical — it never asks *why*, only *whether* and *how long*."""
    from effective.layers import wait_before_retry

    exhausted = wait_before_retry(TransientError("x"), 3, 3, None)
    unretryable = wait_before_retry(RateLimited("x"), 0, 3, None)
    assert exhausted is unretryable is None
