"""`serve` + `retry_domain`: the domain-seam retry.

Retry lives BELOW the seam: it re-invokes the domain call inside one op interpretation, so it
recovers transient failures on the live tail, never fires on replay, and preserves `run_metered`
(the budget cap stays live), which a `compose_domain` wrapper would hide.
"""

from typing import Any

import pytest

from effective.api import step
from effective.budget import MeasuredBudget
from effective.cost import Usage, serve
from effective.domain import AskLLM
from effective.fork import measured_drive
from effective.keys import Key
from effective.layers import TransientError, retry_domain

STEP = 0.001
R = "serve-test"


class FlakyDomain:
    """Raises `TransientError` on the first `fail_times` calls, then returns ('ok', cost)."""

    def __init__(self, fail_times: int = 2, cost: float = STEP):
        self.fail_times = fail_times
        self.cost = cost
        self.attempts = 0

    def run_metered(self, op):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise TransientError(f"boom {self.attempts}")
        return "ok", Usage(cost=self.cost)


def _one_ask():
    return (yield from step("ask", AskLLM(messages=[], response_schema=str)))


def _budget(limit: float = 10 * STEP) -> MeasuredBudget:
    return MeasuredBudget(overall=limit, run_id=R, on_exhaust="park")


def test_serve_retry_domain_recovers_a_transient_call():
    flaky = FlakyDomain(fail_times=2)
    tail = measured_drive(_one_ask, _budget(), serve(retry_domain(2), base=flaky), grants={})
    assert tail.result == "ok"
    assert flaky.attempts == 3  # 2 transient fails + 1 success (retry re-invoked the call)
    assert tail.usage.cost == pytest.approx(STEP)  # only the successful call accrues; fails free


def test_serve_retry_domain_exhaustion_raises_transient():
    always = FlakyDomain(fail_times=99)
    with pytest.raises(TransientError):
        measured_drive(_one_ask, _budget(), serve(retry_domain(1), base=always), grants={})
    assert always.attempts == 2  # 1 try + 1 retry, then it gives up (attempts + 1 total)


def test_serve_preserves_run_metered_so_the_cap_still_fires():
    # $0 budget must park before the ask — proof the trip sees the metered call THROUGH serve
    # (a compose_domain wrapper would expose only `.run` and the cap would go inert).
    served = serve(retry_domain(2), base=FlakyDomain(fail_times=0))
    assert hasattr(served, "run_metered")
    # metered-shaped: `.run` exists only to FAIL LEGIBLY if a serve stack reaches DurableHandler's
    # non-metered arm (T1) — a named TypeError, not a mid-checkpoint AttributeError. ty hides it
    # behind the MeteredDomain return (good); DurableHandler reaches it duck-typed.
    served_any: Any = served
    with pytest.raises(TypeError, match="metered-only"):
        served_any.run(AskLLM(messages=[], response_schema=str))
    tail = measured_drive(_one_ask, _budget(limit=0.0), served, grants={})
    assert tail.tripped_at == Key.parse(f"budget-grant:{R},0")
    assert tail.trace == []


def test_retry_domain_never_fires_on_replay():
    flaky = FlakyDomain(fail_times=1)
    served = serve(retry_domain(2), base=flaky)
    tail = measured_drive(_one_ask, _budget(), served, grants={})
    assert tail.result == "ok"
    before = flaky.attempts  # 2 (1 fail + 1 success)
    # re-drive REPLAYING the recorded prefix — the domain is not called, so retry cannot fire
    replay = measured_drive(_one_ask, _budget(), served, grants={}, recorded=tail.trace)
    assert flaky.attempts == before
    assert replay.result == "ok"


def test_serve_with_no_services_returns_the_base_untouched():
    base = FlakyDomain(fail_times=0)
    assert serve(base=base) is base


def test_serve_rejects_an_op_layer_service_and_names_the_domain_twin():
    """The belt for consumers outside the ty gate (T4): the op-seam `retry` runs perfectly inside
    serve at runtime (identical body), so serve reads the __effective_layer__ marker and refuses,
    pointing at retry_domain."""
    from effective.layers import retry

    # ty catches serve(retry(2)) statically (the in-repo win); this tests the RUNTIME belt for a
    # consumer without a ty gate, so pass it as Any to reach the marker check.
    op_service: Any = retry(2)
    with pytest.raises(TypeError, match="retry_domain") as exc:
        serve(op_service, base=FlakyDomain(fail_times=0))
    # The message names the factory (qualname `retry.<locals>.run`), not the bare closure 'run'.
    assert "retry" in str(exc.value).split("is an")[0]


def test_serve_shape_guards_the_pipe_against_a_bare_value():
    """T2: a service/base that returns a bare value (not the (result, usage) pair) would let
    `result, usage = ...` silently corrupt — the guard turns it into a named error."""

    class BareDomain:  # a mis-written metered domain: returns the result, drops the usage
        def run_metered(self, op):
            return "ok"  # should be ("ok", Usage(...))

    with pytest.raises(TypeError, match="thread the metered"):
        measured_drive(_one_ask, _budget(), serve(retry_domain(2), base=BareDomain()), grants={})


def test_serve_retry_traced_spans_each_attempt_with_an_id_of_its_own():
    """T5/T6 pin: composed retry_domain(outer) + traced(inner) spans EACH attempt — a failed one
    as ERROR, the success as OK. Each attempt has its own span id, so a backend that keys on span
    ids keeps both. Pins the composed order so a reorder is a red diff, not a silent semantic
    change."""
    from effective.telemetry import Span, traced

    spans: list[Span] = []
    domain = serve(retry_domain(2), traced(spans.append, session_id="s"), base=FlakyDomain(1))
    tail = measured_drive(_one_ask, _budget(), domain, grants={})
    assert tail.result == "ok"
    assert [s.status for s in spans] == ["ERROR", "OK"]  # each attempt spanned
    assert spans[0].span_id != spans[1].span_id  # one id per attempt
    assert [s.attempt for s in spans] == [0, 1]


def test_structured_extract_maps_transient_provider_error_to_transient_error():
    import httpx
    import openai

    from effective.interpreters.openai import GPT5_NANO, structured_extract

    class _Completions:
        def parse(self, **_):
            raise openai.APIConnectionError(
                request=httpx.Request("POST", "https://api.openai.com")
            )

    class _Client:
        class chat:
            completions = _Completions()

    with pytest.raises(TransientError):
        structured_extract(
            _Client(), model="gpt-5-nano", messages=[], response_format=str, price=GPT5_NANO
        )


# --- T1: the assembly-time reachability check ------------------------------------------


class _NullCtx:
    """The smallest `TaskContext` a constructor check needs: nothing is ever driven."""

    def step(self, name: Key, thunk, /) -> Any:
        return thunk()

    def await_event(self, name: Key, /) -> Any:
        raise AssertionError("not driven")

    def sleep_until(self, when, /, *, name: Key | None = None) -> None:
        raise AssertionError("not driven")


class _PlainDomain(FlakyDomain):
    """A domain with BOTH arms — the ordinary case a serve stack deliberately lacks."""

    def run(self, op) -> Any:
        return "ok"


def test_a_metered_only_domain_under_V0_is_rejected_at_ASSEMBLY():
    """The T1 hardening beyond the legible `.run` message: under `Contract.V0` EVERY AskLLM takes
    the non-metered arm, so a serve stack cannot work for ANY workflow — a fact known at
    construction. Refuse there, not after the first step has committed."""
    from effective.cost import Contract, serve
    from effective.handlers.durable import DurableHandler

    served = serve(retry_domain(1), base=FlakyDomain(fail_times=0))
    with pytest.raises(TypeError, match="metered-only"):
        # `ty` already rejects this statically (`MeteredDomain` has no `run`, so it is not a
        # `DomainInterpreter`) — which is WHY the assembly check exists: it is the belt for a
        # consumer outside the ty gate, the same rationale as serve()'s @op_layer rejection.
        # Suppressed deliberately: the statically-invalid call IS the case under test.
        DurableHandler(ctx=_NullCtx(), domain=served, contract=Contract.V0)  # ty: ignore[invalid-argument-type]


def test_a_metered_only_domain_under_V1_is_allowed_to_assemble():
    """Deliberately partial: under V1 whether it works depends on the workflow's ops (an
    all-AskLLM run is fine, one CallTool is not), which assembly cannot know. That case keeps
    the runtime `.run` fence — legible, but late."""
    from effective.cost import Contract, serve
    from effective.handlers.durable import DurableHandler

    served = serve(retry_domain(1), base=FlakyDomain(fail_times=0))
    # (statically invalid for the same reason; the runtime path is what is under test)
    DurableHandler(ctx=_NullCtx(), domain=served, contract=Contract.V1)  # ty: ignore[invalid-argument-type]


def test_an_ordinary_domain_is_unaffected_under_either_contract():
    from effective.cost import Contract
    from effective.handlers.durable import DurableHandler

    for contract in (Contract.V0, Contract.V1):
        DurableHandler(ctx=_NullCtx(), domain=_PlainDomain(fail_times=0), contract=contract)
