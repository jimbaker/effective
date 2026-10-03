"""Infra-free unit tests for the permission cascade logic (no Postgres).

Proves the cascade's control flow directly through `drive_through`, with `rules`
tiers only (the `human` tier needs a handler to interpret its injected await — see
`test_replay_permission.py` for the durable suspend/resume proof). Covers: Allow
forwards, Deny raises `Refused`, Escalate defers to the next tier, all-Escalate
fails closed via the default, and first-decisive-wins short-circuits.
"""

from dataclasses import dataclass

import pytest

from effective import RecordingHandler, call_tool
from effective.keys import Key
from effective.layers import drive_through
from effective.ops import AppendLedgerRow, LedgerRow, Step
from effective.permission import Allow, Deny, Escalate, Refused, cascade, human, rules

OP = AppendLedgerRow(row=LedgerRow(event_id=Key.parse("e1"), kind="commitment"))


@dataclass
class _Approval:
    """An approval-shaped value (the human tier duck-types .decision/.rationale).
    Local to keep this substrate test free of any domain model."""

    decision: str
    rationale: str = ""


def _base(op):
    return "forwarded"


def test_allow_forwards_the_op():
    gate = cascade([rules(lambda op: Allow())])
    assert drive_through([gate], OP, _base) == "forwarded"


def test_deny_raises_refused_with_op_and_reason():
    gate = cascade([rules(lambda op: Deny("nope"))])
    with pytest.raises(Refused) as ei:
        drive_through([gate], OP, _base)
    assert ei.value.reason == "nope"
    assert ei.value.op is OP


def test_escalate_defers_to_the_next_tier():
    gate = cascade([rules(lambda op: Escalate()), rules(lambda op: Allow())])
    assert drive_through([gate], OP, _base) == "forwarded"


def test_all_escalate_fails_closed_via_default():
    gate = cascade([rules(lambda op: Escalate())])  # default = Deny
    with pytest.raises(Refused):
        drive_through([gate], OP, _base)


def test_first_decisive_verdict_short_circuits():
    seen: list[object] = []

    def tier2(op):
        seen.append(op)
        return Allow()

    gate = cascade([rules(lambda op: Deny("blocked")), rules(tier2)])
    with pytest.raises(Refused):
        drive_through([gate], OP, _base)
    assert seen == []  # tier 2 never consulted


# --- the human tier in-memory: RecordingHandler now handles a layer-injected await ---


def _act_workflow():
    return (yield from call_tool("act", {}, dict))


def _escalate_act(op):
    if isinstance(op, Step) and op.name == "tool:act":
        return Escalate("needs sign-off")
    return Allow()


def test_human_approve_forwards_through_recording_handler():
    # canned approval -> the injected await resolves -> Allow -> the op forwards
    handler = RecordingHandler(
        responses={"tool:act": {"ok": True}, "approve;step;tool:act": _Approval("approve")},
        op_layers=[cascade([rules(_escalate_act), human(_Approval)])],
    )
    assert handler.run(_act_workflow) == {"ok": True}


def test_human_reject_refuses_through_recording_handler():
    handler = RecordingHandler(
        responses={"tool:act": {"ok": True}, "approve;step;tool:act": _Approval("reject", "no")},
        op_layers=[cascade([rules(_escalate_act), human(_Approval)])],
    )
    with pytest.raises(Refused) as ei:
        handler.run(_act_workflow)
    assert ei.value.reason == "no"


def test_recording_handler_gives_a_clear_error_when_it_cannot_park_a_layer_await():
    # no canned approval: the in-memory recorder can't suspend-from-mid-layer (no call/cc)
    handler = RecordingHandler(
        responses={"tool:act": {"ok": True}},
        op_layers=[cascade([rules(_escalate_act), human(_Approval)])],
    )
    with pytest.raises(NotImplementedError, match="cannot suspend"):
        handler.run(_act_workflow)


# --- allow_table: the data-driven Allow|Escalate tier (B-V0) ----------------

from effective.handlers.base import op_key  # noqa: E402
from effective.permission import PermitPolicy, allow_table  # noqa: E402


def test_allow_table_settles_a_matching_op():
    permit = allow_table(
        PermitPolicy(allow=(op_key(OP).stored(),))
    )  # exact key is a prefix of itself
    assert drive_through([cascade([permit])], OP, _base) == "forwarded"


def test_allow_table_escalates_a_miss_and_fails_closed_alone():
    # a miss -> Escalate; with no next tier the cascade default (FAIL_CLOSED) denies
    miss = allow_table(PermitPolicy(allow=("no-such-prefix",)))
    with pytest.raises(Refused):
        drive_through([cascade([miss])], OP, _base)


def test_allow_table_escalates_to_the_next_tier():
    # allow_table never denies: a miss defers to the following tier (here, an Allow rule)
    miss = allow_table(PermitPolicy(allow=("no-such-prefix",)))
    gate = cascade([miss, rules(lambda op: Allow())])
    assert drive_through([gate], OP, _base) == "forwarded"


def test_permit_policy_is_data_that_round_trips():
    p = PermitPolicy(allow=("tool:read_file", "tool:run_tests"))
    assert PermitPolicy.model_validate(p.model_dump()) == p  # checkpointable -> improve-able
    assert p.permits(Key.parse("tool:read_file,0"))
    assert not p.permits(Key.parse("tool:apply_edit,0"))


# --- the ADVERSARY: can one approval authorize a second, larger charge? --------------------
#
# Written as an attack rather than as coverage, because that is the only framing under which the
# hazard is visible. "Does the tier park?" passes on defective code. "Does approving a $5 charge
# also authorize a $5,000,000 one?" does not.
#
# `op_key` is not occurrence-injective: a `Step` carries the author's bare name, so an agent
# that calls one tool twice yields two ops with one key. The engines restore the missing
# coordinate on the CHECKPOINT axis by suffixing `name#k`; the AUTHORITY axis must restore it
# too, or one delivered approval settles every later occurrence, and a drain loop wires this
# tier into a deployment.


def _charge(amount: int) -> Step:
    from effective.domain import CallTool

    return Step(
        name="tool:charge_card",
        op=CallTool(name="charge_card", args={"amount": amount}, result_schema=dict),
    )


def _park_name(tier, op):
    """The name the tier parks on — the injected `AwaitEvent`, taken at its yield."""
    return next(tier(op)).name


@pytest.mark.adversarial
def test_one_approval_does_not_authorize_a_second_charge_of_the_same_tool():
    """The attack: approve $5, then try to ride that approval into $5,000,000.

    Driven at the tier under a `run_scope` rather than through a handler, because the recorder
    cannot suspend on a layer-injected await (this module's docstring) and the property under
    attack is the NAME, which is decided before any suspend."""
    from effective.layers import run_scope

    tier = human(_Approval)
    with run_scope():
        first = _park_name(tier, _charge(5))
        second = _park_name(tier, _charge(5_000_000))

    assert first != second, (
        "two distinct charges of one tool park on ONE name, so a single delivered approval "
        "settles both: the $5 / $5,000,000 class, live wherever a drain loop wires this tier "
        "into a deployment"
    )
    assert first.stored() == "approve;step;tool:charge_card"
    assert second.stored() == "approve;step;tool:charge_card#2"


@pytest.mark.adversarial
def test_the_first_occurrence_name_carries_no_suffix():
    """Occurrence naming must not orphan recorded runs: `Key.occurrence` returns `self` at
    count 1, so a name that occurs once is unchanged and every park already in a durable store
    still resolves."""
    from effective.layers import run_scope

    with run_scope():
        only = _park_name(human(_Approval), _charge(5))
    assert only.stored() == "approve;step;tool:charge_card", "no `#1` suffix"


@pytest.mark.adversarial
def test_the_count_is_per_RUN_not_per_tier_object():
    """A closure would carry task A's count into task B, so the count lives in
    `layer_run_state` (per `run()`), which is also what
    makes the name a REPLAY recomputes match the name the original parked on."""
    from effective.layers import run_scope

    tier = human(_Approval)  # one tier object, as a registration-time construction would be
    with run_scope():
        run_a = _park_name(tier, _charge(5))
    with run_scope():
        run_b = _park_name(tier, _charge(5))
    assert run_a == run_b == Key.parse("approve;step;tool:charge_card"), (
        "the second run inherited the first run's occurrence count"
    )


@pytest.mark.adversarial
def test_a_gated_sleep_parks_on_its_position_not_on_its_type_name():
    """The attack: gate two sleeps and try to settle both with one approval.

    A sleep is in `LAYERED_OPS`, so it reaches every permission tier — but it has no identity of
    its own, and the local fallback a tier could reach for is `type(op).__name__`. That is
    `"SleepUntil"` for every sleep in the run: one name, so one delivered approval authorizes
    every later sleep. `placed_key` reads the coordinate the walk assigned instead.

    Driven under `placing` rather than a handler, for the reason the module docstring gives: the
    recorder cannot suspend on a layer-injected await, and the property under attack is the NAME,
    decided before any suspend. `placing` is the same rule every drive loop applies."""
    from datetime import UTC, datetime

    from effective.handlers.base import placed_key, placing
    from effective.keys import FramePosition
    from effective.layers import run_scope
    from effective.ops import SleepUntil

    wake = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
    tier = human(_Approval)
    position = FramePosition()  # one frame, so the two sleeps are its 0th and 1st

    with run_scope():
        names = []
        for _ in range(2):
            op = SleepUntil(when=wake)
            with placing(op, position):
                names.append(_park_name(tier, op))

    first, second = names
    assert first != second, f"two gated sleeps parked on one name: {names}"
    assert first.stored() == "approve;sleep:0"
    assert second.stored() == "approve;sleep:1"
    assert "SleepUntil" not in first.stored(), "a type name is not an identity"

    # Off the walk there is no coordinate to read, so this refuses rather than inventing one.
    with pytest.raises(ValueError, match="no standalone op key"):
        placed_key(SleepUntil(when=wake))
