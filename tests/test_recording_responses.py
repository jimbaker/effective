"""`RecordingHandler(responses=…)` as a SEAM — a Mapping that computes, not a table.

The handler consults `responses` as a plain Mapping, which is what lets a fixture answer by
reading the PLACEMENT off the key it is asked for (`_funnel`, `_cart`, `_mcts`, `_coding` all do)
instead of spelling nineteen fully-qualified keys. That seam has a contract, and until this file
existed nothing stated it: it was exercised only through fixtures that happened to satisfy it.
"""

from collections.abc import Mapping

import pytest

from effective.api import Effect, ask_llm, step
from effective.domain import CallTool
from effective.handlers.recording import RecordingHandler


class Computed(Mapping[str, object]):
    """A responder with no items at all — every answer is derived from the key it is asked for.

    `__len__` is 0 and `__iter__` is empty, and that is the HONEST self-report: nothing here can
    be answered without knowing which execution is asking, so there is no bare name to enumerate
    and enumerating one would hand a caller a `KeyError`."""

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def __getitem__(self, key: str) -> object:
        return f"answered {key}"


def _program() -> Effect[str]:
    return (yield from ask_llm("ask", "anything", str))


def _tool_program() -> Effect[str]:
    return (yield from step("probe", CallTool(name="probe", args={}, result_schema=str)))


def test_a_responder_reporting_itself_EMPTY_is_still_consulted():
    """The defect this file was written for. `responses if responses else {}` discarded any
    Mapping whose `len()` was 0 — so a responder that computes every answer and enumerates none
    was replaced by an empty dict, and the handler raised *no canned response* for an op it would
    have answered. The comment directly above that line promised the seam; the line took it away.

    Reddens when: the guard goes back to truthiness. Verified — with `responses if responses
    else {}` restored, this raises `KeyError: no canned response for step 'ask'`."""
    handler = RecordingHandler(responses=Computed())
    assert handler.run(_program) == "answered ask"


def test_a_dict_is_copied_so_a_later_mutation_cannot_leak_in():
    """The other half of the same line, and the reason it is not simply `responses or {}`: a
    caller's dict is the caller's, and 111 fixtures build one inline."""
    table = {"ask": "first"}
    handler = RecordingHandler(responses=table)
    table["ask"] = "mutated after construction"
    assert handler.run(_program) == "first"


def test_a_computing_mapping_is_used_AS_GIVEN_rather_than_flattened():
    """A `dict(...)` copy would collapse a computing responder to whatever it enumerated — which
    for `Computed` is nothing at all, and for `_coding.Answers` would be nothing too.

    Asserted by identity, because "was not copied" is the property; a value check would pass on a
    copy that happened to contain the right entry."""
    responder = Computed()
    assert RecordingHandler(responses=responder).responses is responder


def test_no_responses_at_all_still_builds_and_raises_on_the_first_op():
    """`None` is the case the fallback exists for, and it must stay distinguishable from an empty
    computing Mapping — which is the whole point of testing `is not None`."""
    handler = RecordingHandler()
    assert handler.responses == {}
    with pytest.raises(KeyError, match="no canned response"):
        handler.run(_program)


def test_the_seam_reaches_a_call_tool_op_too():
    """Not only `ask_llm`: the lookup is the Step arm, so a tool call reaches the same responder.
    Named because a fixture that answered one and not the other would look correct until the
    first tool op."""

    assert RecordingHandler(responses=Computed()).run(_tool_program) == "answered probe"
