"""The precise-edit channel: a diff is a `Gated` output channel (design-space §4.4).

Two levels of proof, both no-spend:
- the `Gated` gate directly (render + resolve) — a unique anchor resolves to an `Edit`,
  a non-unique or missing anchor is a `Repair`;
- the bounded repair loop end-to-end via a fake JSON client (a bad anchor re-prompts;
  an unsatisfiable one yields no edit rather than a bad patch).

The gate is pure over the recorded snapshot, so nothing here touches a real model or disk.
"""

import json
from types import SimpleNamespace

from agent.precise_edit import Edit, EditResponse, edit_template, make_precise_editor
from effective.channels import Repair, render
from effective.domain import AskLLM

# a recorded snapshot: `return 1` is unique; `x = 0` appears twice (a bad anchor)
SNAP = {"src/foo.py": "x = 0\ndef foo():\n    x = 0\n    return 1\n"}


def _resp(path: str, old: str, new: str) -> dict:
    return {"edit": {"path": path, "old": old, "new": new}}


# --- the Gated channel directly (no client) --------------------------------


def test_unique_anchor_resolves_to_an_edit():
    prompt = render(edit_template(SNAP, "AssertionError: 1 != 2"), output=EditResponse)
    out = prompt.resolve(_resp("src/foo.py", "return 1", "return 2"))
    assert isinstance(out, EditResponse)
    assert out.edit == Edit(path="src/foo.py", old="return 1", new="return 2")


def test_missing_anchor_is_a_repair():
    prompt = render(edit_template(SNAP, "boom"), output=EditResponse)
    match prompt.resolve(_resp("src/foo.py", "return 99", "return 2")):  # 0 matches
        case Repair() as r:
            assert "exactly once" in r.reason
        case other:
            raise AssertionError(f"expected Repair, got {other!r}")


def test_non_unique_anchor_is_a_repair():
    prompt = render(edit_template(SNAP, "boom"), output=EditResponse)
    match prompt.resolve(_resp("src/foo.py", "x = 0", "x = 1")):  # 2 matches -> ambiguous
        case Repair() as r:
            assert "exactly once" in r.reason
        case other:
            raise AssertionError(f"expected Repair, got {other!r}")


# --- the bounded repair loop via a fake JSON client ------------------------


class _FakeCreate:
    """client.chat.completions stand-in for JSON mode; one content per call."""

    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(
            prompt_tokens=80,
            completion_tokens=15,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        message = SimpleNamespace(content=self.contents.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _client(contents: list[str]):
    completions = _FakeCreate(contents)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def _op(failure: str) -> AskLLM:
    return AskLLM(messages=[{"role": "user", "content": failure}], response_schema=EditResponse)


def _edit_json(path: str, old: str, new: str) -> str:
    return json.dumps(_resp(path, old, new))


def test_bad_anchor_triggers_one_repair_reprompt():
    # first emit: a non-unique anchor (2 matches) -> Repair; second: a unique one
    client, completions = _client(
        [
            _edit_json("src/foo.py", "x = 0", "x = 1"),  # non-unique -> Repair
            _edit_json("src/foo.py", "return 1", "return 2"),  # unique -> accepted
        ]
    )
    edit, usage = make_precise_editor(client, snapshot=SNAP)(_op("AssertionError"))
    assert edit == Edit(path="src/foo.py", old="return 1", new="return 2")
    assert len(completions.calls) == 2  # one repair round-trip
    assert usage.completion_tokens == 30  # both calls metered


def test_unsatisfiable_anchor_yields_no_edit_not_a_bad_patch():
    bad = _edit_json("src/foo.py", "return 99", "x")  # never matches the snapshot
    client, completions = _client([bad, bad, bad])  # max_repairs=2 -> 3 attempts
    edit, _usage = make_precise_editor(client, snapshot=SNAP, max_repairs=2)(_op("boom"))
    assert edit is None
    assert len(completions.calls) == 3
