"""A machine walk's telemetry, joined to its tape by the placed key, on both engines.

ROLE: conformance. `tests/test_span_key_joins_the_tape.py` proves the join exists; this proves it
is worth having, on the runs the machine exists for: "the spans for this step" is a point lookup on
a key the projection already holds, here on a real durable machine walk.

**Why a machine.** A straight-line workflow correlates fine by tool name and order. What degrades
is exactly what `effective.machine` does: revisit a state, call the same tool at visit 0 and again
at visit 3. The span NAME cannot tell those apart (`tool.gate.{iteration}` carries no visit), and
the placed key cannot fail to, because the machine wraps every visit and state in scope frames
(`d:{n}` outside, `state:{value}` inside).

**Duration-only, correctly.** Both embodiments here answer with canned domains and yield no
`AskLLM`, so no span carries a cost and every folded cost is `None`. `None` is the answer for
"nobody measured this", which is why `Node.cost` admits it. The cost axis needs a live model.
"""

from uuid import uuid4

import pytest
from _conformance import (
    CodingSuiteDomain,
    Fault,
    RulingDeployment,
    coding_machine_wf,
    ruling_machine_wf,
)

from effective.graphview import fold_cycles, from_keys
from effective.keys import Index
from effective.layers import compose_domain
from effective.telemetry import Span, measurements, traced

pytestmark = pytest.mark.conformance


def _traced_walk(backend, workflow, base, prefix):
    """Run one machine walk with `traced` COMPOSED INTO THE DOMAIN.

    `register(..., layers=)` installs OP layers, whose alphabet is `Step`; `traced` is a DOMAIN
    layer and a probe at the wrong seam proves nothing about it."""
    spans: list[Span] = []
    run_id = f"r-{uuid4().hex[:8]}"
    domain = compose_domain((traced(spans.append, session_id=run_id),), base)
    name = f"{prefix}-{run_id}"
    backend.register(name, workflow, domain, Fault(None), ())
    task_id = backend.spawn(name, run_id)
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    return spans, run_id, backend.checkpoint_keys(task_id)


def test_only_the_ops_that_reach_the_domain_are_measured(backend):
    """The join, and the exact shape of what it does NOT cover — asked of the graph's own KINDS
    rather than of the key text.

    `traced` is a DOMAIN layer, so it sees only ops reaching the domain —
    `Step(op=AskLLM|CallTool)`.
    `StoreArtifact` and `AppendLedgerRow` are answered inside `DurableHandler._handle` and never
    enter the domain stack, so they are unmeasurable from this seam. That is a statement about
    KINDS, and `Node.kind` is what `kind_of` already computed — asserting it by prefix-matching the
    key would be a second, weaker parse of something the projection hands over."""
    spans, run_id, tape = _traced_walk(backend, coding_machine_wf, CodingSuiteDomain(), "mach")
    measured = measurements(spans)

    assert set(measured) - set(tape) == set(), "a span addressed a node the tape does not have"

    graph = from_keys(run_id, tape, telemetry=measured)
    with_duration = [node for node in graph.nodes if node.duration_ns is not None]
    without = [node for node in graph.nodes if node.duration_ns is None]

    # THE INVARIANT, structurally: measurable iff it reached the domain, i.e. iff it is a step.
    assert {node.kind for node in with_duration} == {"step"}
    assert {node.kind for node in without} == {"ledger", "artifact"}
    assert (len(with_duration), len(without)) == (7, 3)  # anti-vacuity: both halves are non-empty

    # A canned domain yields no `AskLLM`, so nothing carries a cost. `None` is the honest
    # answer for "nobody measured this", which is why `Node.cost` is not a `float`.
    assert all(node.cost is None for node in graph.nodes)
    durations = [node.duration_ns for node in graph.nodes if node.duration_ns is not None]
    assert all(ns > 0 for ns in durations)


def test_a_revisited_states_measurements_sum_across_its_visits(backend):
    """THE PAYOFF, and it needs no new code: `d:{n}` declares `Index` and `state:{name}`
    declares `Name`, so `fold_cycles` collapses the visits and keeps the states. The ruling
    machine walks work -> review -> rule -> work -> review -> rule, so each state is entered twice
    and the fold reports what that state cost across the whole run.

    This is the question a dashboard is actually asked — "which state is expensive?" — and the span
    name cannot answer it, because `tool.gate.{iteration}` is the same string at visit 0 and
    visit 3."""
    spans, run_id, tape = _traced_walk(backend, ruling_machine_wf, RulingDeployment(), "rule")
    measured = measurements(spans)

    assert set(measured) - set(tape) == set()

    graph = fold_cycles(from_keys(run_id, tape, telemetry=measured), drop=(Index,))

    # DISCOVERED from the fold, and drilled down through `Node.members` — which graphview
    # describes as "exactly the drill-down the fold promises". Recovering the members with
    # `key.endswith(node.key)` would be a parse spelled as a string test, and it would survive a
    # change to which coordinate the fold drops while silently matching the wrong rows.
    revisited = [node for node in graph.nodes if node.count == 2]
    # The KEYS, not just how many. A cardinality alone leaves the interesting half unpinned (which
    # three states were revisited), so a fold that collapsed the wrong rows could keep this green,
    # and a document quoting one of these keys would have nothing to reproduce against. Discovered
    # from the fold and then asserted whole, so this stays a structural claim rather than the
    # `endswith` parse the docstring above rejects.
    assert {node.key for node in revisited} == {
        "d:*;state:work;step;tool:gate",
        "d:*;state:review;step;tool:gate",
        "d:*;state:rule;step;tool:ruling",
    }

    for node in revisited:
        visits = [measured[member] for member in node.members]
        assert len(visits) == 2, node.members
        # The folded figure IS the sum of its visits — computed from the same run, so this holds
        # whatever the machine happens to cost on the day.
        assert node.duration_ns == sum(ns for _, ns in visits)
        assert node.cost is None  # no `AskLLM` anywhere in this embodiment

    # Anti-vacuity: the fold must actually have collapsed something.
    assert len(graph.nodes) < len(tape)
