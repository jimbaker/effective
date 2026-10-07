# Three projections, two relational operations

[`src/effective/graphview.py`](../../src/effective/graphview.py).

It is a **pure function of recorded op keys**: no DB, no engine, no `agent` import. A caller hands
it the key sequence a reader produced and gets nodes and edges back, so **nothing here can describe
an edge that did not execute**. It is a graph over a run **tape**, not a program graph.

| operation | what it is | reach | use it when |
|---|---|---|---|
| `fold_cycles` | π with the axes **declared**: the role each coordinate registered at its mint | its quotient is tape-independent, and a function of `build/key-registry.json` as well as of the name | two runs have to be comparable |
| `project` | π with the axes **discovered from the tape** | reaches a coordinate nobody declared, since the tape is the only witness that it varies | the question is *what did this run do* |
| `restrict` | **selection over rows**, contracting edges rather than deleting them | records what it did not draw in `hidden` | filtering is a projection, never a redaction |

A loop that unrolled into a chain folds back into a loop; N identical gather branches fold into one
counted node; a 300-node agent trace is a 6-node cycle graph with counts on the edges.

**Commit order is the edge relation**, so the quotient's nodes are tape-independent and the edges
between concurrent branches are not. Branches with different ops draw as a chain, and branches
folded into one node draw as a self-edge, which [concepts/building-with-effective](building-with-effective.md) shows on real
runs (`draw-concurrent-branches-as-a-fan-task`).

| renderer      | draws                                                                         |
|---------------|-------------------------------------------------------------------------------|
| `to_mermaid`  | a flowchart: the nodes, and the commit-order edges between them               |
| `to_sequence` | a sequence diagram: a participant per branch, each op a message to the world  |
| `to_text`     | a containment tree: each node inside its frames                               |

`states=` **cannot introduce a node**: it is consulted per key, so it can only re-state one
already there. A parked node arrives through `from_keys(pending=…)` or not at all.

## Why this matters beyond the dashboard

An evidence graph queried by family (security, architecture, duplication) is the same algebra one
layer out: a selection over rows annotated with where the evidence came from. The vocabulary
(`Node`, `Edge`, `RunGraph`, `dropped`, `hidden`) is the one to reuse rather than reinvent, with
the caveat that this graph's rows are executions and a program graph's rows are not.
