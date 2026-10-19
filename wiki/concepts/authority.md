# Authority: verdicts, rulings and what an approval binds

A governed op is asked of every policy at its gate, and the answers fold into one verdict. A park
waits for a principal's ruling, which records who ruled and, when the ask binds an effect, which
effect the ruling is about. What is built here is the fold, the typed ruling and the binding; the
laws a delegated organization would need beyond them, attenuation and conservation, are proposed.

## The vocabulary

| term | what it is | in the code |
|---|---|---|
| verdict | one policy's answer about one op: proceed, park with an ask, or refuse with a reason | `Proceed`, `Park`, `Refuse` in `effective.govern` |
| ask | a policy's question at a park: its prompt, its detail, and the digest of the effect it binds, if any | `Ask` |
| ruling | a principal's answer to a park, naming who ruled | `AuthorityRuling` in `effective.ruling` |
| refusal | why a policy refused, naming the policy and the ruler whose ruling caused it | `Refusal` |
| authorization | an op the gate let through: every policy proceeded, after any park was ruled on | the gate forwarding the op |
| grant | a ruling that raises a limit, such as more budget; it binds no effect | an `Ask` with no `binds` |

## The fold is a join over kinds

Order the kinds of verdict $\mathsf{Proceed} < \mathsf{Park} < \mathsf{Refuse}$. The gate's verdict
is the join of its policies' kinds:

```math
\mathrm{kind}\bigl(\mathrm{combine}(v_1, \ldots, v_n)\bigr) = \bigvee_{i=1}^{n} \mathrm{kind}(v_i)
```

So any refusal refuses, else any park parks, else the op proceeds; Proceed is the
join's unit, and a gate is built with at least one policy. The kind is order-free: permuting the
policies cannot change it. The payload keeps order: a refusal carries every refusing policy's reasons, and a park every parking policy's asks,
in the order the gate was assembled, so the merged prompt reads deterministically. Classification
is by constructor, so a refusal that gives no reason still refuses.

Refuse dominating park is the fail-closed direction: a gate never asks a person to grant past a
policy that has already said no. `formal/lean/Effective/Govern.lean` proves the unit, the
dominance, the order-free kind and the fused payload, and shows the inverted polarity diverges;
`tests/test_govern_conformance.py` holds the Python fold to the model's vectors.

## A ruling binds its effect

An ask that is about one effect carries that effect's digest, $a.\mathrm{binds}$, and a ruling on
it names the digest it rules on under `effect`. For a park $P$ and a ruling $r$:

```math
\mathrm{Bound}(P, r) \iff \forall a \in P.\mathrm{asks}.\ \ a.\mathrm{binds} \ne \bot \Rightarrow r[a.\mathrm{policy}].\mathrm{effect} = a.\mathrm{binds}
```

A ruling that is not bound is refused before any policy reads its decision, with every unbound
policy named in the refusal, so an approval of one effect cannot release another. A grant binds
nothing and is read as before.

| what binding establishes | what it rests on, or leaves to others |
|---|---|
| the ruling was made on this effect, as the ask described it | the digest's collision resistance: two effects with one digest would share a ruling |
| a changed amount, a changed plan, or a missing digest is refused | the executor runs the effect the op names; the ERP adapter posts what it is sent |
| a refusal names its policy and its ruler | the ruler is a name the answering surface asserts; authenticating it is the host's |

Binding is matching. It is necessary for an approval and short of authenticated authority: it
does not say the ruler was entitled to rule, and it implements no expiry or revocation. In the
agent firm the bound effect is the settlement's terms row, which carries the digest of its plan, so
a credit and a plan nobody approved run nothing ([concepts/agent-firm](agent-firm.md),
[ADR-0028](../../docs/adr/0028-an-approval-binds-its-effect.md)).

| what holds | pin |
|---|---|
| the fold, its unit, dominance and order-free kind | `tests/test_govern.py`, `tests/test_govern_conformance.py`, `formal/lean/Effective/Govern.lean` |
| an ordinary answer cannot settle an authority park, and a ruling records who ruled | `tests/test_authority_rulings.py` |
| a ruling on another effect is refused before the policy reads it | `tests/test_govern.py`, `tests/test_agent_firm_binding.py` |

## Proposed: attenuation and conservation

Neither law is built; each is stated with the test that would establish it.

**Attenuation.** A child's permissions lie within its parent's. On allow-sets, $P \le Q$ when every
entry of $P$ is covered by some entry of $Q$, the order [concepts/key-order](key-order.md) lifts
from keys. The acceptance test: a spawn whose child policy is not below its parent's is refused,
over every pair of generated allow-sets. Key coverage represents the scope of a permission; it does
not cover identity, expiry, an external resource's version or a consumable allocation.

**Conservation.** Budget delegated down a tree of runs cannot exceed the root's. With the tree's
allocations disjoint, so no reservation is counted at both an ancestor and its descendant:

```math
\sum_i \bigl(\mathrm{outstanding}_i + \mathrm{spent}_i\bigr) \le B_{\mathrm{root}} + \sum_j g_j
```

where $g_j$ are the grants principals made. The acceptance test: a concurrent tree of spawns,
crashed at every op, never spends or reserves past the right side. The structural bounds in
`effective.budget`, such as the spawn-depth ceiling, do not establish this law.

## What names prove

Lean's theorem T1 proves a **structured** key encoding injective, `formal/lean/Effective/Keys.lean`:
two distinct keys never share a name. The bytes the handlers write come from a Python serializer
the proof does not model; its obligation is named in that file and checked by
`tests/test_op_key_injectivity.py` on the cases that threaten it.

## Related

[concepts/agent-firm](agent-firm.md) is the worked case: an approval gate, its sequence, and the
settlement it binds. [concepts/key-order](key-order.md) is the order attenuation would use.
