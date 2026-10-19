# What each claim rests on

A proof establishes a property of a specified model under specified assumptions. What a claim
about the code rests on is that model, those assumptions, and which differences between the model and the
code can change the property: a proof over $\mathbb{N}$ says nothing about a sum of floats. Each law
below names what it observes, where it holds, what it assumes, and how it is known.

| evidence | means |
|---|---|
| proved | a Lean theorem over a model; where the model emits vectors, the row names the suite that runs them against the Python |
| model-checked | a Quint model whose whole reachable graph TLC explores, within the bounds it states |
| tested | pins and property tests over the cases or the generated domain they name |
| sampled | an assumption checked on cases and never discharged |
| proposed | designed, with the test that would accept it, and not built |
| open | stated, with its domain and its acceptance test still to define |

## The inventory

| law | observable | scope | assumptions | evidence |
|---|---|---|---|---|
| names do not alias | two distinct keys store distinct strings | the structured key and the scope grammar | every atom is nonempty and free of the delimiters, which `Segment` refuses otherwise; the Python serializer renders the model's grammar; a left inverse, $\mathrm{parse}(\mathrm{render}(k)) = k$, would prove it | proved for the model ([`Keys.lean`](../../formal/lean/Effective/Keys.lean), [`Scopes.lean`](../../formal/lean/Effective/Scopes.lean)); the serializer sampled by [`tests/test_op_key_injectivity.py`](../../tests/test_op_key_injectivity.py) |
| a step is deterministic | one op and one oracle give one outcome, and replay re-serves it | the step relation of [`Step.lean`](../../formal/lean/Effective/Step.lean) | every I/O passes through a yielded op; the lint enforces the authoring rule, and arbitrary Python can step around it | proved for the model; the boundary linted |
| the ledger only appends | no row is changed or removed, and an append is idempotent by event id | [`Ledger.lean`](../../formal/lean/Effective/Ledger.lean); the ledger table on both engines | a trigger refuses `UPDATE` and `DELETE` | proved for the model; the trigger tested on both engines ([`tests/test_conformance.py`](../../tests/test_conformance.py)) |
| the verdict fold | a gate's ruling is the join of its verdicts' kinds, with payloads in argument order | `govern.combine` | none beyond the vectors' coverage | proved ([`Govern.lean`](../../formal/lean/Effective/Govern.lean)); every vector run against the Python with its expected value decoded from the model alone, and every permutation of each checked for its kind and for its payloads in that permutation's order ([`tests/test_govern_conformance.py`](../../tests/test_govern_conformance.py)) |
| the permission cascade | the first decisive verdict decides; all escalations fall to the default | `permission.decide` | none beyond the vectors' coverage | proved ([`Decide.lean`](../../formal/lean/Effective/Decide.lean)); vectors ([`tests/test_decide_conformance.py`](../../tests/test_decide_conformance.py)) |
| measured spend stops at its ceiling | spend past the limit and its grants is refused, overshooting by at most one call | sequential metered calls; a gather's branches hold no budget, so calls inside one are not enforced against it | `A-money`: the model counts whole units in $\mathbb{N}$, the Python adds floats | proved for the model ([`EnforceMeasured.lean`](../../formal/lean/Effective/EnforceMeasured.lean)); vectors ([`tests/test_enforce_measured_conformance.py`](../../tests/test_enforce_measured_conformance.py)); `A-money` sampled |
| accrual is confluent | a gather's branches accrue the same total in every schedule | the accrual fold | the shared meter is the rejected design, and the model shows its order dependence | proved for the model ([`Budget.lean`](../../formal/lean/Effective/Budget.lean)); a bounded cross-check of eight steps by [`budget_confluence.qnt`](../../formal/quint/budget_confluence.qnt), which models enforcement per branch, which the handler does not do, so it cross-checks the fold and not the handler |
| a retry follows its rule | each failure is retried, waits, or fails now, by its kind | a task's attempts | each tape read's outcome is one of the transition table's rows | proved ([`Retry.lean`](../../formal/lean/Effective/Retry.lean)); vectors ([`tests/test_retry_conformance.py`](../../tests/test_retry_conformance.py)); model-checked ([`retry.qnt`](../../formal/quint/retry.qnt)) |
| an approval binds its effect | a ruling naming another effect, or none, refuses the op before any policy reads it | `govern`'s check on a binding ask | the digest is collision resistant; the host authenticates who ruled | tested ([`tests/test_agent_firm_binding.py`](../../tests/test_agent_firm_binding.py)); not proved, being one equality whose value is where it runs |
| dispatch is licensed by the plan | every act, undo and continuation sends the plan the round's claim recorded and, for an exception, the plan its ruling named; acts and undos come from the claim's holder, and a continuation from any later run | one case, two workers, three plans (the approved one, a changed one, and the clean one a deploy can find), two rounds, two crashes | the executor sends what the plan says; a leg and its answer are one step; a lease is the engine's, outside the model | model-checked ([`plan_license.qnt`](../../formal/quint/plan_license.qnt)); each refused trace tested ([`tests/test_agent_firm_binding.py`](../../tests/test_agent_firm_binding.py)) |
| each apply lands once per receiver | an apply or undo an expired attempt sent lands nothing after the compensation | the ERP's claim on each key, two attempts in each of two rounds | the fence serializes its runs per invoice | model-checked ([`erp_claim.qnt`](../../formal/quint/erp_claim.qnt)); tested on the scripted and live Odoo |
| a race chooses once | a stopped branch admits no op after the choice, and the choice survives a crash | race and quorum on both engines | the flag is published after the choice is saved | model-checked ([`race.qnt`](../../formal/quint/race.qnt)); the race suites run with the GIL off |
| a recorded result is never re-executed | a resume replays a recorded op's result and sends nothing | every op that recorded | an effect completed and not yet recorded may run again; exactly-once beyond that is the receiver's, by its keys | tested by a crash at every op on both engines ([`tests/test_shape_conformance.py`](../../tests/test_shape_conformance.py)) |
| the firm stops | two passes leave the same open cases for the same reasons, none unlanded or compensated, or the pass budget runs out | the firm over one book | `Converged` reports the caller's predicate; a stopped case is asked again and can close later | tested ([`tests/test_agent_firm_fixpoint.py`](../../tests/test_agent_firm_fixpoint.py)) |
| the allow-set is an order | `Key.covers` is a partial order, and a policy depends only on what it permits | the keys a small grammar generates | the generated domain reaches the corners | tested ([`tests/test_allow_set_laws.py`](../../tests/test_allow_set_laws.py)); the Lean proof proposed |
| the frontier is the menu | a nonempty input has a nonempty frontier, membership is order-free, every point left off is dominated by one kept, and no point kept is dominated; a missing objective or a number that is not finite is refused by name | `pareto.frontier` | values finite numbers, every objective present; the first three laws take `dominates` as their oracle, and the fourth checks it independently | tested over every small input of a tied grid ([`tests/test_pareto_laws.py`](../../tests/test_pareto_laws.py)) |
| the interaction is computed | the contrast below, one outcome per cell | four scripted runs | the worlds are scripted, so the test checks the arithmetic and says nothing of the firm | tested ([`tests/test_agent_firm_interaction.py`](../../tests/test_agent_firm_interaction.py)) |
| attenuation and conservation | a spawned child holds no more authority or budget than its parent | proposed in [concepts/authority](authority.md) | | proposed |
| linearizations agree | two valid linearizations of one history project to one ledger | a history's valid linearizations and the projection, both to define | | open |
| dispatch is total | no dispatch by `if` chains or `isinstance` | | | proposed: a gate with a baseline that can only shrink |

## Totality

A transition maps a state and an event to one of continue, done, park, refuse or fail. Totality is
a defined disposition for every declared case; it is not success. Three obligations stay apart:

| obligation | checked by |
|---|---|
| coverage: every declared case has an arm | the type checker, over closed variants and `assert_never` |
| adequacy: each arm keeps the invariants | the tests and the models above |
| behavior over time: safety and liveness | the Quint models |

Arbitrary Python inside a covered arm can still raise or run forever, which coverage does not see.

## `fix` and `fixpoint`

`fix` takes an open recursion and closes it:

```math
\mathrm{fix} : (X \to X) \to X, \qquad X = \mathtt{Callable[P,\ Effect[T]]}
```

The type checker establishes the recursive interface: the argument and the result agree, and the
knot is tied by name. A structural property, such as every external op passing its gate, is proved
by fixed-point induction: the base executes nothing, each unfolding gates its own ops, and the
recursive call cannot bypass the gate, given that every external action passes through the
interpreter.

`fixpoint` applies a step until the caller's predicate accepts a result against its input, or a
budget stops it. `Converged` reports that the predicate held; it is not a proof that $G(x) = x$.
Convergence needs more structure: a monotone step on an order of finite height, a decreasing
well-founded measure, or a contraction. The firm's tape grows while the projection it compares
stabilizes, and one unchanged pass is a fixed point only when the projection determines the next
pass with the outside world held fixed.

## What the types establish

| reach | what |
|---|---|
| established | a recursive call's arguments and result, a step's closure, coverage of declared variants |
| partial | the answer a handler sends back, behind `Effect[T] = Generator[WorkflowOp, Any, T]` and a cast |
| not established | determinism between yields, termination, authority, durable identity |

## Schedules

The schedule instrument (`tests/_turns.py`) runs two attempts through every order of their calls
with at most $k$ switches, where a switch counts when the attempt that ran last is still waiting.
For attempts of $m, n \ge 1$ calls, $s$ switches cut one attempt into $\lceil s/2 \rceil + 1$ runs
and the other into $\lfloor s/2 \rfloor + 1$, either one going first:

```math
N(k) = \sum_{s=0}^{k} \left[ \binom{m-1}{\lceil s/2 \rceil} \binom{n-1}{\lfloor s/2 \rfloor}
     + \binom{n-1}{\lceil s/2 \rceil} \binom{m-1}{\lfloor s/2 \rfloor} \right]
```

which is $2, 6, 14, 18$ for $m = n = 3$ and $k = 0, \ldots, 3$; a binomial out of range is zero.
`tests/test_turns.py` holds the instrument to this count. It covers fixed lengths and two attempts.
A length that depends on the data, a claim that resets which attempt ran last, and a third attempt
are outside it, and the instrument chooses only among the attempts' calls, never among the engine's
own interleavings.

## The interaction contrast is a mixed difference

The interaction compares a critic-enabled organization $o_1$ with one without, $o_0$, under a
knowledge snapshot with a rule, $K_1$, and without it, $K_0$:

```math
I = \bigl[Y(o_1, K_1) - Y(o_0, K_1)\bigr] - \bigl[Y(o_1, K_0) - Y(o_0, K_0)\bigr]
```

$I$ is the mixed second difference of $Y$ on the lattice $\{0, 1\}^2$, so $I \ge 0$ is the
finite-difference form of supermodularity there: the critic adds at least as much with the rule as
without. One $2 \times 2$ design tests that inequality at one point of one lattice, from one outcome
per cell; the tests check the arithmetic on scripted worlds, where any interaction is what the
script put there. A critic the rule makes unnecessary improves the organization by substitution,
which shows as $I < 0$.

## Related

- [concepts/authority](authority.md): the fold, the binding, and the proposed attenuation.
- [concepts/agent-firm](agent-firm.md): the firm's controls, each with the test that checks it.
- [concepts/testing](testing.md): how the substrate is verified, by instrument.
- [concepts/evidence](evidence.md): why a claim is not evidence.
