# ADR-0016: Formalization: the three-tier proof architecture, and the operational semantics with its serialization boundaries named

- **Date:** 2026-07-06
- **Status:** Accepted. Built: the Lean estate `formal/lean/Effective/` (`Keys`, `Scopes`, `Step`,
  `Ledger`, plus `Budget`, `EnforceMeasured`, `Govern`, `Decide` for later decisions), the Quint
  models `formal/quint/` (`effective`, `gather`, `budget_confluence`, `govern_park`, `race`), the
  gates `just formal` and `just formal-verify` (the registered list is
  [`scripts/formal_checks.sh`](../../scripts/formal_checks.sh)). Open: the composed gather-by-scope key lemma (L1.9), and gather
  determinism in Lean.
- **Math:** MathJax `$…$` and `$$…$$`.

## Context

Effective is an interpreter: a workflow yields typed ops and a handler's `match` interprets them,
so it carries an implicit small-step operational semantics. Six bugs found in review (among them
the durable-`gather` key collision, the silent suspend-in-gather, and a family of
canonical-form nondeterminism across processes) were each the violation of one nameable invariant
of that semantics. That recurrence is the case for formalizing: a small op vocabulary generates a
state space small enough to check by construction, where review catches family members one at a
time.

The proofs stop before Python's strings. Lean proves the structured key encoding injective
(`key : Key → List Nat`), while the handlers commit serialized strings (`gather:0,0;step:s` for
a plain name, `gather:0,1;step;a:1;b` for a structured one). A separator bug in the string layer
would pass the Lean gate. The same boundary appears a second time on the value axis, where
`canonical_form` serializes a payload. This ADR fixes the architecture that divides the
obligations, names both boundaries as explicit assumptions, and writes the operational semantics
down with those boundaries in it.

## Decision

### D1: three tiers, each where it is strongest

Formalization is a partition across three tools. Each owns a different kind of obligation.

| tier | method | owns |
|---|---|---|
| Lean | algebra by induction | key injectivity, ledger monotonicity, step determinism: facts over infinite domains a checker cannot enumerate |
| Quint (Apalache, TLC) | model checking the reachable state space | interleavings, suspend-in-gather totality, liveness under fairness: reachability and temporal properties a prover has no good handle on |
| parameterized tests | sampling the correspondence boundary | serialization injectivity, cross-process determinism, cross-backend conformance: the assumptions the proofs ride on, exercised against the real code |

The pure lemmas are what make the mechanical search tractable. `key_injective` lets Quint assume
disjoint namespaces (`gather.qnt` imports it as `assume L15_gatherNamingInjective`) instead of
re-deriving them per state, and determinism lets it use a deterministic action instead of racing
guards. Against an exponential state space, the reduction is what makes the check possible at all.
The tiers form a partition: Lean hands Quint a precondition that shrinks an independent search,
and neither consumes the other's output as a step of one calculation.

### D2: name the boundaries, discharge them in code, and do not extend Lean over strings

Where a proof tier stops, it stops at a named assumption handed to the next tier. Two exist, both
stated formally in "The two serialization boundaries" below and named in `Keys.lean`'s module
docstring.

| assumption | statement | discharged by |
|---|---|---|
| **(A-serialize)** | the runtime name factors as $\mathsf{name}=\mathsf{serialize}\circ\mathsf{key}$, and $\mathsf{serialize}$ is injective on well-formed keys | construction and sampling, in the codebase; Lean proves the $\mathsf{key}$ half |
| **(A-canon)** | $\mathsf{canon}$ (`canonical_form`) is a deterministic function of its value, stable across process, `PYTHONHASHSEED` and backend | sampling only: determinism rather than injectivity, and a value has no structural key to induct over |

Proving either in Lean would drag the kernel into the behavior of Python's `str` and the key
renderer, the regress the `formal/quint` README declines. The model-to-code correspondence at these
boundaries is held by construction and by an adversarial parameter grid, which is where applied
formal methods meet running code.

### D3: the invariant map is the anti-drift ledger

Each of the six invariants is anchored to a regression test and, where a tier owns it, a formal
node, so the model and the suite co-evolve and a divergence surfaces as a red test next to a stale
rule. The map is "Invariant fit" below.

## The operational semantics (small-step, judgment form)

### Syntax

A workflow is a generator that yields a stream of typed **ops**:

$$
o \;::=\; \mathsf{step}(n,\tau) \;\mid\; \mathsf{await}(e,S) \;\mid\; \mathsf{ledger}(\rho)
        \;\mid\; \mathsf{sleep}(t) \;\mid\; \mathsf{gather}(\langle b_0,\dots,b_{k-1}\rangle)
$$

where $n \in \mathsf{Name}$ is a checkpoint name, $\tau$ a *thunk* (an effectful domain
computation, a model or tool call, opaque to the relation), $e$ an event name, $S$ a result schema,
$\rho$ a ledger row carrying a unique $\mathsf{eid}(\rho)$, $t$ a wall-clock deadline, and each
$b_i$ a *branch* (a sub-workflow thunk). A workflow $W$ is a sequence of ops terminated by a return;
$o\,;\,W$ is "yield $o$, then continue as $W$", and $W[v/\bullet]$ resumes the generator with $v$
sent in.

### Configurations

$$
\kappa \;=\; \langle\, W,\; C,\; L,\; E\,\rangle,
\qquad
C : \mathsf{Name} \rightharpoonup \mathsf{Val}\;\text{(checkpoints, disposable)},
\quad
L \in \mathsf{Row}^{*}\;\text{(ledger, append-only)},
\quad
E : \mathsf{Event} \rightharpoonup \mathsf{Val}\;\text{(delivered events)}.
$$

$C$ and $L$ are the two durable bookkeepers: $C$ is rebuildable execution state, $L$ the canonical
record. The relation never reads $C$ to produce $L$ or the reverse (invariant I5).

### The transition relation $\;\kappa \rightarrow \kappa'$

The handler defines $\rightarrow$. Replay is the same relation re-run with a populated $C$:

$$
\frac{\,n \notin \operatorname{dom}(C) \qquad \tau \Downarrow v\,}
     {\langle \mathsf{step}(n,\tau)\,;W,\; C,\; L,\; E\rangle \;\rightarrow\;
      \langle W[v/\bullet],\; C[n\mapsto v],\; L,\; E\rangle}
\;\textsf{(Step-Run)}
\qquad
\frac{\,C(n) = v\,}
     {\langle \mathsf{step}(n,\tau)\,;W,\; C,\; L,\; E\rangle \;\rightarrow\;
      \langle W[v/\bullet],\; C,\; L,\; E\rangle}
\;\textsf{(Step-Replay)}
$$

$\textsf{Step-Replay}$ fires when the checkpoint exists and does not evaluate $\tau$; that premise
is exactly-once under replay. Events, then the ledger (append-only, idempotent by $\mathsf{eid}$):

$$
\frac{\,E(e) = v\,}
     {\langle \mathsf{await}(e,S)\,;W,\,C,L,E\rangle \rightarrow \langle W[v/\bullet],\,C,L,E\rangle}
\;\textsf{(Await-Resume)}
\qquad
\frac{\,e \notin \operatorname{dom}(E)\,}
     {\langle \mathsf{await}(e,S)\,;W,\,C,L,E\rangle \rightarrow \mathsf{Parked}(e)}
\;\textsf{(Await-Park)}
$$

$$
\frac{\,\mathsf{eid}(\rho)\notin \mathsf{ids}(L)\,}
     {\langle \mathsf{ledger}(\rho)\,;W,\,C,L,E\rangle \rightarrow \langle W,\,C,\,L\cdot\rho,\,E\rangle}
\;\textsf{(Ledger-Append)}
\qquad
\frac{\,\mathsf{eid}(\rho)\in \mathsf{ids}(L)\,}
     {\langle \mathsf{ledger}(\rho)\,;W,\,C,L,E\rangle \rightarrow \langle W,\,C,\,L,\,E\rangle}
\;\textsf{(Ledger-Idem)}
$$

### Gather: the applicative rule, where injectivity lives

Let $\rightarrow_{p}$ be the relation with every checkpoint name prefixed by $p$. Branch $i$ of the
$g$-th gather runs under prefix $p_{g,i} = \texttt{gather:}g\texttt{,}i\texttt{;}$:

$$
\frac{\displaystyle \forall i<k.\;\; \langle b_i, C, \varepsilon, E\rangle
        \;\rightarrow^{*}_{p_{g,i}}\; \langle v_i,\; C_i,\; L_i,\; E\rangle}
     {\langle \mathsf{gather}(\langle b_0\dots b_{k-1}\rangle)\,;W,\,C,L,E\rangle
        \;\rightarrow\;
      \langle W[\langle v_0\dots v_{k-1}\rangle/\bullet],\;\; C \uplus \textstyle\biguplus_i C_i,\;\;
              L \cdot L_0 \cdots L_{k-1},\; E\rangle}
\;\textsf{(Gather)}
$$

The result binds by branch index $\langle v_0\dots v_{k-1}\rangle$, never completion order, which
makes the rule deterministic and replay-stable. The disjoint union $\biguplus_i C_i$ is
well-defined only if the per-branch checkpoint domains are disjoint, across siblings and across
gathers $g\neq g'$. That side condition is the injectivity invariant I1. A prefix lacking the $g$
component ($p_i = \texttt{gather:}i$) violates it: a different gather's $\textsf{Step-Replay}$ fires
on a stale checkpoint, which is the collision `keyBad` reproduces in `Keys.lean`. In the runtime,
$g$ is the gather ordinal of the handler's thread of control (`FramePosition.next_gather` in
[`src/effective/keys/frame.py`](../../src/effective/keys/frame.py)), stable across crash-resume because the workflow yields gathers in
deterministic order.

A branch whose head is an await or a sleep parks the whole task on its qualified event name and
resumes after the barrier, on a context with the `peek_event` capability (the SQLite engine and
`ConcurrentAbsurdCtx`). On a context without it the handler raises a legible
`NotImplementedError`, so that position is total by rejection.

**Scope prefixes.** A second prefix family composes with $p_{g,i}$: the handler applies a
`scoped` frame ([ADR-0014](0014-rlm-combinators-run-code.md) §8) through `scope_prefix`, which terminates each atom with the term
separator `;` and refuses an atom carrying one, so nested scopes become [ADR-0020](0020-key-composition-one-grammar.md)'s leading frame
terms (`a:1;b:2;step:s`). The Lean model abstracts this as a scope $s$ drawn from the grammar
$s ::= (\text{atom}\ \texttt{/})^{*}$, atoms nonempty over $\Sigma\setminus\{\texttt{/}\}$,
with every canonical name `/`-free. The flattened name is $s\cdot n$, and well-formedness makes
the flattening injective (split at the last `/` recovers $(s,n)$). This is the one string layer
proved in Lean: `flatten_injective_on` (L1.8) in `Scopes.lean`.

The model and the code differ in two places, and the proof covers the code only up to them:

| model (`Scopes.lean`) | code |
|---|---|
| the separator is `/` | the separator is `;`, [ADR-0020](0020-key-composition-one-grammar.md)'s term separator |
| a canonical name is separator-free | a structured name carries `;` and is spliced as its own terms; the op arm (`step;`, `ledger;`) that opens every op key marks where the frames end |

### The two serialization boundaries

The rules range over structured names and structured values; the handlers commit strings. There
are exactly two places Effective serializes to a normal form, one invariant (serialize to a
deterministic normal form) split by which ambiguity it fights.

**Key axis: factor the naming map.** Write the runtime name as the composite

$$
\mathsf{name}\;=\;\mathsf{serialize}\circ\mathsf{key},
\qquad
\mathsf{key}:\mathsf{Key}\to\mathbb{N}^{*}\ \text{(the frame path)},
\qquad
\mathsf{serialize}:\mathbb{N}^{*}\to\Sigma^{*}\ (\texttt{gather:}g\texttt{,}i\texttt{;}\dots).
$$

Injectivity of $\mathsf{name}$ needs both factors injective:

$$
\underbrace{\mathsf{key}(x)=\mathsf{key}(y)\Rightarrow x=y}_{\textbf{Lean: }\texttt{key\_injective}\ (\text{L1.3})}
\qquad\wedge\qquad
\underbrace{\mathsf{serialize}(u)=\mathsf{serialize}(v)\Rightarrow u=v}_{\textbf{(A-serialize): discharged in code}}
\qquad\Longrightarrow\qquad
\mathsf{name}\ \text{injective.}
$$

Lean discharges the left conjunct by list induction over the frame path. The right conjunct is
discharged in the codebase two ways.

| agency | how |
|---|---|
| construction | `compose_key` ([`src/effective/keys/processor.py`](../../src/effective/keys/processor.py)) is the single op-key producer, and the key it composes is parsed by a grammar ([`src/effective/keys/grammar.py`](../../src/effective/keys/grammar.py)): a value reaching a hole must be a well-formed atom, and an atom may not contain a separator, so delimiter-bearing values are refused and arity is countable from the bytes. Every key opens with its own arm term (`step;…`, `ledger;…`), so the arms' regions are disjoint by construction. `op_key` of a `Gather` raises, so a gather has no aliasable content key and its leaves take only the positional `gather:{g},{i};` frame, which is $\mathsf{key}$'s frame order serialized |
| sampling | [`tests/test_op_key_injectivity.py`](../../tests/test_op_key_injectivity.py) (injectivity across arms, the adjacent-interpolation refusal, `op_key(Gather)` raising, round-trip over every registered shape), [`tests/test_gather.py::test_two_same_arity_gathers_get_distinct_positional_keys`](../../tests/test_gather.py), and [`tests/test_conformance.py::test_durable_gather_keys_distinct_across_two_same_shaped_gathers`](../../tests/test_conformance.py) on both engines |

The aliasing (A-serialize) forecloses is the string-layer sibling of `keyBad`, `Scopes.lean`'s
`flattenBad` foil: a delimiter-free interpolation admits $(\texttt{a},\texttt{12})$ against
$(\texttt{a1},\texttt{2})$. A separator bug in the gather or arm-term rendering would still pass
the Lean gate, and is caught by the grammar's refusal and by the pin grid. Leaving it out of Lean
is deliberate: serialization injectivity is enumerable rather than inductive, a product of op shape
× interpolated-value shape × delimiter × `PYTHONHASHSEED` × backend that a parameter grid covers
and induction cannot see.

**Value axis: a deterministic normal form.** The other boundary serializes a checkpoint or digest
payload. $\mathsf{canon}$ (`canonical_form` in [`src/effective/handlers/base.py`](../../src/effective/handlers/base.py), behind
`content_digest` and `code.canonical`) maps a value to a JSON form by recursively sorting sets. Its
obligation is weaker and different in kind:

$$
\textbf{(A-canon)}\qquad
\mathsf{canon}(v)\ \text{is a deterministic function of}\ v,
\ \text{stable across process, }\texttt{PYTHONHASHSEED}\text{, and backend.}
$$

It has no Lean node and is discharged by sampling: [`tests/test_canonical_determinism.py`](../../tests/test_canonical_determinism.py), and
[`tests/test_run_code.py::test_run_code_rebind_survives_fresh_process_resume_with_a_set_arg`](../../tests/test_run_code.py), which
resumes in a fresh process under a differing `PYTHONHASHSEED`. The asymmetry pins the tiers: the
key axis needs a deterministic normal form and injectivity; the value axis needs only the
deterministic normal form.

### The handlers are interpretations that must agree

Recording $\rightarrow_R$, Replay $\rightarrow_P$, Absurd $\rightarrow_A$ and SQLite
$\rightarrow_S$ implement the same abstract $\rightarrow$. With $L(\kappa)$ the observable (the
committed ledger), correctness is a trace-equivalence metatheorem:

$$
\forall W.\quad
L\big(\,\Downarrow_{R}(W)\,\big) = L\big(\,\Downarrow_{P}(W)\,\big)
                                 = L\big(\,\Downarrow_{A}(W)\,\big)
                                 = L\big(\,\Downarrow_{S}(W)\,\big),
$$

where $\Downarrow_H$ is the terminal observable under handler $H$. The runtime `ReplayMismatch`
check is this theorem for one trace; the cross-backend conformance suite ([`tests/_conformance.py`](../../tests/_conformance.py))
checks it by example on two engines; a model checker discharges it over a reachable space.

### Crash and replay

Model worker death as truncation: a crash at step $j$ keeps $C,L$ (the committed prefix) and
discards $W$'s in-flight frame; resume re-runs $W$ from the start against the surviving $C,L$.
Crash-resume correctness is the diamond

$$
\Downarrow_H(W) \;=\; \Downarrow_H\big(\mathrm{crash}_j(W)\ \text{then resume}\big)
\qquad \text{for every } j,
$$

with side effects (the $\tau\Downarrow v$ premises) firing exactly once, because every committed
$n$ takes $\textsf{Step-Replay}$ on the re-run. [`tests/test_replay_crash.py`](../../tests/test_replay_crash.py) checks this at every
$j$ by example.

## Invariant fit (I1 to I6)

| inv | formal node | state |
|---|---|---|
| **I1 injectivity** | Lean `keyBad_not_injective` (the regression), `key_injective`, `branch_disjoint`, `gather_naming_injective` (L1.5, imported by Quint); `flatten_injective_on` (L1.8) | discharged: the string tier is (A-serialize), held by construction and the grid. L1.9, the composed gather-by-scope frame, is open in Lean; the runtime composite is covered by tests |
| **I2 totality** | Quint `gather.qnt` `total` and `noDeadlock`; the `handlerFixed=false` tooth reproduces suspend-in-gather | await and sleep in a gather are transitions on a peek-capable context and a loud `NotImplementedError` on one without |
| **I3 liveness** | Quint `gather.qnt` `liveness` (TLC, fairness implies termination, non-vacuity guarded by `nonVacuityWitness`) | partial: checks the suspend-in-gather instance. Expired-lease reclaim on Absurd is pinned by [`tests/test_absurd_lease_reclaim.py`](../../tests/test_absurd_lease_reclaim.py) and has no formal node |
| **I4 termination of observers** | none | out of scope: no op carries an observer; watching a run is telemetry |
| **I5 two bookkeepers** | Lean `ledger_monotone`, `ledger_idem`, `advance_nodup`, `L_not_function_of_C`, `C_not_function_of_L`; Quint `effective.qnt` `ledgerNodup` | partial by design: non-derivability is the modest direction; the architectural seal (the append-only trigger) is held by tests |
| **I6 determinism** | Lean `step_deterministic` (for a fixed oracle ω), `replay_stable`; Quint assumes it (`assume T2_determinism`) | partial: covers step, await and ledger; gather determinism is deferred in `Step.lean`, and the index-ordered join is pinned by [`tests/test_gather.py`](../../tests/test_gather.py) |

## Consequences

Every tier that stops does so at a named assumption with a stated discharge, so a reader can audit
(A-serialize) and (A-canon) without re-deriving that they are covered. The organizing thesis is
stated once: serialize to a deterministic normal form, split by which ambiguity you fight (the
value axis fights order; the key axis fights grammar and identity, and additionally needs
injectivity).

Accepted risk and open edges:

| item | standing |
|---|---|
| a separator bug in the gather or arm-term rendering | passes the Lean gate; caught by the grammar's refusal and the pin grid. Proving it in Lean is the regress D2 declines |
| L1.9, the composed gather-by-scope stack as one `Key` frame kind | open |
| I3's lease-reclaim instance | a pytest on Absurd, no formal node |
| gather determinism | argued in code and pinned by tests, no Lean node |

The three-tier win holds only while extraction stays cheap: the Quint and Lean models must stay a
transcription of the handler `match`. If they drift into a hand-maintained parallel artifact, stop.

Related: [ADR-0008](0008-dynamic-workflows-as-ops-applicative-parallelism.md) (applicative `gather` keying is the Gather rule), [ADR-0009](0009-durable-backend-two-regimes-taskcontext.md) (the two-regime
`TaskContext` is the claim that $\rightarrow_A$ and $\rightarrow_S$ refine one $\rightarrow$),
[ADR-0014](0014-rlm-combinators-run-code.md) §8 (the scope frame is the convention layer `Scopes.lean` proves).
