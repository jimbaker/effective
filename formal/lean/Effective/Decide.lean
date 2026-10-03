import Mathlib.Data.Nat.Basic

/-!
# Effective — the permission cascade's decision `decide`, formalized once

The governed-boundaries note (§8.2) named the asymmetry this file closes: budget's measured trip
is one pure transition with a Lean reference and machine-derived conformance vectors
(`EnforceMeasured.lean`), while permission's decision lived *inside* its `@op_layer` driver and had
no formal reference at all. `src/effective/permission.py::decide` extracted the decision; this file
is its model.

The transition is an **ordered fold** over tier verdicts: the first decisive verdict (`allow` /
`deny`) wins, an `escalate` defers to the next, and an all-escalate run falls to the caller's
`default`. What is proved here is the pure-axis half — the *ruling*, not the parking:

- **totality** — by construction, the codomain is `Decision` (`allow | deny`), so an `escalate`
  can never leak out as an answer; even a misconfigured `escalate` default settles (fail-closed);
- **fail-closed** — `no_allow_in_no_allow_out`: nothing in the fold can turn silence into an
  `allow`. This is the safety law the whole guard rests on;
- **suffix absorption** — a decisive verdict absorbs every continuation, which is what makes the
  Python driver's *eager* short-circuit sound. That short-circuit is not an optimization: a later
  tier may park a human, so a settled cascade must never run it.

Determinism needs no theorem — `decide` is a Lean function, so it is a function.

Scope split: reachability and interleavings — an `escalate` reaching a human
tier becoming an `AwaitEvent`, its park/resume under worker-death — are Quint's half, not modeled
here. A reason is an abstract `ℕ` tag; the reason *strings* are a value-axis detail discharged by
sampling in the Python conformance test (the same `(A-money)`-style boundary `EnforceMeasured.lean`
takes for dollars).
-/

namespace Effective.Permission

/-- A tier's ruling: settle (`allow`), block (`deny`), or defer (`escalate`). Mirrors
    `permission.Verdict`. The `deny` reason is an abstract tag — the conformance test maps tags to
    the live reason strings. -/
inductive Verdict
  | allow
  | deny (reason : ℕ)
  | escalate
  deriving DecidableEq, Repr

/-- A *settled* verdict — `decide`'s codomain. `escalate` is a request to keep looking, never an
    answer, so totality is by type rather than by theorem (`permission.Decision`). -/
inductive Decision
  | allow
  | deny (reason : ℕ)
  deriving DecidableEq, Repr

/-- The tag of `permission.FAIL_CLOSED` ("no tier allowed the op"). -/
def failClosedReason : ℕ := 0

/-- The tag of `permission.MISCONFIGURED_DEFAULT` ("cascade default did not decide"). -/
def misconfiguredReason : ℕ := 1

/-- The driver's stop condition (`permission.decisive`): does this verdict settle the cascade? -/
def decisive : Verdict → Bool
  | .escalate => false
  | _ => true

/-- Normalize the caller's `default` into a settled decision. An `escalate` default cannot settle
    anything — a misconfiguration, resolved fail-closed rather than left as a fall-through hole. -/
def settleDefault : Verdict → Decision
  | .allow => .allow
  | .deny r => .deny r
  | .escalate => .deny misconfiguredReason

/-- The cascade decision as ONE total function: an ordered fold, structurally recursive on the
    verdict list (so totality is by definition — the fold cannot diverge). Mirrors
    `permission.decide`. -/
def decide : List Verdict → Verdict → Decision
  | [], d => settleDefault d
  | .allow :: _, _ => .allow
  | .deny r :: _, _ => .deny r
  | .escalate :: rest, d => decide rest d

/-! ## Theorems — the facts every interpreter of the cascade must satisfy -/

/-- **An allow settles immediately** — no later tier is consulted (and, in the driver, no later
    tier runs: this is what licenses skipping a human park). -/
theorem allow_wins (rest : List Verdict) (d : Verdict) :
    decide (.allow :: rest) d = Decision.allow := rfl

/-- **A deny settles immediately, carrying its own reason** — the ruling the driver raises with. -/
theorem deny_wins (r : ℕ) (rest : List Verdict) (d : Verdict) :
    decide (.deny r :: rest) d = Decision.deny r := rfl

/-- **An escalate defers** — and contributes nothing else; the tail decides. -/
theorem escalate_defers (rest : List Verdict) (d : Verdict) :
    decide (.escalate :: rest) d = decide rest d := rfl

/-- **All-escalate falls to the default** (fail-closed when the default is `FAIL_CLOSED`): no
    matter how many tiers defer, silence never settles on its own. -/
theorem all_escalate_falls_to_default (n : ℕ) (d : Verdict) :
    decide (List.replicate n .escalate) d = settleDefault d := by
  induction n with
  | zero => rfl
  | succ k ih => simpa [List.replicate_succ, decide] using ih

/-- **A misconfigured (`escalate`) default still denies** — the total-by-construction arm. The
    Python driver has no "fell through without deciding" path to raise from. -/
theorem misconfigured_default_denies (n : ℕ) :
    decide (List.replicate n .escalate) .escalate = Decision.deny misconfiguredReason := by
  simpa [settleDefault] using all_escalate_falls_to_default n Verdict.escalate

/-- **Fail-closed, as a safety law.** If no tier allowed and the default is not an allow, the fold
    cannot produce an allow. Nothing here turns silence — or a list of deferrals — into permission. -/
theorem no_allow_in_no_allow_out (vs : List Verdict) (d : Verdict)
    (hv : Verdict.allow ∉ vs) (hd : d ≠ Verdict.allow) : decide vs d ≠ Decision.allow := by
  induction vs with
  | nil => cases d <;> simp_all [decide, settleDefault]
  | cons v rest ih => cases v <;> simp_all [decide]

/-- **Suffix absorption** — a decisive verdict absorbs every continuation after it.

    This is the theorem that licenses the driver's eager short-circuit: `cascade` stops running
    tiers at the first decisive verdict and folds only the prefix it collected, yet rules exactly
    as the full tier list would have. The saving is not cycles — it is not parking a human whose
    approval the cascade does not need. -/
theorem suffix_absorbed (pre : List Verdict) (v : Verdict) (hv : decisive v = true)
    (post : List Verdict) (d : Verdict) :
    decide (pre ++ v :: post) d = decide (pre ++ [v]) d := by
  induction pre with
  | nil => cases v <;> simp_all [decide, decisive]
  | cons w ws ih => cases w <;> simp [decide, ih]

/-! ## Teeth (rung-1): a buggy variant that drifts, caught by `decide` -/

/-- The fail-OPEN variant — the classic guard bug: an exhausted cascade permits instead of
    denying. Everything else is identical, so only the silence case distinguishes them. -/
def decideBad : List Verdict → Verdict → Decision
  | [], _ => .allow
  | .allow :: _, _ => .allow
  | .deny r :: _, _ => .deny r
  | .escalate :: rest, d => decideBad rest d

/-- **The fail-open variant diverges** — `decide`-checked, axiom-free. One all-escalate run with
    the fail-closed default separates them: the correct fold denies, the buggy one permits. -/
theorem decideBad_diverges :
    ∃ (vs : List Verdict) (d : Verdict), decide vs d ≠ decideBad vs d := by
  refine ⟨[Verdict.escalate], Verdict.deny failClosedReason, ?_⟩
  decide

/-! ## Conformance vectors — the machine-derived rows every interpreter is held to

Each row is `(verdicts, default) ↦ decision`, `decide`-verified against the model, and emitted to
`formal/decide_vectors.json` by `lake exe decide_vectors` (`just formal-vectors`). The Python
conformance test (`tests/test_decide_conformance.py`) runs the LIVE `permission.decide` — and the
live `cascade` driver — against these rows. -/

/-- Inputs to one conformance row. (`fallback`, not `default`: the latter collides with
    `Inhabited.default` on the generated projection.) -/
structure Vec where
  verdicts : List Verdict
  fallback : Verdict
  deriving Repr

/-- The canonical rows: one per fold arm, plus the order-sensitivity and misconfiguration cases. -/
def conformanceVectors : List (Vec × Decision) :=
  [ -- silence: the default rules, fail-closed
    (⟨[], .deny failClosedReason⟩, .deny failClosedReason),
    (⟨[.escalate], .deny failClosedReason⟩, .deny failClosedReason),
    (⟨[.escalate, .escalate, .escalate], .deny failClosedReason⟩, .deny failClosedReason),
    -- a misconfigured (escalate) default still settles — fail-closed
    (⟨[], .escalate⟩, .deny misconfiguredReason),
    (⟨[.escalate, .escalate], .escalate⟩, .deny misconfiguredReason),
    -- an explicit non-default default is honored (the fold hardcodes no law of its own)
    (⟨[], .allow⟩, .allow),
    (⟨[.escalate], .allow⟩, .allow),
    -- first decisive wins
    (⟨[.allow, .deny 2], .deny failClosedReason⟩, .allow),
    (⟨[.deny 2, .allow], .deny failClosedReason⟩, .deny 2),
    -- ...and the reason travels with the winner (two denies: the FIRST one rules)
    (⟨[.deny 2, .deny 3], .deny failClosedReason⟩, .deny 2),
    -- escalate defers to the next tier, in both directions
    (⟨[.escalate, .allow], .deny failClosedReason⟩, .allow),
    (⟨[.escalate, .deny 3], .deny failClosedReason⟩, .deny 3),
    (⟨[.escalate, .escalate, .allow], .deny failClosedReason⟩, .allow),
    -- a decisive verdict absorbs its suffix (the driver never runs the tail)
    (⟨[.escalate, .deny 2, .allow], .deny failClosedReason⟩, .deny 2),
    (⟨[.allow, .escalate, .deny 3], .deny failClosedReason⟩, .allow) ]

/-- **Every conformance vector holds against the model** — `decide`-checked, axiom-free, so the
    exported table is a machine-verified projection of `decide`, not a hand-written table. -/
theorem conformance_vectors_hold :
    ∀ vd ∈ conformanceVectors, decide vd.1.verdicts vd.1.fallback = vd.2 := by
  decide

/-! ## Trust check -/

#check @decide
#print axioms allow_wins
#print axioms deny_wins
#print axioms escalate_defers
#print axioms all_escalate_falls_to_default
#print axioms misconfigured_default_denies
#print axioms no_allow_in_no_allow_out
#print axioms suffix_absorbed
#print axioms decideBad_diverges
#print axioms conformance_vectors_hold

end Effective.Permission
