import Mathlib.Logic.Function.Defs

/-!
# Effective: names track (T1: no aliasing / invariant I1)

Formalizes durable `gather` checkpoint keying from Effective's operational
semantics. A checkpoint name is generated from its nesting path of
gather frames `(g, i)` (the gather-occurrence id `g` and the branch index `i`)
plus the local step index.

This file is the first two nodes of the proof DAG:

* **L1.1**: the `Key` type and the two encoders (`key`, the correct one;
  `keyBad`, the buggy one).
* **L1.2**: the *regression*, the cross-gather collision, reproduced
  as a `decide`-checked fact. The bug was that the key prefix used only the branch
  index `i` (`gather:i:`), dropping the gather id `g` (`gather:g:i:`), so two
  *different* gathers at the same branch index produced the same name and one
  gather's `Step-Replay` fired on the other's stale checkpoint.

We model the structured encoding `key : Key → List Nat` (node L1.3, which
sidesteps string parsing) and prove *that* injective (`key_injective`).

## The deferred boundary, named

One tier is *deliberately* not proved in Lean: the runtime **serialization**

  `serialize : Key → String`

which is the bytes the handlers emit (`gather:{g},{i};{step}` for a gathered leaf,
`ledger;{id}`, `step;{name}`). The obligation it carries is

> **(A-serialize)** `serialize` is injective on well-formed keys; equivalently,
> it factors through `key` by an injective string encoding, so
> `serialize k₁ = serialize k₂ → k₁ = k₂`.

We do **not** discharge (A-serialize) here: a Lean proof would have to model the
behavior of Python's `str`, the regress the `formal/quint` README refuses
(*we have turtles, but not all the way down*). The model↔code gap is held by
agency rather than a formal harness. (A-serialize) is a **named assumption,
discharged in the codebase** two ways:

* **By construction.** `compose_key` (`src/effective/keys/processor.py`) is the
  single op-key producer, and the statics it composes are *parsed*
  (`src/effective/keys/grammar.py`): a value reaching a hole must be a well-formed
  **atom**, and an atom may not contain a separator, so arity is countable from the
  bytes with no registry. Nothing is escaped; a delimiter-bearing value is
  *refused*. Every key opens with its own **arm term** (`step;ledger:x` is not
  `ledger:x`), so the arms' regions are disjoint by construction. `op_key(Gather)`
  *raises*: a gather has no aliasable content-key, so its leaves are keyed only by
  the positional `gather:{g},{i};` frame, which **is** this file's `key` frame
  structure serialized. The delimiter aliasing this forecloses is `Scopes.lean`'s
  `flattenBad` foil at the key layer: `(a, 12)` vs `(a1, 2)`.

  For the tier argument: the encoding is a grammar with a parser, so injectivity
  is a property of a parse/render pair, which is closer to dischargeable than a
  property of Python's string formatting. It is still not discharged here.
* **By sampling.** `tests/test_op_key_injectivity.py` (`op_key(Gather)` raises;
  adjacent interpolations are refused), `tests/test_gather.py`
  `test_two_same_arity_gathers_get_distinct_positional_keys`, and the
  cross-backend `tests/test_conformance.py`
  `test_durable_gather_keys_distinct_across_two_same_shaped_gathers`.

`Scopes.lean` *does* prove the string layer for the **scope** grammar
(`flatten_injective_on`, L1.8): the one interpolation whose split-at-`/`
argument is clean enough to induct on. The gather-path and arm-tag serializations
stay in the by-construction and sampling tiers by nature: serialization
injectivity is *enumerable, not inductive* (op-shape × value-shape × delimiter ×
`PYTHONHASHSEED` × backend), which a parameter grid covers and induction cannot see.
-/

namespace Effective.Keys

/-- A gather frame: the gather-occurrence id `g` and the branch index `i`. -/
abbrev Frame := Nat × Nat

/-- A checkpoint location: the nesting path of gather frames, outermost first. -/
abbrev Loc := List Frame

/-- A structured checkpoint key: its location and the local step index. -/
structure Key where
  loc : Loc
  step : Nat
deriving DecidableEq, Repr

/-- Correct keying: every frame contributes *both* its gather id and its branch
    index, then the step. Injective — that proof is L1.3, `key_injective` below.

    Runtime correspondence: a `Frame = (g, i)` is the handler's monotonic
    `_gather_seq` (`g`) paired with the branch index (`i`) — see
    `recording.py` / `absurd.py` — so the runtime `gather:{g}:{i}:` path *builds*
    this `key`'s frame order, `flatMap (fun f => [f.1, f.2])`. `op_key(Gather)`
    raises rather than emitting a non-path key, so a gather never contributes an
    aliasable content-key; only its leaves are named, by this structure. -/
def key (k : Key) : List Nat :=
  (k.loc.flatMap (fun f => [f.1, f.2])) ++ [k.step]

/-- Buggy keying (the cross-gather collision): the gather id `g` is dropped, keeping
    only the branch index `i`. Not injective — see `keyBad_not_injective`. -/
def keyBad (k : Key) : List Nat :=
  (k.loc.map (fun f => f.2)) ++ [k.step]

/-- Two distinct sibling gathers (`g = 0` vs `g = 1`), same branch, same step. -/
def collideA : Key := ⟨[(0, 0)], 0⟩
def collideB : Key := ⟨[(1, 0)], 0⟩

/-- **L1.2 (regression, by `decide`).** Under the buggy keying the two distinct
    keys collide: the gather id that would tell them apart was thrown away. -/
example : keyBad collideA = keyBad collideB ∧ collideA ≠ collideB := by decide

/-- The fix, on the *same* witnesses: correct keying keeps `g`, so the two keys
    do not collide. -/
example : key collideA ≠ key collideB := by decide

/-- **L1.2 as the named property.** `keyBad` is not injective — the negative half
    of invariant I1. (The positive half, `Function.Injective key`, is L1.3.) -/
theorem keyBad_not_injective : ¬ Function.Injective keyBad := by
  intro h
  exact absurd (h (show keyBad collideA = keyBad collideB by decide)) (by decide)

/-- Core of L1.3: the structured encoding is injective on the underlying
    `(loc, step)` data. Proven by induction on the two location lists in lockstep —
    the step rides along untouched until the base case, where it is finally
    compared. -/
private theorem enc_inj :
    ∀ (la lb : Loc) (sa sb : Nat),
      la.flatMap (fun f => [f.1, f.2]) ++ [sa]
        = lb.flatMap (fun f => [f.1, f.2]) ++ [sb] →
      la = lb ∧ sa = sb := by
  intro la
  induction la with
  | nil =>
      intro lb sa sb h
      cases lb with
      | nil => simpa using h
      | cons b lb' => obtain ⟨gb, ib⟩ := b; simp at h
  | cons a la' ih =>
      intro lb sa sb h
      obtain ⟨ga, ia⟩ := a
      cases lb with
      | nil => simp at h
      | cons b lb' =>
          obtain ⟨gb, ib⟩ := b
          simp only [List.flatMap_cons, List.cons_append, List.nil_append,
            List.cons.injEq] at h
          obtain ⟨hg, hi, hrest⟩ := h
          obtain ⟨hloc, hs⟩ := ih lb' sa sb hrest
          subst hg; subst hi; subst hloc; subst hs
          exact ⟨rfl, rfl⟩

/-- **L1.3.** The correct keying is injective — distinct checkpoints get distinct
    names. This is the *positive* half of invariant I1, the one `decide` cannot
    reach (it quantifies over all keys, of which there are infinitely many). -/
theorem key_injective : Function.Injective key := by
  rintro ⟨la, sa⟩ ⟨lb, sb⟩ h
  simp only [key] at h
  obtain ⟨hloc, hs⟩ := enc_inj la lb sa sb h
  subst hloc; subst hs
  rfl

/-- The §2.4 prefix `p_{g,i}` on the structured layer: running a checkpoint inside
    branch `i` of gather `g` prepends the frame `(g, i)` to its location path. -/
def underBranch (g i : Nat) (k : Key) : Key := ⟨(g, i) :: k.loc, k.step⟩

/-- **L1.4 (domain disjointness).** Two positions with different `(g, i)` frames —
    different branches of one gather (`i ≠ i'`) *or* different gathers (`g ≠ g'`) —
    never produce the same name, for any inner keys. The cross-gather collision is
    exactly the `g ≠ g'` case of this failing. Follows from `key_injective`: equal
    names force equal keys, hence equal head frames, contradicting `hne`. -/
theorem branch_disjoint {g i g' i' : Nat} (hne : (g, i) ≠ (g', i')) (k k' : Key) :
    key (underBranch g i k) ≠ key (underBranch g' i' k') := by
  intro h
  have hu := key_injective h
  simp only [underBranch, Key.mk.injEq, List.cons.injEq] at hu
  exact hne hu.1.1

/-- Concrete (the sibling case of `branch_disjoint`): branches 0 and 1 of the same
    gather give the *same* inner key distinct names — the bug scenario, now ruled
    out at the branch level. The `(7,0) ≠ (7,1)` side-condition is itself `decide`d. -/
example (k : Key) : key (underBranch 7 0 k) ≠ key (underBranch 7 1 k) :=
  branch_disjoint (by decide) k k

/-- **L1.5 (Gather totality — the export).** Across a whole gather `g`, the name
    assignment `(branch index, inner key) ↦ name` is injective: no two checkpoints
    anywhere in the fan-out share a name. A finite map's disjoint union `⨄ᵢ Cᵢ` is
    defined exactly when its key assignment has no collisions, so this *is* the
    (Gather) side-condition discharged on the encoder side — the lemma the Quint
    model imports as an assumption rather than re-deriving at every state. -/
theorem gather_naming_injective (g : Nat) :
    Function.Injective (fun p : Nat × Key => key (underBranch g p.1 p.2)) := by
  rintro ⟨i, ki, ks⟩ ⟨j, ki', ks'⟩ h
  have hu := key_injective h
  simp only [underBranch, Key.mk.injEq, List.cons.injEq, Prod.mk.injEq] at hu
  obtain ⟨⟨⟨_, hij⟩, hloc⟩, hstep⟩ := hu
  subst hij; subst hloc; subst hstep
  rfl

/-! ## Trust check (click these; output renders in the VS Code Infoview)

`#print axioms` is the structural review check for the whole Lean side: it reports
exactly which axioms a theorem rests on. For this regression it reports **"does not
depend on any axioms"** — `decide` produced a fully axiom-free kernel proof (not
even `propext`/`Classical.choice`/`Quot.sound`). In particular it does **not** list
`Lean.ofReduceBool`, which `native_decide` would add: that absence is the payoff of
`decide` over `native_decide` — the *kernel* verified the computation, so the
compiler stays out of the trusted base. (A stray `sorryAx` here would mean an
unfinished proof; a stray `ofReduceBool`, that we trusted the compiler.) -/

#check keyBad_not_injective   -- ¬ Function.Injective keyBad
#print axioms keyBad_not_injective

#check key_injective          -- Function.Injective key
#print axioms key_injective

#check branch_disjoint
#print axioms branch_disjoint
#check gather_naming_injective
#print axioms gather_naming_injective

-- Frame-order correspondence (§2.4): one gather frame `(g, i)` then a step
-- serializes as `[g, i, step]` — the same order the runtime `gather:{g}:{i}:`
-- path emits. Definitional, so a frame-order regression breaks the build.
example (g i s : Nat) : key ⟨[(g, i)], s⟩ = [g, i, s] := rfl

-- The encodings side by side: the collision, then the fix.
#eval keyBad collideA   -- [0, 0]
#eval keyBad collideB   -- [0, 0]   ← same ⇒ the bug
#eval key    collideA   -- [0, 0, 0]
#eval key    collideB   -- [1, 0, 0] ← differs in the gather id g ⇒ the fix

end Effective.Keys
