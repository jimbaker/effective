import Mathlib.Logic.Function.Defs

/-!
# Effective: names track, the scope grammar (T1; nodes L1.6–L1.8)

Formalizes the scope grammar: the *string-serialization* layer that `Keys.lean`'s
deferred boundary leaves unproved, proved here for scopes. A scoped
durable name is `flatten scope name`: terminated atoms, then a canonical name
free of the terminator. The model spells the terminator `/`; the implementation
terminates each frame atom with the term separator `;` (`scope_prefix` in
`effective.keys.frame`, which refuses an atom carrying it), so a step `s` inside
`scoped(a:1, scoped(b:2, …))` records as `a:1;b:2;step:s`. The model's separator-free name
is a second gap: an implementation name may carry `;`, and the op arm that
opens every op key (`step;`) marks where the frames end.

Three nodes of the proof DAG:

* **L1.6**: the defs: `Atom` / `Scope` / `flatten`, and the delimiter-free
  `flattenBad` foil (what interpolating atoms *without* the terminating `/`
  would compute).
* **L1.7**: the regression. `flattenBad` aliases `("a", "12")` with
  `("a1", "2")`, `keyBad`'s string-layer sibling, `decide`-checked; `flatten`
  separates the same witnesses.
* **L1.8**: the theorem. On the `WF` domain, `flatten` is injective: split
  at the first `/` recovers the head atom, induction recovers the rest.

The composed gather×scope stack (structural `/`-free prefixes over scoped
names) is L1.9, open: it wants a scope-frame kind unified into `Keys.Key`.
-/

namespace Effective.Scopes

/-- An atom: the characters between two `/`s. The grammar constraints
    (nonempty, `/`-free) ride as `WF` hypotheses rather than a subtype,
    keeping `decide` usable on concrete witnesses. -/
abbrev Atom := List Char

/-- A scope: an atom list, flattened as `(atom "/")*`. -/
abbrev Scope := List Atom

/-- The flattening: each atom slash-terminated, then the canonical
    name (the implementation's `"rec:0;" + "step;code:seg,0,audit"`, with `;`). -/
def flatten (s : Scope) (name : List Char) : List Char :=
  (s.flatMap (fun a => a ++ ['/'])) ++ name

/-- The foil: delimiter-free concatenation, which is what a scope interpolation
    *without* the reserved terminator would compute. Not injective (L1.7). -/
def flattenBad (s : Scope) (name : List Char) : List Char :=
  s.flatMap id ++ name

/-- Well-formed: every atom nonempty and `/`-free, plus a `/`-free canonical
    name. Injectivity uses only the `/`-freedom halves;
    nonemptiness is the grammar's legibility choice (no `x//y` keys). -/
def WF (s : Scope) (name : List Char) : Prop :=
  (∀ a ∈ s, a ≠ [] ∧ '/' ∉ a) ∧ '/' ∉ name

/-- The L1.7 witnesses: scope `["a"]` + name `"12"` vs scope `["a1"]` + name `"2"`. -/
def collideA : Scope × List Char := ([['a']], ['1', '2'])
def collideB : Scope × List Char := ([['a', '1']], ['2'])

/-- **L1.7 (regression, by `decide`).** Without the delimiter both flatten to
    `"a12"` — two distinct (scope, name) pairs, one durable key: the aliasing
    class a terminated flattening forecloses. -/
example :
    flattenBad collideA.1 collideA.2 = flattenBad collideB.1 collideB.2
      ∧ collideA ≠ collideB := by decide

/-- The fix, on the *same* witnesses: the terminated flattening separates them
    (`"a/12"` vs `"a1/2"`). -/
example : flatten collideA.1 collideA.2 ≠ flatten collideB.1 collideB.2 := by decide

/-- Core of L1.8: two `/`-free prefixes followed by `/` split a list the same
    way — the "split at the first `/`" argument, as list induction. -/
private theorem split_at_first_slash :
    ∀ (a₁ a₂ t₁ t₂ : List Char), '/' ∉ a₁ → '/' ∉ a₂ →
      a₁ ++ '/' :: t₁ = a₂ ++ '/' :: t₂ → a₁ = a₂ ∧ t₁ = t₂ := by
  intro a₁
  induction a₁ with
  | nil =>
      intro a₂ t₁ t₂ _ h₂ h
      cases a₂ with
      | nil => simpa using h
      | cons c rest =>
          simp only [List.nil_append, List.cons_append, List.cons.injEq] at h
          -- h.1 : '/' = c, so '/' heads a₂ — contradicting its '/'-freedom
          exact absurd (by simp [← h.1]) h₂
  | cons c₁ rest₁ ih =>
      intro a₂ t₁ t₂ h₁ h₂ h
      cases a₂ with
      | nil =>
          simp only [List.cons_append, List.nil_append, List.cons.injEq] at h
          -- h.1 : c₁ = '/', so '/' heads a₁ — contradicting its '/'-freedom
          exact absurd (by simp [h.1]) h₁
      | cons c₂ rest₂ =>
          simp only [List.cons_append, List.cons.injEq] at h
          obtain ⟨hc, hrest⟩ := h
          have h₁' : '/' ∉ rest₁ := fun m => h₁ (List.mem_cons_of_mem _ m)
          have h₂' : '/' ∉ rest₂ := fun m => h₂ (List.mem_cons_of_mem _ m)
          obtain ⟨ha, ht⟩ := ih rest₂ t₁ t₂ h₁' h₂' hrest
          subst hc; subst ha; subst ht
          exact ⟨rfl, rfl⟩

/-- A scoped flattening rearranged to expose its first `/`. -/
private theorem flatten_cons (a : Atom) (s : Scope) (name : List Char) :
    flatten (a :: s) name = a ++ '/' :: flatten s name := by
  simp [flatten, List.flatMap_cons, List.append_assoc]

/-- A cons-headed flattening contains `/`; an empty-scope one is just the name.
    (The disequality engine for the nil-vs-cons cases of L1.8.) -/
private theorem slash_mem_flatten_cons (a : Atom) (s : Scope) (name : List Char) :
    '/' ∈ flatten (a :: s) name := by
  simp [flatten_cons]

/-- **L1.8 (the theorem).** On well-formed inputs the flattening is injective:
    distinct (scope, canonical-name) pairs get distinct durable keys. This is
    the positive half; L1.7 is `flattenBad`'s negative half. -/
theorem flatten_injective_on :
    ∀ (s₁ s₂ : Scope) (n₁ n₂ : List Char),
      WF s₁ n₁ → WF s₂ n₂ →
      flatten s₁ n₁ = flatten s₂ n₂ → s₁ = s₂ ∧ n₁ = n₂ := by
  intro s₁
  induction s₁ with
  | nil =>
      intro s₂ n₁ n₂ w₁ w₂ h
      cases s₂ with
      | nil => exact ⟨rfl, by simpa [flatten] using h⟩
      | cons a rest =>
          -- flatten [] n₁ = n₁ is '/'-free; the right side contains '/'
          exfalso
          have hmem : '/' ∈ flatten (a :: rest) n₂ := slash_mem_flatten_cons a rest n₂
          have hn : flatten ([] : Scope) n₁ = n₁ := by simp [flatten]
          exact w₁.2 (by rw [← h, hn] at hmem; exact hmem)
  | cons a₁ rest₁ ih =>
      intro s₂ n₁ n₂ w₁ w₂ h
      cases s₂ with
      | nil =>
          exfalso
          have hmem : '/' ∈ flatten (a₁ :: rest₁) n₁ := slash_mem_flatten_cons a₁ rest₁ n₁
          have hn : flatten ([] : Scope) n₂ = n₂ := by simp [flatten]
          exact w₂.2 (by rw [h, hn] at hmem; exact hmem)
      | cons a₂ rest₂ =>
          rw [flatten_cons, flatten_cons] at h
          have hfree₁ : '/' ∉ a₁ := (w₁.1 a₁ List.mem_cons_self).2
          have hfree₂ : '/' ∉ a₂ := (w₂.1 a₂ List.mem_cons_self).2
          obtain ⟨ha, ht⟩ := split_at_first_slash a₁ a₂ _ _ hfree₁ hfree₂ h
          have w₁' : WF rest₁ n₁ := ⟨fun a m => w₁.1 a (List.mem_cons_of_mem _ m), w₁.2⟩
          have w₂' : WF rest₂ n₂ := ⟨fun a m => w₂.1 a (List.mem_cons_of_mem _ m), w₂.2⟩
          obtain ⟨hs, hn⟩ := ih rest₂ n₁ n₂ w₁' w₂' ht
          subst ha; subst hs; subst hn
          exact ⟨rfl, rfl⟩

/-! ## Trust check (click these; output renders in the VS Code Infoview)

Same discipline as `Keys.lean`: the `decide` regressions should report
axiom-free; the induction theorem should list at most `propext` — never
`sorryAx` (unfinished) or `Lean.ofReduceBool` (compiler trusted). -/

#check flatten_injective_on
#print axioms flatten_injective_on

-- The encodings side by side: the collision, then the fix.
#eval flattenBad collideA.1 collideA.2  -- "a12".toList
#eval flattenBad collideB.1 collideB.2  -- "a12".toList  ← same ⇒ the aliasing
#eval flatten collideA.1 collideA.2     -- "a/12".toList
#eval flatten collideB.1 collideB.2     -- "a1/2".toList ← differs at the delimiter ⇒ the fix

end Effective.Scopes
