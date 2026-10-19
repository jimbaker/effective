# ADR-0028: An approval binds its effect

- **Date:** 2026-10-14
- **Status:** Accepted, and built.
- **Relates to:** [ADR-0027](0027-the-firm-settles-to-a-fixpoint.md) (the terms row this binds),
  [ADR-0018](0018-measured-spend-accrual.md) (the recorded format an answer is read under).

## 1. The decision

An approval covers exactly the effect it was asked about. The approver's ruling names the digest of
that effect, and the gate refuses the op unless its effect has that digest.

| question | answered by |
|---|---|
| what an approval names | the digest of the gated op's effect (§2) |
| where the check runs | in `govern`, when a ruling arrives, before the policy reads it (§3) |
| how a settlement's legs are covered | the approved terms row carries the digest of the plan, and the plan is data (§4) |
| who refused | every refusal names its policy and its ruler (§5) |
| what an older record does | an answer to a binding ask without a digest is refused by name (§6) |

## 2. An effect has a digest

`effect_of(op)` is the canonical value an approval is about: a tool and its arguments, a ledger row,
a wake time. The digest of an effect is `digest_atom(effect_of(op))`, an algorithm-prefixed atom.
Two effects that differ in any field digest apart; the idempotency key is outside the effect, so a
retried step keeps its digest.

## 3. A binding ask, and the ruling that answers it

An `Ask` may bind the effect it asks about: `binds` holds the op's digest, and the ask's row shows
it beside the terms. A ruling on a binding ask names the effect it rules on under `effect`, for an
approval and a denial alike.

| the ruling | the gate |
|---|---|
| names the bound digest | hands the answer to the policy, as for any ask |
| names another digest | refuses the op, `Unbound`, naming the policy, the ruler and both digests |
| names no digest | refuses the op the same way |

The check runs when the ruling arrives, before the gate records who ruled and before any policy
reads the answer, so a ruling on another effect reaches no policy. Its refusal is not a denial: the
ruler rejected nothing, and the op changed under the ruling.

The park's name does not change. A name carrying the digest would park a resumed claim again
under a fresh name after the claim had committed, and nothing would answer it.

The case this closes is a resume that derives different terms. A run approved for credit C crashes
after the ruling and before its claim commits; a resume whose code or knowledge derives C′ parks
under the same name, replay delivers the ruling on C, and the gate refuses C′.

An approver finds the digest on the firm's ask row, which names its park and the frames it runs
under, so two gathered settlements of one case are told apart.

## 4. The settlement's plan is data, and rides in its terms

A settlement is a plan: one entry per leg, each naming the step it sends and the step that undoes
it.

| a step | its form |
|---|---|
| the step's name, the tool and the arguments | data |
| an argument an earlier leg landed, such as the memo's entry | the name of the leg that landed it |

An argument is a JSON value, the form the digest reads, so two plans that digest alike send the
same values; each send gets its own copy, so what a tool does with its arguments leaves the plan as
approved.

One executor builds the saga's legs from the plan. The terms row the approval gates carries `plan`,
the digest of the whole plan, so approving the row approves every step the settlement can send and
every undo. A deploy that changes a step changes the digest:

| where the changed code meets the run | what happens |
|---|---|
| before the claim, or after it on a resume | the gate re-reads the approval and refuses before any leg |
| a continuation of an unlanded collection | `resume` compares the plan with the row's and refuses, `Replanned`, before its leg |
| a resume whose code finds the case clean | the terms ask nobody, so the gate passes them unasked; `settle` compares the plan with the round's claim and refuses, `Replanned`, before any leg |

A refused run fails, and its round stays claimed with no ending, so no later round opens over it;
settling what its landed legs left is a decision for whoever deployed the change.

The executor is the code the approval trusts, as it trusts the ERP adapter: a plan binds what the
legs send, and the executor is the one function that sends it. A grantor is outside the plan, since
it decides how many attempts a collection gets, and the attempts send the step the plan names.

## 5. A refusal names who refused

A gate folds every refusing policy into one verdict, so one refusal can carry several reasons. Each
reason is a `Refusal`: the policy, the sentence, and who ruled when a ruling caused it. `Refused`
carries the tuple; a refusal the substrate raises outside any policy carries none. A binding
refusal gives one per policy whose ask the ruling missed, with a fixed sentence; its message names
the digests.

| the firm reads | as |
|---|---|
| the approval policy refusing alone, with a ruler | a denial, `Denied`, naming the ruler |
| a binding refusal | raised: the op changed under its ruling |
| a budget refusal | raised: every later metered op is refused the same way |
| any other policy's refusal | raised |

A child's refusal climbs a spawn tree with its type and message; its rulers stay in the child.

## 6. Format

An answer to a binding ask that names no digest is refused by name. The recorded format that
carries rulings has not shipped with a binding ask, and no banked run parks at one, so the rule
needs no new format version.

## 7. What this does not rule

| question | what holds |
|---|---|
| a separate authorization type beside the answer's `effect` | not taken; one surface approves, and the answer's `effect` carries its digest |
| the `human` tier's park, which names its op by key | `permission`'s own question; its ask binds nothing |
| revoking an approval | not built |
| a recorded `Proceed`, which would keep a committed op's verdict across a deploy | not built |
| a park a fork renamed, which the ask row's lookup does not find | not built |
| a race branch's refusal, re-served with its sentence alone | not built |
| the ruler of a budget's `stop`, which its refusal does not name | not built |
| a dashboard reader of the firm's ask row | not built |
