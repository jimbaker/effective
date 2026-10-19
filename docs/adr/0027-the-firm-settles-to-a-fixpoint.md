# ADR-0027: The firm settles to a fixpoint

- **Date:** 2026-10-13
- **Status:** Accepted, and built.
- **Relates to:** [ADR-0025](0025-race-and-quorum.md) (the gather a round's cases run in), the
  agent firm ([`wiki/concepts/agent-firm.md`](../../wiki/concepts/agent-firm.md)).

## 1. The decision

The AR firm runs `reconcile`, `resolve` and `settle` until the ledger agrees with itself. Each pass
is a round of the books, a round starts only when the one before it has ended, and what the firm
cannot settle comes back as dissent rather than as an error.

| the firm's question | answered by |
|---|---|
| when may a round start | when the earlier round's ending is a ledger row (§3) |
| what happens to a case the earlier round ended | its ending decides (§3) |
| what a pass reads | the ledger for what the firm did; the books for what it did not (§4) |
| when the passes stop | when the open set stops changing, or the budget runs out (§5) |
| who owns an ERP match two rounds share | every live claim on it; the last release removes it (§2) |

## 2. The match is owned by its claims

An apply claims its key in the ERP, in the transaction that makes the match or finds it standing.
A claim records the partial reconciles it made or adopted, so the claims on one match can be
counted. An unapply releases its key's claim, and removes the match only when that claim was the
last live one; it reports which happened. In Odoo the claim is a row on the invoice and each match
it holds a row on the match, which Odoo deletes with the match, whoever removes it. Every fence
run on an invoice writes that invoice's row first, so two runs on one invoice serialize: Odoo
retries the later one on a snapshot holding the earlier one's claims and holds.

| the unapply finds | it | and answers |
|---|---|---|
| other live claims on the match | releases its claim, leaves the match | released, not removed |
| no other live claim | releases its claim, removes the match | released and removed |

The last release removes the match whoever made it, a match a person made by hand and the firm
adopted included: a compensation undoes the books to what they were before the round's claims,
and the ERP cannot tell an adopted match's history from its own. A match a person removes takes
its holds with it, so the claims on it own nothing until a key holds a match again.

**Falsified if:** through the firm's own ops, an earlier round's late unapply removes a match a
later round's live claim holds; an unapply reports removing a match it held that still stands; or
two fence runs at once leave a match no claim holds, or a hold on a match that is gone.
**Checked by** `test_an_earlier_rounds_undo_leaves_the_match_a_later_round_adopted` and
`test_an_unapply_says_whether_it_removed_the_match`, on scripted and live Odoo, the three races in
`tests/test_agent_firm_erp_port.py` that hold a fence run open on live Odoo, and the claim model
`formal/quint/erp_claim.qnt`'s `liveClaimHasMatch`. The model finds the count and the ending to be
two layers: with rounds that wait and an unlanded round continued, no two rounds' claims are live
at once, so the count is what keeps a match when rounds overlap. The model's step is one whole
fence run, which the serialization makes true.

## 3. Rounds end in the ledger

A settlement appends one more ledger row after its saga, naming its round and its ending:
`completed`, `compensated` or `unlanded`, and a continuation of an unlanded round appends its own.
A round opens only when the round before it, or its last continuation, ended compensated; `settle`
answers `Unopened` otherwise, and a run that finds the earlier round claimed and not ended leaves
the case open as in flight.

| round k ended | round k+1 |
|---|---|
| completed | closes the case |
| superseded | follows its holder's ending, which is the round's |
| compensated | reopens the case, under round k+1's keys |
| stopped (a model gave no usable answer) | asks again |
| unlanded (a retriable leg ran out) | continues round k's retriable leg under round k's keys; no new round |
| denied (the approver refused) | leaves the case open, denied |

Within one round, the first approved proposal the run resumes owns it. A gathered park exposes one
wake, its lowest parked branch's, and no engine waits on any of several events, so an approver who
wants a different proposal to own the round refuses the first.

**Falsified if:** a round's settlement starts before the earlier round's ending is a row; a
completed case is credited again; an unlanded case opens a new round.
**Checked by** `test_a_round_waits_for_the_earlier_round_to_end`,
`test_a_round_does_not_open_after_one_that_completed_or_ended_unlanded`, a pin per ending in
`tests/test_agent_firm_fixpoint.py`, and the claim model's `noRoundBeforeEnded` and `creditedOnce`.

## 4. What a pass reads

A case's state is the ledger's: its latest round's terms and ending. A settled case can owe money
(a credit smaller than the deduction leaves the rest owed and reminded), so the books cannot say
whether the firm is done with it. The books say what the firm did not do, such as a second payment,
through a recorded port op, `Read(invoice) -> Statement`, which a replay serves without the ERP.

**Falsified if:** a pass reopens a settled case because its invoice still owes; or a replay calls
the ERP to read.
**Checked by** `test_a_read_answers_the_books_as_they_stand_under_any_key` on both arms,
`test_a_case_whose_invoice_was_paid_closes_asking_no_model`, and
`test_a_replayed_pass_reads_its_statement_from_the_record`.

## 5. The fixpoint, and dissent

`firm(books, cases)` is the substrate's `fixpoint`, scoped `firm:{books}` so the `retry:` and
`repair:` fixpoints inside a settlement keep their own levels. Each iteration is a pass over the
open cases, and each case is in the round its ledger says. The value compared between two passes
is the open cases, each with its round and what keeps it open, and the firm does not converge
while an open case is unlanded or compensated, since a pass can still run it:

| an open case's last disposition | means |
|---|---|
| stopped | a model gave no usable answer |
| denied | the approver refused the proposal |
| in flight | another run holds a round that has not ended |
| unlanded | a retriable leg ran out; the next pass continues it |
| compensated | the settlement was undone; the next pass resolves the case in the next round |

`Converged` with open cases is preserved dissent, each open case stopped, denied or in flight:
what the firm agreed and what it did not. `Unconverged` is the budget running out while a case
still moved or could still be run. A denial is a value of its case, and only the approval's
refusal of the round's claim is one, so one denied case leaves its siblings settling.

Two runs of the firm over the same books, the first converged with every case closed, equal one in
the books and in what the customer is told. A case the first left stopped is asked again by the
second, and may settle there. They differ in recorded reads and ERP calls, since each read is a step of
the run that made it.

**Falsified if:** the firm, run again after it converged with every case closed, changes the books
or tells the customer again; a denial ends the run; the firm converges with a case it could still run; the firm fails to
converge on the canonical cases within two passes, three with a stopped case.
**Checked by** the convergence pins and the done pins in `tests/test_agent_firm_fixpoint.py`: the
canonical firm run twice, two runs of one case in every schedule within one switch, and one run
crashed at every op, each settling a credit that leaves the invoice owing.

## 6. What this does not rule

| question | where it goes |
|---|---|
| binding an approval to its effect and its compensation as one pair | not built; this ADR fixes the terms row it binds |
| waking a gathered park on any branch's ask | not built; no engine wakes a gathered park on any branch's ask |
| keying the payment's apply without the round, which dissolves adoption inside the firm | not taken; the payment's apply is keyed by its round |
| `settle` as a substrate combinator choosing within an envelope | not built; the firm's loop is named `firm` |
