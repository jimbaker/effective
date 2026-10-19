# The agent firm

An agent firm is a top-down organization of agents whose substrate keeps the tape, the budget and
the verdict. Its worked example is an accounts-receivable firm, `src/examples/agent_firm/`: a
customer pays short of an invoice, and the firm explains the gap, has an outside critic check
the explanation against the contract, parks the settlement for an approver other than its
proposer, and settles the case by saga through an ERP port. `uv run python -m
examples.agent_firm sla-cap` runs one case; the module docstring has the others.

## The composition

| stage | combinator | module |
|---|---|---|
| admit | a manifest of the run's inputs in its spawn params: the case by content, the round, the knowledge snapshot it reads, the arm, code, models and budget | `manifest.py` |
| know | `load_knowledge`: the pinned snapshot of the firm's rules, read at a recorded step and refused when its contents do not hash to the pin | `knowledge.py` |
| observe | `reconcile`: a `gather` over the cases, each held to the invariants | `reconcile.py` |
| explain | an ask through a `Gated` channel: a credit within the shortfall, on the case's documents and the firm's rules | `engage.py`, `resolve.py` |
| engage | an outside critic's review of the proposal, re-asked with the reason when a citation is not held: a `fixpoint` under `repair:engage`, one level per ask | `engage.py`, `critic.py` |
| claim | the round, read and then claimed on the ledger by the settlement's claimant, a name minted from its task and placement that replay reads back: a settlement whose round another holds ends `Superseded` whatever its credit, before it asks when the read finds the holder and at the claim when the holder claimed after the read | `ar.py` |
| authorize | a `govern` policy parking the claim on any case that breaks an invariant, a $0 credit included, until a principal other than its proposer approves it; the ask names the credit and the deduction, so a credit nobody approved holds no round; each ask and each ruling is a ledger row, paired by pass | `resolve.py` |
| act | `saga`: compensable legs, at most one pivot, then retriable legs, checked at construction; a retriable leg is a `fixpoint` under `retry:{leg}`, one level per attempt, which a grantor may extend | `saga.py`, `ar.py` |
| end | how the settlement ended, appended to the ledger once its saga does: completed, compensated or unlanded; a round after the first opens only once the round before it ended compensated | `ar.py` |
| settle the books | `firm`: a `fixpoint` under `firm:{books}` whose passes advance every open case concurrently by what the ledger says of its rounds, until two passes leave the same cases open in the same rounds for the same reasons, none of them one a pass can still run (ADR-0027) | `firm.py` |

| leg | kind | ERP op or tool | undone by |
|---|---|---|---|
| `credit-memo` | compensable | `PostMemo` | `Reverse` |
| `apply-payment` | compensable | `Apply` | `Unapply` |
| `apply-memo` | compensable | `Apply` of the memo's entry | `Unapply` |
| `notify` | pivot | the customer's notice | nothing |
| `remind` | retriable | a collection notice for what is still owed | nothing |

Every leg is a step keyed by the fact it is about: the case's invoice and payment in a round of
its books, the leg, and its attempt. A step resumed after a crash reaches the ERP or the customer
under the key it was first sent with. One settlement owns a round: a second settlement of it
performs nothing, so no other settlement's apply can land between the owner's unapply of the memo
and its reversal. A settlement that cannot recover is another process's to finish. A refused spend
ends a retry or a repair and is raised, since every later metered op is refused the same way.
`settlement` takes a grantor for the collection, and no resolution passes one yet.

## The firm settles to a fixpoint

A case's state is the ledger's, so a pass reads it round by round and acts on what it finds.

| the ledger says of the case | the pass | the case is |
|---|---|---|
| its latest round completed | does nothing | closed |
| a round claimed and not ended | does nothing | open, in flight |
| its latest round ended unlanded | continues the leg that did not land, under the round's keys, as a numbered continuation | as the leg ends |
| its latest round compensated | resolves the case in the next round | as resolved |
| no claim of its first open round | reads the invoice's statement, and resolves the case there unless the invoice owes nothing | as resolved |

A settled case can still owe, since a credit smaller than the deduction leaves the rest reminded,
so the books cannot say whether the firm is done with a case; the statement says what the firm did
not do, such as a second payment. A denied case stays open and the next pass asks nobody; only
the approval's refusal of the round's claim is a denial. `Converged` with open cases is the firm's
dissent, each open case stopped, denied or in flight; the firm passes again while a case is
unlanded or compensated, and answers `Unconverged` when the budget runs out first. Two runs of the
firm, the first converged with every case closed, equal one in the books and in what the customer
is told; they differ in
the reads each records.

## Knowledge and the factors

Knowledge (K) is a stock. A snapshot of the firm's rules is read by digest at a recorded step, so a replay
reads the rules the first attempt read whatever the store has gained since. An adoption reads its parent's row, then claims
`knowledge-adopted:{books},{height}` on the ledger, naming parent, child and adopter: one whose
parent is not the snapshot at the height below is unlinked, the first adopter to claim a height
holds it, and one with another child is outrun. A read answers from the reader's own lineage, and a
claim of an id the other lineage holds is refused, so a hypothetical adoption neither moves the
canonical head nor answers a canonical adopter. The driver folds the
head by reading adoptions upward and admits a run there; the run never reads the head itself.

`factors.py` projects one resolution's factors of production from the record that holds each:
assignment from the manifest, human judgment (H) from the ask and ruling rows of the run's own
lineage, agent labor (A) from the tape's model steps and capital (C) from their recorded spend,
knowledge (K) from the pin. `score.py` reads the outcome (Y) off the books against each case's gold
credit, outside the firm, so a superseded run scores as the credit its round holds. `interaction`
measures how much more the critic moves the outcome with the SLA rule held than without, from each
run's manifest, given both arms' snapshots, and refuses a stray run, a doubled outcome, or a
missing one. The letters are the production function's:

## The ERP port

`erp/port.py` is a typed port: its ops are answered by `Posted`, `Applied`, `Unapplied`,
`Statement` or `Refused`. One adapter (`erp/odoo.py`) runs over two transports, XML-RPC to Odoo Community in the
pod `just odoo-up` starts, and a scripted Odoo (`erp/scripted.py`) that answers the same calls in
memory.

| clause | what holds |
|---|---|
| a post or a reversal lands once per key | the key is claimed by creating its unique `ir.model.data` row; a draft loaded without the key loses to the claim and is deleted |
| an apply lands once per key, and an unapply once per apply it names | each claims the apply's key in an `ir.model.data` module naming what it records, from one server action that makes or removes the match in the same transaction; an apply whose undo's claim is held is `Refused`, as a post whose memo was reversed is |
| a match is owned by its claims | an apply holds each match it made or found, in a row Odoo deletes with the match; an unapply releases its key's holds and removes a match no other key holds, and says whether it removed it; the last release removes the match whoever made it |
| fence runs on one invoice serialize | each writes the invoice's row first, so Odoo retries the later of two overlapping runs on a snapshot holding the earlier one's claims and holds; three races held open on live Odoo pin it |
| a read lands nothing | `Read` answers what the invoice owes and the entries on its customer's account, as they stand, under any key |
| a refusal is data | `Refused` carries a reason and the values it is about, and nothing lands |
| a key names one move | an op whose key names a move differing in kind, customer, amount or original is `Refused`, and the move stands |
| a fault is unknown | a call whose outcome could not be read raises `Unknown`, and the replay resolves it |

## The controls the tests check

| control | what holds | pin |
|---|---|---|
| crash-safe settlement | a settlement posts one memo and applies it and the payment once, at every crash point, on both engines and both ERPs, so the invoice owes the gap less the credit | `tests/test_agent_firm_saga.py`, `tests/test_agent_firm_resolve.py` |
| evidence doesn't move money | the firm's committed effects do not read the notes and tickets, over generated variants of them | `tests/test_agent_firm_resolve.py` |
| a review acts through its answer | a critique changes the run at its own result and nowhere else; an uncited one posts nothing | `tests/test_agent_firm_resolve.py` |
| deduction review | a critic finds the contract's cap on the customer's deduction, and its run replays without asking a model | `tests/test_agent_firm_engage.py` pins the program, and replays the three haiku runs `src/examples/agent_firm/deduction_review.py` banked from a live model, each finding the cap |
| segregation of duties | an approval by the proposer does not count, and the memo parks again | `tests/test_agent_firm_resolve.py` |
| approved as proposed | the attempt resumed after the approval replays the recorded proposal and review, and asks no model | `tests/test_agent_firm_resolve.py` |
| rejection unwinds | a rejected notice unapplies the memo and the payment, then reverses the memo, every entry posted and matched | `tests/test_agent_firm_saga.py` |
| each apply once | an apply sent by an attempt whose lease expired lands nothing after the compensation that undid it, for the payment and the memo; each undo names the apply it releases, and an earlier round's undo leaves a match a later round holds. `formal/quint/erp_claim.qnt` checks the protocol over every interleaving of two attempts in each of two rounds; the books hold wherever a first attempt's lease expires and wherever its held op lands, on both engines | `tests/test_agent_firm_stale_apply.py`, `tests/test_agent_firm_erp_port.py` |
| credit within the shortfall | a settlement refuses a credit its payment's shortfall does not absorb, in whole cents from $0 to the shortfall, before anything is claimed or applied; the firm's proposal is held to the same rule | `tests/test_agent_firm_resolve.py` |
| no step without its undo | a saga whose order cannot be compensated is refused before it runs | `tests/test_agent_firm_saga.py` |
| period close | a reversal whose period locked posts in the open period | `tests/test_agent_firm_saga.py` |
| no duplicate settlement | a case resolved twice in one round settles once, in sequence or gathered, and a second settlement of the round makes no ERP call, whichever claims first, in every schedule of their ops within two switches and every crash at one; a second credit is superseded and does nothing; a denied or unanswered credit holds no round; a round starts only once the round before it has ended | `tests/test_agent_firm_resolve.py`, `tests/test_agent_firm_saga.py` |
| the books settle | the firm settles the canonical cases in two passes, three when a proposal stops once; a case stopped every time, or denied, stays open beside its settling siblings; a completed case is not credited again, a compensated one settles in the next round, and an unlanded one continues its round | `tests/test_agent_firm_fixpoint.py` |
| twice is once | two runs of the firm, the first converged with every case closed, equal one in the books and the customer's notices: the canonical firm run twice, two runs of one case in every schedule within one switch, and one run crashed at every op | `tests/test_agent_firm_fixpoint.py` |
| admitted case | a run reads the case its manifest admitted, on every attempt and every replay | `tests/test_agent_firm_resolve.py` |
| knowledge snapshot | an old run replays after the head advances; a no-op delta briefs byte for byte; two adopters leave one row; a mismatched snapshot is refused; hypothetical arms in books of their own leave the canonical round alone | `tests/test_agent_firm_knowledge.py` |
| factors | model calls match the world's, short only by a crash's lost record; each ruler is named from the ledger alone; the books score a wrong credit wrong; the example's own run, spawned under the v1 contract, spends what its tape recorded | `tests/test_agent_firm_factors.py`, `tests/test_agent_firm_example.py` |
| interaction test | the instrument measures no interaction in a null or additive world, flips with its labels, refuses a missing outcome, and misses an interaction read off completion | `tests/test_agent_firm_interaction.py` |

The port's own suite, `tests/test_agent_firm_erp_port.py`, faults every call each op makes, before
it and after it committed, and explores every schedule of two overlapping attempts of one op under
one key with at most three switches (`tests/_turns.py`). `just odoo-test` runs the live arms.

## What it does not show

| | why |
|---|---|
| another vendor | QuickBooks and Xero keep keys for a limited time; the port's contract assumes the ERP keeps them |
| an approval bound to its effect and its compensation | the approval names the claim of the round; binding it to the effect and its undo as one pair is not built |
| a gathered park waking on any branch | a gathered park exposes its lowest branch's ask alone, so the first approved proposal the run resumes owns a round; no engine wakes a gathered park on any branch |
| the firm improving its own knowledge | an adoption is made by hand; `evolve`, and the complementarity of agent labor and organizational knowledge it would test, are proposed. The interaction test runs on scripted worlds, which tests the instrument and nothing about the firm |
