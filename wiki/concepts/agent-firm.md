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

| leg | kind | ERP op or tool | undone by |
|---|---|---|---|
| `credit-memo` | compensable | `PostMemo` | `Reverse` |
| `apply-payment` | compensable | `Apply` | `Unapply` |
| `apply-memo` | compensable | `Apply` of the memo's entry | `Unapply` |
| `notify` | pivot | the customer's notice | nothing |
| `collect` | retriable | the collection of what is still owed | nothing |

Every leg is a step keyed by the fact it is about: the case's invoice and payment in a round of
its books, the leg, and its attempt. A step resumed after a crash reaches the ERP or the customer
under the key it was first sent with. One settlement owns a round: a second settlement of it
performs nothing, so no other settlement's apply can land between the owner's unapply of the memo
and its reversal. A settlement that cannot recover is another process's to finish. A case reopened after
a settlement that was compensated or left unlanded is resolved in the next round, under fresh
keys. A refused spend ends a retry or a repair and is raised, since every later metered op is refused the same way.
`settlement` takes a grantor for the collection, and no resolution passes one yet.

## Knowledge and the account

K is a stock. A snapshot of the firm's rules is read by digest at a recorded step, so a replay
reads the rules the first attempt read whatever the store has gained since. An adoption reads its parent's row, then claims
`knowledge-adopted:{books},{height}` on the ledger, naming parent, child and adopter: one whose
parent is not the snapshot at the height below is unlinked, the first adopter to claim a height
holds it, and one with another child is outrun. A read answers from the reader's own lineage, and a
claim of an id the other lineage holds is refused, so a hypothetical adoption neither moves the
canonical head nor answers a canonical adopter. The driver folds the
head by reading adoptions upward and admits a run there; the run never reads the head itself.

`account.py` projects one resolution's factors from the record that holds each: assignment from
the manifest, H from the ask and ruling rows of the run's own lineage, A and C from the tape's
model steps, K from the pin. `score.py` reads Y off the books against each case's gold credit, outside the firm, so a
superseded run scores as the credit its round holds. `interaction` measures the 2×2 of K and the
critic from each run's manifest, given both arms' snapshots, and refuses a stray run, a doubled
cell, or a missing one.

## The ERP port

`erp/port.py` is a typed port: its ops are answered by `Posted`, `Applied`, `Unapplied` or
`Refused`. One adapter (`erp/odoo.py`) runs over two transports, XML-RPC to Odoo Community in the
pod `just odoo-up` starts, and a scripted Odoo (`erp/scripted.py`) that answers the same calls in
memory.

| clause | what holds |
|---|---|
| a post or a reversal lands once per key | the key is claimed by creating its unique `ir.model.data` row; a draft loaded without the key loses to the claim and is deleted |
| an apply or an unapply is read from its match | its key names nothing in the ERP: the match it makes is the state the replay reads |
| a refusal is data | `Refused` carries a reason and the values it is about, and nothing lands |
| a key names one move | an op whose key names a move differing in kind, customer, amount or original is `Refused`, and the move stands |
| a fault is unknown | a call whose outcome could not be read raises `Unknown`, and the replay resolves it |

## The questions the tests answer

| question | what holds | pin |
|---|---|---|
| R1 | a settlement posts one memo and applies it and the payment once, at every crash point, on both engines and both ERPs, so the invoice owes the gap less the credit | `tests/test_agent_firm_saga.py`, `tests/test_agent_firm_resolve.py` |
| R2 | the firm's committed effects do not read the notes and tickets, over generated variants of them | `tests/test_agent_firm_resolve.py` |
| R3 | a critique changes the run at its own result and nowhere else; an uncited one posts nothing | `tests/test_agent_firm_resolve.py` |
| R4 | a critic finds the contract's cap, and its run replays without asking a model | `tests/test_agent_firm_engage.py` pins the program; a live model is measured outside the suite, three haiku runs banked and each replayed by `research/agent_firm/probe_r4.py --verify` |
| R5 | an approval by the proposer does not count, and the memo parks again | `tests/test_agent_firm_resolve.py` |
| R11 | a rejected notice unapplies the memo and the payment, then reverses the memo, every entry posted and matched | `tests/test_agent_firm_saga.py` |
| R12 | a saga whose order cannot be compensated is refused before it runs | `tests/test_agent_firm_saga.py` |
| R13 | a reversal whose period locked posts in the open period | `tests/test_agent_firm_saga.py` |
| twice | a case resolved twice in one round settles once, in sequence or gathered, and a second settlement of the round makes no ERP call, whichever claims first; a second credit is superseded and does nothing; a denied or unanswered credit holds no round; the next round settles a case the first left unfinished | `tests/test_agent_firm_resolve.py` |
| admitted | a run reads the case its manifest admitted, on every attempt and every replay | `tests/test_agent_firm_resolve.py` |
| K | an old run replays after the head advances; a no-op delta briefs byte for byte; two adopters leave one row; a mismatched snapshot is refused; hypothetical arms in books of their own leave the canonical round alone | `tests/test_agent_firm_knowledge.py` |
| account | model calls match the world's, short only by a crash's lost record; each ruler is named from the ledger alone; the books score a wrong credit wrong; the example's own run, spawned under the v1 contract, spends what its tape recorded | `tests/test_agent_firm_account.py`, `tests/test_agent_firm_example.py` |
| 2×2 | the instrument measures no interaction in a null or additive world, flips with its labels, refuses a missing cell, and misses an interaction read off completion | `tests/test_agent_firm_interaction.py` |

The port's own suite, `tests/test_agent_firm_erp_port.py`, faults every call each op makes, before
it and after it committed, and explores every schedule of two overlapping attempts of one op under
one key with at most three switches (`tests/_turns.py`). `just odoo-test` runs the live arms.

## What it does not show

| | why |
|---|---|
| another vendor | QuickBooks and Xero keep keys for a limited time; the port's contract assumes the ERP keeps them |
| the firm improving its own knowledge | an adoption is made by hand; `evolve`, and the complementarity of agent labor and organizational knowledge it would test, are proposed. The 2×2 runs on scripted worlds, which tests the instrument and nothing about the firm |
