BEGIN TRANSACTION;
CREATE TABLE checkpoints (
  task_id UUID NOT NULL,
  name TEXT NOT NULL,
  state TEXT,
  PRIMARY KEY (task_id, name)
);
INSERT INTO "checkpoints" VALUES('01a12e6a-d837-7381-98e4-a102053150d0','repair:engage;d:0;step:engage','{"result": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207", "INV-1001", "PAY-5501"], "reason": "The proposed $3,000 credit is wrong on three counts under MSA-NW-2025.\n\n1. Commitment threshold (4.1): credits apply only to unavailability beyond 99.9% monthly availability. September has 30 days (720 hours), so the allowance is 0.1% \u00d7 720 = 0.72 hours (about 43 minutes). The outage was 6h15m (TCK-881), so the excess is about 5.53 hours, which gives 5 full hours. The proposal counted all six hours and ignored the commitment.\n\n2. Per-hour amount and cap (4.2): 5 hours \u00d7 0.5% \u00d7 $100,000 = $2,500. Section 4.2 caps credits at $1,000 in any calendar month, so the supported credit is $1,000.\n\n3. Set-off (7.4): the customer may withhold only a credit issued under 4.2. The $3,000 deduction in PAY-5501 (paid $97,000 against $100,000 in INV-1001) is therefore $2,000 more than the contract allows. The $2,000 balance is still owed. INV-1001 was issued 2026-09-01 on Net 30 terms, so it was due 2026-10-01. Under 3.2 it is past due and accrues 1% per month as of today (2026-10-11).\n\nThe claim itself is timely: CRM-2207 (2026-09-26) is within 30 days of the 2026-09-14 incident, as 4.2 requires. Recommended action: issue a $1,000 credit memo, and collect the remaining $2,000 with late accrual. The proposal should be revised to $1,000 and should cite the MSA cap and set-off clauses, not only the customer''s claim."}, "usage": {"prompt_tokens": 2372, "completion_tokens": 2122, "cache_read_input_tokens": 2032, "cache_creation_input_tokens": 338, "cost": 0.00114912, "latency_s": 0.0}}');
INSERT INTO "checkpoints" VALUES('01a12e6a-d837-7381-98e4-a102053150d0','repair:engage;d:1;step:engage','{"result": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207", "PAY-5501"], "reason": "The proposed $3,000 credit is not supported by the contract. Under MSA-NW-2025 \u00a74.2, the credit is 0.5% of the monthly platform fee for each full hour of unavailability beyond the \u00a74.1 commitment, capped at $1,000 per calendar month. The proposal ignores both the 99.9% commitment and the cap.\n\nCalculation: TCK-881 records 6h15m of unavailability on 2026-09-14. A 99.9% monthly commitment over September (720 hours) allows about 0.72 hours of downtime, leaving roughly 5.5 hours beyond the commitment, or 5 full hours. That is 5 \u00d7 0.5% \u00d7 $100,000 = $2,500, which the $1,000 monthly cap reduces to $1,000. Even if all 6 full hours were counted, the result would be 6 \u00d7 $500 = $3,000, which the cap also reduces to $1,000. The supported credit is $1,000 under either reading.\n\nThe customer''s deduction of $3,000 in CRM-2207 is also not permitted beyond the credit. \u00a77.4 prohibits withholding or set-off except a \u00a74.2 credit. Under PAY-5501 the customer paid $97,000 against the $100,000 invoice (INV-1001). With a $1,000 credit memo, the amount due is $99,000, leaving $2,000 open. The customer should be told to remit that balance. CRM-2207 was sent within the 30-day written-request window in \u00a74.2, so the claim itself is valid, but only for $1,000. The account manager''s note in CRM-2210 about renewal does not change the contractual amount."}, "usage": {"prompt_tokens": 2429, "completion_tokens": 1815, "cache_read_input_tokens": 715, "cache_creation_input_tokens": 1712, "cost": 0.0012572500000000001, "latency_s": 0.0}}');
INSERT INTO "checkpoints" VALUES('01a12e6a-d837-7381-98e4-a102053150d0','repair:engage;d:2;step:engage','{"result": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposed $3,000 credit does not follow the contract the firm holds (MSA-NW-2025). Section 4.2 credits only apply to unavailability beyond the 99.9% commitment in 4.1. September has 30 days (720 hours), so the allowance is 0.1% \u00d7 720 = 0.72 hours. The incident ran 6h15m (TCK-881), leaving about 5.53 hours beyond the commitment, or 5 full hours. At 0.5% of the $100,000 monthly fee ($500 per hour), that is $2,500. Section 4.2 also caps credits at $1,000 per calendar month, so the credit is $1,000. The cap binds under any reading: even the customer''s six-hour count would be $3,000 before the cap, which is still limited to $1,000. The proposal''s rationale, which cites only CRM-2207 and the six-hour figure, ignores the 4.1 threshold and the 4.2 cap. Section 7.4 also bars Northwind from setting off any amount except a 4.2 credit, so the $3,000 deduction in PAY-5501 (paid $97,000 against INV-1001''s $100,000) is not authorized. After the correct $1,000 credit, the amount owed is $99,000, which leaves $2,000 unpaid. Under the Net 30 term in 3.2, that balance was due 2026-10-01 and is now past due, so it may accrue 1% per month. The firm should issue a $1,000 credit memo, which is the 4.2 credit, and pursue the $2,000 shortfall. The CRM-2207 request was timely: it was sent 12 days after the incident, well within the 30-day window in 4.2. CRM-2210 reflects a relationship concern and does not change the contract amount, so I did not rely on it."}, "usage": {"prompt_tokens": 2430, "completion_tokens": 2036, "cache_read_input_tokens": 2087, "cache_creation_input_tokens": 341, "cost": 0.00110727, "latency_s": 0.0}}');
CREATE TABLE events (
  name TEXT NOT NULL PRIMARY KEY,
  payload TEXT,
  emitted_at REAL NOT NULL DEFAULT (unixepoch('subsec'))
);
CREATE TABLE ledger (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,
  workflow_run_id TEXT,
  payload TEXT NOT NULL,
  recorded_at REAL NOT NULL,
  hypothetical INTEGER NOT NULL DEFAULT 0, -- counterfactual (D); canonical view filters it
  -- WHO wrote the row (`effective.ops.Writer`). Nullable: a direct writer append (a fork's
  -- genesis/seal) has no placement, and rows written before this column existed have none
  -- either — both must keep meaning 'unknown', which the collision check reads as allow.
  writer_task TEXT,
  writer_placement TEXT
);
CREATE TABLE tasks (
  task_id UUID PRIMARY KEY,
  name TEXT NOT NULL,
  params TEXT NOT NULL,
  state TEXT NOT NULL,
  attempt INTEGER NOT NULL DEFAULT 1,
  max_attempts INTEGER NOT NULL DEFAULT 5,
  available_at REAL NOT NULL DEFAULT 0,
  waiting_event TEXT,
  claimed_by TEXT,
  claim_expires_at REAL,
  result TEXT,
  failure TEXT,
  idempotency_key TEXT
);
INSERT INTO "tasks" VALUES('01a12e6a-d837-7381-98e4-a102053150d0','critique','{}','completed',1,5,1791780116.5359504,NULL,NULL,1791780176.3439412,'{"critique": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposed $3,000 credit does not follow the contract the firm holds (MSA-NW-2025). Section 4.2 credits only apply to unavailability beyond the 99.9% commitment in 4.1. September has 30 days (720 hours), so the allowance is 0.1% \u00d7 720 = 0.72 hours. The incident ran 6h15m (TCK-881), leaving about 5.53 hours beyond the commitment, or 5 full hours. At 0.5% of the $100,000 monthly fee ($500 per hour), that is $2,500. Section 4.2 also caps credits at $1,000 per calendar month, so the credit is $1,000. The cap binds under any reading: even the customer''s six-hour count would be $3,000 before the cap, which is still limited to $1,000. The proposal''s rationale, which cites only CRM-2207 and the six-hour figure, ignores the 4.1 threshold and the 4.2 cap. Section 7.4 also bars Northwind from setting off any amount except a 4.2 credit, so the $3,000 deduction in PAY-5501 (paid $97,000 against INV-1001''s $100,000) is not authorized. After the correct $1,000 credit, the amount owed is $99,000, which leaves $2,000 unpaid. Under the Net 30 term in 3.2, that balance was due 2026-10-01 and is now past due, so it may accrue 1% per month. The firm should issue a $1,000 credit memo, which is the 4.2 credit, and pursue the $2,000 shortfall. The CRM-2207 request was timely: it was sent 12 days after the incident, well within the 30-day window in 4.2. CRM-2210 reflects a relationship concern and does not change the contract amount, so I did not rely on it."}, "refused": null}',NULL,NULL);
CREATE TRIGGER ledger_no_update BEFORE UPDATE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (UPDATE blocked)'); END;
CREATE TRIGGER ledger_no_delete BEFORE DELETE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (DELETE blocked)'); END;
CREATE UNIQUE INDEX tasks_idempotency ON tasks (name, idempotency_key);
CREATE INDEX tasks_by_idempotency_key ON tasks (idempotency_key);
DELETE FROM "sqlite_sequence";
COMMIT;
