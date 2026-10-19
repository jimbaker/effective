BEGIN TRANSACTION;
CREATE TABLE checkpoints (
  task_id UUID NOT NULL,
  name TEXT NOT NULL,
  state TEXT,
  PRIMARY KEY (task_id, name)
);
INSERT INTO "checkpoints" VALUES('01a14d45-b9aa-7636-8b2d-11255b5668f6','repair:engage;d:0;step;llm:engage','{"result": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207", "INV-1001", "PAY-5501"], "reason": "The proposed $3,000 credit does not match the contract. Clause 4.2 caps the service credit at $1,000 per calendar month, and the cap binds under any reading of the clause. The proposal''s arithmetic (six hours at 0.5% of the fee, or $3,000) ignores both the cap and the threshold. 4.2 pays only for unavailability beyond the 4.1 commitment of 99.9% monthly availability. In September (30 days, 43,200 minutes), that commitment allows about 43.2 minutes of downtime. The TCK-881 outage was 6h15m (375 minutes), leaving about 331.8 minutes beyond the commitment, which is five full hours. Five hours at 0.5% of $100,000 is $2,500, which the $1,000 cap reduces to $1,000. Even if every outage hour were counted with no threshold, six hours would be $3,000 before the cap, so the correct credit is $1,000. The claim was timely: CRM-2207 (2026-09-26) is within 30 days of the 2026-09-14 incident, so the credit is valid and the customer is entitled to a $1,000 credit memo. Clause 7.4 prohibits any other withholding or set-off. The customer''s $3,000 deduction is therefore $2,000 beyond what 4.2 allows. Against the invoice INV-1001 ($100,000, issued 2026-09-01, Net 30, so due 2026-10-01), the net amount due after the $1,000 credit is $99,000. PAY-5501 paid $97,000, so $2,000 remains past due as of 2026-10-17, and late charges under 3.2 may apply. The account manager''s note (CRM-2210) about renewal is not a contractual basis for the amount and was not relied on. The proposal should be revised to a $1,000 credit memo, with a demand for the remaining $2,000 and a note that the customer''s deduction exceeded clause 4.2."}, "usage": {"prompt_tokens": 2373, "completion_tokens": 2139, "cache_read_input_tokens": 2032, "cache_creation_input_tokens": 339, "cost": 0.00115782, "latency_s": 0.0}}');
INSERT INTO "checkpoints" VALUES('01a14d45-b9aa-7636-8b2d-11255b5668f6','repair:engage;d:1;step;llm:engage','{"result": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposed $3,000 credit does not match the contract. Under MSA-NW-2025 \u00a74.2, the credit is 0.5% of the monthly fee ($100,000, \u00a73.1, so $500) for each full hour of unavailability beyond the \u00a74.1 commitment, capped at $1,000 per calendar month. Three errors in the proposal:\n\n1. It ignores the 99.9% availability commitment. September has 30 days, or 43,200 minutes. The allowed downtime is 0.1%, which is 43.2 minutes. TCK-881 records 6h15m (375 minutes) of unavailability, so 331.8 minutes exceed the commitment. That is 5 full hours, which gives 5 \u00d7 $500 = $2,500 before the cap.\n2. It ignores the cap. \u00a74.2 limits credits to $1,000 in any calendar month, so the supportable credit is $1,000.\n3. It treats the customer''s claimed deduction as a right. \u00a77.4 bars set-off except a \u00a74.2 credit. The $2,000 difference between the $3,000 claimed and the $1,000 credit is not a permitted deduction, and the customer may not withhold it.\n\nThe credit was requested in writing on 2026-09-26 (CRM-2207), within the 30-day window in \u00a74.2, so the request is timely. The credit memo should be $1,000. After it, the invoice INV-1001 balance is $99,000. The $97,000 payment leaves $2,000 unpaid. That balance has been past the net-30 due date of 2026-10-01 (\u00a73.2) since then, so it is overdue, and the 1% monthly late charge in \u00a73.2 may apply to it. The account manager''s note (CRM-2210) about the Q1 renewal does not change the contract terms. The proposal''s citations should be corrected: the operative source is MSA-NW-2025, and CRM-2207 records only the customer''s claim."}, "usage": {"prompt_tokens": 2429, "completion_tokens": 2601, "cache_read_input_tokens": 715, "cache_creation_input_tokens": 1712, "cost": 0.00165025, "latency_s": 0.0}}');
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
INSERT INTO "tasks" VALUES('01a14d45-b9aa-7636-8b2d-11255b5668f6','critique','{"__recorded_format__": "v1"}','completed',1,5,1792297777.5787475,NULL,NULL,1792297831.0246892,'{"critique": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposed $3,000 credit does not match the contract. Under MSA-NW-2025 \u00a74.2, the credit is 0.5% of the monthly fee ($100,000, \u00a73.1, so $500) for each full hour of unavailability beyond the \u00a74.1 commitment, capped at $1,000 per calendar month. Three errors in the proposal:\n\n1. It ignores the 99.9% availability commitment. September has 30 days, or 43,200 minutes. The allowed downtime is 0.1%, which is 43.2 minutes. TCK-881 records 6h15m (375 minutes) of unavailability, so 331.8 minutes exceed the commitment. That is 5 full hours, which gives 5 \u00d7 $500 = $2,500 before the cap.\n2. It ignores the cap. \u00a74.2 limits credits to $1,000 in any calendar month, so the supportable credit is $1,000.\n3. It treats the customer''s claimed deduction as a right. \u00a77.4 bars set-off except a \u00a74.2 credit. The $2,000 difference between the $3,000 claimed and the $1,000 credit is not a permitted deduction, and the customer may not withhold it.\n\nThe credit was requested in writing on 2026-09-26 (CRM-2207), within the 30-day window in \u00a74.2, so the request is timely. The credit memo should be $1,000. After it, the invoice INV-1001 balance is $99,000. The $97,000 payment leaves $2,000 unpaid. That balance has been past the net-30 due date of 2026-10-01 (\u00a73.2) since then, so it is overdue, and the 1% monthly late charge in \u00a73.2 may apply to it. The account manager''s note (CRM-2210) about the Q1 renewal does not change the contract terms. The proposal''s citations should be corrected: the operative source is MSA-NW-2025, and CRM-2207 records only the customer''s claim."}, "refused": null}',NULL,NULL);
CREATE TRIGGER ledger_no_update BEFORE UPDATE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (UPDATE blocked)'); END;
CREATE TRIGGER ledger_no_delete BEFORE DELETE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (DELETE blocked)'); END;
CREATE UNIQUE INDEX tasks_idempotency ON tasks (name, idempotency_key);
CREATE INDEX tasks_by_idempotency_key ON tasks (idempotency_key);
DELETE FROM "sqlite_sequence";
COMMIT;
