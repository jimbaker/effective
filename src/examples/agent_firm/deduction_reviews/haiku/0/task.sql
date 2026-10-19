BEGIN TRANSACTION;
CREATE TABLE checkpoints (
  task_id UUID NOT NULL,
  name TEXT NOT NULL,
  state TEXT,
  PRIMARY KEY (task_id, name)
);
INSERT INTO "checkpoints" VALUES('01a12e6a-abaa-7144-8e1f-d102f8710115','repair:engage;d:0;step:engage','{"result": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposed $3,000 credit is not supported by the contract. MSA 4.2 sets the credit at 0.5% of the monthly platform fee ($100,000, per 3.1) for each full hour of unavailability beyond the 99.9% commitment in 4.1, and caps it at $1,000 per calendar month. The ticket (TCK-881) shows 6h15m of unavailability on 2026-09-14, which is 6 full hours. Even after netting out the commitment allowance (0.1% of September''s 43,200 minutes, about 43 minutes), at least 5 full hours fall outside it. Any count of 2 or more hours hits the $1,000 cap, so the credit is $1,000. The customer''s own calculation (6 hours \u00d7 0.5% = $3,000) ignores the cap. The customer''s 30-day written claim (CRM-2207, dated 2026-09-26) is timely, so the $1,000 credit is valid. The customer''s $3,000 deduction is not permitted: MSA 7.4 bars withholding or set-off except a 4.2 credit, and 4.2 makes credits the sole remedy. The proposal also cites CRM-2207 and TCK-881 but omits MSA-NW-2025, the governing document. The correct resolution is a $1,000 credit memo against INV-1001 (adjusted balance $99,000). Against the $97,000 received (PAY-5501), $2,000 remains open and past due (Net 30 from 2026-09-01 ran to 2026-10-01, per 3.2), so it should be collected, with the 1% monthly late accrual applied as the contract allows. The account manager''s note (CRM-2210) asks for goodwill but does not change the contractual amount, so it is not relied on."}, "usage": {"prompt_tokens": 2373, "completion_tokens": 2293, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 2371, "cost": 0.0016208999999999998, "latency_s": 0.0}}');
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
INSERT INTO "tasks" VALUES('01a12e6a-abaa-7144-8e1f-d102f8710115','critique','{}','completed',1,5,1791780105.1308816,NULL,NULL,1791780146.4753604,'{"critique": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposed $3,000 credit is not supported by the contract. MSA 4.2 sets the credit at 0.5% of the monthly platform fee ($100,000, per 3.1) for each full hour of unavailability beyond the 99.9% commitment in 4.1, and caps it at $1,000 per calendar month. The ticket (TCK-881) shows 6h15m of unavailability on 2026-09-14, which is 6 full hours. Even after netting out the commitment allowance (0.1% of September''s 43,200 minutes, about 43 minutes), at least 5 full hours fall outside it. Any count of 2 or more hours hits the $1,000 cap, so the credit is $1,000. The customer''s own calculation (6 hours \u00d7 0.5% = $3,000) ignores the cap. The customer''s 30-day written claim (CRM-2207, dated 2026-09-26) is timely, so the $1,000 credit is valid. The customer''s $3,000 deduction is not permitted: MSA 7.4 bars withholding or set-off except a 4.2 credit, and 4.2 makes credits the sole remedy. The proposal also cites CRM-2207 and TCK-881 but omits MSA-NW-2025, the governing document. The correct resolution is a $1,000 credit memo against INV-1001 (adjusted balance $99,000). Against the $97,000 received (PAY-5501), $2,000 remains open and past due (Net 30 from 2026-09-01 ran to 2026-10-01, per 3.2), so it should be collected, with the 1% monthly late accrual applied as the contract allows. The account manager''s note (CRM-2210) asks for goodwill but does not change the contractual amount, so it is not relied on."}, "refused": null}',NULL,NULL);
CREATE TRIGGER ledger_no_update BEFORE UPDATE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (UPDATE blocked)'); END;
CREATE TRIGGER ledger_no_delete BEFORE DELETE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (DELETE blocked)'); END;
CREATE UNIQUE INDEX tasks_idempotency ON tasks (name, idempotency_key);
CREATE INDEX tasks_by_idempotency_key ON tasks (idempotency_key);
DELETE FROM "sqlite_sequence";
COMMIT;
