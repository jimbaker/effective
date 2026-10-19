BEGIN TRANSACTION;
CREATE TABLE checkpoints (
  task_id UUID NOT NULL,
  name TEXT NOT NULL,
  state TEXT,
  PRIMARY KEY (task_id, name)
);
INSERT INTO "checkpoints" VALUES('01a12e6b-4ce7-72a0-a406-8660f0f6d3d8','repair:engage;d:0;step:engage','{"result": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposed $3,000 is wrong on three counts. (1) Section 4.2 credits only downtime beyond the 99.9% monthly commitment. September has 30 days (720 hours), so the allowance is 0.72 hours. TCK-881 records 6h15m of unavailability (02:10 to 08:25 UTC), leaving about 5.53 hours beyond the commitment. Only full hours count, so that is 5 full hours, not 6. (2) The credit is 0.5% of the $100,000 monthly fee per hour, so 5 hours yields $2,500. (3) Section 4.2 caps credits at $1,000 per calendar month, so the credit is $1,000. The customer''s six-hour, $3,000 claim ignores both the commitment threshold and the cap. Section 7.4 bars set-off except a 4.2 credit, so the customer''s $2,000 deduction beyond the $1,000 credit is not a permitted withhold. The customer''s claim was timely: CRM-2207 (2026-09-26) is within 30 days of the 2026-09-14 incident, and the request was in writing. Net result: credit memo $1,000; invoice INV-1001 is $100,000 less $1,000 = $99,000 due, of which $97,000 was paid, leaving $2,000 past due (Net 30 from 2026-09-01 ran to 2026-10-01), which accrues 1% per month under 3.2. Assumption: the 30-day month and the hour-counting reading of 4.2 are the most natural ones; if the contract intends a different availability basis, the credit could change, but it cannot exceed $1,000."}, "usage": {"prompt_tokens": 2375, "completion_tokens": 1687, "cache_read_input_tokens": 2032, "cache_creation_input_tokens": 341, "cost": 0.00093222, "latency_s": 0.0}}');
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
INSERT INTO "tasks" VALUES('01a12e6b-4ce7-72a0-a406-8660f0f6d3d8','critique','{}','completed',1,5,1791780146.4070742,NULL,NULL,1791780184.6696594,'{"critique": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposed $3,000 is wrong on three counts. (1) Section 4.2 credits only downtime beyond the 99.9% monthly commitment. September has 30 days (720 hours), so the allowance is 0.72 hours. TCK-881 records 6h15m of unavailability (02:10 to 08:25 UTC), leaving about 5.53 hours beyond the commitment. Only full hours count, so that is 5 full hours, not 6. (2) The credit is 0.5% of the $100,000 monthly fee per hour, so 5 hours yields $2,500. (3) Section 4.2 caps credits at $1,000 per calendar month, so the credit is $1,000. The customer''s six-hour, $3,000 claim ignores both the commitment threshold and the cap. Section 7.4 bars set-off except a 4.2 credit, so the customer''s $2,000 deduction beyond the $1,000 credit is not a permitted withhold. The customer''s claim was timely: CRM-2207 (2026-09-26) is within 30 days of the 2026-09-14 incident, and the request was in writing. Net result: credit memo $1,000; invoice INV-1001 is $100,000 less $1,000 = $99,000 due, of which $97,000 was paid, leaving $2,000 past due (Net 30 from 2026-09-01 ran to 2026-10-01), which accrues 1% per month under 3.2. Assumption: the 30-day month and the hour-counting reading of 4.2 are the most natural ones; if the contract intends a different availability basis, the credit could change, but it cannot exceed $1,000."}, "refused": null}',NULL,NULL);
CREATE TRIGGER ledger_no_update BEFORE UPDATE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (UPDATE blocked)'); END;
CREATE TRIGGER ledger_no_delete BEFORE DELETE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (DELETE blocked)'); END;
CREATE UNIQUE INDEX tasks_idempotency ON tasks (name, idempotency_key);
CREATE INDEX tasks_by_idempotency_key ON tasks (idempotency_key);
DELETE FROM "sqlite_sequence";
COMMIT;
