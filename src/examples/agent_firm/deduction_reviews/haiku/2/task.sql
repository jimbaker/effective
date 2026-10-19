BEGIN TRANSACTION;
CREATE TABLE checkpoints (
  task_id UUID NOT NULL,
  name TEXT NOT NULL,
  state TEXT,
  PRIMARY KEY (task_id, name)
);
INSERT INTO "checkpoints" VALUES('01a1387f-4994-707d-9f8b-48b7aa380b39','repair:engage;d:0;step:engage','{"result": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposal''s $3,000 is wrong. Under MSA-NW-2025 \u00a74.2, the credit is 0.5% of the monthly fee ($500) per full hour of unavailability beyond the \u00a74.1 commitment, and the total for any calendar month may not exceed $1,000. The cap controls the result. Even on the most generous reading, six full hours at $500 is $3,000, which the cap reduces to $1,000. The proposal also ignores the 99.9% monthly availability allowance: September 2026 has 720 hours, so about 43 minutes of downtime is permitted. TCK-881 records 6h15m of unavailability, which leaves roughly 5.5 hours beyond the commitment (five full hours, or $2,500 before the cap). Either way the supported credit is $1,000. The request was made in writing on 2026-09-26, within 30 days of the 2026-09-14 incident, so the credit is valid. The customer''s $3,000 deduction is a set-off, which \u00a77.4 prohibits except for a \u00a74.2 credit. Only the $1,000 credit may be applied. The remaining $2,000 of INV-1001 is still owed. The invoice was due 2026-10-01 under Net 30, so it is now past due, and \u00a73.2 late-payment accrual of 1% per month may apply. The credit memo should be issued for $1,000 and the $2,000 shortfall pursued as an unpaid balance. CRM-2210 was reviewed and does not change the contractual amount."}, "usage": {"prompt_tokens": 2374, "completion_tokens": 1649, "cache_read_input_tokens": 2032, "cache_creation_input_tokens": 340, "cost": 0.0009130200000000001, "latency_s": 0.0}}');
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
INSERT INTO "tasks" VALUES('01a1387f-4994-707d-9f8b-48b7aa380b39','critique','{"__recorded_format__": "v1"}','completed',1,5,1791949228.4367971,NULL,NULL,1791949267.5126255,'{"critique": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposal''s $3,000 is wrong. Under MSA-NW-2025 \u00a74.2, the credit is 0.5% of the monthly fee ($500) per full hour of unavailability beyond the \u00a74.1 commitment, and the total for any calendar month may not exceed $1,000. The cap controls the result. Even on the most generous reading, six full hours at $500 is $3,000, which the cap reduces to $1,000. The proposal also ignores the 99.9% monthly availability allowance: September 2026 has 720 hours, so about 43 minutes of downtime is permitted. TCK-881 records 6h15m of unavailability, which leaves roughly 5.5 hours beyond the commitment (five full hours, or $2,500 before the cap). Either way the supported credit is $1,000. The request was made in writing on 2026-09-26, within 30 days of the 2026-09-14 incident, so the credit is valid. The customer''s $3,000 deduction is a set-off, which \u00a77.4 prohibits except for a \u00a74.2 credit. Only the $1,000 credit may be applied. The remaining $2,000 of INV-1001 is still owed. The invoice was due 2026-10-01 under Net 30, so it is now past due, and \u00a73.2 late-payment accrual of 1% per month may apply. The credit memo should be issued for $1,000 and the $2,000 shortfall pursued as an unpaid balance. CRM-2210 was reviewed and does not change the contractual amount."}, "refused": null}',NULL,NULL);
CREATE TRIGGER ledger_no_update BEFORE UPDATE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (UPDATE blocked)'); END;
CREATE TRIGGER ledger_no_delete BEFORE DELETE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (DELETE blocked)'); END;
CREATE UNIQUE INDEX tasks_idempotency ON tasks (name, idempotency_key);
CREATE INDEX tasks_by_idempotency_key ON tasks (idempotency_key);
DELETE FROM "sqlite_sequence";
COMMIT;
