BEGIN TRANSACTION;
CREATE TABLE checkpoints (
  task_id UUID NOT NULL,
  name TEXT NOT NULL,
  state TEXT,
  PRIMARY KEY (task_id, name)
);
INSERT INTO "checkpoints" VALUES('01a14d46-1542-71ed-ae83-b1817f6d776e','repair:engage;d:0;step;llm:engage','{"result": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposed $3,000 credit is not supported by the contract. Section 4.2 credits only full hours of unavailability beyond the 99.9% monthly commitment in 4.1, and caps the credit at $1,000 per calendar month. The proposal applied 0.5% per hour to all six hours with no cap.\n\nCalculation: September has 30 days, or 720 hours. At 99.9% availability, 0.72 hours of downtime is allowed. The incident lasted 6h15m (TCK-881), so about 5.53 hours exceed the commitment, which is 5 full hours. Five hours at 0.5% of $100,000 is $2,500. Section 4.2 caps this at $1,000, so the supported credit is $1,000. The cap binds even if the 99.9% allowance is ignored, since six full hours would give $3,000, which is also capped at $1,000. The $1,000 result holds under either reading.\n\nThe customer''s claim in CRM-2207 is timely, since it was sent on 2026-09-26, within 30 days of the 2026-09-14 incident. The request is therefore valid, but the amount is not. Under 7.4, the customer may set off only a 4.2 credit. The customer deducted $3,000 and paid $97,000, so the $2,000 above the $1,000 credit is an unauthorized set-off. After a $1,000 credit memo, $2,000 of INV-1001 remains open and should be collected. The payment itself was received on time under Net 30 (due 2026-10-01)."}, "usage": {"prompt_tokens": 2375, "completion_tokens": 1914, "cache_read_input_tokens": 2032, "cache_creation_input_tokens": 341, "cost": 0.00104572, "latency_s": 0.0}}');
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
INSERT INTO "tasks" VALUES('01a14d46-1542-71ed-ae83-b1817f6d776e','critique','{"__recorded_format__": "v1"}','completed',1,5,1792297801.0263195,NULL,NULL,1792297841.2293105,'{"critique": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposed $3,000 credit is not supported by the contract. Section 4.2 credits only full hours of unavailability beyond the 99.9% monthly commitment in 4.1, and caps the credit at $1,000 per calendar month. The proposal applied 0.5% per hour to all six hours with no cap.\n\nCalculation: September has 30 days, or 720 hours. At 99.9% availability, 0.72 hours of downtime is allowed. The incident lasted 6h15m (TCK-881), so about 5.53 hours exceed the commitment, which is 5 full hours. Five hours at 0.5% of $100,000 is $2,500. Section 4.2 caps this at $1,000, so the supported credit is $1,000. The cap binds even if the 99.9% allowance is ignored, since six full hours would give $3,000, which is also capped at $1,000. The $1,000 result holds under either reading.\n\nThe customer''s claim in CRM-2207 is timely, since it was sent on 2026-09-26, within 30 days of the 2026-09-14 incident. The request is therefore valid, but the amount is not. Under 7.4, the customer may set off only a 4.2 credit. The customer deducted $3,000 and paid $97,000, so the $2,000 above the $1,000 credit is an unauthorized set-off. After a $1,000 credit memo, $2,000 of INV-1001 remains open and should be collected. The payment itself was received on time under Net 30 (due 2026-10-01)."}, "refused": null}',NULL,NULL);
CREATE TRIGGER ledger_no_update BEFORE UPDATE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (UPDATE blocked)'); END;
CREATE TRIGGER ledger_no_delete BEFORE DELETE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (DELETE blocked)'); END;
CREATE UNIQUE INDEX tasks_idempotency ON tasks (name, idempotency_key);
CREATE INDEX tasks_by_idempotency_key ON tasks (idempotency_key);
DELETE FROM "sqlite_sequence";
COMMIT;
