BEGIN TRANSACTION;
CREATE TABLE checkpoints (
  task_id UUID NOT NULL,
  name TEXT NOT NULL,
  state TEXT,
  PRIMARY KEY (task_id, name)
);
INSERT INTO "checkpoints" VALUES('01a14d45-880c-7466-a470-90da44c81360','repair:engage;d:0;step;llm:engage','{"result": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposed $3,000 credit is not supported by the contract. The supported credit is $1,000, and the customer''s $3,000 deduction is an unauthorized set-off.\n\n1. Allowed downtime (MSA 4.1): 99.9% monthly availability over September (30 days = 720 hours) allows 0.1% \u00d7 720 = 0.72 hours (43.2 minutes) of unavailability before any credit applies.\n2. Downtime (TCK-881): 02:10 to 08:25 UTC on 2026-09-14 is 6h15m (6.25 hours).\n3. Hours beyond commitment: 6.25 \u2212 0.72 = 5.53 hours, which contains 5 full hours. The proposal''s \"six hours\" ignores the 4.1 allowance and the \"full hour\" rule in 4.2.\n4. Credit per hour (4.2): 0.5% \u00d7 $100,000 = $500. Five hours gives $2,500.\n5. Cap (4.2): credits \"not to exceed $1,000 in any calendar month.\" The supported credit is therefore $1,000.\n\nThe cap makes the result robust: any reasonable reading that yields two or more qualifying hours produces $1,000, so the $3,000 figure cannot be reached under the contract.\n\nThe claim is timely (CRM-2207 dated 2026-09-26, within 30 days of the outage), so the $1,000 credit is valid.\n\nThe $2,000 remainder is not covered. MSA 7.4 prohibits withholding or set-off except a 4.2 credit, so the customer''s $3,000 deduction exceeds what they may withhold. After the $1,000 credit memo, the balance due is $99,000 and the customer paid $97,000, leaving $2,000 unpaid. The invoice was due 2026-10-01 (net 30 from 2026-09-01), so that $2,000 is past due as of 2026-10-17, and late balances accrue 1% per month under 3.2. Recommend issuing the $1,000 credit memo and pursuing the $2,000 shortfall.\n\nCRM-2210 (account manager note on renewal) does not bear on the amount and was not relied on."}, "usage": {"prompt_tokens": 2373, "completion_tokens": 2420, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 2371, "cost": 0.0016843999999999997, "latency_s": 0.0}}');
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
INSERT INTO "tasks" VALUES('01a14d45-880c-7466-a470-90da44c81360','critique','{"__recorded_format__": "v1"}','completed',1,5,1792297764.8763948,NULL,NULL,1792297807.5764761,'{"critique": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposed $3,000 credit is not supported by the contract. The supported credit is $1,000, and the customer''s $3,000 deduction is an unauthorized set-off.\n\n1. Allowed downtime (MSA 4.1): 99.9% monthly availability over September (30 days = 720 hours) allows 0.1% \u00d7 720 = 0.72 hours (43.2 minutes) of unavailability before any credit applies.\n2. Downtime (TCK-881): 02:10 to 08:25 UTC on 2026-09-14 is 6h15m (6.25 hours).\n3. Hours beyond commitment: 6.25 \u2212 0.72 = 5.53 hours, which contains 5 full hours. The proposal''s \"six hours\" ignores the 4.1 allowance and the \"full hour\" rule in 4.2.\n4. Credit per hour (4.2): 0.5% \u00d7 $100,000 = $500. Five hours gives $2,500.\n5. Cap (4.2): credits \"not to exceed $1,000 in any calendar month.\" The supported credit is therefore $1,000.\n\nThe cap makes the result robust: any reasonable reading that yields two or more qualifying hours produces $1,000, so the $3,000 figure cannot be reached under the contract.\n\nThe claim is timely (CRM-2207 dated 2026-09-26, within 30 days of the outage), so the $1,000 credit is valid.\n\nThe $2,000 remainder is not covered. MSA 7.4 prohibits withholding or set-off except a 4.2 credit, so the customer''s $3,000 deduction exceeds what they may withhold. After the $1,000 credit memo, the balance due is $99,000 and the customer paid $97,000, leaving $2,000 unpaid. The invoice was due 2026-10-01 (net 30 from 2026-09-01), so that $2,000 is past due as of 2026-10-17, and late balances accrue 1% per month under 3.2. Recommend issuing the $1,000 credit memo and pursuing the $2,000 shortfall.\n\nCRM-2210 (account manager note on renewal) does not bear on the amount and was not relied on."}, "refused": null}',NULL,NULL);
CREATE TRIGGER ledger_no_update BEFORE UPDATE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (UPDATE blocked)'); END;
CREATE TRIGGER ledger_no_delete BEFORE DELETE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (DELETE blocked)'); END;
CREATE UNIQUE INDEX tasks_idempotency ON tasks (name, idempotency_key);
CREATE INDEX tasks_by_idempotency_key ON tasks (idempotency_key);
DELETE FROM "sqlite_sequence";
COMMIT;
