BEGIN TRANSACTION;
CREATE TABLE checkpoints (
  task_id UUID NOT NULL,
  name TEXT NOT NULL,
  state TEXT,
  PRIMARY KEY (task_id, name)
);
INSERT INTO "checkpoints" VALUES('01a1387e-7037-73e2-a182-5c02f059ae0c','repair:engage;d:0;step:engage','{"result": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207", "INV-1001", "PAY-5501"], "reason": "The proposed $3,000 credit exceeds what the contract allows. Under MSA 4.2 the credit is 0.5% of the monthly fee ($100,000) per full hour of unavailability beyond the 4.1 commitment, but it \"not [to] exceed $1,000 in any calendar month.\" The September 14 outage was 6h15m (TCK-881). Even after the 99.9% allowance (about 43 minutes in a 720-hour month), at least five full hours fall outside the commitment, so the credit is triggered. The uncapped figure is $2,500 to $3,000, and the $1,000 monthly cap binds. The supported credit memo is $1,000. The proposal''s rationale, citing only CRM-2207''s \"six hours at 0.5% per hour,\" omits the cap and the contract itself.\n\nThe customer''s $3,000 deduction also breaches MSA 7.4, which bars withholding or set-off except a credit issued under 4.2. Only $1,000 of the $3,000 short-pay is a permitted credit. Invoice INV-1001 is $100,000 and PAY-5501 paid $97,000. After a $1,000 credit memo, $2,000 remains open. That amount was due 2026-10-01 under Net 30 (MSA 3.2), so it is now overdue as of 2026-10-13 and accrues 1% per month. It should be collected, not written off.\n\nThe credit request was timely: it was made in writing on 2026-09-26, within 30 days of the outage (MSA 4.2). CRM-2210 is not a contractual basis for any credit. The account manager explicitly says to follow the contract, so I did not rely on it.\n\nRecommended action: issue a $1,000 credit memo against INV-1001, notify Northwind AP in writing that the $2,000 balance is past due and that the unauthorized set-off is not permitted under 7.4, and collect the remaining $2,000 with late charges per 3.2."}, "usage": {"prompt_tokens": 2374, "completion_tokens": 2766, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 2372, "cost": 0.0018575999999999998, "latency_s": 0.0}}');
INSERT INTO "checkpoints" VALUES('01a1387e-7037-73e2-a182-5c02f059ae0c','repair:engage;d:1;step:engage','{"result": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposed $3,000 credit does not match the contract. MSA-NW-2025 \u00a74.2 sets the credit at 0.5% of the monthly platform fee ($500) for each full hour of unavailability beyond the 99.9% commitment in \u00a74.1, and caps the credit at $1,000 per calendar month. TCK-881 records 6h15m of unavailability on 2026-09-14, so the calculation is 6 hours at $500 = $3,000 before the cap, and the $1,000 cap then limits the credit. Excluding the monthly allowance under \u00a74.1 would give 5 full hours, or $2,500, before the cap, so the cap controls under either reading and the supported credit is $1,000. CRM-2207 was sent 2026-09-26, within the 30-day written-request window in \u00a74.2, so the claim is timely, but the amount is capped. Under \u00a74.2 the credit is the customer''s sole remedy for unavailability, and \u00a77.4 bars withholding or set-off except a \u00a74.2 credit. Northwind paid $97,000 against a $100,000 invoice (INV-1001). Only the $1,000 credit may be deducted; the remaining $2,000 is an unauthorized set-off and stays due. The invoice was due 2026-10-01 under the Net 30 terms in \u00a73.2, so the $2,000 balance is now past due and may accrue 1% per month under \u00a73.2. The credit memo should be issued for $1,000 and the $2,000 shortfall collected."}, "usage": {"prompt_tokens": 2428, "completion_tokens": 1815, "cache_read_input_tokens": 715, "cache_creation_input_tokens": 1711, "cost": 0.00125705, "latency_s": 0.0}}');
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
INSERT INTO "tasks" VALUES('01a1387e-7037-73e2-a182-5c02f059ae0c','critique','{"__recorded_format__": "v1"}','completed',1,5,1791949172.7918942,NULL,NULL,1791949224.7815664,'{"critique": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposed $3,000 credit does not match the contract. MSA-NW-2025 \u00a74.2 sets the credit at 0.5% of the monthly platform fee ($500) for each full hour of unavailability beyond the 99.9% commitment in \u00a74.1, and caps the credit at $1,000 per calendar month. TCK-881 records 6h15m of unavailability on 2026-09-14, so the calculation is 6 hours at $500 = $3,000 before the cap, and the $1,000 cap then limits the credit. Excluding the monthly allowance under \u00a74.1 would give 5 full hours, or $2,500, before the cap, so the cap controls under either reading and the supported credit is $1,000. CRM-2207 was sent 2026-09-26, within the 30-day written-request window in \u00a74.2, so the claim is timely, but the amount is capped. Under \u00a74.2 the credit is the customer''s sole remedy for unavailability, and \u00a77.4 bars withholding or set-off except a \u00a74.2 credit. Northwind paid $97,000 against a $100,000 invoice (INV-1001). Only the $1,000 credit may be deducted; the remaining $2,000 is an unauthorized set-off and stays due. The invoice was due 2026-10-01 under the Net 30 terms in \u00a73.2, so the $2,000 balance is now past due and may accrue 1% per month under \u00a73.2. The credit memo should be issued for $1,000 and the $2,000 shortfall collected."}, "refused": null}',NULL,NULL);
CREATE TRIGGER ledger_no_update BEFORE UPDATE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (UPDATE blocked)'); END;
CREATE TRIGGER ledger_no_delete BEFORE DELETE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (DELETE blocked)'); END;
CREATE UNIQUE INDEX tasks_idempotency ON tasks (name, idempotency_key);
CREATE INDEX tasks_by_idempotency_key ON tasks (idempotency_key);
DELETE FROM "sqlite_sequence";
COMMIT;
