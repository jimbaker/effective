BEGIN TRANSACTION;
CREATE TABLE checkpoints (
  task_id UUID NOT NULL,
  name TEXT NOT NULL,
  state TEXT,
  PRIMARY KEY (task_id, name)
);
INSERT INTO "checkpoints" VALUES('01a1387e-c61f-7720-9202-c97c93d4cfa4','repair:engage;d:0;step:engage','{"result": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207", "PAY-5501"], "reason": "The proposed $3,000 is not supported by the contract. Under 4.2 the credit is 0.5% of the monthly fee per full hour beyond the 4.1 commitment, not per hour of outage. The fee is $100,000 (3.1), so 0.5% is $500 per hour. September has 30 days, so the 99.9% commitment allows about 43.2 minutes of unavailability. TCK-881 records 6h15m, leaving about 5h32m beyond the commitment, which is 5 full hours. That gives 5 x $500 = $2,500. The 4.2 cap of $1,000 per calendar month then limits the credit to $1,000. The customer''s six-hour calculation ignores both the commitment threshold and the cap. The proposal also cites CRM-2207, which is the customer''s own claim, as if it were evidence of the amount. The outage date and the 2026-09-26 written request are within the 30-day window in 4.2, so the claim is timely, but the amount is wrong. Under 7.4 the customer may set off only a 4.2 credit, so the remaining $2,000 of the $3,000 deduction is not a permitted set-off. PAY-5501 paid $97,000 against a $100,000 invoice, so the correct credit memo is $1,000 and the invoice should show $2,000 still open (with 3.2 late-payment interest if it is past due). Assumption: TCK-881 is the only unavailability in September; if other incidents exist, the credit could be higher, but it stays capped at $1,000."}, "usage": {"prompt_tokens": 2374, "completion_tokens": 1527, "cache_read_input_tokens": 2032, "cache_creation_input_tokens": 340, "cost": 0.00085202, "latency_s": 0.0}}');
INSERT INTO "checkpoints" VALUES('01a1387e-c61f-7720-9202-c97c93d4cfa4','repair:engage;d:1;step:engage','{"result": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207", "INV-1001", "PAY-5501"], "reason": "The proposed $3,000 credit is not supported by the contract. Under MSA-NW-2025 section 4.2, the credit is 0.5% of the monthly platform fee ($100,000, per section 3.1) for each full hour of unavailability beyond the 99.9% commitment in 4.1, and it \"shall not exceed $1,000 in any calendar month.\" The proposal''s own math gives $3,000 (six hours at $500), which exceeds that cap, so the credit is $1,000.\n\nThe proposal''s arithmetic is also wrong on the inputs. TCK-881 records 6h15m of unavailability (02:10 to 08:25 UTC on 2026-09-14). The 99.9% commitment for a 30-day month (720 hours) allows about 0.72 hours of downtime, so roughly 5.5 hours fall beyond the commitment, which is 5 full hours. Five hours at $500 is $2,500, which the cap reduces to $1,000. Even under the most generous reading (all six full hours counted with no allowance), the cap still binds at $1,000.\n\nThe $3,000 deduction is also an unauthorized set-off. Section 7.4 bars withholding or set-off except for a credit issued under 4.2. Only $1,000 of the $3,000 short-payment (PAY-5501: $97,000 against INV-1001''s $100,000) is a permitted credit. The remaining $2,000 is unpaid and was due on net-30 terms (section 3.2) by 2026-10-01, so it is past due as of 2026-10-13 and may accrue the 1% monthly late charge. The credit request in CRM-2207 (written, 2026-09-26) is timely under the 30-day rule in 4.2, so the credit itself is valid, only its amount is overstated. The account manager note CRM-2210 is internal and does not change the contractual amount, so I did not rely on it."}, "usage": {"prompt_tokens": 2430, "completion_tokens": 2752, "cache_read_input_tokens": 2087, "cache_creation_input_tokens": 341, "cost": 0.00146527, "latency_s": 0.0}}');
INSERT INTO "checkpoints" VALUES('01a1387e-c61f-7720-9202-c97c93d4cfa4','repair:engage;d:2;step:engage','{"result": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposed $3,000 credit overstates what the contract allows, for three reasons.\n\n1. Cap. MSA-NW-2025 \u00a74.2 caps service credits at $1,000 in any calendar month. Even the customer''s own arithmetic (six hours at 0.5% of the $100,000 fee per hour, which is $500 per hour) comes to $3,000, which exceeds the cap. The most that can be credited is $1,000.\n\n2. Commitment threshold. \u00a74.1 commits to 99.9% monthly availability. Over a 30-day September (720 hours), that permits about 0.72 hours (43 minutes) of unavailability. TCK-881 records 6h15m (02:10 to 08:25 UTC on 2026-09-14), so about 5.53 hours fall beyond the commitment, which is 5 full hours. Credits under \u00a74.2 accrue only for full hours beyond the commitment, so the contractual base is 5 \u00d7 $500 = $2,500, which the cap reduces to $1,000. The customer''s six-hour figure ignores the commitment threshold. The cap binds whenever there are at least two full hours beyond the commitment, so $1,000 holds under any reasonable reading of the outage record.\n\n3. Set-off. \u00a77.4 bars withholding or set-off except for a \u00a74.2 credit. The customer''s $3,000 deduction (CRM-2207) is therefore improper beyond the $1,000 credit. The customer''s written request was made within 30 days, so the credit itself is valid.\n\nConsequence: issue a $1,000 credit memo against INV-1001. The invoice is $100,000 and the payment of $97,000 (PAY-5501) leaves a $2,000 shortfall after the credit, so the net balance is $1,000 after the credit memo and the $2,000 short-pay remains owed. Net 30 from 2026-09-01 made the due date 2026-10-01, so the balance is now past due and may accrue 1% per month under \u00a73.2. CRM-2210 says to \"follow the contract,\" which supports this result.\n\nAssumption: no other unavailability occurred in September. Any additional outage would only increase the credit, and the $1,000 cap would still apply."}, "usage": {"prompt_tokens": 2428, "completion_tokens": 2189, "cache_read_input_tokens": 2087, "cache_creation_input_tokens": 339, "cost": 0.00118337, "latency_s": 0.0}}');
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
INSERT INTO "tasks" VALUES('01a1387e-c61f-7720-9202-c97c93d4cfa4','critique','{"__recorded_format__": "v1"}','completed',1,5,1791949194.783349,NULL,NULL,1791949258.4352658,'{"critique": {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025", "TCK-881", "CRM-2207"], "reason": "The proposed $3,000 credit overstates what the contract allows, for three reasons.\n\n1. Cap. MSA-NW-2025 \u00a74.2 caps service credits at $1,000 in any calendar month. Even the customer''s own arithmetic (six hours at 0.5% of the $100,000 fee per hour, which is $500 per hour) comes to $3,000, which exceeds the cap. The most that can be credited is $1,000.\n\n2. Commitment threshold. \u00a74.1 commits to 99.9% monthly availability. Over a 30-day September (720 hours), that permits about 0.72 hours (43 minutes) of unavailability. TCK-881 records 6h15m (02:10 to 08:25 UTC on 2026-09-14), so about 5.53 hours fall beyond the commitment, which is 5 full hours. Credits under \u00a74.2 accrue only for full hours beyond the commitment, so the contractual base is 5 \u00d7 $500 = $2,500, which the cap reduces to $1,000. The customer''s six-hour figure ignores the commitment threshold. The cap binds whenever there are at least two full hours beyond the commitment, so $1,000 holds under any reasonable reading of the outage record.\n\n3. Set-off. \u00a77.4 bars withholding or set-off except for a \u00a74.2 credit. The customer''s $3,000 deduction (CRM-2207) is therefore improper beyond the $1,000 credit. The customer''s written request was made within 30 days, so the credit itself is valid.\n\nConsequence: issue a $1,000 credit memo against INV-1001. The invoice is $100,000 and the payment of $97,000 (PAY-5501) leaves a $2,000 shortfall after the credit, so the net balance is $1,000 after the credit memo and the $2,000 short-pay remains owed. Net 30 from 2026-09-01 made the due date 2026-10-01, so the balance is now past due and may accrue 1% per month under \u00a73.2. CRM-2210 says to \"follow the contract,\" which supports this result.\n\nAssumption: no other unavailability occurred in September. Any additional outage would only increase the credit, and the $1,000 cap would still apply."}, "refused": null}',NULL,NULL);
CREATE TRIGGER ledger_no_update BEFORE UPDATE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (UPDATE blocked)'); END;
CREATE TRIGGER ledger_no_delete BEFORE DELETE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (DELETE blocked)'); END;
CREATE UNIQUE INDEX tasks_idempotency ON tasks (name, idempotency_key);
CREATE INDEX tasks_by_idempotency_key ON tasks (idempotency_key);
DELETE FROM "sqlite_sequence";
COMMIT;
