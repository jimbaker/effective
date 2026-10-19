"""The live Odoo arm of the ERP port's tests: the pod `just odoo-up` starts, or a skip.

`EFFECTIVE_REQUIRE_ODOO=1` turns the skip into a failure, so a lane that asks for Odoo cannot pass
on a run that never reached it.
"""

import os
import socket
from collections.abc import Callable, Sequence
from decimal import Decimal
from functools import cache
from urllib.parse import urlsplit
from uuid import uuid4

import pytest

from examples.agent_firm.ar import Round
from examples.agent_firm.cases import SLA_CAP, Case
from examples.agent_firm.erp.odoo import OdooErp, Transport, XmlRpc, seed
from examples.agent_firm.erp.port import Erp
from examples.agent_firm.erp.scripted import ScriptedOdoo

ERPS = ["scripted", "odoo"]
"""The arms a test of the port runs on."""


@cache
def _connected() -> XmlRpc | str:
    url = urlsplit(os.environ.get("EFFECTIVE_ODOO_URL", "http://127.0.0.1:18069"))
    try:
        socket.create_connection((url.hostname, url.port or 80), timeout=1).close()
        return XmlRpc.from_env()
    except OSError as error:
        return f"no Odoo at {url.netloc}: {error}"


def odoo() -> XmlRpc:
    match _connected():
        case XmlRpc() as transport:
            return transport
        case reason:
            if os.environ.get("EFFECTIVE_REQUIRE_ODOO") in ("1", "true", "yes"):
                pytest.fail(f"EFFECTIVE_REQUIRE_ODOO is set and there is {reason}")
            pytest.skip(f"{reason}; `just odoo-up` starts one")


type Opened = tuple[Transport, dict[str, int]]
"""An Odoo, and the move of each invoice and payment it holds by the document's id."""


def opened(arm: str, cases: Sequence[Case]) -> Callable[[], Opened]:
    """Fresh books for `cases` on each call, on the scripted or the live Odoo."""
    match arm:
        case "scripted":

            def scripted() -> Opened:
                odoo = ScriptedOdoo.seeded(cases)
                return odoo, odoo.documents

            return scripted
        case "odoo":
            transport = odoo()
            return lambda: (transport, seed(transport, cases, uuid4().hex[:8]))
        case _:
            raise ValueError(f"no ERP arm {arm!r}")


def books(arm: str, cases: Sequence[Case]) -> Callable[[], Erp]:
    """A fresh ERP holding `cases`' books on each call, on the scripted or the live Odoo."""
    open_ = opened(arm, cases)
    return lambda: OdooErp(*open_())


TERMS = "30% Now, Balance 60 Days"
"""A payment term Odoo ships, which splits an invoice's receivable into two installment lines."""


def another_invoice(arm: str, amount: Decimal, installments: bool = False) -> tuple[OdooErp, str]:
    """`SLA_CAP`'s books holding a second invoice of `amount` for its customer, due at once or,
    with `installments`, under `TERMS`, and the new invoice's move."""
    transport, documents = opened(arm, [SLA_CAP])()
    match transport:
        case ScriptedOdoo() as odoo:
            customer = odoo.moves[documents[SLA_CAP.invoice.id]].partner
            shares = (Decimal("0.3"), Decimal("0.7")) if installments else (Decimal(1),)
            move = odoo.posted("out_invoice", customer, "2026-09-01", "INV-1004", amount, shares)
        case _:
            (held,) = transport.call(
                "account.move", "read", [documents[SLA_CAP.invoice.id]], ["partner_id"]
            )
            line = {"name": "Onboarding", "quantity": 1, "price_unit": float(amount)}
            invoice = {
                "move_type": "out_invoice",
                "partner_id": held["partner_id"][0],
                "invoice_date": "2026-09-01",
                "invoice_line_ids": [(0, 0, line | {"tax_ids": []})],
            }
            if installments:
                (terms,) = transport.call("account.payment.term", "search", [("name", "=", TERMS)])
                invoice["invoice_payment_term_id"] = terms
            move = transport.call("account.move", "create", invoice)
            transport.call("account.move", "action_post", [move])
    return OdooErp(transport, documents), str(move)


def fresh_round() -> Round:
    """The first round of books no other run keeps, since the test databases keep every run's
    ledger and the live Odoo every run's keys, under one case's ids."""
    return Round(f"books-{uuid4().hex[:12]}")
