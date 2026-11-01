"""treg: verify and find on a budget, tagged per customer — no real calls."""

import csv
import json
import os
import tempfile

import httpx
import pytest
import pytest_asyncio

from harvey.integrations import email_finder as ef
from harvey.integrations import treg
from harvey.models.campaign import Campaign, EmailStep
from harvey.models.prospect import Prospect
from harvey.state import StateManager

RealAsyncClient = httpx.AsyncClient


@pytest_asyncio.fixture
async def state():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = StateManager(os.path.join(tmpdir, "test.db"))
        await sm.init_db()
        yield sm


class FakeTreg:
    """Stands in for treg's call API and records what Harvey sent."""

    def __init__(self, answers):
        self.answers = answers  # endpoint id -> (status, json, headers)
        self.requests = []

    def client(self, *args, **kwargs):
        def handler(request: httpx.Request):
            self.requests.append(request)
            endpoint = request.url.path.removeprefix("/call/")
            status, body, headers = self.answers[endpoint]
            return httpx.Response(status, json=body, headers=headers)
        return RealAsyncClient(transport=httpx.MockTransport(handler))


def _paid(cost_micro, call_id="call_1", served_by="provider_a", **extra):
    return {"X-Treg-Cost-Micro": str(cost_micro), "X-Treg-Call-Id": call_id,
            "X-Treg-Served-By": served_by, **extra}


@pytest.fixture
def treg_on(monkeypatch):
    monkeypatch.setenv("TREG_TOKEN", "tok_test")
    monkeypatch.setenv("TREG_CUSTOMER", "Acme Corp")

    async def google_mx(domain):
        return "aspmx.l.google.com"
    monkeypatch.setattr(ef, "get_mx_host", google_mx)
    monkeypatch.setattr(ef, "_hunter_api_key", lambda: "")


def _install(monkeypatch, fake):
    monkeypatch.setattr(treg.httpx, "AsyncClient", fake.client)


# ── tags ──


def test_tag_value_is_safe():
    assert treg.tag_value("INNOVERIC Labs!") == "innoveric-labs"
    assert "@" not in treg.tag_value("jane@example.com")
    assert len(treg.tag_value("x" * 300)) == 128


def test_meta_header_carries_customer_seat_feature(monkeypatch):
    monkeypatch.setenv("TREG_CUSTOMER", "Acme Corp")
    monkeypatch.setenv("TREG_SEAT", "teammate-1")
    assert treg.meta_header("people.email.find") == \
        "customer=acme-corp, seat=teammate-1, feature=people.email.find"


# ── budget ──


@pytest.mark.asyncio
async def test_off_without_token(state):
    assert not await treg.treg_available(state)


@pytest.mark.asyncio
async def test_off_without_state(treg_on):
    assert not await treg.treg_available(None)


@pytest.mark.asyncio
async def test_monthly_budget_stops_calls(state, treg_on, monkeypatch):
    monkeypatch.setenv("TREG_MONTHLY_BUDGET", "0.10")
    monkeypatch.setenv("TREG_MAX_COST_PER_LOOKUP", "0.05")
    await state.increment_setting(treg._spend_key(), 60_000)  # $0.06 spent
    assert not await treg.treg_available(state)  # 0.06 + 0.05 > 0.10


# ── verify + find through find_email ──


@pytest.mark.asyncio
async def test_treg_verifies_the_pattern_candidate(state, treg_on, monkeypatch):
    fake = FakeTreg({"treg.people.email.verify": (
        200, {"output": {"valid": True, "status": "valid"}}, _paid(1500))})
    _install(monkeypatch, fake)

    result = await ef.find_email("Jane", "Doe", "example.com", state=state)

    assert (result.email, result.status) == ("jane.doe@example.com", "verified")
    sent = fake.requests[0]
    assert sent.headers["X-Treg-Token"] == "tok_test"
    assert sent.headers["X-Treg-Meta"] == "customer=acme-corp, feature=people.email.verify"
    assert json.loads(sent.content) == {"email": "jane.doe@example.com"}
    assert await treg.treg_spent_this_month(state) == pytest.approx(0.0015)


@pytest.mark.asyncio
async def test_treg_finds_when_candidate_is_invalid(state, treg_on, monkeypatch):
    fake = FakeTreg({
        "treg.people.email.verify": (200, {"output": {"valid": False, "status": "invalid"}},
                                     _paid(1000, "call_v")),
        "treg.people.email.find": (200, {
            "output": {"email": "jdoe@example.com"},
            "_treg": {"domain_match": True, "verification": {"verdict": "valid"}},
        }, _paid(5000, "call_f")),
    })
    _install(monkeypatch, fake)

    result = await ef.find_email("Jane", "Doe", "example.com", state=state)

    assert (result.email, result.status) == ("jdoe@example.com", "verified")
    assert result.pattern == "{fi}{l}"
    find = fake.requests[1]
    assert find.headers["X-Treg-Route-Verify"] == "true"
    assert json.loads(find.content) == {
        "domain": "example.com", "first_name": "Jane", "last_name": "Doe"}
    assert await treg.treg_spent_this_month(state) == pytest.approx(0.006)


@pytest.mark.asyncio
async def test_hit_on_another_domain_is_discarded(state, treg_on, monkeypatch):
    fake = FakeTreg({
        "treg.people.email.verify": (200, {"output": {"status": "unknown"}}, _paid(1000)),
        "treg.people.email.find": (200, {
            "output": {"email": "jane@example.net"}, "_treg": {"domain_match": False},
        }, _paid(5000)),
    })
    _install(monkeypatch, fake)

    result = await ef.find_email("Jane", "Doe", "example.com", state=state)

    assert result.email == "jane.doe@example.com"
    assert result.status == "guess"


@pytest.mark.asyncio
async def test_treg_refusal_is_not_a_result(state, treg_on, monkeypatch):
    fake = FakeTreg({"treg.people.email.verify": (
        402, {"error": "insufficient_balance", "topup_url": "https://treg.to/topup"},
        {"X-Treg-Error": "1"})})
    _install(monkeypatch, fake)
    assert await treg.treg_verify("jane.doe@example.com", state) is None


@pytest.mark.asyncio
async def test_call_id_and_cost_are_logged(state, treg_on, monkeypatch):
    fake = FakeTreg({"treg.people.email.verify": (
        200, {"output": {"status": "valid"}}, _paid(1500, "call_abc"))})
    _install(monkeypatch, fake)
    await treg.treg_verify("jane.doe@example.com", state)

    import aiosqlite
    async with aiosqlite.connect(state.db_path) as db:
        async with db.execute(
            "SELECT details_json FROM actions WHERE action_type = 'treg_call'"
        ) as cur:
            rows = [json.loads(r[0]) for r in await cur.fetchall()]
    assert rows == [{"feature": "people.email.verify", "call_id": "call_abc",
                     "cost_micro": 1500, "served_by": "provider_a",
                     "customer": "acme-corp"}]


# ── drafts export ──


@pytest.mark.asyncio
async def test_drafts_export_fills_each_prospect(state, tmp_path, monkeypatch):
    from harvey import export

    monkeypatch.setattr(export, "_do_not_contact", lambda config: [])
    await state.add_prospect(Prospect(
        id="p1", first_name="Jane", company="Acme", email="jane@example.com",
        email_status="verified"))
    await state.add_prospect(Prospect(
        id="p2", first_name="Sam", company="Beta", email="sam@example.org",
        email_status="guess"))  # not deliverable: left out
    seq = [EmailStep(step=i, subject=f"Note {i} for {{{{company}}}}",
                     body=f"Hi {{{{first_name}}}}, part {i}") for i in (1, 2, 3)]
    await state.add_campaign(Campaign(id="c1", sequence=seq, prospect_ids=["p1", "p2"]))
    await state.add_outbox_item(
        prospect_id="p1", campaign_id="c1", to_email="jane@example.com",
        subject="A question about Acme", body="Hi Jane, written for Acme.",
        send_at="2026-01-01T00:00:00")

    out = tmp_path / "drafts.csv"
    assert await export.export_drafts_csv(state, str(out)) == 1
    row = next(csv.DictReader(out.open()))
    assert row["email"] == "jane@example.com"
    assert (row["subject_1"], row["body_1"]) == ("A question about Acme", "Hi Jane, written for Acme.")
    assert (row["subject_2"], row["body_2"]) == ("Note 2 for Acme", "Hi Jane, part 2")
    assert row["body_3"] == "Hi Jane, part 3"
