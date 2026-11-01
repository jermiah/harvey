"""treg — pay-per-call email finding and verification, on a hard budget.

treg (https://treg.to) routes one request to the cheapest of ~30 contact
data providers and bills only the answer it returns. Harvey uses two of
its routed tools, and only when the free pattern pipeline couldn't settle
an address:

- ``treg.people.email.verify`` checks the one candidate Harvey built from
  the company's email pattern (fractions of a cent).
- ``treg.people.email.find`` looks the person up by name + domain when the
  candidate failed or couldn't be graded, verifying the hit in the same call.

Both are optional: without ``TREG_TOKEN`` in .env nothing here runs.

Spend is capped twice — per call (``TREG_MAX_COST_PER_LOOKUP``, sent to treg
as X-Treg-Route-Max-Cost) and per calendar month (``TREG_MONTHLY_BUDGET``,
counted from the X-Treg-Cost-Micro header of every answer). A call is only
made when the month has at least one full per-call cap left, so the budget
is never overrun. Spend is counted in the state store; with no store there
is nothing to count against, so treg is not used.

Attribution: every call goes through ``_call`` and carries an X-Treg-Meta
tag set here, in code, never by a model: ``customer`` is the company this
checkout prospects for (``TREG_CUSTOMER``, else persona.company), ``seat``
the teammate running it (``TREG_SEAT``, optional), and ``feature`` the tool.
treg's ledger then splits spend per customer and per seat
(``harvey treg``), and each call's X-Treg-Call-Id and cost are kept in the
local actions log to join the two.
"""

import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger("harvey.treg")

TREG_BASE = "https://treg.to"
TREG_CALL_URL = TREG_BASE + "/call/"
TREG_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)

DEFAULT_MAX_COST_PER_LOOKUP = 0.05
DEFAULT_MONTHLY_BUDGET = 10.0
MICRO = 1_000_000

_customer_cache: Optional[str] = None


def treg_token() -> str:
    return os.getenv("TREG_TOKEN", "").strip()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return default


def max_cost_per_lookup() -> float:
    return _env_float("TREG_MAX_COST_PER_LOOKUP", DEFAULT_MAX_COST_PER_LOOKUP)


def monthly_budget() -> float:
    return _env_float("TREG_MONTHLY_BUDGET", DEFAULT_MONTHLY_BUDGET)


def tag_value(raw: str) -> str:
    """A string treg accepts as a tag value: letters, digits and ``. _ - :``,
    at most 128 characters, never shaped like an email address."""
    value = re.sub(r"[^A-Za-z0-9._:-]+", "-", (raw or "").strip().lower()).strip("-")
    return value[:128]


def customer_tag() -> str:
    """Who this checkout's treg spend is billed to."""
    global _customer_cache
    explicit = tag_value(os.getenv("TREG_CUSTOMER", ""))
    if explicit:
        return explicit
    if _customer_cache is None:
        try:
            from harvey.config import load_config
            _customer_cache = tag_value(load_config().persona.company)
        except Exception:
            _customer_cache = ""
        if _customer_cache in ("", "your-company"):
            _customer_cache = "unassigned"
    return _customer_cache


def meta_header(feature: str) -> str:
    tags = [f"customer={customer_tag()}"]
    seat = tag_value(os.getenv("TREG_SEAT", ""))
    if seat:
        tags.append(f"seat={seat}")
    tags.append(f"feature={tag_value(feature)}")
    return ", ".join(tags)


def _spend_key() -> str:
    return f"treg_spend_micro:{datetime.now(timezone.utc):%Y-%m}"


async def treg_spent_this_month(state) -> float:
    """USD spent on treg this calendar month (UTC), by this checkout."""
    if state is None:
        return 0.0
    try:
        return int(await state.get_setting(_spend_key(), "0") or "0") / MICRO
    except (ValueError, TypeError):
        return 0.0


async def treg_available(state) -> bool:
    """A token is set and the month has room for one more capped call."""
    if not treg_token() or state is None:
        return False
    cap = max_cost_per_lookup()
    if cap <= 0:
        return False
    return await treg_spent_this_month(state) + cap <= monthly_budget()


async def _record(state, feature: str, call_id: str, cost_micro: int, served_by: str) -> None:
    """Count spend toward the monthly budget and keep the call id, so
    treg's ledger and Harvey's own log can be joined later."""
    try:
        if cost_micro > 0:
            await state.increment_setting(_spend_key(), cost_micro)
        await state.log_action(
            action_type="treg_call",
            agent="scout",
            details={"feature": feature, "call_id": call_id, "cost_micro": cost_micro,
                     "served_by": served_by, "customer": customer_tag()},
        )
    except Exception as e:
        logger.debug(f"treg usage write failed: {e}")


def _refusal(resp: httpx.Response) -> None:
    """Log one of treg's own refusals (X-Treg-Error: 1). Balance details
    stay in the log; they are never shown as a lookup result."""
    try:
        body = resp.json()
    except ValueError:
        body = {}
    error = body.get("error", "") if isinstance(body, dict) else ""
    if error == "route_max_cost":
        logger.info("treg: lookup would cost more than TREG_MAX_COST_PER_LOOKUP; skipped.")
    elif error in ("tag_spend_cap_reached", "tag_blocked"):
        logger.warning(f"treg: {body.get('dim')}={body.get('val')} is over its treg budget "
                       f"({error}); skipping treg until it resets.")
    elif resp.status_code == 402:
        logger.warning("treg: out of balance — top up at https://treg.to to keep using it.")
    elif resp.status_code in (401, 403):
        logger.warning("treg: token refused — check TREG_TOKEN in .env.")
    else:
        logger.warning(f"treg refused the call ({resp.status_code} {error or 'no detail'}).")


async def _call(endpoint: str, body: dict, state, headers: dict | None = None) -> Optional[dict]:
    """POST one routed treg tool. The only place Harvey talks to treg's
    call API. Returns the JSON answer, or None."""
    if not await treg_available(state):
        return None
    feature = endpoint.removeprefix("treg.")
    request_headers = {
        "X-Treg-Token": treg_token(),
        "X-Treg-Meta": meta_header(feature),
        "X-Treg-Route-Max-Cost": f"{max_cost_per_lookup():.4f}",
        **(headers or {}),
    }
    try:
        async with httpx.AsyncClient(timeout=TREG_TIMEOUT) as client:
            resp = await client.post(TREG_CALL_URL + endpoint, json=body, headers=request_headers)
    except httpx.HTTPError as e:
        logger.debug(f"treg {endpoint} request failed: {e}")
        return None

    try:
        cost_micro = int(resp.headers.get("X-Treg-Cost-Micro", "0") or 0)
    except ValueError:
        cost_micro = 0
    served_by = resp.headers.get("X-Treg-Served-By", "")
    await _record(state, feature, resp.headers.get("X-Treg-Call-Id", ""), cost_micro, served_by)

    if resp.headers.get("X-Treg-Error") == "1":
        _refusal(resp)
        return None
    if resp.status_code != 200:
        logger.debug(f"treg {endpoint}: provider answered {resp.status_code}")
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    logger.info(f"treg {endpoint}: served by {served_by or '?'}, ${cost_micro / MICRO:.4f}")
    return data


def _verdict(output: dict) -> Optional[dict]:
    """treg's verify output → the {'status', 'catch_all'} shape Harvey's
    other verifiers return."""
    if not isinstance(output, dict):
        return None
    status = str(output.get("status") or "").lower().replace("-", "_")
    if not status:
        if output.get("valid") is True:
            status = "valid"
        elif output.get("valid") is False:
            status = "invalid"
        else:
            return None
    return {"status": status, "catch_all": status in ("accept_all", "catch_all")}


async def treg_verify(email: str, state) -> Optional[dict]:
    """Verify one address. Returns {'status', 'catch_all'} or None."""
    data = await _call("treg.people.email.verify", {"email": email}, state)
    if not data:
        return None
    return _verdict(data.get("output") or {})


async def treg_find(first_name: str, last_name: str, domain: str, state) -> Optional[tuple[str, Optional[dict]]]:
    """Look up a person's work email at ``domain``.

    Returns (email, verdict_or_None), or None on a miss. A hit on another
    domain (a LinkedIn-sourced former employer) is discarded.
    """
    data = await _call(
        "treg.people.email.find",
        {"domain": domain, "first_name": first_name, "last_name": last_name},
        state,
        headers={"X-Treg-Route-Verify": "true"},
    )
    if not data:
        return None
    output = data.get("output") or {}
    email = str(output.get("email") or "").strip().lower()
    if not email:
        return None
    meta = data.get("_treg") or {}
    if meta.get("domain_match") is False or email.partition("@")[2] != domain:
        logger.info(f"treg: a hit for {first_name} {last_name} is not at {domain}; discarded")
        return None
    verification = meta.get("verification") or {}
    verdict = None
    if verification.get("verdict"):
        verdict = _verdict({"status": verification.get("verdict")})
    elif output.get("verified") is True:
        verdict = {"status": "valid", "catch_all": False}
    return email, verdict


async def usage_by_tag(key: str = "customer", days: int = 30) -> dict:
    """treg's ledger, grouped by one tag, for the org in TREG_ORG_ID.

    This is the number to bill from (the per-call log is shed under load).
    Raises ValueError when the ledger doesn't reconcile
    (attributed + unattributed must equal total).
    """
    org = os.getenv("TREG_ORG_ID", "").strip()
    if not org:
        raise ValueError("Set TREG_ORG_ID in .env (your treg team's numeric id).")
    async with httpx.AsyncClient(timeout=TREG_TIMEOUT) as client:
        resp = await client.get(
            f"{TREG_BASE}/orgs/{org}/usage/by-tag",
            params={"key": key, "days": days},
            headers={"X-Treg-Token": treg_token()},
        )
    resp.raise_for_status()
    data = resp.json()
    attributed = int(data.get("attributed_micro", 0) or 0)
    unattributed = int(data.get("unattributed_micro", 0) or 0)
    total = int(data.get("total_micro", 0) or 0)
    if attributed + unattributed != total:
        raise ValueError(f"treg ledger doesn't reconcile for '{key}': "
                         f"{attributed} + {unattributed} != {total} micro-USD")
    return data
