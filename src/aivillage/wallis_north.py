"""Wallis-North transaction sector of the village, sized in token spend.

Wallis & North (1986) measure the transaction sector as (1) all resources of
the transaction industries plus (2) the wages of transaction occupations inside
the other industries. The village analogue, built on costs.parquet:

- "GDP" is total model spend.
- Exchange actions are the transaction industries: chat, room moves, history
  search, helper/outreach requests, sign-in handoffs. Counted in full, exactly.
- Work turns are everything else an agent does in computer use. Some are
  transaction work (emailing, commenting, reviewing another agent's PR,
  trading, monitoring others): the in-firm transaction occupations. Their
  share can't be read off the action. It's estimated per month from an LLM's
  labels on cost-weighted samples of work turns (work_turn_llm_labels.csv,
  see `classify`), corrected for the LLM's error rates on the hand-labelled
  turns (work_turn_labels.csv). Months without LLM labels fall back to the
  hand-labelled share for their regime (before or after the 2026-03-24
  perma-computer-use change).
- Memory consolidation, waiting and session start/stop decisions are kept as
  their own categories: whether they're transaction costs is a judgment call.
"""

import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import numpy as np

from . import costs

EXCHANGE = (
    "AGENT_TALK", "send_message_back_to_chat", "SEARCH_HISTORY", "search_history",
    "ENTER_ROOM", "move_to_room", "REQUEST_HUMAN_HELPER", "request_human_helper",
    "CANCEL_REQUEST_FOR_HUMAN_HELPER", "cancel_request_for_human_helper",
    "STOP_HUMAN_USE_SESSION", "OUTREACH_APPROVAL_REQUEST",
    "request_approval_for_unsolicited_outreach", "REQUEST_GOOGLE_SIGN_IN",
    "request_Google_sign_in",
)  # fmt: skip
WAITING = ("WAIT", "wait", "PAUSE", "pause")
CATEGORIES = ("exchange", "work", "waiting", "memory", "session")
REGIME_CHANGE = "2026-03-24"
LABELS = Path(__file__).with_name("work_turn_labels.csv")
LLM_LABELS = Path(__file__).with_name("work_turn_llm_labels.csv")


def _quoted(xs) -> str:
    return ", ".join(f"'{x}'" for x in xs)


CATEGORY_SQL = f"""CASE
  WHEN kind IN ({_quoted(EXCHANGE)}) THEN 'exchange'
  WHEN kind IN ({_quoted(WAITING)}) THEN 'waiting'
  WHEN kind = 'CONSOLIDATE' THEN 'memory'
  WHEN kind IN ('START_USING_COMPUTER', 'STOP_USING_COMPUTER') THEN 'session'
  ELSE 'work' END"""


@dataclass(frozen=True)
class Share:
    p: float
    lo: float  # 95% Wilson interval
    hi: float
    n: int


def wilson(k: int, n: int, z: float = 1.96) -> Share:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return Share(p, centre - half, centre + half, n)


def work_transaction_share() -> dict[str, Share]:
    """Transaction share of work-turn spend, by regime, from the hand labels.

    The samples are drawn with probability proportional to cost, so the share of
    labels is an estimate of the share of spend. Unclear turns are left out.
    """
    counts = {"pre": [0, 0], "post": [0, 0]}
    with LABELS.open() as f:
        for row in csv.DictReader(f):
            if row["label"] == "U":
                continue
            c = counts["pre" if row["created_date"] < REGIME_CHANGE else "post"]
            c[0] += row["label"] == "T"
            c[1] += 1
    return {regime: wilson(k, n) for regime, (k, n) in counts.items()}


def _rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def classifier_rates(sample: str | None = None) -> dict[str, tuple[int, int]]:
    """How often the LLM says T, by hand label: {"T": (k, n), "P": (k, n), "U": (k, n)}.

    k/n for T is the true-positive rate, for P the false-positive rate. An LLM
    U counts as not-T. `sample` restricts to the "train" or "holdout" labels.
    """
    llm = {r["turn_id"]: r["label"] for r in _rows(LLM_LABELS)}
    out = {"T": [0, 0], "P": [0, 0], "U": [0, 0]}
    for r in _rows(LABELS):
        if sample in (None, r["sample"]) and r["turn_id"] in llm:
            c = out[r["label"]]
            c[0] += llm[r["turn_id"]] == "T"
            c[1] += 1
    return {h: (k, n) for h, (k, n) in out.items()}


def corrected(observed: float, tpr: float, fpr: float) -> float:
    """The true share behind an observed classifier share, clipped to [0, 1]."""
    return min(1.0, max(0.0, (observed - fpr) / (tpr - fpr)))


def llm_work_shares(draws: int = 4000, seed: int = 0) -> dict:
    """Transaction share of work-turn spend per month, from the LLM labels.

    true = (observed - FPR) / (TPR - FPR), with TPR and FPR measured on all the
    hand-labelled turns. The samples are cost-weighted, so label shares
    estimate spend shares. The 95% interval is a percentile bootstrap that
    resamples each month's sample and the T and P validation turns, so it
    includes the uncertainty in TPR and FPR. Draws share one TPR/FPR per
    replicate, as the months share one classifier.
    """
    rates = classifier_rates()
    (kt, nt), (kp, np_) = rates["T"], rates["P"]
    months: dict = {}
    for r in _rows(LLM_LABELS):
        if r["monthly_sample"] == "1":
            c = months.setdefault(date.fromisoformat(r["month"] + "-01"), [0, 0])
            c[0] += r["label"] == "T"
            c[1] += 1
    rng = np.random.default_rng(seed)
    tpr = rng.binomial(nt, kt / nt, draws) / nt
    fpr = rng.binomial(np_, kp / np_, draws) / np_
    out = {}
    for month, (k, n) in sorted(months.items()):
        obs = rng.binomial(n, k / n, draws) / n
        boot = np.clip((obs - fpr) / (tpr - fpr), 0, 1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[month] = Share(corrected(k / n, kt / nt, kp / np_), float(lo), float(hi), n)
    return out


def population() -> dict:
    """Per month: active days, agent-days and agent-hours.

    An agent is active on a day if it made any model call. Its active hours that
    day run from its first call to its last, which tracks the village schedule
    (4h/day, 8h/day from 2026-06-29) without hard-coding it. Days are village
    (Pacific) days: the 9am-5pm PT window crosses UTC midnight. A fixed -8h
    shift is enough, as sessions never come near PT midnight.
    """
    sql = f"""
    WITH d AS (
      SELECT agent_id, (created_at - INTERVAL 8 HOUR)::DATE AS dy,
             (epoch(max(created_at)) - epoch(min(created_at))) / 3600 AS hours
      FROM '{costs.path()}' WHERE agent_id IS NOT NULL GROUP BY ALL
    )
    SELECT date_trunc('month', dy)::DATE AS month, count(DISTINCT dy) AS days,
           count(*) AS agent_days, sum(hours) AS agent_hours,
           count(DISTINCT agent_id) AS agents
    FROM d GROUP BY ALL"""
    return {
        month: {"days": days, "agent_days": ad, "agent_hours": ah, "agents": agents}
        for month, days, ad, ah, agents in duckdb.sql(sql).fetchall()
    }


def monthly(price: str = "cost") -> list[dict]:
    """One row per month: spend by category, the transaction sector, unit and per-capita costs.

    `price` picks the cost column: "cost", "cost_constant" or "cost_uncached".
    """
    shares = work_transaction_share()
    llm = llm_work_shares() if LLM_LABELS.exists() else {}
    sql = f"""
    SELECT date_trunc('month', created_at)::DATE AS month, {CATEGORY_SQL} AS category,
      sum({price}) AS usd, sum((total_input + output) * weight) AS tokens,
      sum(weight) AS calls, (date_trunc('month', created_at) >= DATE '{REGIME_CHANGE}') AS post
    FROM '{costs.path()}' GROUP BY ALL ORDER BY month"""
    rows = duckdb.sql(sql).fetchall()
    pop = population()
    out: dict = {}
    for month, cat, usd, tokens, calls, post in rows:
        m = out.setdefault(month, {"month": month, "post": post})
        m[f"{cat}_usd"] = usd or 0.0
        m[f"{cat}_tokens"] = tokens or 0.0
        m[f"{cat}_calls"] = calls or 0.0
    result = []
    for m in out.values():
        for cat in CATEGORIES:
            for k in ("usd", "tokens", "calls"):
                m.setdefault(f"{cat}_{k}", 0.0)
        s = llm.get(m["month"]) or shares["post" if m["post"] else "pre"]
        m["work_share_source"] = "llm" if m["month"] in llm else "hand"
        m["work_transaction_share"] = s.p
        m["work_transaction_share_lo"] = s.lo
        m["work_transaction_share_hi"] = s.hi
        total = sum(m[f"{c}_usd"] for c in CATEGORIES)
        m["total_usd"] = total
        m["total_tokens"] = sum(m[f"{c}_tokens"] for c in CATEGORIES)
        m["work_transaction_usd"] = m["work_usd"] * s.p
        m["transaction_share"] = (m["exchange_usd"] + m["work_usd"] * s.p) / total
        m["transaction_share_lo"] = (m["exchange_usd"] + m["work_usd"] * s.lo) / total
        m["transaction_share_hi"] = (m["exchange_usd"] + m["work_usd"] * s.hi) / total
        m["usd_per_exchange"] = m["exchange_usd"] / m["exchange_calls"]
        m["tokens_per_exchange"] = m["exchange_tokens"] / m["exchange_calls"]
        p = pop[m["month"]]
        m["agents_in_month"] = p["agents"]
        m["agents_per_day"] = p["agent_days"] / p["days"]
        m["agent_days"] = p["agent_days"]
        m["agent_hours"] = p["agent_hours"]
        m["hours_per_agent_day"] = p["agent_hours"] / p["agent_days"]
        m["usd_per_agent_hour"] = total / p["agent_hours"]
        m["transaction_usd_per_agent_hour"] = m["transaction_share"] * total / p["agent_hours"]
        m["tokens_per_agent_hour"] = m["total_tokens"] / p["agent_hours"]
        m["exchanges_per_agent_hour"] = m["exchange_calls"] / p["agent_hours"]
        result.append(m)
    return result
