"""Dollar cost of a model call at the list price in force when it was made.

Prices come from pydantic's genai-prices package (pinned in pyproject.toml,
since the price data ships inside it). It knows dated price changes (e.g.
the o3 cut on 2025-06-10), long-context tiers applied per request, and
time-of-day discounts. `price_overrides.csv` corrects or fills models it gets
wrong or lacks; every row cites its source.

Pass `when` as the call's timestamp for current-price (nominal) dollars, or a
fixed reference date for constant-price dollars.
"""

import csv
import functools
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from genai_prices import Usage as GPUsage
from genai_prices import calc_price

from .usage import Usage

_OVERRIDES = Path(__file__).with_name("price_overrides.csv")
_MTOK = Decimal(1_000_000)


@dataclass(frozen=True)
class _Override:
    start: date
    input: Decimal
    cache_read: Decimal
    cache_write: Decimal
    output: Decimal


@functools.cache
def _overrides() -> dict[str, list[_Override]]:
    out: dict[str, list[_Override]] = {}
    with _OVERRIDES.open() as f:
        for row in csv.DictReader(f):
            inp = Decimal(row["input_mtok"])
            out.setdefault(row["model"], []).append(
                _Override(
                    date.fromisoformat(row["start"]),
                    inp,
                    Decimal(row["cache_read_mtok"] or inp),
                    Decimal(row["cache_write_mtok"] or inp),
                    Decimal(row["output_mtok"]),
                )
            )
    for rows in out.values():
        rows.sort(key=lambda r: r.start)
    return out


def normalize_model(model: str) -> str | None:
    """Map a village model string to a priceable one, or None if it has no list price."""
    if model.startswith("tinker://"):  # fine-tuned checkpoints
        return None
    return model.removeprefix("claude-code::").removeprefix("models/")


def cost(model: str, usage: Usage, when: datetime) -> Decimal | None:
    """Dollars for one call, or None if the model has no known price."""
    name = normalize_model(model)
    if name is None:
        return None
    if when.tzinfo is None:  # dataset timestamps are naive UTC
        when = when.replace(tzinfo=UTC)
    rows = [r for r in _overrides().get(name, []) if r.start <= when.date()]
    if rows:
        r = rows[-1]
        return (
            usage.input * r.input
            + usage.cache_read * r.cache_read
            + usage.cache_write * r.cache_write
            + usage.output * r.output
        ) / _MTOK
    gp_usage = GPUsage(
        input_tokens=usage.total_input,  # genai-prices counts cached tokens inside input
        cache_read_tokens=usage.cache_read,
        cache_write_tokens=usage.cache_write,
        output_tokens=usage.output,
    )
    try:
        return calc_price(gp_usage, model_ref=name, genai_request_timestamp=when).total_price
    except LookupError:
        return None
