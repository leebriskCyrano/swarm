from datetime import datetime
from decimal import Decimal

from aivillage import prices
from aivillage.usage import Usage

MTOK_IN = Usage(1_000_000, 0, 0, 0, "openai")
MTOK_OUT = Usage(0, 0, 0, 1_000_000, "openai")


def test_dated_price_change():
    # o3 list price was cut on 2025-06-10
    assert prices.cost("o3-2025-04-16", MTOK_OUT, datetime(2025, 5, 1, 12)) == 40
    assert prices.cost("o3-2025-04-16", MTOK_OUT, datetime(2025, 7, 1, 12)) == 8


def test_cache_buckets():
    u = Usage(input=0, cache_read=1_000_000, cache_write=1_000_000, output=0, provider="anthropic")
    # Opus 4.5: cache read $0.50, 5-minute cache write $6.25 per MTok
    assert prices.cost("claude-opus-4-5-20251101", u, datetime(2026, 1, 1)) == Decimal("6.75")


def test_override_applies_from_start_date():
    assert prices.cost("deepseek-reasoner", MTOK_OUT, datetime(2026, 1, 1)) == Decimal("0.42")


def test_model_normalization():
    when = datetime(2026, 2, 1)
    assert prices.cost("claude-code::claude-opus-4-5-20251101", MTOK_IN, when) == 5
    assert prices.cost("models/gemini-2.5-pro-preview-05-06", MTOK_IN, when) is not None


def test_unpriced():
    assert prices.cost("tinker://abc/sampler_weights/x", MTOK_IN, datetime(2026, 6, 1)) is None
    assert prices.cost("no-such-model-xyz", MTOK_IN, datetime(2026, 6, 1)) is None
