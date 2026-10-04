import pytest

from aivillage import wallis_north as wn


def test_wilson_interval():
    s = wn.wilson(30, 100)
    assert (
        s.p == 0.3
        and s.lo == pytest.approx(0.219, abs=1e-3)
        and s.hi == pytest.approx(0.396, abs=1e-3)
    )


def test_work_transaction_share_from_labels():
    shares = wn.work_transaction_share()
    assert set(shares) == {"pre", "post"}
    for s in shares.values():
        assert 0 < s.lo < s.p < s.hi < 1 and s.n > 50


def test_categories_cover_exchange_kinds():
    assert "send_message_back_to_chat" in wn.EXCHANGE and "AGENT_TALK" in wn.EXCHANGE
    assert "CONSOLIDATE" in wn.CATEGORY_SQL
