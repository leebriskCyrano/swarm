"""End-to-end check of the cost table's dedup rules on a tiny synthetic village."""

import gzip
import json

import duckdb
import pytest

from aivillage import calls, costs

AGENT = "a-claude"
OAI = "a-gpt"
T0 = "2026-05-01 10:00:"


def _anthropic(rid, cache_read=1000, out=10):
    return {
        "id": rid,
        "model": "claude-opus-4-8",
        "usage": {
            "input_tokens": 100,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": 0,
            "output_tokens": out,
        },
    }


def _turn(tid, sid, sec, action, msgs):
    return {
        "id": tid,
        "session_id": sid,
        "created_at": f"{T0}{sec:02d}",
        "agent_action": action,
        "agent_messages": msgs,
    }


TURNS = [
    # one response that issued two actions: counted once, split in half
    _turn("t1", "s1", 0, {"action": "left_click"}, _anthropic("msg_aaaaaaaaaaaaaaaaaaaa1")),
    _turn("t2", "s1", 1, {"action": "key"}, _anthropic("msg_aaaaaaaaaaaaaaaaaaaa1")),
    # chat turn, mirrored by an AGENT_TALK event 1s later
    _turn("t3", "s1", 10, {"action": "send_message_back_to_chat"}, _anthropic("msg_b2")),
    # scripted bootstrap turn
    _turn("t4", "s1", 20, {"action": "mouse_move"}, _anthropic("msg_" + "0" * 32, 0, 0)),
    # OpenAI-family chat turn without usage; its event reports the tokens
    _turn("t5", "s2", 30, {"action": "send_message_back_to_chat"}, [{"id": "rs_x1"}]),
    # OpenAI-family click without usage: imputed
    _turn("t6", "s2", 40, {"action": "left_click"}, [{"id": "rs_x2"}]),
]
EVENTS = [
    {
        "id": "e1",
        "created_at": f"{T0}11",
        "data": {"actionType": "AGENT_TALK", "speakerId": AGENT, "inputTokens": 100},
    },
    {
        "id": "e2",
        "created_at": f"{T0}31",
        "data": {
            "actionType": "AGENT_TALK",
            "speakerId": OAI,
            "inputTokens": 5000,
            "outputTokens": 50,
        },
    },
    {  # an outer-loop call: counted
        "id": "e3",
        "created_at": f"{T0}50",
        "data": {"actionType": "WAIT", "agentId": OAI, "inputTokens": 7000, "outputTokens": 5},
    },
    {  # a human-side event: not a model call
        "id": "e4",
        "created_at": f"{T0}51",
        "data": {"actionType": "OUTREACH_APPROVAL_RESPONSE", "inputTokens": 0, "outputTokens": 0},
    },
]
CLAUDE_CODE = [  # one response logged once per content block
    {
        "id": f"c{i}",
        "agent_id": AGENT,
        "created_at": f"{T0}5{i}",
        "message_type": "assistant",
        "content": {"message": _anthropic("msg_cc1")},
    }
    for i in range(3)
]


def _write(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def table(tmp_path, monkeypatch):
    monkeypatch.setenv("AIV_DATA_DIR", str(tmp_path))
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw / "computer_use_turns.jsonl.gz", TURNS)
    _write(raw / "events.jsonl.gz", EVENTS)
    _write(raw / "claude_code_messages.jsonl.gz", CLAUDE_CODE)
    _write(
        raw / "agents.jsonl.gz",
        [{"id": AGENT, "model_string": "claude-opus-4-8"}, {"id": OAI, "model_string": "gpt-5.5"}],
    )
    _write(
        raw / "computer_use_sessions.jsonl.gz",
        [{"id": "s1", "agent_id": AGENT}, {"id": "s2", "agent_id": OAI}],
    )
    for t in calls.EXTRACTORS:
        calls.build(t)
    out = costs.build(workers=1)
    rows = duckdb.sql(f"SELECT * FROM '{out}'").fetchall()
    cols = duckdb.sql(f"SELECT * FROM '{out}'").columns
    return {r[cols.index("id")]: dict(zip(cols, r, strict=True)) for r in rows}


def test_dedup(table):
    # mirrored events e1/e2 and the non-call e4 are gone; Claude Code collapses to one row
    sources = sorted(r["source"] for r in table.values())
    assert sources.count("claude_code_messages") == 1
    assert {k for k, r in table.items() if r["source"] != "claude_code_messages"} == {
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
        "t6",
        "e3",
    }


def test_multi_action_response_split(table):
    assert table["t1"]["weight"] == table["t2"]["weight"] == 0.5
    assert table["t1"]["cost"] == pytest.approx(table["t3"]["cost"] / 2)


def test_bootstrap_turn_is_free(table):
    assert table["t4"]["total_input"] == 0 and table["t4"]["cost"] == 0


def test_reported_and_imputed_tokens(table):
    assert (table["t5"]["total_input"], table["t5"]["output"], table["t5"]["imputed"]) == (
        5000,
        50,
        False,
    )
    assert table["t6"]["imputed"] and table["t6"]["total_input"] > 0
    assert not table["t5"]["cache_known"]
    assert table["t5"]["cost_uncached"] >= table["t5"]["cost"]
