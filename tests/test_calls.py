import gzip
import json

import duckdb

from aivillage import calls

USAGE = {"input_tokens": 5, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 0}
TURN = {
    "id": "t1",
    "session_id": "s1",
    "created_at": "2026-05-01 10:00:00.1",
    "agent_action": {"command": "gh issue list\n--limit 5"},
    "agent_messages": {"model": "claude-opus-4-8", "usage": {**USAGE, "output_tokens": 7}},
}
TALK = {
    "id": "e1",
    "created_at": "2026-05-01 10:00:01",
    "data": {
        "actionType": "AGENT_TALK",
        "speakerId": "a1",
        "roomId": "r1",
        "inputTokens": 105,
        "outputTokens": 7,
        "output": [{"type": "function_call"}],
    },
}
USER_TALK = {"id": "e2", "created_at": "2026-05-01 10:00:02", "data": {"actionType": "USER_TALK"}}


def test_turn_rows():
    (row,) = calls.turn_rows([TURN])
    r = dict(zip(calls.COLUMNS, row, strict=True))
    assert r["kind"] == "command"
    assert r["detail"] == "gh issue list"
    assert (r["model"], r["input"], r["cache_read"], r["output"]) == ("claude-opus-4-8", 5, 100, 7)


def test_event_rows_skip_humans_and_keep_reported_tokens():
    rows = list(calls.event_rows([TALK, USER_TALK]))
    assert len(rows) == 1
    r = dict(zip(calls.COLUMNS, rows[0], strict=True))
    assert (r["agent_id"], r["kind"], r["provider"]) == ("a1", "AGENT_TALK", None)
    assert (r["reported_input"], r["reported_output"]) == (105, 7)


def test_build_writes_parquet(tmp_path, monkeypatch):
    monkeypatch.setenv("AIV_DATA_DIR", str(tmp_path))
    (tmp_path / "raw").mkdir()
    with gzip.open(tmp_path / "raw" / "events.jsonl.gz", "wt") as f:
        for row in (TALK, USER_TALK):
            f.write(json.dumps(row) + "\n")
    out = calls.build("events")
    got = duckdb.sql(f"SELECT kind, reported_input, created_at FROM '{out}'").fetchall()
    assert len(got) == 1 and got[0][:2] == ("AGENT_TALK", 105)
