import gzip
import json

import pytest

from aivillage import load, screenshots, tables

ROWS = [
    {"id": 1, "agent": "a", "data": {"actionType": "AGENT_TALK", "text": "hi"}},
    {"id": 2, "agent": "b", "data": {"actionType": "WAIT"}},
    {"id": 3, "agent": "a", "data": {"actionType": "AGENT_TALK", "text": "it's fine"}},
]


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AIV_DATA_DIR", str(tmp_path))
    raw = tmp_path / "raw"
    raw.mkdir()
    with gzip.open(raw / "events.jsonl.gz", "wt") as f:
        for row in ROWS:
            f.write(json.dumps(row) + "\n")
    return tmp_path


def test_iter_rows(data_dir):
    assert list(load.iter_rows("events")) == ROWS
    assert len(list(load.iter_rows("events", limit=2))) == 2


def test_iter_rows_missing(data_dir):
    with pytest.raises(FileNotFoundError, match="aiv download agents"):
        list(load.iter_rows("agents"))


def test_connect_reads_jsonl(data_dir):
    with load.connect() as con:
        rel = con.sql("SELECT count(*) FROM events WHERE data.actionType = 'AGENT_TALK'")
        assert rel.fetchone()[0] == 2


def test_schema_inference_sees_late_rows(data_dir):
    # DuckDB's default sample is 20480 rows. A key that first appears after it,
    # or a type that changes after it, must still be handled.
    rows = [{"id": i, "data": {"actionType": "WAIT", "msg": ["a"]}} for i in range(20480)]
    rows.append({"id": 20480, "data": {"actionType": "STOP", "endReason": "done", "msg": "x"}})
    with gzip.open(data_dir / "raw" / "events.jsonl.gz", "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with load.connect() as con:
        assert con.sql("SELECT max(data.endReason) FROM events").fetchone()[0] == "done"
        assert con.sql("SELECT count(data.msg) FROM events").fetchone()[0] == len(rows)


def test_to_parquet_projects_columns(data_dir):
    out = load.to_parquet("events", columns=["id", "agent"])
    assert out.exists()
    with load.connect() as con:
        assert con.sql("SELECT * FROM events ORDER BY id").columns == ["id", "agent"]
        assert con.sql("SELECT count(*) FROM events").fetchone()[0] == 3


def test_tiers():
    assert tables.get("agents").tier == "small"
    assert tables.get("events").tier == "medium"
    assert tables.get("computer_use_turns").tier == "large"
    with pytest.raises(KeyError):
        tables.get("nope")


@pytest.mark.parametrize(
    "created_at, day",
    [
        ("2025-04-02T18:00:00", "2025-04-02"),
        ("2025-04-03T03:00:00", "2025-04-02"),  # 8pm PDT the previous day
        ("2025-12-01T07:59:59+00:00", "2025-11-30"),  # PST
    ],
)
def test_village_day(created_at, day):
    assert screenshots.village_day(created_at) == day
