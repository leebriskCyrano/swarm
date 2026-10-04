"""Per-call usage extracts, the input to the cost table.

Each extract is one row per model call, with normalized token buckets (see
`usage`) and enough context to classify the call later. Extracts are written
as Parquet under data/derived/.
"""

import csv
import gzip
import json
from collections.abc import Callable, Iterator
from pathlib import Path

import duckdb

from . import hub, tables, usage

COLUMNS = [
    "source",  # table the call came from
    "id",  # row id in that table
    "created_at",
    "agent_id",  # null for turns: join session_id to computer_use_sessions
    "session_id",
    "kind",  # actionType (events) or agent_action name (turns)
    "detail",  # first line of a bash command, or the room/message target
    "model",  # model reported by the response, if any
    "response_id",  # provider response id; repeated when one call is logged twice
    "provider",  # usage block format, if any
    "input",
    "cache_read",
    "cache_write",
    "output",
    "reported_input",  # events only: data.inputTokens / outputTokens
    "reported_output",
    "text",  # turns: the model's narration, then the command or typed text (truncated)
]


def derived_dir() -> Path:
    return hub.data_dir() / "derived"


_TOKEN_COLUMNS = {
    "input",
    "cache_read",
    "cache_write",
    "output",
    "reported_input",
    "reported_output",
}


def _usage_cols(response) -> list:
    head = [usage.model(response), usage.response_id(response)]
    u = usage.extract(response)
    if u is None:
        return [*head, None, None, None, None, None]
    return [*head, u.provider, u.input, u.cache_read, u.cache_write, u.output]


_NARRATION_KEYS = ("text", "thinking", "summary")


def narration(response, limit: int = 400) -> str:
    """The model's own words in a raw response: visible text, thinking or its summary."""
    parts: list[str] = []

    def walk(o, depth: int = 0) -> None:
        if depth > 6 or sum(map(len, parts)) >= limit:
            return
        if isinstance(o, dict):
            for k in _NARRATION_KEYS:
                v = o.get(k)
                if isinstance(v, str) and len(v.strip()) > 20:
                    parts.append(v.strip())
            children = o.values()
        elif isinstance(o, list):
            children = o
        else:
            return
        for child in children:
            walk(child, depth + 1)

    walk(response)
    return " | ".join(parts)[:limit]


def _first_line(s: str | None, n: int = 200) -> str | None:
    return s.strip().split("\n", 1)[0][:n] if isinstance(s, str) else None


def turn_rows(rows: Iterator[dict]) -> Iterator[list]:
    for r in rows:
        a = r.get("agent_action")
        if isinstance(a, dict):
            kind = a.get("action") or ("command" if "command" in a else None)
            detail = _first_line(a.get("command") or a.get("text") or a.get("room"))
            acted = a.get("command") or a.get("text") or ""
        else:
            kind = detail = None
            acted = ""
        yield [
            "computer_use_turns",
            r["id"],
            r["created_at"],
            None,
            r["session_id"],
            kind,
            detail,
            *_usage_cols(r.get("agent_messages")),
            None,
            None,
            f"{narration(r.get('agent_messages'))} ⟂ {acted[:400]}",
        ]


def event_rows(rows: Iterator[dict]) -> Iterator[list]:
    for r in rows:
        d = r.get("data") or {}
        if "inputTokens" not in d and "output" not in d:
            continue  # human events (USER_TALK, renames) aren't model calls
        yield [
            "events",
            r["id"],
            r["created_at"],
            d.get("agentId") or d.get("speakerId"),
            d.get("computerUseSessionId"),
            d.get("actionType"),
            d.get("roomId"),
            *_usage_cols(d.get("output")),
            d.get("inputTokens"),
            d.get("outputTokens"),
            None,
        ]


def claude_code_rows(rows: Iterator[dict]) -> Iterator[list]:
    for r in rows:
        if r.get("message_type") != "assistant":
            continue  # user/system/result entries aren't model calls
        yield [
            "claude_code_messages",
            r["id"],
            r["created_at"],
            r.get("agent_id"),
            r.get("sdk_session_id"),
            r.get("message_type"),
            None,
            *_usage_cols(r.get("content")),
            None,
            None,
            narration(r.get("content")),
        ]


EXTRACTORS: dict[str, Callable[[Iterator[dict]], Iterator[list]]] = {
    "computer_use_turns": turn_rows,
    "events": event_rows,
    "claude_code_messages": claude_code_rows,
}


def _iter_raw(table: str) -> Iterator[dict]:
    path = hub.raw_path(tables.get(table))
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build(table: str) -> Path:
    """Write data/derived/<table>_calls.parquet from the downloaded raw table."""
    out = derived_dir() / f"{table}_calls.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = out.with_suffix(".csv.tmp")
    with tmp_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(EXTRACTORS[table](_iter_raw(table)))
    # read_csv maps names to columns by position, so keep COLUMNS order
    types = {c: "BIGINT" if c in _TOKEN_COLUMNS else "VARCHAR" for c in COLUMNS}
    types["created_at"] = "TIMESTAMP"
    with duckdb.connect() as con:
        con.execute(
            f"COPY (SELECT * FROM read_csv('{tmp_csv}', header=true, columns={types!r})) "
            f"TO '{out}' (FORMAT parquet, COMPRESSION zstd)"
        )
    tmp_csv.unlink()
    return out
