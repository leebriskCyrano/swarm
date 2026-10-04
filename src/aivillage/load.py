"""Reading tables: row iteration, Parquet conversion, and a DuckDB session.

The Hub files are gzipped JSON Lines, which can't be range-read, so the
workflow is: download (or stream) once, convert to Parquet keeping only the
columns you need, then query the Parquet.
"""

import gzip
import itertools
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import duckdb

from . import hub, tables
from .tables import Table

# Rows in computer_use_turns / claude_code_messages can be large.
_MAX_OBJECT_SIZE = 256 * 2**20


def _table(t: Table | str) -> Table:
    return tables.get(t) if isinstance(t, str) else t


def iter_rows(t: Table | str, *, remote: bool = False, limit: int | None = None) -> Iterator[dict]:
    """Yield rows as dicts, from the local download or streamed from the Hub."""
    table = _table(t)
    if remote:
        f = hub.open_remote(table)
    else:
        path = hub.raw_path(table)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `aiv download {table.name}` first")
        f = gzip.open(path, "rt", encoding="utf-8")
    with f:
        for line in itertools.islice(f, limit):
            if line.strip():
                yield json.loads(line)


def _quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _ident(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def _json_source(table: Table) -> str:
    path = hub.raw_path(table)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `aiv download {table.name}` first")
    # sample_size=-1 infers the schema from every row, not the first 20480.
    # With sampling, keys that first appear later (e.g. events.data.endReason)
    # are silently dropped, and type conflicts (claude_code_messages'
    # message.content is a string in some rows, an array in others) fail
    # mid-query. Costs an extra pass over the file when the view is bound.
    return (
        f"read_json({_quote(str(path))}, format='newline_delimited', compression='gzip', "
        f"maximum_object_size={_MAX_OBJECT_SIZE}, sample_size=-1)"
    )


def parquet_path(t: Table | str) -> Path:
    return hub.parquet_dir() / f"{_table(t).name}.parquet"


def to_parquet(t: Table | str, *, columns: Sequence[str] | None = None) -> Path:
    """Convert a downloaded table to zstd Parquet, optionally projecting columns."""
    table = _table(t)
    out = parquet_path(table)
    out.parent.mkdir(parents=True, exist_ok=True)
    select = ", ".join(_ident(c) for c in columns) if columns else "*"
    tmp = out.with_suffix(".parquet.tmp")
    with duckdb.connect() as con:
        con.execute(
            f"COPY (SELECT {select} FROM {_json_source(table)}) "
            f"TO {_quote(str(tmp))} (FORMAT parquet, COMPRESSION zstd)"
        )
    tmp.replace(out)
    return out


def connect(database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """DuckDB connection with a view per locally available table.

    Prefers the Parquet copy; falls back to reading the .jsonl.gz directly.
    """
    con = duckdb.connect(database)
    for table in tables.TABLES.values():
        pq = parquet_path(table)
        if pq.exists():
            source = f"read_parquet({_quote(str(pq))})"
        elif hub.raw_path(table).exists():
            source = _json_source(table)
        else:
            continue
        con.execute(f"CREATE OR REPLACE VIEW {_ident(table.name)} AS SELECT * FROM {source}")
    return con
