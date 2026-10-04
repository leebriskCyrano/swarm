"""Command-line entry point: `aiv <command>`."""

import argparse
import json
import sys

from . import hub, load, tables


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000:
            return f"{n:.0f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def cmd_tables(args: argparse.Namespace) -> None:
    for t in tables.TABLES.values():
        local = []
        if hub.raw_path(t).exists():
            local.append("raw")
        if load.parquet_path(t).exists():
            local.append("parquet")
        print(
            f"{t.name:24} {t.tier:7} {_fmt_bytes(t.approx_bytes):>8}  "
            f"{','.join(local) or '-':12} {t.description}"
        )


def cmd_download(args: argparse.Namespace) -> None:
    targets = [tables.get(n) for n in args.tables]
    if args.tier:
        tiers = {
            "small": ["small"],
            "medium": ["small", "medium"],
            "all": ["small", "medium", "large"],
        }
        targets += tables.by_tier(*tiers[args.tier])
    files = [t.filename for t in dict.fromkeys(targets)]
    if args.docs or not files:
        files = list(tables.DOC_FILES) + files
    if args.transcript:
        files.append(tables.TRANSCRIPT_FILE)
    for f in files:
        print(f"downloading {f} ...", file=sys.stderr)
        print(hub.download_file(f, force=args.force))


def cmd_peek(args: argparse.Namespace) -> None:
    for row in load.iter_rows(args.table, remote=args.remote, limit=args.n):
        print(json.dumps(row, ensure_ascii=False, indent=None if args.compact else 2))


def cmd_parquet(args: argparse.Namespace) -> None:
    columns = args.columns.split(",") if args.columns else None
    for name in args.tables:
        print(load.to_parquet(name, columns=columns))


def cmd_sql(args: argparse.Namespace) -> None:
    with load.connect() as con:
        print(con.sql(args.query))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="aiv", description=__doc__)
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("tables", help="list tables, tiers and local status")
    s.set_defaults(func=cmd_tables)

    s = sub.add_parser("download", help="download tables into data/raw (docs only if none given)")
    s.add_argument("tables", nargs="*")
    s.add_argument("--tier", choices=["small", "medium", "all"], help="include tables up to tier")
    s.add_argument("--docs", action="store_true", help="also fetch README/SCHEMA/CHANGELOG")
    s.add_argument("--transcript", action="store_true", help="also fetch village-transcript.json")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("peek", help="print the first rows of a table")
    s.add_argument("table")
    s.add_argument("-n", type=int, default=3)
    s.add_argument("--remote", action="store_true", help="stream from the Hub, no download")
    s.add_argument("--compact", action="store_true", help="one row per line")
    s.set_defaults(func=cmd_peek)

    s = sub.add_parser("parquet", help="convert downloaded tables to Parquet")
    s.add_argument("tables", nargs="+")
    s.add_argument("--columns", help="comma-separated columns to keep")
    s.set_defaults(func=cmd_parquet)

    s = sub.add_parser("sql", help="run SQL against local tables (one view per table)")
    s.add_argument("query")
    s.set_defaults(func=cmd_sql)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
