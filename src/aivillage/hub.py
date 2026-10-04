"""Access to the dataset on the Hugging Face Hub.

Auth comes from the HF_TOKEN environment variable (huggingface_hub reads it
automatically). Set AIV_REVISION to a commit sha to pin a dataset snapshot,
and AIV_DATA_DIR to change where files land (default: ./data).
"""

import gzip
import io
import os
from pathlib import Path
from typing import IO

from huggingface_hub import HfFileSystem, hf_hub_download

from .tables import REPO_ID, Table


def data_dir() -> Path:
    return Path(os.environ.get("AIV_DATA_DIR", "data"))


def raw_dir() -> Path:
    return data_dir() / "raw"


def parquet_dir() -> Path:
    return data_dir() / "parquet"


def revision() -> str | None:
    return os.environ.get("AIV_REVISION") or None


def raw_path(table: Table) -> Path:
    return raw_dir() / table.filename


def download_file(filename: str, *, force: bool = False) -> Path:
    """Download one repo file into data/raw (skipped if already up to date)."""
    path = hf_hub_download(
        REPO_ID,
        filename,
        repo_type="dataset",
        revision=revision(),
        local_dir=raw_dir(),
        force_download=force,
    )
    return Path(path)


def download(table: Table, *, force: bool = False) -> Path:
    return download_file(table.filename, force=force)


def hub_path(filename: str) -> str:
    return f"datasets/{REPO_ID}/{filename}"


def open_remote(table: Table) -> IO[str]:
    """Stream a table straight from the Hub as decompressed text, no local copy.

    Reading only the first N lines fetches only the first few MB, so this is
    the cheap way to inspect the large tables.
    """
    fs = HfFileSystem()
    raw = fs.open(hub_path(table.filename), "rb", revision=revision(), block_size=8 * 2**20)
    return io.TextIOWrapper(gzip.GzipFile(fileobj=raw), encoding="utf-8")
