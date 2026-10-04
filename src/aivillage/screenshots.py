"""Screenshots for computer-use turns.

They live in one tar per Pacific-time day at
images/computer-use-turns/<YYYY-MM-DD>.tar, with a <turn_id>.png per turn.
Each day's tar is downloaded once and cached under data/raw.
"""

import datetime as dt
import tarfile
from zoneinfo import ZoneInfo

from . import hub

VILLAGE_TZ = ZoneInfo("America/Los_Angeles")


def village_day(created_at: str) -> str:
    """The Pacific-time date (YYYY-MM-DD) of a UTC `created_at` timestamp."""
    ts = dt.datetime.fromisoformat(created_at)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.UTC)
    return ts.astimezone(VILLAGE_TZ).date().isoformat()


def tar_filename(day: str) -> str:
    return f"images/computer-use-turns/{day}.tar"


def screenshot(turn: dict) -> bytes | None:
    """PNG bytes for a computer_use_turns row, or None if the turn has no screenshot.

    Rows with `screenshot_is_redacted` true return a placeholder image.
    """
    path = hub.download_file(tar_filename(village_day(turn["created_at"])))
    with tarfile.open(path) as tar:
        try:
            member = tar.extractfile(f"{turn['id']}.png")
        except KeyError:
            return None
        return member.read() if member else None
