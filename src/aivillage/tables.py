"""Registry of the tables in the AI Village dataset.

Sizes are a snapshot (Sept 2026) and only used to bucket tables into tiers;
the dataset is refreshed roughly weekly, so treat them as approximate.
"""

from dataclasses import dataclass
from typing import Literal

REPO_ID = "aidigestorg/ai-village"

Tier = Literal["small", "medium", "large"]

# Non-table files worth having locally. SCHEMA.md and CHANGELOG.md are gated.
DOC_FILES = ("README.md", "SCHEMA.md", "CHANGELOG.md", "manifest.json", "example.py")
TRANSCRIPT_FILE = "village-transcript.json"  # ~360 MB, chat-only convenience export


@dataclass(frozen=True)
class Table:
    name: str
    approx_bytes: int  # compressed size on the Hub
    approx_rows: int | None
    description: str

    @property
    def filename(self) -> str:
        return f"{self.name}.jsonl.gz"

    @property
    def tier(self) -> Tier:
        if self.approx_bytes < 10_000_000:
            return "small"
        if self.approx_bytes < 500_000_000:
            return "medium"
        return "large"


_TABLES = [
    Table("agents", 5_280, 31, "Agent metadata: name, emoji, model, goal, token usage"),
    Table("villages", 327, 1, "The village record"),
    Table("village_goals", 4_449, 45, "Village-wide goals with start/end times"),
    Table("agent_goals", 3_973, None, "Per-agent individual goals"),
    Table("chat_rooms", 1_711, 5, "Chat rooms"),
    Table("claude_code_sessions", 12_662, 300, "Session records for Claude Code agents"),
    Table("summaries", 2_851_293, 800, "LLM-generated daily/agent/goal summaries (secondary)"),
    Table("computer_use_sessions", 40_080_397, 37_000, "Computer-use sessions with session_goal"),
    Table("chat_messages", 52_543_996, 123_000, "Chat messages: room, speaker, content, time"),
    Table("claude_code_messages", 104_002_877, 245_000, "Claude Code message stream"),
    Table("events", 328_621_853, 233_000, "Activity timeline; data.actionType, event_index"),
    Table("agent_memories", 2_438_234_633, 165_000, "Long-term memories from consolidation"),
    Table("computer_use_turns", 2_475_319_119, 1_140_000, "Turn-by-turn computer use"),
]

TABLES: dict[str, Table] = {t.name: t for t in _TABLES}


def get(name: str) -> Table:
    try:
        return TABLES[name]
    except KeyError:
        raise KeyError(f"unknown table {name!r}; known: {', '.join(TABLES)}") from None


def by_tier(*tiers: Tier) -> list[Table]:
    return [t for t in _TABLES if t.tier in tiers]
