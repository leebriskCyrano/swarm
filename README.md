# swarm

Tools for processing the [AI Village dataset](https://huggingface.co/datasets/aidigestorg/ai-village)
(`aidigestorg/ai-village`): a near-verbatim export of AI Digest's long-running
multi-agent village: event timeline, chat, computer-use sessions and turns,
agent memories, goals, and screenshots.

## Access

The dataset is gated with manual review. Request access on the dataset page,
then expose a read-only Hugging Face token as `HF_TOKEN`. The dataset's terms
apply to everything here: research and analysis use only, no training without
permission, no re-identification, and cite AI Digest / AI Village.

Downloaded data goes to `data/` (git-ignored). Never commit dataset contents.

## Setup

```sh
uv sync
uv run aiv tables          # list tables, size tier, what's local
```

Environment variables:

| Variable       | Purpose                                              |
| -------------- | ---------------------------------------------------- |
| `HF_TOKEN`     | Hugging Face token with access to the dataset        |
| `AIV_DATA_DIR` | Where files go (default `./data`)                    |
| `AIV_REVISION` | Pin a dataset commit sha (the Hub copy updates weekly) |

## Data layout and sizes

Every table is a gzipped JSON Lines file. Compressed sizes:

- **small** (< 10 MB): `agents`, `villages`, `village_goals`, `agent_goals`,
  `chat_rooms`, `claude_code_sessions`, `summaries`
- **medium** (40–330 MB): `computer_use_sessions`, `chat_messages`,
  `claude_code_messages`, `events`
- **large** (~2.4 GB each): `agent_memories`, `computer_use_turns`

Gzip can't be range-read, so the workflow is to download once, convert to
Parquet with only the columns you need, and query the Parquet.

## Workflow

```sh
uv run aiv download                        # docs: README, SCHEMA, CHANGELOG, manifest
uv run aiv download --tier medium          # all small + medium tables (~600 MB)
uv run aiv peek computer_use_turns --remote -n 2   # inspect a large table without downloading it
uv run aiv download computer_use_turns     # big: ~2.5 GB
uv run aiv parquet computer_use_turns --columns id,session_id,created_at,agent_action
uv run aiv sql "SELECT data.actionType AS t, count(*) FROM events GROUP BY t ORDER BY 2 DESC"
```

From Python:

```python
import aivillage
from aivillage.screenshots import screenshot

con = aivillage.connect()  # DuckDB, one view per local table
df = con.sql("SELECT * FROM chat_messages LIMIT 1000").pl()  # polars

for row in aivillage.iter_rows("agent_memories", remote=True, limit=100):
    ...  # streamed from the Hub, nothing stored

png = screenshot(turn_row)  # fetches that day's tar once, then cached
```

`connect()` reads the Parquet copy of a table if there is one, otherwise the
raw `.jsonl.gz`.

## Caveats from the dataset authors

- Read `CHANGELOG.md` (fetched by `aiv download`) before drawing conclusions
  about behaviour over time. It records scaffolding changes such as prompts,
  tools, model upgrades and memory.
- Agents misreport. Treat narration as a claim, and check screenshots.
- `summaries` are LLM-generated without seeing inside computer sessions, so
  treat them as secondary.
- Timestamps (`created_at`, UTC) are authoritative. The village clock is
  Pacific time.

## Research

`docs/methodology.md` describes the Wallis-North transaction-sector method
built on this package; `docs/papers/` has the reference papers.

## Development

```sh
uv run pytest
uv run ruff check . && uv run ruff format .
```
