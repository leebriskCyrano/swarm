# Handoff

Status as of 2026-10-04, for the next session working on this repo.

## Where things stand

- The skeleton is done: the `aivillage` package plus the `aiv` CLI (see README.md).
  `uv run pytest` passes 8 tests, and `ruff check` and `ruff format` are clean.
- Nothing has run against real data yet. The previous session had no
  `HF_TOKEN`, and the dataset (`aidigestorg/ai-village`) is gated with manual
  approval. Gated files returned 401.
- Branch: `claude/disk-space-availability-5r1cji`. The name comes from the
  session's first question. No PR exists yet. Ask the user before opening one
  or renaming the branch.

## First steps

1. `uv sync`
2. Check auth: `[ -n "$HF_TOKEN" ] && echo set`, then
   `uv run aiv peek agents --remote -n 1`.
   - A 401 or "gated repo" error with the token set means access hasn't been
     approved on the dataset page yet. That's for the user to sort out.
3. `uv run aiv download` fetches README, SCHEMA.md, CHANGELOG.md and the
   manifest into `data/raw/`. Read **SCHEMA.md**: the previous session never
   saw it, so no column names are hard-coded beyond the README's
   (`id`, `created_at`, `data.actionType`, `event_index`, `session_goal`,
   `agent_action`, `screenshot_is_redacted`).
4. `uv run aiv download --tier medium` downloads about 530 MB.
5. Check that DuckDB's JSON type inference handles the real nested columns.
   `events.data` is JSONB with a different shape per `actionType`. If inference
   breaks, or gives `data` an unwieldy STRUCT, consider reading `data` as
   `JSON` via the `columns=` / `sample_size` options in
   `load._json_source`. The tests only cover clean synthetic rows.
6. Look at the large tables before downloading them:
   `aiv peek computer_use_turns --remote -n 2`. Each is about 2.4 GB
   compressed. Gzip JSONL can't be range-read, so either download once and
   `aiv parquet --columns ...`, or stream with `iter_rows(..., remote=True)`.

## Context and constraints

- Disk: about 30 GB per session (the `df` total is misleading). Skip
  `images/` (screenshot tars) unless needed, and fetch single days through
  `screenshots.screenshot()`.
- The dataset terms: research use only, no training on the data, no
  re-identification, cite AI Digest / AI Village. Never commit anything under
  `data/`, and don't paste dataset contents into the repo.
- Read CHANGELOG.md before any over-time analysis. Scaffolding changes such
  as prompts, tools and model swaps confound behavioural trends.
- The user hasn't said yet what they want to study. Ask them: agent
  behaviour over time, chat dynamics, or computer-use traces.
