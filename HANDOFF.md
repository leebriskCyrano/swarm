# Handoff

Status as of 2026-10-04 (second session), for the next session working on this repo.

## Where things stand

- The `aivillage` package and `aiv` CLI work against the real data (see README.md).
  `uv run pytest` passes 9 tests, and `ruff check` and `ruff format` are clean.
- Access works. `HF_TOKEN` authenticates and the gated dataset is approved. The
  cloud environment needs network access to `huggingface.co` and to `*.hf.co`:
  LFS/Xet files redirect to `us.aws.cdn.hf.co`, and `hf_xet` may also use
  `cas-*.xethub.hf.co`.
- Downloaded in this session (`data/` is per-container, so a new session must
  re-download): docs, plus the small and medium tiers, 504 MB in total. The
  large tables (`agent_memories`, `computer_use_turns`) were only peeked at
  remotely.
- Export in use: `exportedAt` 2026-09-20. The manifest's row counts are well
  above SCHEMA.md's approximations: events 381,610 (SCHEMA says ~235k),
  computer_use_turns 2,510,487 (~1.16M), agent_memories 246,151 (~166k),
  agents 46 (31). Use the manifest, not SCHEMA.md or `tables.approx_rows`.
- Branch: `claude/peaceful-dirac-a5epoh`. The first session's branch was
  `claude/disk-space-availability-5r1cji`. No PR exists yet. Ask the user
  before opening one.

## What the first real-data run found

- **DuckDB's sampled schema inference was wrong.** Fixed in
  `load._json_source` with `sample_size=-1`, and covered by
  `test_schema_inference_sees_late_rows`. Two failures with the default
  20,480-row sample:
  - `events.data.endReason` / `endComment` (`STOP_HUMAN_USE_SESSION`) first
    appear at row 29,483. They were silently missing from the STRUCT, so
    querying them gave a Binder Error.
  - In `claude_code_messages.content`, `message.content` is a string in some
    rows and an array in others. Any query that parsed `content` failed at
    line 49,148, and so would `aiv parquet claude_code_messages`.
- Cost of the fix: binding a raw `.jsonl.gz` view now takes a full pass, so
  `connect()` over the medium tier takes about 13s instead of about 5s. All 11
  local tables now parse completely. With a large table raw on disk, that
  pass would take minutes on every `connect()`. Convert large tables with
  `aiv parquet` first: `connect()` prefers the Parquet copy.
- `events.data` is a wide sparse STRUCT (about 46 keys, the union over all
  `actionType`s). It works, with dot access. `data.output` (provider-shaped,
  list or dict) is inferred as JSON. `cost`, `inputTokens` and `outputTokens`
  are integers in every row.
- Large tables, from remote peeks: rows are ordered by `id` (UUID), not by
  time, so the first N rows are a scattered sample across dates. In
  `computer_use_turns`, `agent_messages` is a list (OpenAI Responses) or a dict
  (Gemini `candidates`, Anthropic), and `agent_action` holds `{command}` for
  bash. In `agent_memories`, `content` is 20–50 KB of markdown per row.

## Next steps

1. Ask the user what they want to study (agent behaviour over time, chat
   dynamics, or computer-use traces). That decides which large table, if any,
   is worth the 2.4 GB download.
2. Read `data/raw/CHANGELOG.md` before any over-time analysis.
3. For a large table: `aiv download <table>`, then
   `aiv parquet <table> --columns ...` straight away, so later `connect()`
   calls skip the full-file inference.

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
