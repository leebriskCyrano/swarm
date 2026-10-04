# Handoff

Status as of 2026-10-04 (second session, later the same day), for the next session working on this repo.

## Where things stand

- The `aivillage` package and `aiv` CLI work against the real data (see README.md).
  `uv run pytest` passes 34 tests, and `ruff check` and `ruff format` are clean.
- **The research goal (from the user):** apply Wallis & North's transaction-sector
  measure to the village, sizing transactions in token spend (tokens and dollars),
  and plot it over time. Papers: `docs/papers/` (notes + `fetch.sh`; PDFs are not
  committed because the repo is public).
- Access works. `HF_TOKEN` authenticates and the gated dataset is approved. The
  cloud environment needs network access to `huggingface.co` and to `*.hf.co`:
  LFS/Xet files redirect to `us.aws.cdn.hf.co`, and `hf_xet` may also use
  `cas-*.xethub.hf.co`.
- Downloaded (`data/` is per-container, so a new session must re-download):
  docs, the small and medium tiers, and `computer_use_turns` (2.5 GB).
  `agent_memories` was only peeked at.
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

## The Wallis-North pipeline

Rebuild from scratch (about 10 minutes after downloads):

```sh
uv run aiv download --tier medium && uv run aiv download computer_use_turns
uv run python -c "from aivillage import calls; [calls.build(t) for t in calls.EXTRACTORS]"
uv run python -c "from aivillage import costs; costs.build()"
uv run python scripts/plot_wallis_north.py   # data/figures/wallis_north.{png,csv}
```

- `usage.py` normalizes provider usage blocks; `prices.py` prices a call at the
  list price in force (pinned `genai-prices` plus `price_overrides.csv`, each
  row sourced); `calls.py` extracts one row per call; `costs.py` builds
  `data/derived/costs.parquet`; `wallis_north.py` makes the monthly series.
- The dedup and imputation rules are documented in `costs.py`, each checked on
  the data. The important ones: events mirroring a turn are the same model call;
  Claude Code logs repeat a response per content block; OpenAI-compatible
  responses carry no usage (tokens from mirrored events, else calibrated
  imputation); Anthropic's `inputTokens` on events counts uncached input only.
- Totals: about $110k of model spend (upper bound $160k if no OpenAI-family
  input was cached), 105B input and 0.95B output tokens.
- Classification: exchange actions (chat, rooms, requests) are counted exactly
  by action type. Transaction work inside computer use is estimated from 250
  hand-labelled, cost-weighted work turns (`work_turn_labels.csv`, ids only):
  40% (95% CI 29-52%) before 2026-03-24 and 32% (25-40%) after. Keyword rules
  on narration were tried and rejected: on a frozen holdout they flagged 40%
  of production turns (precision 0.48).
- Results: the transaction sector is about 40-48% of spend through March 2026,
  about 32% after the scaffolding change (the step coincides with the regime
  change, so behaviour and scaffolding aren't separable there). Tokens per
  exchange action rose from about 9k to about 48k; dollars per exchange stayed
  around $0.03-0.09 as prices fell and caching grew.

## Next steps

1. A better in-work classifier: an LLM labelling a large stratified sample of
   turns (by month) would give a monthly series instead of two regime-level
   shares. There is no API key in the environment; it needs the user's
   go-ahead and budget.
2. Decide the judgment calls with the user: are waiting, memory consolidation
   and session decisions transaction costs? They're plotted separately.
3. Verify the `deepseek-reasoner` price override against DeepSeek's archived
   pricing pages (web.archive.org was unreachable from the container). It's
   under 1% of spend, so low priority.

## Context and constraints

- Disk: about 30 GB per session (the `df` total is misleading). Skip
  `images/` (screenshot tars) unless needed, and fetch single days through
  `screenshots.screenshot()`.
- The dataset terms: research use only, no training on the data, no
  re-identification, cite AI Digest / AI Village. Never commit anything under
  `data/`, and don't paste dataset contents into the repo.
- Read CHANGELOG.md before any over-time analysis. Scaffolding changes such
  as prompts, tools and model swaps confound behavioural trends.
