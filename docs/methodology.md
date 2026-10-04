# Measuring a Wallis-North transaction sector in a multi-agent LLM system

This is the method we used to size the "transaction sector" of the AI Village
(`aidigestorg/ai-village`, export of 2026-09-20) in model spend. It's written
so it can be reused on another dataset of LLM agents that work and talk to
each other. The AI Village specifics appear as worked examples, marked
**Village:**. Code references are to this repo (`src/aivillage/`).

The method has five stages:

1. Define the economy: what counts as GDP, and what a "transaction" is.
2. Build a per-call ledger: one row per model call, with tokens.
3. Price the ledger at the prices in force when each call ran.
4. Classify spend into transaction and transformation activity.
5. Aggregate per period and per capita, with uncertainty, and plot it.

Each stage ends with the checks that caught real errors in our data. Run
them on a new dataset: every one of them failed at least once on ours.

## 1. The framework

Wallis & North (1986) define **transaction costs** as the value of the inputs
used to make exchanges, as opposed to **transformation costs**, the inputs used
to turn inputs into outputs. They can't observe transaction costs directly, so
they measure the **transaction sector** in two parts:

1. **Transaction industries** (finance, insurance, real estate, wholesale and
   retail trade): all their resources count.
2. **Transaction occupations** inside every other industry (managers,
   supervisors, clerks, purchasing and sales staff): only their wages count.

They report the sector as a share of GNP, about 25% in 1870 rising to about
45% in 1970. Their 1988 follow-up argues that most of it is already counted
as intermediate product, so it shouldn't be subtracted from GNP. Citations
and notes are in `docs/papers/README.md`.

The mapping to an agent economy:

| Wallis-North | Agent economy |
| --- | --- |
| GNP | Total model spend (tokens, and dollars at list prices) |
| Resource input | A model call: its input and output tokens |
| Transaction industry | Actions whose only purpose is exchange: chat messages, room or channel moves, searching shared history, requests to humans, approval requests, credential handoffs |
| Transaction occupation | Work-mode actions that exchange, coordinate or monitor: emailing, posting and commenting, reviewing another agent's work, trading, checking what others did |
| Transformation | Producing or changing artifacts: code, documents, analysis, testing one's own work, operating tools, playing a game that is itself the task |

One advantage over the original: Wallis & North could observe only
*marketed* transaction services. Waiting, searching and monitoring done by
the buyer themselves were invisible to them. In an agent log they are visible
as model calls. We keep them as separate, explicitly labelled categories
rather than deciding up front whether they belong in the sector:

- **Waiting / pausing**: calls in which an agent decides to idle. Wallis &
  North's "waiting in line" is a transaction cost they couldn't measure.
- **Memory consolidation**: an agent contracting with its future self.
  Arguably the firm-internal coordination of their hierarchy argument,
  arguably overhead.
- **Session start/stop decisions**: choosing what to work on next.

Decide where these belong with the people who'll use the result, and show
them separately in the meantime.

## 2. The per-call ledger

**Goal:** one row per model call that was paid for, with:

- timestamp
- agent
- model
- input tokens, split into uncached, cache read and cache write
- output tokens
- an activity label (the action the call produced)
- the model's own narration of what it was doing

Code: `usage.py` (parsing usage blocks), `calls.py` (per-table extracts) and
`costs.py` (the joined, deduplicated ledger).

### 2a. Find every source of model calls

Agent platforms usually log calls in more than one place. **Village:** three
tables carry calls:

- `computer_use_turns`: one row per action in work mode, with the raw provider
  response
- `events`: the activity timeline, with chat and decisions made outside work
  mode
- `claude_code_messages`: one agent that ran on a different harness

### 2b. Normalize provider usage blocks

The raw response is usually stored as the provider returned it, and providers
count cached tokens differently. Normalize everything to four billable
buckets: uncached input, cache read, cache write, and output (including
reasoning or thinking tokens).

| Provider format | Uncached input | Cache read | Cache write | Output |
| --- | --- | --- | --- | --- |
| Anthropic `usage` | `input_tokens` (already excludes cache) | `cache_read_input_tokens` | `cache_creation_input_tokens` | `output_tokens` |
| OpenAI chat `usage` | `prompt_tokens − cached_tokens` | `prompt_tokens_details.cached_tokens` | — | `completion_tokens` (includes reasoning) |
| OpenAI Responses `usage` | `input_tokens − cached_tokens` | `input_tokens_details.cached_tokens` | — | `output_tokens` |
| Gemini `usageMetadata` | `promptTokenCount − cachedContentTokenCount` | `cachedContentTokenCount` | — | `candidatesTokenCount + thoughtsTokenCount` |

Find the usage block by searching the response structure: SDK wrappers nest
it. Also keep the provider's **response id**, which is what deduplication
needs. Treat placeholder ids (a run of zeros, or short counters like
`bash_11`) as missing.

### 2c. Deduplicate

The same call is often logged more than once. Check each of these on a new
dataset:

- **Mirrored events.** A chat or tool event is frequently a copy of the
  work-mode turn that produced it. Match them by shared response id, or by the
  same agent issuing the matching tool call within a few seconds. Confirm the
  match by comparing usage where both sides have it. **Village:** identical in
  100% of 28,538 pairs. All chat events after the 2026-03-24 scaffolding change
  were mirrors.
- **Per-block logging.** Some harnesses write one log entry per content block
  and repeat the response's usage on each. Collapse them by response id.
  **Village:** 168,266 Claude Code rows were only 65,571 calls. Counting rows
  would have overstated that agent's input 2.6×.
- **Multi-action responses.** One response can issue several actions, logged
  as several turns. Price the call once, at its full size so per-request price
  tiers apply, and split the cost evenly across its actions with a weight of
  1/k. The split matters for classification, because the actions can differ
  in kind.
- **Synthetic calls.** Scaffold-generated bootstrap turns carry zero usage or
  placeholder ids. Count them as zero, and never impute tokens for them.
- **Non-calls.** Human-side and system events (approval responses, restarts)
  carry no tokens. Drop them.

### 2d. Fill in missing usage

Some providers' responses carry no usage at all. **Village:** none of the
OpenAI-compatible responses did (OpenAI, DeepSeek, Kimi, GLM, Grok), about 42%
of turns. Fill the gaps in this order:

1. **Exact counts from a mirror.** If a mirrored event reports token counts,
   use them. First check what the field means for each provider. **Village:**
   for Anthropic, the event's `inputTokens` is the uncached part only (equal
   to `input_tokens` in 100% of rows checked). So Anthropic calls with only
   that field are priced as fully uncached and flagged as lower bounds.
2. **Calibrated imputation for everything else.** Impute each call from the
   same agent's same-day median over its calls with known tokens, scaled by
   the action type's typical ratio to that pool, times one calibration
   constant. Fit the ratios and the constant on agents whose usage *is* known,
   then **validate by hiding their usage and imputing it.** **Village:** the
   naive version overstated input by 23%. That was because outer-loop calls
   such as memory consolidation have much larger prompts than ordinary turns.
   Restricting the pool to mirrored work turns, scaling by action type and
   calibrating gave per-agent-month input within about 0.9–1.27× of the truth,
   and output within about 0.5–1.8×, with aggregates exact by construction.
   Flag imputed rows.
3. **Unknown cache split.** If a provider doesn't report cache usage, give a
   central estimate and a bound. **Village:** the central estimate applies the
   cache-read share observed on reporting providers that week; the upper bound
   assumes no caching. Total spend was $110k central, $160k upper bound.

**Checks for stage 2:**

- Usage coverage by model: it's usually all-or-nothing per provider.
- The share of input tokens that are exact, reported, imputed and lower bounds
  (**Village:** 63%, 5%, 31%, 1%).
- Duplicate response ids within and across sources, and what the most
  repeated ids look like.

## 3. Prices

Price each call at the list price **in force when it ran**, and also at one
fixed reference date. The first is current-price "GDP"; the second is a
constant-price series. They differ where list prices changed. Keep raw token
volume as a third, price-free series.

- **Source:** pydantic's `genai-prices` package, pinned to an exact version
  (its price data ships in the package). It records dated price changes,
  per-request long-context tiers and time-of-day discounts, so **price each
  call individually**; never price aggregated tokens. It matched Anthropic's
  official price page exactly, including model-specific cache multipliers.
- **Cross-check:** LiteLLM's `model_prices_and_context_window.json`, sampled
  weekly from its git history, as an independent price history.
- **Overrides:** a small CSV for models the package lacks or gets wrong, with
  the source cited on every row (`price_overrides.csv`). **Village:** five
  models, under 1% of spend. Models routed through aggregators such as
  OpenRouter have blended prices that can swing several-fold within weeks;
  mark them approximate.
- **No list price** (for example fine-tuned checkpoints): leave the cost
  empty and report the share of calls this affects. Don't guess a price.
- Never fill a price table from memory. Model prices change faster than any
  one person's (or model's) knowledge of them.

## 4. Classification

### 4a. By action type (exact)

Map each call's action or event type to exchange, waiting, memory, session
or work. This needs no judgment beyond the mapping itself. **Village:**
`wallis_north.EXCHANGE` and the related constants.

### 4b. Transaction work inside work mode (estimated)

An action type like "click" or "run a command" doesn't reveal its purpose.
The model's narration of the turn does. Use a labelled sample to estimate the
share:

1. **Write the rubric before sampling:**
   - **T:** the turn exists to exchange with, coordinate with or monitor
     another party.
   - **P:** producing or changing artifacts, including research for one's own
     production and testing one's own work.
   - **U:** unclear.

   Record the debatable calls explicitly. Ours: prediction-market trading is
   T, since Wallis & North treat finance as a transaction industry; games
   played as the task are P; questionnaires are U.
2. **Draw the sample with probability proportional to cost**
   (`ORDER BY -ln(random()) / cost`). The share of labels then estimates the
   share of *spend* directly, with no reweighting.
3. **Label, then estimate shares with Wilson intervals,** leaving out U.
   **Village:** 250 labels gave 37% of work spend as transaction work overall:
   40% (95% CI 29–52%) before the scaffolding change and 32% (25–40%) after.
4. **Any automatic classifier must be scored on a holdout labelled after the
   classifier was frozen.** **Village:** keyword rules on narration looked
   fine in-sample (recall 0.91, false-positive rate 0.19) but flagged 40% of
   production turns on the holdout (precision 0.48), so we rejected them.
5. With a validated classifier (an LLM is the obvious candidate), apply it to
   a large sample stratified by month. Correct its shares for its measured
   error rates, `true = (observed − FPR) / (TPR − FPR)`, and propagate the
   uncertainty in those rates. Without one, report the transaction-work share
   per regime from the labels and say so.

**Data handling.** Sending narration to an external API exposes whatever it
contains, including names and email addresses of real people in agent logs.
Check the dataset's terms. Mask emails and handles before sending. Use
providers that don't train on or retain prompts.

## 5. Aggregation and per-capita measures

- **Transaction sector** = exchange spend + work spend × transaction-work
  share, as a share of total spend per month, with the interval carried
  through from the share estimate.
- **Unit cost of an exchange:** dollars and tokens per exchange action.
  **Village:** tokens per exchange rose more than 5× (about 9k → 48k) as
  contexts grew, while dollars per exchange stayed around $0.03–0.09 as
  prices fell and caching grew. So exchange got more expensive in real terms,
  while cheaper prices hid it in dollar terms.
- **Population:** count agents from the calls themselves, not from an agents
  table. Measure **agent-hours** as each agent's span from first to last call
  per local day, which follows schedule changes without hard-coding them.
  Group by the platform's own local day: **Village:** a 9am–5pm PT window
  crosses UTC midnight, and UTC days produced impossible 16-hour agent-days.
- **Per-capita series:** spend, transaction-sector spend, tokens and exchange
  actions per agent-hour. **Village:** active agents per day grew from 4 to
  about 31, so most GDP growth was population growth. Spend per agent-hour
  roughly halved after March 2026.

## Confounds to check before interpreting a trend

- **Scaffolding changes.** Read the platform's changelog. A change in how
  actions are logged moves the action-type categories mechanically.
  **Village:** the 2026-03-24 change moved chat from separate outer-loop calls
  to single turns inside work mode. Exchange spend fell from about 15–20% to
  about 4% of GDP partly for that reason, and the transaction sector's step
  from about 45% to 32% can't be separated from it. Mark such changes on
  every chart.
- **Model mix.** Current-price shares move when relative prices change, even
  if behaviour doesn't. Compare the current-price, constant-price and token
  series; where they diverge, the cause is prices.
- **Population and schedule.** Totals scale with agents and hours. Look at
  per-capita series before reading behaviour into totals.
- **Estimation.** In-work transaction shares estimated by regime are constant
  within each regime by construction. Only the exact components vary
  month to month.

## Reproducing on the AI Village

```sh
uv sync
uv run aiv download --tier medium && uv run aiv download computer_use_turns
uv run python -c "from aivillage import calls; [calls.build(t) for t in calls.EXTRACTORS]"
uv run python -c "from aivillage import costs; costs.build()"
uv run python scripts/plot_wallis_north.py   # data/figures/wallis_north.{png,csv}
```

About 10 minutes after the downloads, on 4 cores. `HANDOFF.md` has the
current status and open questions.

## Adapting to another dataset

1. Write an extractor per table that logs model calls (`calls.EXTRACTORS`):
   timestamp, agent or session, action kind, narration, raw response.
2. Check `usage.normalize` against the new provider formats, and extend it if
   needed.
3. Redo the deduplication checks (2c) from scratch: the mirror rules
   (`costs._MIRRORS`) are platform-specific.
4. Redo the imputation validation (2d) if any provider lacks usage blocks.
5. Rewrite the action-type mapping (4a) for the platform's actions.
6. Draw and label new cost-weighted samples (4b) with the same rubric.
   Labels don't transfer between platforms.
7. Read the platform's changelog and mark its regime changes.
