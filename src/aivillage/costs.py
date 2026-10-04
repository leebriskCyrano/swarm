"""The per-call cost table: one row per model call the village paid for.

Built from the extracts in `calls` (run `calls.build` for computer_use_turns,
events and claude_code_messages first). The rules, each checked against the
data before it was written down:

- An event that mirrors a computer-use turn is the same model call (identical
  usage in 100% of pairs where both carry it), so it isn't counted twice. A
  mirror shares a real response id with a turn, or follows a turn by the same
  agent within 5s whose action is the event's tool (AGENT_TALK after
  send_message_back_to_chat, PAUSE after pause, ...). Its reported token
  counts are kept for the turn when the turn has no usage block.
- Events without tokens or usage (approval responses, sign-in restarts) aren't
  model calls.
- Claude Code SDK logs repeat each response once per content block: those are
  collapsed by response id, and responses already present as turns dropped.
- Turns sharing a real response id are one call that issued several actions:
  each row keeps the call's full tokens (so per-request price tiers apply) and
  `weight` = 1/k splits it. Sum `tokens * weight` and `cost` (already weighted).
- Placeholder ids (a run of 16 zeros) mark scripted bootstrap turns: zero tokens.

OpenAI-compatible providers' responses carry no usage. Their tokens come from
mirrored events where possible, otherwise they are imputed (`imputed`): see
_SQL_IMPUTE. Checked on Anthropic/Gemini agents by hiding their usage, the
rule's calibration fixes aggregate totals; per agent-month, input lands within
about 0.9-1.25x of the truth and output within about 0.5-2x.

Anthropic's reported input counts only uncached tokens, so Anthropic calls
without a usage block (mostly events before November 2025) are priced as
uncached and flagged `input_lower_bound`: the cached part is missing.

For the others the cache split is unknown: `cost` assumes the cache-read share
observed on Anthropic and Gemini calls that week; `cost_uncached` is the upper
bound with no caching.
"""

import csv
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import duckdb

from . import calls, hub, prices, tables
from .usage import Usage

_VALID_ID = "(length({c}) >= 16 AND NOT regexp_matches({c}, '0{{16}}'))"
_MIRRORS = {  # event actionType -> the turn action it mirrors
    "AGENT_TALK": "send_message_back_to_chat",
    "ENTER_ROOM": "move_to_room",
    "PAUSE": "pause",
    "WAIT": "wait",
    "SEARCH_HISTORY": "search_history",
    "REQUEST_GOOGLE_SIGN_IN": "request_Google_sign_in",
    "REQUEST_HUMAN_HELPER": "request_human_helper",
    "CANCEL_REQUEST_FOR_HUMAN_HELPER": "cancel_request_for_human_helper",
    "OUTREACH_APPROVAL_REQUEST": "request_approval_for_unsolicited_outreach",
}
CONSTANT_PRICE_DATE = datetime(2026, 9, 15)


def path() -> Path:
    return calls.derived_dir() / "costs.parquet"


def _raw(name: str) -> str:
    return str(hub.raw_path(tables.get(name)))


def _sql_calls(d: Path) -> str:
    valid = _VALID_ID.format(c="response_id")
    mirrors = " UNION ALL ".join(f"SELECT '{e}' ekind, '{t}' tkind" for e, t in _MIRRORS.items())
    return f"""
CREATE TEMP TABLE agents AS
  SELECT id::VARCHAR id, model_string FROM read_json('{_raw("agents")}');
CREATE TEMP TABLE sess AS
  SELECT id::VARCHAR id, agent_id::VARCHAR agent_id
  FROM read_json('{_raw("computer_use_sessions")}');
CREATE TEMP TABLE turns AS
  SELECT t.*, s.agent_id agent, {valid} valid_id,
         coalesce(regexp_matches(response_id, '0{{16}}'), false) synthetic
  FROM '{d}/computer_use_turns_calls.parquet' t LEFT JOIN sess s ON s.id = t.session_id;
CREATE TEMP TABLE events AS
  SELECT *, {valid} valid_id FROM '{d}/events_calls.parquet'
  WHERE provider IS NOT NULL OR coalesce(reported_input, 0) + coalesce(reported_output, 0) > 0;
CREATE TEMP TABLE mirrors AS {mirrors};

-- each event's mirrored turn, by response id or by timing + action
CREATE TEMP TABLE ev_turn AS
WITH by_id AS (
  SELECT e.id eid, any_value(t.id) tid FROM events e JOIN turns t USING (response_id)
  WHERE e.valid_id GROUP BY e.id
), prev AS (
  SELECT e.id eid, e.kind ekind, t.id tid, t.kind tkind,
         epoch(e.created_at) - epoch(t.created_at) dt
  FROM events e ASOF JOIN turns t ON t.agent = e.agent_id AND t.created_at <= e.created_at
), by_time AS (
  SELECT eid, tid FROM prev JOIN mirrors USING (ekind, tkind) WHERE dt < 5
)
SELECT eid, coalesce(by_id.tid, by_time.tid) tid
FROM by_id FULL JOIN by_time USING (eid);

-- tokens reported by mirroring events, for turns whose response has no usage
CREATE TEMP TABLE turn_reported AS
  SELECT m.tid, max(e.reported_input) ri, max(e.reported_output) ro
  FROM ev_turn m JOIN events e ON e.id = m.eid
  WHERE e.provider IS NULL GROUP BY m.tid;

CREATE TEMP TABLE raw_calls AS
SELECT 'computer_use_turns' AS source, t.id, t.created_at, t.agent agent_id, t.kind, t.detail,
  t.model, t.provider,
  CASE WHEN t.synthetic THEN 0 ELSE coalesce(t.input + t.cache_read + t.cache_write, r.ri) END
    total_input,
  CASE WHEN t.synthetic THEN 0 ELSE t.cache_read END cache_read,
  CASE WHEN t.synthetic THEN 0 ELSE t.cache_write END cache_write,
  CASE WHEN t.synthetic THEN 0 ELSE coalesce(t.output, r.ro) END output,
  1.0 / count(*) OVER (PARTITION BY CASE WHEN t.valid_id THEN t.response_id ELSE t.id END)
    weight,
  t.id IN (SELECT tid FROM ev_turn WHERE tid IS NOT NULL) mirrored
FROM turns t LEFT JOIN turn_reported r ON r.tid = t.id
UNION ALL
SELECT 'events', e.id, e.created_at, e.agent_id, e.kind, NULL, e.model, e.provider,
  coalesce(e.input + e.cache_read + e.cache_write, e.reported_input), e.cache_read,
  e.cache_write, coalesce(e.output, e.reported_output), 1.0, false
FROM events e WHERE e.id NOT IN (SELECT eid FROM ev_turn WHERE tid IS NOT NULL)
UNION ALL
SELECT 'claude_code_messages', any_value(c.id), min(c.created_at), any_value(c.agent_id),
  'claude_code', NULL, any_value(c.model), any_value(c.provider),
  max(c.input + c.cache_read + c.cache_write), max(c.cache_read), max(c.cache_write),
  max(c.output), 1.0, false
FROM '{d}/claude_code_messages_calls.parquet' c
WHERE c.response_id NOT IN (SELECT response_id FROM turns WHERE valid_id)
GROUP BY c.response_id;
"""


_SQL_IMPUTE = """
-- Turns without tokens get: the agent's same-day median over its mirrored turns
-- (the turns whose tokens are known for every provider), times the action's
-- typical ratio to that pool, times a constant calibrating the totals. Ratios
-- and constants are fitted on turns with usage, where the truth is known.
CREATE TEMP TABLE known AS
  SELECT agent_id, created_at, coalesce(kind, 'none') kind, total_input ti, output o, mirrored,
         provider IS NOT NULL has_usage
  FROM raw_calls WHERE source = 'computer_use_turns' AND total_input > 0;
CREATE TEMP TABLE pool_day AS
  SELECT agent_id, created_at::DATE d, median(ti) mi, median(o) mo
  FROM known WHERE mirrored GROUP BY ALL;
CREATE TEMP TABLE pool_agent AS
  SELECT agent_id, median(ti) mi, median(o) mo FROM known WHERE mirrored GROUP BY ALL;
-- fallback for agents with no mirrored turns (before chatting from computer use)
CREATE TEMP TABLE pool_village AS
  SELECT created_at::DATE d, median(ti) mi, median(o) mo FROM known GROUP BY ALL;
CREATE TEMP TABLE kind_ratio AS
  WITH per_kind AS (
    SELECT agent_id, kind, median(ti) mi, median(o) mo FROM known WHERE has_usage GROUP BY ALL
  )
  SELECT kind, median(k.mi / p.mi) ri, median(k.mo / p.mo) ro
  FROM per_kind k JOIN pool_agent p USING (agent_id) GROUP BY kind;
CREATE TEMP TABLE base AS
  SELECT k.*,
    coalesce(d.mi, a.mi) * coalesce(r.ri, 1) ei,
    coalesce(d.mo, a.mo) * coalesce(r.ro, 1) eo
  FROM known k
  LEFT JOIN pool_day d ON d.agent_id = k.agent_id AND d.d = k.created_at::DATE
  LEFT JOIN pool_agent a ON a.agent_id = k.agent_id
  LEFT JOIN kind_ratio r ON r.kind = k.kind;
CREATE TEMP TABLE calib AS
  SELECT sum(ti) / sum(ei) ci, sum(o) / sum(eo) co FROM base WHERE has_usage AND NOT mirrored;

CREATE TEMP TABLE cache_week AS
  SELECT date_trunc('week', created_at) w, sum(cache_read) / sum(total_input) AS share
  FROM raw_calls WHERE provider IS NOT NULL AND total_input > 0 GROUP BY ALL;

CREATE TEMP TABLE cost_calls AS
WITH est AS (
  SELECT c.*,
    coalesce(d.mi, pa.mi, pv.mi) * coalesce(r.ri, 1) * calib.ci ei,
    coalesce(d.mo, pa.mo, pv.mo) * coalesce(r.ro, 1) * calib.co eo
  FROM raw_calls c
  CROSS JOIN calib
  LEFT JOIN pool_day d ON d.agent_id = c.agent_id AND d.d = c.created_at::DATE
  LEFT JOIN pool_agent pa ON pa.agent_id = c.agent_id
  LEFT JOIN pool_village pv ON pv.d = c.created_at::DATE
  LEFT JOIN kind_ratio r ON r.kind = coalesce(c.kind, 'none')
)
SELECT c.source, c.id, c.created_at, c.agent_id,
  coalesce(c.model, a.model_string) model, c.kind, c.detail, c.weight,
  c.total_input IS NULL imputed,
  round(coalesce(c.total_input, c.ei))::BIGINT total_input,
  round(coalesce(c.output, c.eo))::BIGINT output,
  CASE
    WHEN c.cache_read IS NOT NULL THEN c.cache_read
    WHEN lower_bound THEN 0
    ELSE round(coalesce(c.total_input, c.ei) * coalesce(w.share, 0))
  END::BIGINT cache_read,
  coalesce(c.cache_write, 0)::BIGINT cache_write,
  c.provider IS NOT NULL OR lower_bound cache_known,
  lower_bound input_lower_bound
FROM (
  -- Anthropic's reported input counts uncached tokens only: without a usage block
  -- the cached part is unknown, so these rows are priced uncached as a lower bound
  SELECT *, provider IS NULL AND total_input IS NOT NULL
            AND coalesce(model, '') || coalesce(
              (SELECT model_string FROM agents WHERE id = agent_id), '') LIKE '%claude%'
            lower_bound
  FROM est
) c
LEFT JOIN agents a ON a.id = c.agent_id
LEFT JOIN cache_week w ON w.w = date_trunc('week', c.created_at);
"""


def _price_chunk(rows: list[tuple]) -> list[tuple]:
    out = []
    for rid, src, model, created, tin, cr, cw, o, known, w in rows:
        if model is None or tin is None or o is None:
            out.append((rid, src, None, None, None))
            continue
        u = Usage(tin - cr - cw, cr, cw, o, "")
        nominal = prices.cost(model, u, created)
        constant = prices.cost(model, u, CONSTANT_PRICE_DATE)
        uncached = nominal if known else prices.cost(model, Usage(tin, 0, 0, o, ""), created)
        out.append(
            (
                rid,
                src,
                *(None if x is None else float(x) * w for x in (nominal, constant, uncached)),
            )
        )
    return out


def build(workers: int = 4) -> Path:
    """Write data/derived/costs.parquet."""
    d = calls.derived_dir()
    out = path()
    priced_csv = out.with_suffix(".priced.csv.tmp")
    with duckdb.connect() as con:
        con.execute("SET enable_progress_bar = false")
        con.execute(_sql_calls(d))
        con.execute(_SQL_IMPUTE)
        rows = con.execute(
            "SELECT id, source, model, created_at, total_input, cache_read, cache_write, output,"
            " cache_known, weight FROM cost_calls"
        ).fetchall()
        chunks = [rows[i : i + 50_000] for i in range(0, len(rows), 50_000)]
        with priced_csv.open("w", newline="") as f, ProcessPoolExecutor(workers) as ex:
            w = csv.writer(f)
            w.writerow(["id", "source", "cost", "cost_constant", "cost_uncached"])
            for priced in ex.map(_price_chunk, chunks):
                w.writerows(priced)
        con.execute(
            "COPY (SELECT c.*, p.cost, p.cost_constant, p.cost_uncached FROM cost_calls c"
            f" JOIN read_csv('{priced_csv}', header=true, columns={{'id': 'VARCHAR',"
            " 'source': 'VARCHAR', 'cost': 'DOUBLE', 'cost_constant': 'DOUBLE',"
            " 'cost_uncached': 'DOUBLE'}) p USING (id, source) ORDER BY c.created_at)"
            f" TO '{out}' (FORMAT parquet, COMPRESSION zstd)"
        )
    priced_csv.unlink()
    return out
