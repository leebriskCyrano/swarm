"""LLM labels for computer-use work turns: transaction (T), production (P) or unclear (U).

Methodology section 4b. An LLM applies the hand-labelling rubric to a
cost-weighted sample of work turns in every month. Its error rates are
measured on the 250 hand-labelled turns (work_turn_labels.csv), and
`wallis_north.llm_work_shares` corrects the monthly shares for them.

    uv run python -m aivillage.classify validate        # label the 250 hand-labelled turns
    uv run python -m aivillage.classify run             # label the monthly sample
    uv run python -m aivillage.classify export          # write work_turn_llm_labels.csv

Privacy: only a turn's `text` (narration, then the command or typed text) is
sent, with emails and @handles masked, and only to OpenRouter endpoints that
neither train on nor retain prompts (`zdr`, `data_collection: deny`).

Spend: every run checks the key's recorded usage against a cap before it
starts, and stops sending when the cap is reached. Results are cached under
data/derived/llm_labels/, one JSONL file per configuration, so a dropped run
resumes where it stopped. The committed CSV holds ids and labels only.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import duckdb

from . import calls, costs
from . import wallis_north as wn

API = "https://openrouter.ai/api/v1"
MODEL = "deepseek/deepseek-v4.1-flash"  # pinned: the ~flash-latest alias can move mid-run
MAX_PRICE = {"prompt": 0.3, "completion": 1.2}  # USD per million tokens
PROVIDER = {
    "zdr": True,  # zero data retention endpoints only
    "data_collection": "deny",  # no providers that train on or store prompts
    "quantizations": ["fp8"],  # one quantization, so labels don't depend on routing
    "order": ["deepinfra/fp8", "coreweave/fp8", "novita/fp8", "nextbit/fp8"],
    "allow_fallbacks": True,
    "max_price": MAX_PRICE,
}
CONFIGS = {  # name -> request settings
    "fast": {"reasoning": {"enabled": False}, "max_tokens": 8},
    "think": {"reasoning": {"effort": "low", "exclude": True}, "max_tokens": 4000},
}
SALT = "wallis-north-2026-10"  # fixes the monthly sample

SYSTEM = """\
You label single turns of AI agents working on computers in a multi-agent \
village. Each turn shows the agent's narration (its own words about what it is \
doing) and the command or text it entered. Emails and @handles are masked.

Decide what the turn is for:

T - transaction work: the turn exists to exchange with, coordinate with or \
monitor another party (another agent, a human, an organisation, a market). \
For example: writing, sending or reading email; posting, commenting or \
replying; messaging; reviewing another agent's work; checking what others \
did or said. Trading on a prediction market is T.

P - production work: producing or changing artifacts. For example: writing \
code, documents or analysis; editing files; operating tools to build \
something; research for one's own production; testing one's own work. \
Playing a game that is itself the task is P.

U - unclear: the purpose can't be told from the turn. Filling in a \
questionnaire or survey is U.

Answer with exactly one letter: T, P or U."""

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_HANDLE = re.compile(r"(?<![\w@])@\w[\w.-]*")


def mask(text: str) -> str:
    """Replace email addresses and @handles before text leaves the machine."""
    return _HANDLE.sub("@handle", _EMAIL.sub("<email>", text))


def messages(text: str) -> list[dict]:
    narration, _, acted = text.partition(" ⟂ ")
    turn = f"Narration: {narration.strip() or '(none)'}\nAction: {acted.strip() or '(none)'}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": mask(turn)}]


def parse(content: str | None) -> str | None:
    """The single label in a reply, or None if there isn't exactly one."""
    found = set(re.findall(r"\b([TPU])\b", content or ""))
    return found.pop() if len(found) == 1 else None


def config(name: str) -> dict:
    return {"model": MODEL, "system": SYSTEM, "provider": PROVIDER, **CONFIGS[name]}


def cache_path(name: str) -> Path:
    digest = hashlib.sha256(json.dumps(config(name), sort_keys=True).encode()).hexdigest()
    return calls.derived_dir() / "llm_labels" / f"{name}-{digest[:10]}.jsonl"


def load_cache(name: str) -> dict[str, dict]:
    p = cache_path(name)
    if not p.exists():
        return {}
    with p.open() as f:
        return {r["id"]: r for r in map(json.loads, f)}


# --- OpenRouter ---------------------------------------------------------------


class BudgetError(RuntimeError):
    pass


def _key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_TOKEN")
    if not key:
        raise RuntimeError("set OPENROUTER_API_KEY")
    return key


def _request(path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"{API}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def spent() -> float:
    """USD spent on this key so far: OpenRouter's count, or the cached costs if higher.

    The key's count lags by a minute or more, so it misses a run just finished.
    """
    cached = 0.0
    for p in (calls.derived_dir() / "llm_labels").glob("*.jsonl"):
        with p.open() as f:
            cached += sum(json.loads(line)["cost"] for line in f)
    return max(float(_request("/key")["data"]["usage"]), cached)


_RETRY = {408, 429, 500, 502, 503, 504, 520, 522, 524, 529}


def send(name: str, text: str, tries: int = 6) -> dict:
    """Label one turn. Retries rate limits, server errors and dropped connections."""
    cfg = CONFIGS[name]
    body = {
        "model": MODEL,
        "messages": messages(text),
        "temperature": 0,
        "max_tokens": cfg["max_tokens"],
        "reasoning": cfg["reasoning"],
        "provider": PROVIDER,
        "usage": {"include": True},
    }
    for attempt in range(tries):
        try:
            d = _request("/chat/completions", body)
            if "choices" not in d:  # an upstream error relayed with status 200
                raise ConnectionError(str(d.get("error"))[:200])
            content = d["choices"][0]["message"].get("content")
            u = d.get("usage") or {}
            return {
                "label": parse(content),
                "reply": (content or "")[:20],
                "cost": float(u.get("cost") or 0),
                "provider": d.get("provider"),
                "input_tokens": u.get("prompt_tokens"),
                "output_tokens": u.get("completion_tokens"),
            }
        except urllib.error.HTTPError as e:
            if e.code not in _RETRY or attempt == tries - 1:
                raise
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == tries - 1:
                raise
        time.sleep(min(60, 2**attempt) * (1 + random.random()))
    raise AssertionError("unreachable")


def label(
    texts: dict[str, str],
    name: str = "fast",
    budget: float = 10.0,
    workers: int = 16,
    send: Callable[[str, str], dict] = send,
    spent: Callable[[], float] = spent,
) -> dict[str, dict]:
    """Label every turn in `texts` (id -> text), resuming from the cache.

    `budget` caps the key's total spend in USD, earlier runs included. A run
    that would exceed it on the mean cost per call so far (or on the worst
    case, before 50 calls have been made) is refused before it starts.
    """
    done = load_cache(name)
    todo = [i for i in texts if i not in done]
    if todo:
        start = spent()
        costs_so_far = [r["cost"] for r in done.values()]
        if len(costs_so_far) >= 50:
            per_call = sum(costs_so_far) / len(costs_so_far)
        else:
            chars = len(SYSTEM) + max(len(texts[i]) for i in todo)
            per_call = (chars * MAX_PRICE["prompt"] + CONFIGS[name]["max_tokens"] * 3 *
                        MAX_PRICE["completion"]) / 3e6  # fmt: skip
        projected = per_call * len(todo)
        print(f"{len(todo)} to label, {len(done)} cached; spent ${start:.4f}, "
              f"projected ${projected:.4f}, cap ${budget:.2f}", file=sys.stderr)  # fmt: skip
        if start + projected > budget:
            raise BudgetError(f"${start:.2f} spent + ${projected:.2f} projected > ${budget:.2f}")
        path = cache_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = threading.Lock()
        running = [start]
        failed = 0

        def one(i: str) -> None:
            r = {"id": i, **send(name, texts[i])}
            with lock, path.open("a") as f:
                f.write(json.dumps(r) + "\n")
                running[0] += r["cost"]
                done[i] = r

        with ThreadPoolExecutor(workers) as pool:
            queue, pending = iter(todo), set()
            while True:
                while len(pending) < workers and running[0] < budget:
                    i = next(queue, None)
                    if i is None:
                        break
                    pending.add(pool.submit(one, i))
                if not pending:
                    break
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    if fut.exception() is not None:
                        failed += 1
                        print(f"failed: {fut.exception()!r}"[:300], file=sys.stderr)
                if len(done) % 500 < len(finished):
                    print(f"{len(done)} labelled, ${running[0]:.4f}", file=sys.stderr)
        print(f"done: {len(done)} labelled, {failed} failed, ~${running[0]:.4f} spent",
              file=sys.stderr)  # fmt: skip
        if running[0] >= budget:
            raise BudgetError(f"stopped at the ${budget:.2f} cap")
    return {i: done[i] for i in texts if i in done}


# --- Samples ------------------------------------------------------------------


def texts(ids: list[str]) -> dict[str, str]:
    """Each turn's narration and command/typed text, from the extract."""
    extract = calls.derived_dir() / "computer_use_turns_calls.parquet"
    with duckdb.connect() as con:
        con.execute("CREATE TEMP TABLE want (id VARCHAR)")
        con.executemany("INSERT INTO want VALUES (?)", [(i,) for i in ids])
        rows = con.execute(
            f"SELECT id, coalesce(text, '') FROM '{extract}' JOIN want USING (id)"
        ).fetchall()
    return dict(rows)


def validation_ids() -> list[str]:
    with wn.LABELS.open() as f:
        return [r["turn_id"] for r in csv.DictReader(f)]


def sample(per_month: int = 1000) -> list[tuple[str, str]]:
    """(turn id, month) of a cost-weighted sample of work turns in every month.

    Weighted sampling without replacement, `ORDER BY -ln(u) / cost` per month,
    where u is a uniform drawn from a hash of the turn id, so the same sample
    comes back on every run.
    """
    u = f"((hash(id || '{SALT}')::DOUBLE + 0.5) / 18446744073709551616.0)"
    sql = f"""
    SELECT id, strftime(created_at, '%Y-%m') AS month FROM '{costs.path()}'
    WHERE source = 'computer_use_turns' AND cost > 0 AND {wn.CATEGORY_SQL} = 'work'
    QUALIFY row_number() OVER (PARTITION BY month ORDER BY -ln({u}) / cost) <= {per_month}
    ORDER BY month, id"""
    return duckdb.sql(sql).fetchall()


def export(name: str = "fast", per_month: int = 1000) -> Path:
    """Write the committed label file: ids, months and labels only."""
    cached = load_cache(name)
    in_sample = dict(sample(per_month))
    val = set(validation_ids())
    missing = [i for i in [*in_sample, *val] if i not in cached]
    if missing:
        raise RuntimeError(f"{len(missing)} turns not labelled yet: run `run` and `validate`")
    with duckdb.connect() as con:
        con.execute("CREATE TEMP TABLE want (id VARCHAR)")
        con.executemany("INSERT INTO want VALUES (?)", [(i,) for i in val - set(in_sample)])
        months = dict(con.execute(
            f"SELECT id, strftime(created_at, '%Y-%m') FROM '{costs.path()}' JOIN want USING (id)"
        ).fetchall())  # fmt: skip
    months.update(in_sample)
    with wn.LLM_LABELS.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["turn_id", "month", "label", "monthly_sample"])
        for i in sorted(months, key=lambda i: (months[i], i)):
            w.writerow([i, months[i], cached[i]["label"] or "U", int(i in in_sample)])
    return wn.LLM_LABELS


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m aivillage.classify")
    ap.add_argument("command", choices=["validate", "run", "export"])
    ap.add_argument("--config", default="fast", choices=CONFIGS)
    ap.add_argument("--per-month", type=int, default=1000)
    ap.add_argument("--budget", type=float, default=10.0, help="cap on the key's total spend")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    if a.command == "export":
        print(export(a.config, a.per_month))
        return
    ids = validation_ids() if a.command == "validate" else [i for i, _ in sample(a.per_month)]
    label(texts(ids), a.config, a.budget, a.workers)


if __name__ == "__main__":
    main()
