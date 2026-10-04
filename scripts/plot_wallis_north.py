"""Plot the village's Wallis-North transaction sector over time.

    uv run python scripts/plot_wallis_north.py

Writes data/figures/wallis_north.png and wallis_north.csv (the same numbers as
a table). Needs data/derived/costs.parquet (see aivillage.costs).
"""

import csv
from datetime import date

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from aivillage import hub
from aivillage import wallis_north as wn

# Reference palette, light mode (validated: dataviz skill, scripts/validate_palette.js)
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
TRANSFORMATION = "#d9d8d3"
REGIME = date(2026, 3, 24)


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axvline(REGIME, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def line(ax, months, ys, color=BLUE, label=None):
    ax.plot(months, ys, color=color, linewidth=2, marker="o", markersize=4, label=label)


def main() -> None:
    rows = wn.monthly("cost")
    const = {r["month"]: r["total_usd"] for r in wn.monthly("cost_constant")}
    upper = {r["month"]: r["total_usd"] for r in wn.monthly("cost_uncached")}
    months = [r["month"] for r in rows]
    width = 22  # days
    xlim = (date(2025, 3, 15), date(2026, 9, 20))

    fig = plt.figure(figsize=(14, 17))
    fig.patch.set_facecolor(SURFACE)
    grid = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1.5, 1], hspace=0.6, wspace=0.15)
    ax_gdp, ax_pop = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax_pc, ax_freq = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    ax_share = fig.add_subplot(grid[2, :])
    ax_usd, ax_tok = fig.add_subplot(grid[3, 0]), fig.add_subplot(grid[3, 1])

    # Village GDP
    style(ax_gdp, "Village GDP: model spend per month (USD)")
    ax_gdp.fill_between(months, [r["total_usd"] for r in rows], [upper[m] for m in months],
                        color=BLUE, alpha=0.15, linewidth=0,
                        label="Upper bound: no caching on OpenAI-family calls")  # fmt: skip
    ax_gdp.plot(months, [r["total_usd"] for r in rows], color=BLUE, linewidth=2,
                label="Prices in force at the time")  # fmt: skip
    ax_gdp.plot(months, [const[m] for m in months], color=ORANGE, linewidth=2,
                label="Constant prices (2026-09-15)")  # fmt: skip
    ax_gdp.yaxis.set_major_formatter(lambda v, _: f"${v / 1000:.0f}k")
    ax_gdp.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")

    # Population
    style(ax_pop, "Population: agents active per village day (monthly mean)")
    line(ax_pop, months, [r["agents_per_day"] for r in rows])
    ax_pop.axvline(date(2026, 6, 29), color=MUTED, linewidth=1, linestyle=(0, (1, 2)))
    ax_pop.text(date(2026, 6, 29), 2, " 4h → 8h days\n (Jun 29)", fontsize=7, color=MUTED)
    ax_pop.set_ylim(bottom=0)

    # Per capita spend
    style(ax_pc, "Spend per agent-hour (USD)")
    line(ax_pc, months, [r["usd_per_agent_hour"] for r in rows], BLUE, "All spend")
    line(ax_pc, months, [r["transaction_usd_per_agent_hour"] for r in rows], ORANGE,
         "Transaction sector")  # fmt: skip
    ax_pc.set_ylim(bottom=0)
    ax_pc.yaxis.set_major_formatter(lambda v, _: f"${v:.0f}")
    ax_pc.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="center right")

    # Exchange frequency
    style(ax_freq, "Exchange actions per agent-hour")
    line(ax_freq, months, [r["exchanges_per_agent_hour"] for r in rows])
    ax_freq.set_ylim(bottom=0)

    # Transaction sector share, stacked
    ax = ax_share
    style(ax, "Transaction sector as a share of village GDP")
    layers = [
        ("Exchange actions (chat, rooms, requests)", BLUE, None, lambda r: r["exchange_usd"]),
        ("Transaction work inside computer use (estimated)", ORANGE, "////",
         lambda r: r["work_transaction_usd"]),
        ("Waiting / pausing", AQUA, None, lambda r: r["waiting_usd"]),
        ("Memory consolidation", YELLOW, None, lambda r: r["memory_usd"]),
        ("Session start/stop decisions", MAGENTA, None, lambda r: r["session_usd"]),
        ("Transformation work", TRANSFORMATION, None,
         lambda r: r["work_usd"] - r["work_transaction_usd"]),
    ]  # fmt: skip
    bottom = [0.0] * len(rows)
    handles = []
    for label, color, hatch, f in layers:
        vals = [f(r) / r["total_usd"] for r in rows]
        ax.bar(months, vals, width=width, bottom=bottom, color=color, hatch=hatch,
               edgecolor=SURFACE, linewidth=1)  # fmt: skip
        bottom = [b + v for b, v in zip(bottom, vals, strict=True)]
        handles.append(Patch(facecolor=color, hatch=hatch, edgecolor=SURFACE, label=label))
    core = [r["transaction_share"] for r in rows]
    ax.errorbar(months, core,
                yerr=[[c - r["transaction_share_lo"] for c, r in zip(core, rows, strict=True)],
                      [r["transaction_share_hi"] - c for c, r in zip(core, rows, strict=True)]],
                fmt="o", markersize=4, color=INK, ecolor=INK, elinewidth=1, capsize=2)  # fmt: skip
    sector = "Transaction sector (exchange + transaction work), 95% CI"
    handles.insert(2, plt.Line2D([], [], color=INK, marker="o", markersize=4, linewidth=1,
                                 label=sector))  # fmt: skip
    for x, y in ((rows[5]["month"], core[5]), (rows[-1]["month"], core[-1])):
        ax.annotate(f"{y:.0%}", (x, y), xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color=INK)  # fmt: skip
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK_2,
              loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4)  # fmt: skip

    # Unit cost of one exchange action
    style(ax_usd, "Cost of one exchange action (USD, prices in force)")
    line(ax_usd, months, [r["usd_per_exchange"] for r in rows])
    ax_usd.set_ylim(bottom=0)
    ax_usd.yaxis.set_major_formatter(lambda v, _: f"${v:.2f}")
    style(ax_tok, "Tokens per exchange action (input + output)")
    line(ax_tok, months, [r["tokens_per_exchange"] for r in rows])
    ax_tok.set_ylim(bottom=0)
    ax_tok.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")

    for a in fig.axes:
        a.set_xlim(*xlim)
        a.text(REGIME, a.get_ylim()[1], " always-in-computer-use\n scaffolding (Mar 24)",
               fontsize=7, color=MUTED, va="top")  # fmt: skip

    fig.suptitle("AI Village: a Wallis-North transaction sector, sized in model spend",
                 x=0.06, y=0.985, ha="left", fontsize=15, color=INK)  # fmt: skip
    fig.text(0.06, 0.012,
             "Agent-hours: each agent's span from first to last model call per village (PT) day. "
             "Transaction work inside computer use = hand-labelled transaction share of work spend "
             "(separately before/after Mar 24) x work spend.\nSource: AI Digest / AI Village "
             "dataset (aidigestorg/ai-village), export of 2026-09-20. Spend from "
             "data/derived/costs.parquet.",
             fontsize=7, color=MUTED)  # fmt: skip
    fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.05)

    out_dir = hub.data_dir() / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "wallis_north.png", dpi=160, facecolor=SURFACE)
    keys = [k for k in rows[0] if k != "post"]
    with (out_dir / "wallis_north.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["month", "constant_price_total_usd",
                                          "no_cache_total_usd", *keys[1:]])  # fmt: skip
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r[k] for k in keys}, "constant_price_total_usd": const[r["month"]],
                        "no_cache_total_usd": upper[r["month"]]})  # fmt: skip
    print(out_dir / "wallis_north.png")


if __name__ == "__main__":
    main()
