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


def main() -> None:
    rows = wn.monthly("cost")
    const = {r["month"]: r["total_usd"] for r in wn.monthly("cost_constant")}
    upper = {r["month"]: r["total_usd"] for r in wn.monthly("cost_uncached")}
    months = [r["month"] for r in rows]
    width = 22  # days

    fig, axes = plt.subplots(
        4, 1, figsize=(10, 13), sharex=True, gridspec_kw={"height_ratios": [1, 1.6, 0.8, 0.8]}
    )
    fig.patch.set_facecolor(SURFACE)

    # A. Village GDP
    ax = axes[0]
    style(ax, "Village GDP: model spend per month (USD)")
    ax.fill_between(months, [r["total_usd"] for r in rows], [upper[m] for m in months],
                    color=BLUE, alpha=0.15, linewidth=0,
                    label="Upper bound: no caching on OpenAI-family calls")  # fmt: skip
    ax.plot(months, [r["total_usd"] for r in rows], color=BLUE, linewidth=2,
            label="Prices in force at the time")  # fmt: skip
    ax.plot(months, [const[m] for m in months], color=ORANGE, linewidth=2,
            label="Constant prices (2026-09-15)")  # fmt: skip
    ax.yaxis.set_major_formatter(lambda v, _: f"${v / 1000:.0f}k")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")

    # B. Transaction sector share, stacked
    ax = axes[1]
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
              loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2)  # fmt: skip

    # C, D. Unit cost of one exchange action
    ax = axes[2]
    style(ax, "Cost of one exchange action (USD, prices in force)")
    ax.plot(months, [r["usd_per_exchange"] for r in rows], color=BLUE, linewidth=2,
            marker="o", markersize=4)  # fmt: skip
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:.2f}")
    ax = axes[3]
    style(ax, "Tokens per exchange action (input + output)")
    ax.plot(months, [r["tokens_per_exchange"] for r in rows], color=BLUE, linewidth=2,
            marker="o", markersize=4)  # fmt: skip
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    for a in axes:
        a.text(REGIME, a.get_ylim()[1], " always-in-computer-use\n scaffolding (Mar 24)",
               fontsize=7, color=MUTED, va="top")  # fmt: skip

    fig.suptitle("AI Village: a Wallis-North transaction sector, sized in model spend",
                 x=0.06, ha="left", fontsize=14, color=INK)  # fmt: skip
    fig.text(0.06, 0.005,
             "Spend from data/derived/costs.parquet. Transaction work inside computer use is the "
             "hand-labelled transaction share of work spend (separately before/after Mar 24) "
             "times work spend.\nSource: AI Digest / AI Village dataset (aidigestorg/ai-village), "
             "export of 2026-09-20.",
             fontsize=7, color=MUTED)  # fmt: skip
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

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
