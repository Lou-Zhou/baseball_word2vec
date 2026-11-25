"""Report figure: one batter and his nearest neighbors in the embedding space.

    python src/neighbor_figure.py --run runs/2024 --batter "Arraez"

Left: the k most similar batters (cosine similarity of centered embeddings).
Right: strip plots of a few stats, with every batter with >= --min-pa PA in gray,
the neighbors in blue and the searched batter in orange, to show that the
neighbors share his place in each distribution.

Writes <run>/figures/<name>_neighbors.png (and .csv with the plotted numbers).
"""

import argparse
import re
from pathlib import Path

import numpy as np

from batter_neighbors import build_table, neighbors

# Validated reference palette (categorical slots 1 and 2, light surface) plus neutral inks.
QUERY, NEIGHBOR = "#eb6834", "#2a78d6"
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#c9c8c3", "#e8e7e3"

STATS = {
    "k_pct": ("Strikeout rate", lambda v: f"{100 * v:.0f}%", 100),
    "whiff_pct": ("Whiff rate (per swing)", lambda v: f"{100 * v:.0f}%", 100),
    "chase_pct": ("Chase rate", lambda v: f"{100 * v:.0f}%", 100),
    "bb_pct": ("Walk rate", lambda v: f"{100 * v:.0f}%", 100),
    "ev": ("Average exit velocity (mph)", lambda v: f"{v:.0f}", 1),
    "la": ("Average launch angle (deg)", lambda v: f"{v:.0f}", 1),
    "hard_pct": ("Hard-hit rate (95+ mph)", lambda v: f"{100 * v:.0f}%", 100),
    "woba": ("wOBA", lambda v: f"{v:.3f}".lstrip("0"), 1),
    "xwoba": ("xwOBA", lambda v: f"{v:.3f}".lstrip("0"), 1),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--batter", required=True, help="Text in the batter's name, e.g. Arraez")
    parser.add_argument("--k", type=int, default=8, help="Neighbors to show")
    parser.add_argument("--min-pitches", type=int, default=500, help="Training pitches a neighbor needs")
    parser.add_argument("--min-pa", type=int, default=300, help="PA for the gray comparison batters")
    parser.add_argument("--stats", nargs="+", default=["k_pct", "whiff_pct", "ev", "xwoba"], choices=list(STATS))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table, unit, season = build_table(args.run)
    hits = np.flatnonzero(table.name.str.contains(args.batter, case=False).to_numpy())
    if len(hits) != 1:
        parser.error(f"{args.batter!r} matches {len(hits)} batters: {list(table.name.iloc[hits])}")
    row = hits[0]
    query = table.iloc[row]
    near = neighbors(table, unit, row, args.k, args.min_pitches)
    league = table.loc[table.pa >= args.min_pa]
    last_name = query["name"].split()[-1]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2})
    fig = plt.figure(figsize=(13, 1.0 + 1.25 * len(args.stats)), facecolor=SURFACE)
    grid = fig.add_gridspec(len(args.stats), 2, width_ratios=[1, 1.35], wspace=0.32, hspace=0.9,
                            left=0.13, right=0.97, top=0.80, bottom=0.12)

    # ---- left: similarity ranking ----
    ax = fig.add_subplot(grid[:, 0], facecolor=SURFACE)
    y = np.arange(len(near))[::-1]
    ax.barh(y, near.similarity, height=0.55, color=NEIGHBOR, edgecolor=SURFACE, linewidth=2)
    for yi, s in zip(y, near.similarity):
        ax.text(s + 0.012, yi, f"{s:.2f}", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y, near.name, fontsize=10, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Cosine similarity to " + last_name, fontsize=9)
    ax.set_title(f"{args.k} most similar batters", loc="left", fontsize=11, color=INK, pad=10)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    # ---- right: where they sit in each distribution ----
    rng = np.random.default_rng(0)
    for i, stat in enumerate(args.stats):
        label, fmt, _ = STATS[stat]
        ax = fig.add_subplot(grid[i, 1], facecolor=SURFACE)
        values = league[stat].dropna()
        ax.scatter(values, rng.uniform(-0.3, 0.3, len(values)), s=14, color=MUTED, linewidth=0, zorder=1)
        # Line spans the dot band only, so it never crosses the batter's label below it.
        ax.axvline(values.median(), ymin=0.33, ymax=0.85, color=INK_2, linewidth=1, linestyle=(0, (3, 3)), zorder=2)
        ax.text(values.median(), 0.62, f"median {fmt(values.median())}", ha="center", fontsize=8, color=INK_2)
        ax.scatter(near[stat], rng.uniform(-0.12, 0.12, len(near)), s=55, color=NEIGHBOR,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.scatter([query[stat]], [0], s=120, color=QUERY, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(f"{last_name} {fmt(query[stat])}", (query[stat], 0), xytext=(0, -20),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
        ax.set_ylim(-0.75, 0.85)
        ax.set_yticks([])
        ax.set_title(label, loc="left", fontsize=10, color=INK, pad=3)
        ax.tick_params(axis="x", labelsize=8.5, length=0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _, f=fmt: f(v)))
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=QUERY, label=query["name"]),
               plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=NEIGHBOR,
                          label=f"{args.k} nearest neighbors"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=5, color=MUTED,
                          label=f"All batters, {args.min_pa}+ PA")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.93), ncol=3, frameon=False,
               fontsize=9, handletextpad=0.3, columnspacing=1.4)
    fig.text(0.02, 0.95, f"{query['name']}'s nearest neighbors in the {season} batter embedding space",
             fontsize=14, fontweight="bold", color=INK)
    fig.text(0.02, 0.905, "Neighbors are found from the 16-dimensional embedding alone; "
             "the stats on the right were not model inputs.", fontsize=9.5, color=INK_2)
    fig.text(0.02, 0.025, f"Three-head pitch model trained on {season} Statcast data. Cosine similarity of "
             f"embeddings centered on the average batter; neighbors need {args.min_pitches}+ training pitches.",
             fontsize=8, color=INK_2)

    slug = re.sub(r"[^a-z0-9]+", "_", query["name"].lower()).strip("_")
    output = args.output or args.run / "figures" / f"{slug}_neighbors.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    columns = ["name", "similarity", "train_pitches", *args.stats]
    near.assign(similarity=near.similarity).loc[:, columns].to_csv(output.with_suffix(".csv"), index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
