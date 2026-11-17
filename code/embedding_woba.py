"""Can a batter's learned embedding predict his wOBA?

Reads a train.py run (batter_embeddings.csv + config.json), computes wOBA from
the same Parquet data, and fits a cross-validated ridge regression from the
embedding dimensions to wOBA.

Example (from the repository root):
    python code/embedding_woba.py --run runs/20261116_211400

Targets (one row of results each):
  season_<year>   full-season wOBA for every season in the data
  held_out        wOBA from --validation-start onward; these plate appearances
                  were never seen while training the embeddings (the honest test)

Each target notes how much of it overlaps the training window. If a season is
inside the training window, the embedding has already seen those plate
appearances, so a good fit there is descriptive, not predictive.

Feature sets compared:
  mean              predict the average wOBA (R^2 = 0 by construction)
  train_woba        the batter's own wOBA over the training window
  embedding         ridge regression on the embedding dimensions
  embedding+train   both together

The reliability column is the split-half reliability of the target itself
(Spearman-Brown corrected). It is roughly the highest R^2 any model could get,
because the rest of the variation in wOBA over that many PAs is luck.

Outputs go to <run>/woba/: report.md, metrics.csv, predictions.csv, scatter plots.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# FanGraphs Guts! linear weights (approximate; edit if you want exact values).
WOBA_WEIGHTS = {
    2023: {"walk": 0.696, "hit_by_pitch": 0.726, "single": 0.883, "double": 1.244,
           "triple": 1.569, "home_run": 2.004},
    2024: {"walk": 0.689, "hit_by_pitch": 0.720, "single": 0.882, "double": 1.254,
           "triple": 1.590, "home_run": 2.050},
}
# PA-ending events left out of the wOBA denominator.
NOT_IN_DENOMINATOR = {"intent_walk", "sac_bunt", "sac_bunt_double_play", "catcher_interf", "truncated_pa"}


def plate_appearances(raw):
    """One row per PA-ending pitch with its wOBA numerator value and denominator flag."""
    pa = raw.loc[raw.events.notna(), ["batter", "game_date", "events"]].copy()
    pa["game_date"] = pd.to_datetime(pa.game_date).dt.normalize()
    pa["season"] = pa.game_date.dt.year
    missing = sorted(set(pa.season) - set(WOBA_WEIGHTS))
    if missing:
        print(f"Warning: no wOBA weights for {missing}; using {max(WOBA_WEIGHTS)} weights")
    weights = pa.season.map(lambda s: WOBA_WEIGHTS.get(s, WOBA_WEIGHTS[max(WOBA_WEIGHTS)]))
    pa["woba_value"] = [w.get(e, 0.0) for w, e in zip(weights, pa.events)]
    pa["woba_denom"] = (~pa.events.isin(NOT_IN_DENOMINATOR)).astype(float)
    return pa.loc[pa.woba_denom.eq(1)]


def woba_by_batter(pa):
    grouped = pa.groupby("batter")
    return pd.DataFrame({"woba": grouped.woba_value.sum() / grouped.woba_denom.sum(),
                         "pa": grouped.woba_denom.sum()})


def split_half_reliability(pa, batters, seed):
    """Correlation of wOBA between random halves of each batter's PAs, Spearman-Brown corrected."""
    rng = np.random.default_rng(seed)
    pa = pa.loc[pa.batter.isin(batters)]
    half = rng.integers(0, 2, len(pa)).astype(bool)
    a, b = woba_by_batter(pa.loc[half]).woba, woba_by_batter(pa.loc[~half]).woba
    r = pd.concat([a, b], axis=1, join="inner").corr().iloc[0, 1]
    return float(2 * r / (1 + r))


def cross_validated(x, y, folds, repeats, seed, model_factory):
    """Out-of-fold predictions for each repeat (repeats x rows)."""
    out = np.zeros((repeats, len(y)))
    for repeat in range(repeats):
        for train, test in KFold(folds, shuffle=True, random_state=seed + repeat).split(x):
            model = model_factory().fit(x[train], y[train])
            out[repeat, test] = model.predict(x[test])
    return out


def scores(predictions, y):
    rows = []
    for p in predictions:
        rows.append({"r2": 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                     "rmse": float(np.sqrt(((y - p) ** 2).mean())),
                     "pearson_r": float(np.corrcoef(p, y)[0, 1]) if p.std() > 0 else 0.0})
    frame = pd.DataFrame(rows)
    return {"r2": frame.r2.mean(), "r2_sd": frame.r2.std(), "rmse": frame.rmse.mean(),
            "pearson_r": frame.pearson_r.mean()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="A train.py output folder")
    parser.add_argument("--data", type=Path, nargs="+", default=None,
                        help="Parquet file(s); default is the data listed in the run's config.json")
    parser.add_argument("--min-pa", type=int, default=300, help="Minimum PAs for a full-season target")
    parser.add_argument("--min-pa-held-out", type=int, default=100, help="Minimum PAs for the held-out target")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    data_paths = args.data or [Path(p) for p in config["data"]]
    validation_start = pd.Timestamp(config["validation_start"])
    output_dir = args.run / "woba"
    output_dir.mkdir(exist_ok=True)

    embeddings = pd.read_csv(args.run / "batter_embeddings.csv").set_index("mlbam_id")
    dims = [c for c in embeddings.columns if c.startswith("dim_")]
    raw = pd.concat([pd.read_parquet(p) for p in data_paths], ignore_index=True)
    if "game_type" in raw:
        raw = raw.loc[raw.game_type.eq("R")]
    pa = plate_appearances(raw)
    train_pa = pa.loc[pa.game_date < validation_start]
    train_woba = woba_by_batter(train_pa)

    targets = {}
    for season in sorted(pa.season.unique()):
        part = pa.loc[pa.season == season]
        overlap = (part.game_date < validation_start).mean()
        targets[f"season_{season}"] = (part, args.min_pa, overlap)
    targets["held_out"] = (pa.loc[pa.game_date >= validation_start], args.min_pa_held_out, 0.0)

    metrics, predictions = [], []
    ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
    linear = lambda: make_pipeline(StandardScaler(), LinearRegression())
    for name, (part, min_pa, overlap) in targets.items():
        target = woba_by_batter(part)
        target = target.loc[target.pa >= min_pa].join(embeddings, how="inner")
        target = target.join(train_woba.rename(columns={"woba": "train_woba", "pa": "train_pa"}), how="inner")
        if len(target) < 4 * args.folds:
            print(f"Skipping {name}: only {len(target)} batters")
            continue
        y = target.woba.to_numpy()
        reliability = split_half_reliability(part, target.index, args.seed)
        feature_sets = {
            "mean": (np.zeros((len(y), 1)), linear),
            "train_woba": (target[["train_woba"]].to_numpy(), linear),
            "embedding": (target[dims].to_numpy(), ridge),
            "embedding+train": (target[dims + ["train_woba"]].to_numpy(), ridge),
        }
        for feature_name, (x, factory) in feature_sets.items():
            oof = cross_validated(x, y, args.folds, args.repeats, args.seed, factory)
            metrics.append({"target": name, "features": feature_name, "batters": len(y),
                            "median_pa": float(target.pa.median()),
                            "share_in_training_window": round(float(overlap), 3),
                            "reliability": reliability, **scores(oof, y)})
            predictions.append(pd.DataFrame({
                "target": name, "features": feature_name, "mlbam_id": target.index,
                "pa": target.pa.to_numpy(), "actual_woba": y, "predicted_woba": oof.mean(axis=0)}))
        print(f"{name}: {len(y)} batters, reliability {reliability:.2f}")

    metrics = pd.DataFrame(metrics)
    predictions = pd.concat(predictions, ignore_index=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(metrics.target.unique())
        fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.6), squeeze=False)
        for ax, name in zip(axes[0], names):
            for features, color in (("train_woba", "tab:gray"), ("embedding", "tab:blue")):
                part = predictions.loc[(predictions.target == name) & (predictions.features == features)]
                r2 = metrics.set_index(["target", "features"]).loc[(name, features), "r2"]
                ax.scatter(part.predicted_woba, part.actual_woba, s=8, alpha=0.5, color=color,
                           label=f"{features} (R² {r2:.2f})")
            lo, hi = part.actual_woba.min(), part.actual_woba.max()
            ax.plot([lo, hi], [lo, hi], color="black", lw=0.8)
            ax.set_title(name), ax.set_xlabel("predicted wOBA (out of fold)"), ax.set_ylabel("actual wOBA")
            ax.legend(fontsize=8)
        fig.tight_layout(), fig.savefig(output_dir / "woba_scatter.png", dpi=120), plt.close(fig)
    except ImportError:
        print("matplotlib not installed; skipping plot")

    table = metrics.copy()
    for column in ("r2", "r2_sd", "rmse", "pearson_r", "reliability"):
        table[column] = table[column].map(lambda v: f"{v:.3f}")
    header = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in table.itertuples(index=False)]
    report = [
        "# Batter embeddings vs wOBA", "",
        f"Run: {args.run}. Embeddings trained on pitches before {validation_start.date()}.",
        f"{args.folds}-fold cross-validation repeated {args.repeats} times; R² and RMSE are out of fold "
        "(r2_sd is the spread across repeats).", "",
        "## How to read this", "",
        "- **held_out** is the real test: those plate appearances were never used to train the embeddings.",
        "- Targets with `share_in_training_window` above 0 overlap the training data, so the embedding "
        "has partly seen the answer.",
        "- **embedding vs train_woba**: does the embedding know more about a hitter than his own wOBA "
        "over the same pitches? **embedding+train** above train_woba means it adds something.",
        "- **reliability** is about the most R² any model could reach, because wOBA over that many PAs is noisy.", "",
        "\n".join([header, rule, *body]), "",
        "Files: metrics.csv, predictions.csv (one row per batter, target and feature set), woba_scatter.png.",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n" + metrics.loc[:, ["target", "features", "batters", "share_in_training_window",
                                 "reliability", "r2", "rmse", "pearson_r"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nReport: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
