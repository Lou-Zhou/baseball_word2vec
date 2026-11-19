"""How well do batter embeddings encode season wOBA and xwOBA? (model-free check)

    python src/embedding_woba.py --runs runs/2024
    python src/embedding_woba.py --runs runs/2024 runs/2024_other   # compare runs on the same batters

Fits a small PyTorch MLP (WobaMLP; edit it to change the architecture) from a
run's batter embedding to each batter's Baseball Savant season wOBA and xwOBA,
scoring every prediction out of fold (repeated k-fold over batters). Unlike the
readouts, this does not use the trained heads at all: it asks whether the
information is in the vectors.

Rows for target wOBA:  mean, xwoba (the batter's own xwOBA), embedding:<run>,
                       embedding+xwoba:<run> (does the embedding add anything?)
Rows for target xwOBA: mean, embedding:<run>

Outputs go to --output-dir (default <run>/embedding_woba for one run,
runs/embedding_woba_<timestamp> for several): report.md, metrics.csv,
predictions.csv, woba_scatter.png.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from metrics import markdown_table
from paths import RUNS_DIR, run_seasons, savant_woba

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}


class WobaMLP(nn.Module):
    """Embedding (+ optional extra inputs) -> hidden layers -> one number (standardized target)."""

    def __init__(self, in_features, hidden=(32,), dropout=0.0, activation="relu"):
        super().__init__()
        layers, width = [], in_features
        for size in hidden:
            layers += [nn.Linear(width, size), ACTIVATIONS[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = size
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def fit_predict(x_train, y_train, x_test, settings, seed):
    """Train a WobaMLP on one fold (early stopping on an inner split) and predict the held-out batters."""
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0) + 1e-8
    y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
    x = torch.tensor((x_train - x_mean) / x_std, dtype=torch.float32)
    y = torch.tensor((y_train - y_mean) / y_std, dtype=torch.float32)
    x_out = torch.tensor((x_test - x_mean) / x_std, dtype=torch.float32)

    order = torch.randperm(len(y), generator=generator)
    n_valid = max(1, int(len(y) * settings.inner_valid))
    valid, train = order[:n_valid], order[n_valid:]

    model = WobaMLP(x.shape[1], settings.hidden, settings.dropout, settings.activation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    best_loss, best_state, stale = float("inf"), None, 0
    for _ in range(settings.max_epochs):
        model.train()
        for batch in torch.randperm(len(train), generator=generator).split(settings.batch_size):
            index = train[batch]
            loss = nn.functional.mse_loss(model(x[index]), y[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = nn.functional.mse_loss(model(x[valid]), y[valid]).item()
        if valid_loss < best_loss - 1e-5:
            best_loss, stale = valid_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= settings.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(x_out).numpy() * y_std + y_mean


def kfold_indices(n, folds, seed):
    fold = np.random.default_rng(seed).permutation(n) % folds
    return [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(folds)]


def cross_validated(x, y, settings):
    """Out-of-fold WobaMLP predictions for each repeat (repeats x rows)."""
    out = np.zeros((settings.repeats, len(y)))
    for repeat in range(settings.repeats):
        for fold, (train, test) in enumerate(kfold_indices(len(y), settings.folds, settings.seed + repeat)):
            out[repeat, test] = fit_predict(x[train], y[train], x[test], settings,
                                            seed=settings.seed + 1000 * repeat + fold)
    return out


def summarize(predictions, actual, pa):
    """Error metrics averaged over repeats; predictions is (repeats x rows)."""
    rows = []
    for p in np.atleast_2d(predictions):
        miss = p - actual
        rows.append({"rmse": np.sqrt((miss ** 2).mean()), "rmse_pa_weighted": np.sqrt(np.average(miss ** 2, weights=pa)),
                     "mae": np.abs(miss).mean(), "bias": miss.mean(),
                     "pearson_r": np.corrcoef(p, actual)[0, 1] if p.std() > 0 else np.nan})
    frame = pd.DataFrame(rows)
    return {**frame.mean().astype(float).to_dict(), "rmse_sd": float(frame.rmse.std(ddof=0))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="train.py output folder(s)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-pa", type=int, default=300, help="Minimum plate appearances in the season")
    network = parser.add_argument_group("network (WobaMLP)")
    network.add_argument("--hidden", type=int, nargs="*", default=[32],
                         help="Hidden layer sizes; pass no values for a linear model")
    network.add_argument("--activation", choices=sorted(ACTIVATIONS), default="relu")
    # Defaults chosen on 2024 (~290 batters): weaker regularization overfits.
    network.add_argument("--dropout", type=float, default=0.3)
    network.add_argument("--weight-decay", type=float, default=1.0)
    network.add_argument("--lr", type=float, default=1e-3)
    network.add_argument("--batch-size", type=int, default=32)
    network.add_argument("--max-epochs", type=int, default=500)
    network.add_argument("--patience", type=int, default=30)
    network.add_argument("--inner-valid", type=float, default=0.2,
                         help="Share of each training fold held out for early stopping")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    runs = {run.resolve().name: run for run in args.runs}
    if len(runs) != len(args.runs):
        parser.error("Run folder names must be different")
    output_dir = args.output_dir or (args.runs[0] / "embedding_woba" if len(runs) == 1
                                     else RUNS_DIR / f"embedding_woba_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seasons = sorted(set.intersection(*(set(run_seasons(r)) for r in runs.values())))
    if len(seasons) != 1:
        parser.error(f"Runs must share exactly one season (found {seasons})")
    season = seasons[0]

    embeddings = {}
    for name, run in runs.items():
        table = pd.read_csv(run / "batter_embeddings.csv").set_index("mlbam_id")
        embeddings[name] = table.loc[:, [c for c in table.columns if c.startswith("dim_")]]
    args.hidden = tuple(args.hidden)
    torch.set_num_threads(1)  # tiny networks: threading overhead outweighs the work

    batters = savant_woba(season)
    batters = batters.loc[(batters.pa >= args.min_pa) & batters.xwoba.notna()]
    for table in embeddings.values():
        batters = batters.loc[batters.index.isin(table.index)]
    pa, xwoba = batters.pa.to_numpy(), batters.xwoba.to_numpy()
    print(f"{season}: {len(batters)} batters with >= {args.min_pa} PA and an embedding in every run")

    metrics, predictions = [], []
    for target in ("woba", "xwoba"):
        y = batters[target].to_numpy()
        mean = np.zeros(len(y))
        for train, test in kfold_indices(len(y), args.folds, args.seed):
            mean[test] = y[train].mean()
        predicted = {"mean": mean[None, :]}
        if target == "woba":
            predicted["xwoba"] = xwoba[None, :]
        for name, table in embeddings.items():
            dims = table.reindex(batters.index).to_numpy()
            print(f"  fitting {target} <- embedding:{name}")
            predicted[f"embedding:{name}"] = cross_validated(dims, y, args)
            if target == "woba":
                predicted[f"embedding+xwoba:{name}"] = cross_validated(np.column_stack([dims, xwoba]), y, args)
        for model, values in predicted.items():
            metrics.append({"target": target, "model": model, "batters": len(y), **summarize(values, y, pa)})
            predictions.append(pd.DataFrame({"target": target, "model": model, "mlbam_id": batters.index,
                                             "pa": pa, "actual": y, "predicted": values.mean(axis=0)}))

    metrics = pd.DataFrame(metrics).sort_values(["target", "rmse"])
    predictions = pd.concat(predictions, ignore_index=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        panels = list(predictions.loc[predictions.model != "mean", ["target", "model"]]
                      .drop_duplicates().itertuples(index=False))
        columns = min(4, len(panels))
        rows = int(np.ceil(len(panels) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(3.8 * columns, 3.8 * rows), squeeze=False)
        indexed = metrics.set_index(["target", "model"])
        for ax, (target, model) in zip(axes.flat, panels):
            part = predictions.loc[(predictions.target == target) & (predictions.model == model)]
            ax.scatter(part.predicted, part.actual, s=6, alpha=0.5)
            lo, hi = part.actual.min(), part.actual.max()
            ax.plot([lo, hi], [lo, hi], color="black", lw=0.8)
            ax.set_title(f"{target} <- {model}\nRMSE {indexed.loc[(target, model), 'rmse']:.4f}", fontsize=8)
            ax.set_xlabel("predicted", fontsize=8), ax.set_ylabel(f"actual {target}", fontsize=8)
        for ax in list(axes.flat)[len(panels):]:
            ax.axis("off")
        fig.tight_layout(), fig.savefig(output_dir / "woba_scatter.png", dpi=110), plt.close(fig)
    except ImportError:
        print("matplotlib not installed; skipping plot")

    shown = metrics.loc[:, ["target", "model", "batters", "rmse", "rmse_sd", "rmse_pa_weighted", "bias", "pearson_r"]]
    report = [
        f"# Batter embeddings vs {season} season wOBA and xwOBA", "",
        "Runs: " + ", ".join(f"{name} ({run})" for name, run in runs.items()) + ".",
        f"Targets: Baseball Savant season wOBA and xwOBA for batters with >= {args.min_pa} PA and an "
        "embedding in every run.",
        f"WobaMLP: hidden {args.hidden}, {args.activation}, dropout {args.dropout}, weight decay "
        f"{args.weight_decay}; predictions out of fold ({args.folds}-fold CV x {args.repeats} repeats; "
        "rmse_sd is the spread across repeats).", "",
        "## How to read this", "",
        "- **rmse** is the typical miss (0.030 = 30 points of wOBA); lower is better.",
        "- **woba <- xwoba** is the gap left by batted-ball luck; an embedding near that RMSE encodes about "
        "as much as xwOBA does.",
        "- **embedding+xwoba** below **xwoba** means the embedding knows something xwOBA does not.", "",
        markdown_table(shown), "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n" + shown.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nReport: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
