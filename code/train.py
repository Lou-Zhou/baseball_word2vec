"""Train the pitch outcome model and write an evaluation report.

Example (from the repository root):
    python code/train.py --data data/pitches_2023_2024.parquet \
        --validation-start 2024-08-01 --test-start 2024-09-01

Models compared on the validation and test splits:
  nn_pitch     the neural network without player vectors
  nn_full      the neural network with batter and pitcher vectors (the model)

Everything is written to --output-dir: report.md (start here), metrics.csv,
calibration and confusion tables, plots, player embeddings and model.pt.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from pitch_model import (
    CONTINUOUS_FEATURES, OUTCOMES, PITCH_INPUTS, PitchOutcomeModel, PitchPreprocessor,
    chronological_split, prepare_pitches,
)

SWING = ["swinging_strike", "foul", "out_in_play", "single", "double", "triple", "home_run"]
IN_PLAY = ["out_in_play", "single", "double", "triple", "home_run"]
HITS = ["single", "double", "triple", "home_run"]


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def to_device(tensors, device):
    return [t.to(device) for t in tensors]


@torch.inference_mode()
def predict_proba(model, features, batch_size=16384):
    model.eval()
    n = len(features[0])
    out = [
        model(*(f[start:start + batch_size] for f in features)).softmax(-1).cpu()
        for start in range(0, n, batch_size)
    ]
    return torch.cat(out).numpy().astype("float64")


@torch.inference_mode()
def mean_loss(model, features, target, batch_size=16384):
    model.eval()
    total = 0.0
    for start in range(0, len(target), batch_size):
        logits = model(*(f[start:start + batch_size] for f in features))
        total += F.cross_entropy(logits, target[start:start + batch_size], reduction="sum").item()
    return total / len(target)


def train_network(name, model, train, valid, args, device):
    """AdamW with early stopping on validation loss; restores the best epoch."""
    (*train_x, train_y), (*valid_x, valid_y) = train, valid
    embedding_params = [p for n, p in model.named_parameters() if "embedding" in n]
    other_params = [p for n, p in model.named_parameters() if "embedding" not in n]
    optimizer = torch.optim.AdamW([
        {"params": other_params, "weight_decay": args.weight_decay},
        {"params": embedding_params, "weight_decay": args.embedding_weight_decay},
    ], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    history, best_loss, best_state, stale = [], float("inf"), None, 0
    n = len(train_y)
    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time, running = time.time(), 0.0
        order = torch.randperm(n, device=device)
        for start in range(0, n, args.batch_size):
            index = order[start:start + args.batch_size]
            loss = F.cross_entropy(model(*(f[index] for f in train_x)), train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(index)
        train_loss, valid_loss = running / n, mean_loss(model, valid_x, valid_y)
        scheduler.step(valid_loss)
        history.append({"model": name, "epoch": epoch, "train_loss": train_loss,
                        "valid_loss": valid_loss, "lr": optimizer.param_groups[0]["lr"],
                        "seconds": time.time() - start_time})
        print(f"  [{name}] epoch {epoch:3d}  train {train_loss:.4f}  valid {valid_loss:.4f}"
              f"  ({history[-1]['seconds']:.0f}s)")
        if valid_loss < best_loss - 1e-4:
            best_loss, stale = valid_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"  [{name}] early stop; best valid {best_loss:.4f}")
                break
    model.load_state_dict(best_state)
    return history


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def log_loss(proba, y):
    return float(-np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1)).mean())


def brier(proba, y):
    onehot = np.eye(len(OUTCOMES))[y]
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def expected_calibration_error(proba, y, bins=15):
    confidence, predicted = proba.max(axis=1), proba.argmax(axis=1)
    correct = predicted == y
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(confidence, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(ece)


def group_proba(proba, names):
    return proba[:, [OUTCOMES.index(n) for n in names]].sum(axis=1)


def summary_metrics(proba, y):
    onehot = np.eye(len(OUTCOMES))[y]
    swing_p, swing_y = group_proba(proba, SWING), group_proba(onehot, SWING)
    contact = ["foul"] + IN_PLAY
    # Conditional probabilities predicted for each pitch, compared with what happened.
    whiff_mask = swing_y == 1
    take_mask = ~whiff_mask & (y != OUTCOMES.index("hit_by_pitch"))
    inplay_mask = group_proba(onehot, IN_PLAY) == 1
    pred_whiff = proba[:, OUTCOMES.index("swinging_strike")] / np.clip(swing_p, 1e-9, None)
    pred_called = proba[:, OUTCOMES.index("called_strike")] / np.clip(
        proba[:, [OUTCOMES.index("ball"), OUTCOMES.index("called_strike")]].sum(axis=1), 1e-9, None)
    pred_hit = group_proba(proba, HITS) / np.clip(group_proba(proba, IN_PLAY), 1e-9, None)

    def binary_ll(p, actual):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-(actual * np.log(p) + (1 - actual) * np.log(1 - p)).mean())

    return {
        "rows": int(len(y)),
        "log_loss": log_loss(proba, y),
        "brier": brier(proba, y),
        "accuracy": float((proba.argmax(axis=1) == y).mean()),
        "top_label_ece": expected_calibration_error(proba, y),
        "swing_log_loss": binary_ll(swing_p, swing_y),
        "whiff_given_swing_log_loss": binary_ll(
            pred_whiff[whiff_mask], (y[whiff_mask] == OUTCOMES.index("swinging_strike")).astype(float)),
        "called_strike_given_take_log_loss": binary_ll(
            pred_called[take_mask], (y[take_mask] == OUTCOMES.index("called_strike")).astype(float)),
        "hit_given_in_play_log_loss": binary_ll(
            pred_hit[inplay_mask], np.isin(y[inplay_mask], [OUTCOMES.index(h) for h in HITS]).astype(float)),
        "contact_classes": contact,
    }


def calibration_table(proba, y, model, split, bins=10):
    rows = []
    for k, name in enumerate(OUTCOMES):
        p, actual = proba[:, k], (y == k).astype(float)
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
        which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            mask = which == b
            if mask.any():
                rows.append({"model": model, "split": split, "outcome": name, "bin": b,
                             "n": int(mask.sum()), "mean_predicted": float(p[mask].mean()),
                             "observed_rate": float(actual[mask].mean())})
    return rows


def rate_table(proba, y, model, split):
    onehot = np.eye(len(OUTCOMES))[y]
    return [{"model": model, "split": split, "outcome": name,
             "predicted_rate": float(proba[:, k].mean()), "actual_rate": float(onehot[:, k].mean())}
            for k, name in enumerate(OUTCOMES)]


def per_row_loss(proba, y):
    return -np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1))


# --------------------------------------------------------------------------
# Plots and report
# --------------------------------------------------------------------------

def make_plots(history, calibration, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return
    history = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, part in history.groupby("model"):
        ax.plot(part.epoch, part.train_loss, "--", label=f"{name} train")
        ax.plot(part.epoch, part.valid_loss, "-", label=f"{name} valid")
    ax.set_xlabel("epoch"), ax.set_ylabel("cross-entropy"), ax.legend(), ax.set_title("Training curves")
    fig.tight_layout(), fig.savefig(output_dir / "loss_curves.png", dpi=120), plt.close(fig)

    calibration = pd.DataFrame(calibration)
    test = calibration.loc[calibration.split == "test"]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
    for ax, name in zip(axes.flat, OUTCOMES):
        part = test.loc[test.outcome == name]
        top = max(part.mean_predicted.max(), part.observed_rate.max()) * 1.05
        ax.plot([0, top], [0, top], color="gray", lw=0.8)
        for model, rows in part.groupby("model"):
            ax.plot(rows.mean_predicted, rows.observed_rate, "o-", ms=3, label=model)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("predicted", fontsize=8), ax.set_ylabel("observed", fontsize=8)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Test-set calibration by outcome (on the diagonal = calibrated)")
    fig.tight_layout(), fig.savefig(output_dir / "calibration.png", dpi=120), plt.close(fig)


def markdown_table(frame, floatfmt=".4f"):
    frame = frame.copy()
    for column in frame.select_dtypes("number").columns:
        if column not in ("rows", "n", "pitches"):
            frame[column] = frame[column].map(lambda v: format(v, floatfmt))
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True, help="Parquet file(s) from load_data.py")
    parser.add_argument("--validation-start", required=True, help="First date of the validation split")
    parser.add_argument("--test-start", required=True, help="First date of the test split")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: runs/<timestamp>")
    parser.add_argument("--min-pitches", type=int, default=100,
                        help="Players with fewer training pitches share the unknown-player vector")
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir or Path(__file__).resolve().parents[1] / "runs" / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to {output_dir}")

    # ---- data ----
    raw = pd.concat([pd.read_parquet(path) for path in args.data], ignore_index=True)
    pitches, clean_report = prepare_pitches(raw)
    print(f"Kept {clean_report['retained_rows']:,} of {clean_report['input_rows']:,} rows")
    train, valid, test = chronological_split(pitches, args.validation_start, args.test_start)
    splits = {"train": train, "valid": valid, "test": test}
    for name, part in splits.items():
        print(f"  {name:5s} {len(part):>9,} pitches  {part.game_date.min().date()} to {part.game_date.max().date()}")

    preprocessor = PitchPreprocessor.fit(train, min_pitches=args.min_pitches)
    num_batters = len(preprocessor.player_maps["batter"])
    num_pitchers = len(preprocessor.player_maps["pitcher"])
    print(f"Player vectors: {num_batters} batters, {num_pitchers} pitchers (>= {args.min_pitches} train pitches)")
    encoded = {
        name: to_device([*preprocessor.encode(part), torch.from_numpy(part.target.to_numpy())], device)
        for name, part in splits.items()
    }

    # ---- models ----
    probabilities, history = {}, []
    configs = {}
    for name, use_players in (("nn_pitch", False), ("nn_full", True)):
        print(f"Training {name}...")
        torch.manual_seed(args.seed)
        configs[name] = dict(
            num_batters=num_batters, num_pitchers=num_pitchers, num_pitch_features=len(PITCH_INPUTS),
            embedding_dim=args.embedding_dim, hidden_sizes=tuple(args.hidden), dropout=args.dropout,
            use_players=use_players,
        )
        model = PitchOutcomeModel(**configs[name]).to(device)
        history += train_network(name, model, encoded["train"], encoded["valid"], args, device)
        probabilities[name] = {s: predict_proba(model, encoded[s][:-1]) for s in ("valid", "test")}
        if use_players:
            full_model = model

    torch.save({
        "model_config": configs["nn_full"],
        "model_state": {k: v.cpu() for k, v in full_model.state_dict().items()},
        "preprocessor": preprocessor.state_dict(),
        "outcomes": list(OUTCOMES),
    }, output_dir / "model.pt")

    # ---- evaluation ----
    metrics, calibration, rates = [], [], []
    for model_name, by_split in probabilities.items():
        for split, proba in by_split.items():
            y = splits[split].target.to_numpy()
            row = summary_metrics(proba, y)
            row.pop("contact_classes")
            metrics.append({"model": model_name, "split": split, **row})
            calibration += calibration_table(proba, y, model_name, split)
            rates += rate_table(proba, y, model_name, split)
    metrics = pd.DataFrame(metrics)
    pitch_ll = metrics.set_index(["split", "model"]).log_loss.xs("nn_pitch", level="model")
    metrics["log_loss_gain_vs_nn_pitch"] = metrics.split.map(pitch_ll) - metrics.log_loss
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(calibration).to_csv(output_dir / "calibration.csv", index=False)
    pd.DataFrame(rates).to_csv(output_dir / "outcome_rates.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    y_test = test.target.to_numpy()
    full_test = probabilities["nn_full"]["test"]
    confusion = pd.crosstab(
        pd.Categorical(np.array(OUTCOMES)[y_test], categories=OUTCOMES, ordered=True),
        pd.Categorical(np.array(OUTCOMES)[full_test.argmax(axis=1)], categories=OUTCOMES, ordered=True),
        rownames=["actual"], colnames=["predicted"], dropna=False,
    )
    confusion.to_csv(output_dir / "confusion_matrix_test.csv")

    # Where do player vectors help? Split test log loss by count and by player coverage.
    losses = pd.DataFrame({name: per_row_loss(p["test"], y_test) for name, p in probabilities.items()})
    losses["count"] = (test.balls.astype(str) + "-" + test.strikes.astype(str)).to_numpy()
    batter_known = test.batter.isin(preprocessor.player_maps["batter"]).to_numpy()
    pitcher_known = test.pitcher.isin(preprocessor.player_maps["pitcher"]).to_numpy()
    losses["players_with_vectors"] = np.select(
        [batter_known & pitcher_known, batter_known, pitcher_known],
        ["both", "batter only", "pitcher only"], "neither")
    by_count = losses.groupby("count").agg(pitches=("nn_full", "size"), **{
        m: (m, "mean") for m in probabilities}).reset_index()
    by_coverage = losses.groupby("players_with_vectors").agg(pitches=("nn_full", "size"), **{
        m: (m, "mean") for m in probabilities}).reset_index()
    by_count.to_csv(output_dir / "test_log_loss_by_count.csv", index=False)
    by_coverage.to_csv(output_dir / "test_log_loss_by_player_coverage.csv", index=False)

    # Player vectors with MLBAM IDs and training pitch counts, for later analysis.
    for role, embedding in (("batter", full_model.batter_embedding), ("pitcher", full_model.pitcher_embedding)):
        ids = pd.Series(preprocessor.player_maps[role]).sort_values()
        vectors = embedding.weight.detach().cpu().numpy()[ids.to_numpy()]
        table = pd.DataFrame(vectors, columns=[f"dim_{i}" for i in range(vectors.shape[1])])
        table.insert(0, "train_pitches", train[role].value_counts().reindex(ids.index).to_numpy())
        table.insert(0, "mlbam_id", ids.index)
        table.to_csv(output_dir / f"{role}_embeddings.csv", index=False)

    make_plots(history, calibration, output_dir)

    # ---- report ----
    headline = metrics.loc[:, ["model", "split", "rows", "log_loss", "log_loss_gain_vs_nn_pitch",
                               "brier", "accuracy", "top_label_ece"]]
    parts = metrics.loc[:, ["model", "split", "swing_log_loss", "whiff_given_swing_log_loss",
                            "called_strike_given_take_log_loss", "hit_given_in_play_log_loss"]]
    rate_frame = pd.DataFrame(rates)
    rate_frame = rate_frame.loc[rate_frame.split == "test"].pivot(index="outcome", columns="model", values="predicted_rate")
    rate_frame.insert(0, "actual", pd.DataFrame(rates).query("split == 'test' and model == 'nn_full'")
                      .set_index("outcome").actual_rate)
    rate_frame = rate_frame.loc[list(OUTCOMES)].reset_index()
    report = [
        "# Pitch outcome model report", "",
        f"Data: {', '.join(str(p) for p in args.data)}",
        f"Train {len(train):,} / valid {len(valid):,} / test {len(test):,} pitches; "
        f"validation from {args.validation_start}, test from {args.test_start}.",
        f"Player vectors for {num_batters} batters and {num_pitchers} pitchers "
        f"(at least {args.min_pitches} training pitches); everyone else uses the shared unknown vector.",
        f"Pitch inputs ({len(PITCH_INPUTS)}): {', '.join(PITCH_INPUTS)}.", "",
        "## How to read this", "",
        "- **log_loss** is the main score (lower is better).",
        "- **nn_full vs nn_pitch** is what the batter and pitcher vectors add beyond the pitch itself; "
        "`log_loss_gain_vs_nn_pitch` is that difference (positive = player vectors help).",
        "- **top_label_ece** near 0 means confident predictions are right about as often as claimed. "
        "`calibration.png` shows calibration for every outcome.",
        "- Accuracy is mostly 'predict ball' and says little; use it only as a sanity check.", "",
        "## Overall", "", markdown_table(headline), "",
        "## Decision-level log loss", "",
        "Each column scores one baseball decision on the pitches where it applies "
        "(e.g. whiff given swing only on swings).", "", markdown_table(parts), "",
        "## Predicted vs actual outcome rates (test)", "", markdown_table(rate_frame), "",
        "## Test log loss by player coverage", "",
        "If player vectors help, nn_full should beat nn_pitch most where both players have vectors.", "",
        markdown_table(by_coverage), "",
        "## Test log loss by count", "", markdown_table(by_count), "",
        "## Confusion matrix (nn_full, test; rows = actual)", "",
        markdown_table(confusion.reset_index().rename(columns={"actual": "actual \\ predicted"})), "",
        "## Data cleaning", "", "```json", json.dumps(clean_report, indent=2), "```", "",
        "## Files", "",
        "metrics.csv, calibration.csv, outcome_rates.csv, training_history.csv, confusion_matrix_test.csv, "
        "test_log_loss_by_count.csv, test_log_loss_by_player_coverage.csv, batter_embeddings.csv, "
        "pitcher_embeddings.csv, loss_curves.png, calibration.png, model.pt (load with pitch_model.load_checkpoint).",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(
        {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         "data": [str(p) for p in args.data], "continuous_features": list(CONTINUOUS_FEATURES)},
        indent=2), encoding="utf-8")

    print("\n" + headline.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nFull report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
