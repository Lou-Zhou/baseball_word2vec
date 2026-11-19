"""Evaluation helpers shared by the training and readout scripts."""

import numpy as np
import pandas as pd

from features import OUTCOMES

SWING = ["swinging_strike", "foul", "out_in_play", "single", "double", "triple", "home_run"]
IN_PLAY = ["out_in_play", "single", "double", "triple", "home_run"]
HITS = ["single", "double", "triple", "home_run"]


# --------------------------------------------------------------------------
# Pitch-outcome head
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
    """Overall log loss / Brier / accuracy / calibration plus log losses of the batter's decisions."""
    onehot = np.eye(len(OUTCOMES))[y]
    swing_p, swing_y = group_proba(proba, SWING), group_proba(onehot, SWING)
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
    }


def calibration_table(proba, y, split, bins=10):
    rows = []
    for k, name in enumerate(OUTCOMES):
        p, actual = proba[:, k], (y == k).astype(float)
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
        which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            mask = which == b
            if mask.any():
                rows.append({"split": split, "outcome": name, "bin": b, "n": int(mask.sum()),
                             "mean_predicted": float(p[mask].mean()), "observed_rate": float(actual[mask].mean())})
    return rows


def rate_table(proba, y, split):
    onehot = np.eye(len(OUTCOMES))[y]
    return [{"split": split, "outcome": name, "predicted_rate": float(proba[:, k].mean()),
             "actual_rate": float(onehot[:, k].mean())} for k, name in enumerate(OUTCOMES)]


# --------------------------------------------------------------------------
# Season-level comparisons
# --------------------------------------------------------------------------

def errors(predicted, actual):
    miss = np.asarray(predicted, float) - np.asarray(actual, float)
    r = np.corrcoef(predicted, actual)[0, 1] if np.std(predicted) > 0 else np.nan
    return {"rmse": float(np.sqrt((miss ** 2).mean())), "mae": float(np.abs(miss).mean()),
            "bias": float(miss.mean()), "pearson_r": float(r)}


def calibrated_out_of_fold(raw, actual, folds, seed):
    """actual ~ a + b * raw, fit on the other folds (2 parameters), predicted for each held-out fold."""
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(raw)) % folds
    out = np.zeros(len(raw))
    for f in range(folds):
        b, a = np.polyfit(raw[fold != f], actual[fold != f], 1)
        out[fold == f] = a + b * raw[fold == f]
    return out


def mean_out_of_fold(actual, folds, seed):
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(actual)) % folds
    return np.array([actual[fold != f].mean() for f in fold])


# --------------------------------------------------------------------------
# Plots and report tables
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
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, name in zip(axes, ("total", "outcome_ce", "contact_mse", "value_mse")):
        ax.plot(history.epoch, history[f"train_{name}"], "--", label="train")
        ax.plot(history.epoch, history[f"valid_{name}"], "-", label="valid")
        ax.set_title(name), ax.set_xlabel("epoch")
    axes[0].legend()
    fig.tight_layout(), fig.savefig(output_dir / "loss_curves.png", dpi=120), plt.close(fig)

    calibration = pd.DataFrame(calibration)
    test = calibration.loc[calibration.split == "test"]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
    for ax, name in zip(axes.flat, OUTCOMES):
        part = test.loc[test.outcome == name]
        top = max(part.mean_predicted.max(), part.observed_rate.max()) * 1.05
        ax.plot([0, top], [0, top], color="gray", lw=0.8)
        ax.plot(part.mean_predicted, part.observed_rate, "o-", ms=3)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("predicted", fontsize=8), ax.set_ylabel("observed", fontsize=8)
    fig.suptitle("Test-set calibration by outcome (on the diagonal = calibrated)")
    fig.tight_layout(), fig.savefig(output_dir / "calibration.png", dpi=120), plt.close(fig)


def markdown_table(frame, floatfmt=".4f"):
    frame = frame.copy()
    for column in frame.select_dtypes("number").columns:
        if column not in ("rows", "n", "pitches", "batters", "balls_in_play", "plate_appearances"):
            frame[column] = frame[column].map(lambda v: format(v, floatfmt))
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])
