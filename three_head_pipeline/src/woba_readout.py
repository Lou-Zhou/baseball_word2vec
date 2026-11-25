"""Read wOBA out of the outcome head: batter vector -> simulated plate appearances -> wOBA.

    python src/woba_readout.py --run runs/2024

Nothing is fit to the batters. For each batter vector:
  1. Score a fixed reference set of real pitches (random sample, default 2000)
     with the batter's vector and the mean pitcher vector (weighted by training
     pitches), using each reference pitch's own count and characteristics.
  2. Average the outcome probabilities within each of the 12 counts.
  3. Run the count chain from 0-0: balls and strikes move the count, a foul with
     two strikes leaves it unchanged, ball four is a walk, strike three is a
     strikeout, hit-by-pitch and balls in play end the plate appearance.
  4. Score the plate-appearance outcomes with wOBA linear weights.
This is an alternative to the value head's direct readout (train.py), and also
yields K%, BB% and HR% for each batter.

Outputs go to <run>/woba_readout/: report.md, metrics.csv, projections.csv
(every batter with a vector), reference_pitches.csv and woba_readout.png.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from features import OUTCOMES, prepare_pitches
from metrics import calibrated_out_of_fold, errors, markdown_table, mean_out_of_fold
from model import load_checkpoint
from paths import run_data_paths, savant_woba

# FanGraphs Guts! linear weights. Seasons not listed use the latest one; the
# 2-parameter calibration absorbs small differences in scale.
WOBA_WEIGHTS = {
    2023: {"walk": 0.696, "hit_by_pitch": 0.726, "single": 0.883, "double": 1.244, "triple": 1.569, "home_run": 2.004},
    2024: {"walk": 0.689, "hit_by_pitch": 0.720, "single": 0.882, "double": 1.254, "triple": 1.590, "home_run": 2.050},
}
PA_OUTCOMES = ("walk", "strikeout", "hit_by_pitch", "out_in_play", "single", "double", "triple", "home_run")
IDX = {name: i for i, name in enumerate(OUTCOMES)}


@torch.inference_mode()
def count_outcome_table(model, batter_vectors, pitcher_vector, counts, pitch, chunk=64):
    """(batters x 12 counts x outcomes): mean probabilities over the reference pitches in each count."""
    n_ref, n_batters = len(counts), len(batter_vectors)
    count_onehot = F.one_hot(counts, num_classes=12).double()
    count_sizes = count_onehot.sum(0)
    out = np.zeros((n_batters, 12, len(OUTCOMES)))
    for start in range(0, n_batters, chunk):
        b = batter_vectors[start:start + chunk]
        k = len(b)
        logits = model.outcome_logits_from_vectors(
            b.repeat_interleave(n_ref, 0), pitcher_vector.expand(k * n_ref, -1), counts.repeat(k), pitch.repeat(k, 1))
        probs = logits.softmax(-1).double().view(k, n_ref, -1)
        out[start:start + k] = (torch.einsum("brc,rk->bkc", probs, count_onehot) / count_sizes[None, :, None]).numpy()
    return out


def plate_appearance_outcomes(table):
    """Exact count-chain probabilities of each plate-appearance outcome (dict of arrays)."""
    n = table.shape[0]
    reach = np.zeros((n, 4, 3))
    reach[:, 0, 0] = 1.0
    result = {name: np.zeros(n) for name in PA_OUTCOMES}
    for balls in range(4):
        for strikes in range(3):
            r = reach[:, balls, strikes]
            p = table[:, balls * 3 + strikes]
            ball = p[:, IDX["ball"]]
            strike = p[:, IDX["called_strike"]] + p[:, IDX["swinging_strike"]]
            foul = p[:, IDX["foul"]]
            if strikes == 2:
                # Fouls with two strikes keep the count: condition on leaving it.
                r = r / np.clip(1 - foul, 1e-9, None)
                foul = np.zeros(n)
            result["hit_by_pitch"] += r * p[:, IDX["hit_by_pitch"]]
            for name in ("out_in_play", "single", "double", "triple", "home_run"):
                result[name] += r * p[:, IDX[name]]
            if balls == 3:
                result["walk"] += r * ball
            else:
                reach[:, balls + 1, strikes] += r * ball
            if strikes == 2:
                result["strikeout"] += r * strike
            else:
                reach[:, balls, strikes + 1] += r * (strike + foul)
    return result


def woba_from_outcomes(outcomes, weights):
    total = sum(outcomes[name] for name in PA_OUTCOMES)
    return sum(weights[name] * outcomes[name] for name in weights) / total


def actual_rates(raw):
    """Per-batter K%, BB%, HR% from the run's own pitch data."""
    excluded = ["truncated_pa", "intent_walk", "catcher_interf", "sac_bunt", "sac_bunt_double_play"]
    pa = raw.loc[raw.events.notna() & ~raw.events.isin(excluded)]
    grouped = pa.groupby("batter").events
    return pd.DataFrame({
        "actual_k_rate": grouped.apply(lambda e: e.isin(["strikeout", "strikeout_double_play"]).mean()),
        "actual_bb_rate": grouped.apply(lambda e: e.eq("walk").mean()),
        "actual_hr_rate": grouped.apply(lambda e: e.eq("home_run").mean()),
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="A train.py output folder")
    parser.add_argument("--reference-pitches", type=int, default=2000)
    parser.add_argument("--min-pa", type=int, default=300, help="Minimum PA for the wOBA comparison")
    parser.add_argument("--folds", type=int, default=5, help="Folds for the out-of-fold calibration")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    output_dir = args.run / "woba_readout"
    output_dir.mkdir(exist_ok=True)
    model, preprocessor, _, _ = load_checkpoint(args.run / "model.pt")
    raw = pd.concat([pd.read_parquet(p) for p in run_data_paths(args.run)], ignore_index=True)
    pitches, _ = prepare_pitches(raw)
    seasons = sorted(pitches.game_date.dt.year.unique())
    if len(seasons) != 1:
        parser.error("woba_readout needs a single-season run")
    season = int(seasons[0])
    weights = WOBA_WEIGHTS.get(season, WOBA_WEIGHTS[max(WOBA_WEIGHTS)])

    # Reference pitches: a random sample of real pitches, each with its own count.
    reference = pitches.sample(n=min(args.reference_pitches, len(pitches)), random_state=args.seed)
    count_index = reference.balls * 3 + reference.strikes
    count_sizes = count_index.value_counts().reindex(range(12), fill_value=0)
    if (count_sizes == 0).any():
        raise ValueError("Some counts have no reference pitches; increase --reference-pitches")
    reference.loc[:, ["game_pk", "game_date", "pitcher", "balls", "strikes", "pitch_type",
                      "release_speed", "plate_x", "plate_z"]].to_csv(output_dir / "reference_pitches.csv", index=False)
    counts = torch.from_numpy(count_index.to_numpy(dtype=np.int64))
    pitch = torch.from_numpy(preprocessor.pitch_matrix(reference))

    # Mean pitcher vector, weighted by training pitches.
    pitcher_table = pd.read_csv(args.run / "pitcher_embeddings.csv")
    dims = [c for c in pitcher_table.columns if c.startswith("dim_")]
    pitcher_vector = torch.tensor(np.average(pitcher_table[dims].to_numpy(), axis=0,
                                             weights=pitcher_table.train_pitches), dtype=torch.float32)[None, :]
    batters = pd.read_csv(args.run / "batter_embeddings.csv")
    batter_vectors = torch.tensor(batters[dims].to_numpy(), dtype=torch.float32)
    print(f"Reading out {len(batters)} batters against {len(reference)} reference pitches...")
    outcomes = plate_appearance_outcomes(count_outcome_table(model, batter_vectors, pitcher_vector, counts, pitch))

    projections = pd.DataFrame({"mlbam_id": batters.mlbam_id, "train_pitches": batters.train_pitches})
    for name in PA_OUTCOMES:
        projections[f"{name}_rate"] = outcomes[name]
    projections["readout_woba"] = woba_from_outcomes(outcomes, weights)
    projections = projections.set_index("mlbam_id")

    # ---- compare with actual season stats ----
    compare = projections.join(savant_woba(season), how="inner").join(actual_rates(raw), how="left")
    compare = compare.loc[(compare.pa >= args.min_pa) & compare.xwoba.notna()]
    y, xwoba, readout = compare.woba.to_numpy(), compare.xwoba.to_numpy(), compare.readout_woba.to_numpy()
    calibrated = calibrated_out_of_fold(readout, y, args.folds, args.seed)
    rows = [
        {"target": "woba", "prediction": "readout (raw)", **errors(readout, y)},
        {"target": "woba", "prediction": "readout (calibrated, out of fold)", **errors(calibrated, y)},
        {"target": "woba", "prediction": "xwoba", **errors(xwoba, y)},
        {"target": "woba", "prediction": "league mean (out of fold)", **errors(mean_out_of_fold(y, args.folds, args.seed), y)},
        {"target": "xwoba", "prediction": "readout (raw)", **errors(readout, xwoba)},
        {"target": "xwoba", "prediction": "readout (calibrated, out of fold)",
         **errors(calibrated_out_of_fold(readout, xwoba, args.folds, args.seed), xwoba)},
    ]
    for component, column in (("k_rate", "strikeout_rate"), ("bb_rate", "walk_rate"), ("hr_rate", "home_run_rate")):
        rows.append({"target": component, "prediction": "readout (raw)",
                     **errors(compare[column].to_numpy(), compare[f"actual_{component}"].to_numpy())})
    metrics = pd.DataFrame(rows)
    metrics.insert(1, "batters", len(compare))
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    projections.reset_index().to_csv(output_dir / "projections.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        panels = (("readout wOBA (calibrated)", calibrated, y, "actual wOBA"),
                  ("readout K%", compare.strikeout_rate, compare.actual_k_rate, "actual K%"),
                  ("readout BB%", compare.walk_rate, compare.actual_bb_rate, "actual BB%"),
                  ("readout HR%", compare.home_run_rate, compare.actual_hr_rate, "actual HR%"))
        for ax, (xlabel, x, actual, ylabel) in zip(axes, panels):
            ax.scatter(x, actual, s=6, alpha=0.5)
            lo, hi = np.nanmin(actual), np.nanmax(actual)
            ax.plot([lo, hi], [lo, hi], color="black", lw=0.8)
            ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
            ax.set_title(f"r = {np.corrcoef(np.asarray(x, float), np.asarray(actual, float))[0, 1]:.2f}")
        fig.tight_layout(), fig.savefig(output_dir / "woba_readout.png", dpi=110), plt.close(fig)
    except ImportError:
        print("matplotlib not installed; skipping plot")

    league = projections.loc[:, [f"{n}_rate" for n in PA_OUTCOMES] + ["readout_woba"]].mean()
    report = [
        "# wOBA readout through the outcome head", "",
        f"Run: {args.run}. Season {season}; wOBA weights {weights}.",
        f"Each batter's vector was scored on {len(reference)} random reference pitches against the mean "
        "pitcher vector (weighted by training pitches), then run through the count chain.",
        "Nothing is fit to the batters except the 2-parameter calibration rows (fit on other folds).", "",
        "## How to read this", "",
        "- **readout (raw)** is the model's own wOBA; its bias shows how far the mean-pitcher, random-pitch "
        "setup is from the real league. The calibrated rows remove a constant and a scale.",
        "- Compare the calibrated rows with **league mean** (no information) and **xwoba** (the reference).",
        "- The K%, BB% and HR% rows check the components against this season's actual rates.", "",
        markdown_table(metrics), "",
        "## Readout league averages (all batters with a vector)", "",
        ", ".join(f"{k}: {v:.3f}" for k, v in league.items()), "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n" + metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nReport: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
