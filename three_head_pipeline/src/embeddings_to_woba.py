"""Turn a year of batter embeddings into each batter's wOBA, using a trained model's value head.

WHAT YOU NEED
  1. The Python packages in requirements.txt (pip install -r requirements.txt).
  2. A trained model folder to decode with (--run). It must contain:
       model.pt                 the trained three-head model
       pitcher_embeddings.csv   used to build the "average pitcher"
       batter_embeddings.csv    used to fit the calibration
       config.json              tells the script which season the model was trained on
     Every folder written by train.py or sequential.py has these, e.g. runs/sequential/2026.
     Build them with:  python src/sequential.py --seasons 2022 2023 2024 2025 2026 <flags>
  3. A CSV of batter embeddings (--embeddings) with:
       mlbam_id                 the batter's MLB ID
       dim_0 ... dim_<n-1>      the vector, same number of dimensions as the model (32 for the tuned model)
       season                   optional; if present, pick a year with --season
     The vectors must be in the model's coordinates. All runs/sequential seasons share one
     space, so their embeddings, or a transition model's projections of them, work with
     any runs/sequential/<season> model.
  4. Internet access the first time a season is used: names and actual wOBA come from the
     Baseball Savant leaderboard (cached in data/savant/). Seasons not played yet are fine.

HOW TO RUN (from the three_head_pipeline folder)
  One season's own embeddings:
    python src/embeddings_to_woba.py --run runs/sequential/2025 --embeddings runs/sequential/2025/batter_embeddings.csv

  One year out of the combined file:
    python src/embeddings_to_woba.py --run runs/sequential/2026 --embeddings runs/sequential/batter_embeddings_by_season.csv --season 2026

  Projected embeddings for a future season (decode with the latest season's model):
    python src/embeddings_to_woba.py --run runs/sequential/2026 --embeddings projected_2027.csv --season 2027

  Options: --output PATH (where to write), --min-pa N (PA for the calibration fit and the
  comparison, default 300), --no-calibration (report the raw readout).

WHAT IT DOES, for each batter:
  1. raw = value_head(batter vector, mean pitcher vector), on the wOBA scale.
     The mean pitcher is the run's pitcher vectors averaged, weighted by training pitches.
  2. predicted_woba = a + b x raw. a and b are fit once on the run's own season:
     its batters' raw readouts against their Savant wOBA (batters with >= --min-pa PA).
     --no-calibration skips this step.

OUTPUT
  <embeddings folder>/woba_<season>.csv (or --output), sorted by predicted wOBA:
    mlbam_id, name, season, raw_readout, predicted_woba
    plus pa, woba, xwoba when Savant has that season; the RMSE and correlation
    against actual wOBA are then printed too.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import load_checkpoint
from paths import run_config, savant_names, savant_woba


@torch.inference_mode()
def raw_readout(model, value_scaler, mean_pitcher, vectors):
    batter = torch.as_tensor(vectors, dtype=torch.float32)
    pitcher = torch.as_tensor(mean_pitcher, dtype=torch.float32).expand_as(batter)
    return value_scaler.decode(model.value_from_vectors(batter, pitcher).numpy().astype(np.float64))


def load_run(run_dir):
    """(model, value scaler, mean pitcher vector, dimension columns, run season)."""
    model, _, _, value_scaler = load_checkpoint(run_dir / "model.pt")
    pitchers = pd.read_csv(run_dir / "pitcher_embeddings.csv")
    dims = [c for c in pitchers.columns if c.startswith("dim_")]
    mean_pitcher = np.average(pitchers[dims].to_numpy(), axis=0, weights=pitchers.train_pitches)
    return model, value_scaler, mean_pitcher, dims, int(run_config(run_dir)["seasons"][0])


def calibration(model, value_scaler, mean_pitcher, run_dir, dims, season, min_pa):
    """(a, b) with woba ~ a + b x raw, fit on the run's own batters and season."""
    batters = pd.read_csv(run_dir / "batter_embeddings.csv").set_index("mlbam_id")
    raw = pd.Series(raw_readout(model, value_scaler, mean_pitcher, batters[dims].to_numpy()), index=batters.index)
    savant = savant_woba(season)
    savant = savant.loc[(savant.pa >= min_pa) & savant.woba.notna()]
    both = raw.index.intersection(savant.index)
    b, a = np.polyfit(raw.loc[both], savant.woba.loc[both], 1)
    return float(a), float(b), len(both)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Trained model folder (model.pt, embeddings)")
    parser.add_argument("--embeddings", type=Path, required=True, help="CSV of batter vectors (mlbam_id, dim_*)")
    parser.add_argument("--season", type=int, default=None,
                        help="Season of the embeddings (selects rows if the file has a season column; "
                             "default: the file's only season, else the run's season)")
    parser.add_argument("--min-pa", type=int, default=300, help="PA for the calibration fit and comparison")
    parser.add_argument("--no-calibration", action="store_true", help="Report the raw value-head readout")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    model, value_scaler, mean_pitcher, dims, run_season = load_run(args.run)
    table = pd.read_csv(args.embeddings)
    if "season" in table:
        seasons = sorted(table.season.unique())
        if args.season is None and len(seasons) > 1:
            parser.error(f"{args.embeddings} has seasons {seasons}; pick one with --season")
        season = args.season or int(seasons[0])
        table = table.loc[table.season == season]
    else:
        season = args.season or run_season
    missing = set(dims) - set(table.columns)
    if missing:
        parser.error(f"Embeddings are missing {sorted(missing)} (the run uses {len(dims)} dimensions)")
    if table.empty:
        parser.error(f"No embeddings for season {season}")

    out = pd.DataFrame({"mlbam_id": table.mlbam_id.to_numpy(), "season": season,
                        "raw_readout": raw_readout(model, value_scaler, mean_pitcher, table[dims].to_numpy())})
    if args.no_calibration:
        out["predicted_woba"] = out.raw_readout
        note = "raw value-head readout (no calibration)"
    else:
        a, b, n = calibration(model, value_scaler, mean_pitcher, args.run, dims, run_season, args.min_pa)
        out["predicted_woba"] = a + b * out.raw_readout
        note = f"calibrated with a = {a:.4f}, b = {b:.4f} (fit on {n} {run_season} batters with >= {args.min_pa} PA)"

    names = {}
    for name_season in dict.fromkeys([season, run_season]):
        try:
            names = {**savant_names(name_season), **names}
        except Exception:  # no leaderboard yet (e.g. a projected future season)
            pass
    out.insert(1, "name", out.mlbam_id.map(names).fillna(out.mlbam_id.map(lambda i: f"ID {i}")))
    try:
        actual = savant_woba(season)
    except Exception:
        actual = None
    if actual is not None and not actual.empty:
        out = out.join(actual, on="mlbam_id")
        scored = out.loc[(out.pa >= args.min_pa) & out.woba.notna()]
        if len(scored) > 2:
            miss = scored.predicted_woba - scored.woba
            print(f"vs {season} actual wOBA ({len(scored)} batters with >= {args.min_pa} PA): "
                  f"RMSE {np.sqrt((miss ** 2).mean()):.4f}, bias {miss.mean():+.4f}, "
                  f"r {np.corrcoef(scored.predicted_woba, scored.woba)[0, 1]:.3f}")

    out = out.sort_values("predicted_woba", ascending=False)
    output = args.output or args.embeddings.parent / f"woba_{season}.csv"
    out.to_csv(output, index=False)
    print(f"Decoded with {args.run} ({run_season} heads); {note}")
    print(out.head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{len(out)} batters -> {output}")


if __name__ == "__main__":
    main()
