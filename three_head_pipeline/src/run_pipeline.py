"""Run the whole pipeline for one or more seasons: download -> train -> evaluate.

    python src/run_pipeline.py --season 2024
    python src/run_pipeline.py --season 2024 2025 2026
    python src/run_pipeline.py --season 2024 --value-weight 3 --epochs 20   # unknown options go to train.py

For each season, independently (every season is self-contained):
  1. load_data.py         data/pitches_<season>.parquet (skipped if it exists)
  2. train.py             runs/<season>/ (model, embeddings, report.md with all three heads)
  3. woba_readout.py      runs/<season>/woba_readout/   (count-chain wOBA, K%, BB%, HR%)
  4. embedding_woba.py    runs/<season>/embedding_woba/ (MLP from embedding to wOBA / xwOBA)
  5. batter_neighbors.py  runs/<season>/batter_neighbors.html
"""

import argparse
from pathlib import Path

import batter_neighbors
import embedding_woba
import load_data
import train
import woba_readout
from paths import RUNS_DIR


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, nargs="+", required=True)
    parser.add_argument("--data", type=Path, default=None,
                        help="Use this Parquet file instead of data/pitches_<season>.parquet (one season only)")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="Parent folder for run outputs")
    parser.add_argument("--repeats", type=int, default=10, help="Cross-validation repeats for embedding_woba")
    parser.add_argument("--min-pa", type=int, default=300, help="Minimum PA for the season wOBA comparisons")
    parser.add_argument("--skip-train", action="store_true", help="Only re-run the evaluations")
    args, train_args = parser.parse_known_args(argv)
    if args.data and len(args.season) > 1:
        parser.error("--data works with a single --season")

    for season in args.season:
        print(f"\n===== {season} =====")
        data = args.data or load_data.season_path(season)
        if not data.exists():
            load_data.main(["--season", str(season)])
        run_dir = args.runs_dir / str(season)
        if not args.skip_train:
            print("\n--- train ---")
            train.main(["--data", str(data), "--output-dir", str(run_dir), "--min-pa", str(args.min_pa)] + train_args)
        print("\n--- count-chain wOBA readout ---")
        woba_readout.main(["--run", str(run_dir), "--min-pa", str(args.min_pa)])
        print("\n--- embedding -> wOBA MLP ---")
        embedding_woba.main(["--runs", str(run_dir), "--min-pa", str(args.min_pa), "--repeats", str(args.repeats)])
        print("\n--- nearest neighbors ---")
        batter_neighbors.main(["--run", str(run_dir)])
        print(f"\n{season} done: start with {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
