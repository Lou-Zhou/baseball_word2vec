"""Each batter's predicted season wOBA from a run's value-head readout, with names.

    python src/season_woba.py --run runs/2024_tuned

Reads <run>/batter_value_readout.csv (written by train.py for every batter with
a vector), adds names, and writes <run>/season_woba.csv sorted by predicted
wOBA: mlbam_id, name, pa, qualified, predicted_woba, woba, xwoba.
predicted_woba is the calibrated value-head readout a + b x readout. For
qualified batters (>= train.py's --min-pa PA) a and b are fit out of fold; for
the rest they are fit on all qualified batters. Low-PA predictions are noisier.
"""

import argparse
from pathlib import Path

import pandas as pd

from paths import run_config, savant_names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="A train.py output folder")
    parser.add_argument("--output", type=Path, default=None, help="Default <run>/season_woba.csv")
    args = parser.parse_args(argv)

    season = run_config(args.run)["seasons"][0]
    table = pd.read_csv(args.run / "batter_value_readout.csv")
    table["name"] = table.mlbam_id.map(savant_names(season))
    if "qualified" not in table:  # runs trained before every batter was exported
        table["qualified"] = True
    table["name"] = table.name.fillna(table.mlbam_id.map(lambda i: f"ID {i}"))
    out = (table[["mlbam_id", "name", "pa", "qualified", "readout_calibrated_woba", "woba", "xwoba"]]
           .rename(columns={"readout_calibrated_woba": "predicted_woba"})
           .sort_values("predicted_woba", ascending=False))
    output = args.output or args.run / "season_woba.csv"
    out.to_csv(output, index=False)
    print(out.head(15).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{len(out)} batters -> {output}")


if __name__ == "__main__":
    main()
