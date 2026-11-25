"""Download one or more seasons of pitch-level Statcast data and save them as Parquet.

    python src/load_data.py --season 2024
    python src/load_data.py --season 2024 2025 2026      # one file per season

Each season is saved to data/pitches_<season>.parquet (regular season only by
default), so every season stays self-contained.
"""

import argparse
from pathlib import Path

from paths import DATA_DIR

ID_COLUMNS = [
    "game_pk", "game_date", "game_type", "at_bat_number", "pitch_number",
    "batter", "pitcher", "stand", "p_throws", "balls", "strikes",
]
# Measured after the pitch: training targets only, never model inputs.
#   launch_speed / launch_angle            contact head (balls in play)
#   woba_value / woba_denom                Statcast's per-PA wOBA credit (denom 0 for IBB, sac bunts, ...)
#   estimated_woba_using_speedangle        xwOBA of a batted ball (value head)
OUTCOME_COLUMNS = [
    "description", "events", "launch_speed", "launch_angle",
    "woba_value", "woba_denom", "estimated_woba_using_speedangle",
]
# Measured before the ball reaches the batter; nothing here depends on the result.
PITCH_COLUMNS = [
    "pitch_type",
    "release_speed", "release_spin_rate", "spin_axis", "release_extension",
    "release_pos_x", "release_pos_z",
    "pfx_x", "pfx_z",
    "plate_x", "plate_z", "sz_top", "sz_bot",
]
COLUMNS = ID_COLUMNS + OUTCOME_COLUMNS + PITCH_COLUMNS


def season_path(season):
    return DATA_DIR / f"pitches_{season}.parquet"


def download_season(season, output_path=None, regular_season_only=True):
    """Download one calendar year of Statcast pitches, keep COLUMNS, save and return them."""
    from pybaseball import statcast

    pitches = statcast(start_dt=f"{season}-01-01", end_dt=f"{season}-12-31")
    missing = set(COLUMNS) - set(pitches.columns)
    if missing:
        raise ValueError(f"Statcast response is missing columns: {sorted(missing)}")
    pitches = pitches.loc[:, COLUMNS].copy()
    if regular_season_only:
        pitches = pitches.loc[pitches.game_type.eq("R")].copy()
    output_path = Path(output_path or season_path(season))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pitches.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    return pitches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, nargs="+", required=True, help="Season(s) to download")
    parser.add_argument("--output", type=Path, default=None,
                        help="Destination file (one season only; default data/pitches_<season>.parquet)")
    parser.add_argument("--all-game-types", action="store_true",
                        help="Keep spring training, exhibition and postseason pitches too")
    parser.add_argument("--overwrite", action="store_true", help="Re-download seasons that already exist")
    args = parser.parse_args(argv)
    if args.output and len(args.season) > 1:
        parser.error("--output works with a single --season")
    for season in args.season:
        path = args.output or season_path(season)
        if path.exists() and not args.overwrite:
            print(f"{season}: {path} already exists (use --overwrite to download again)")
            continue
        print(f"{season}: downloading Statcast (this takes several minutes)...")
        pitches = download_season(season, path, regular_season_only=not args.all_game_types)
        print(f"{season}: saved {len(pitches):,} pitches to {path}")


if __name__ == "__main__":
    main()
