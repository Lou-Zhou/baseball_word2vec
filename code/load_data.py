"""Download pitch-level Statcast data and save it as Parquet."""

import argparse
from pathlib import Path

from pybaseball import statcast


ID_COLUMNS = [
    "game_pk", "game_date", "game_type", "at_bat_number", "pitch_number",
    "batter", "pitcher", "stand", "p_throws", "balls", "strikes",
]
OUTCOME_COLUMNS = ["description", "events"]
# Measured before the ball reaches the batter; nothing here depends on the result.
PITCH_COLUMNS = [
    "pitch_type",
    "release_speed", "release_spin_rate", "spin_axis", "release_extension",
    "release_pos_x", "release_pos_z",
    "pfx_x", "pfx_z",
    "plate_x", "plate_z", "sz_top", "sz_bot",
]
COLUMNS = ID_COLUMNS + OUTCOME_COLUMNS + PITCH_COLUMNS


def get_batter_pitcher_outcome(start_year, end_year, output_path=None, regular_season_only=True):
    """Save and return pitches from start_year inclusive to end_year exclusive.
    For example, (2024, 2025) downloads the 2024 calendar year.
    By default only regular-season games (game_type "R") are kept, and the file
    is saved to this project's data/pitches_2024.parquet (or
    pitches_START_LAST.parquet for multiple years). Supply output_path to
    choose another location; parent directories are created automatically.
    """
    pitches = statcast(
        start_dt=f"{start_year}-01-01", end_dt=f"{end_year - 1}-12-31"
    )
    missing = set(COLUMNS) - set(pitches.columns)
    if missing:
        raise ValueError(f"Statcast response is missing columns: {sorted(missing)}")
    pitches = pitches.loc[:, COLUMNS].copy()
    if regular_season_only:
        pitches = pitches.loc[pitches.game_type.eq("R")].copy()
    if output_path is None:
        years = str(start_year) if end_year == start_year + 1 else f"{start_year}_{end_year - 1}"
        output_path = Path(__file__).resolve().parents[1] / "data" / f"pitches_{years}.parquet"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pitches.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    return pitches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, required=True, help="First year to download (inclusive)")
    parser.add_argument("--end-year", type=int, required=True, help="End year (exclusive)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Destination Parquet file (default: data/pitches_<years>.parquet)")
    parser.add_argument("--all-game-types", action="store_true",
                        help="Keep spring training, exhibition and postseason pitches too")
    args = parser.parse_args(argv)
    if args.end_year <= args.start_year:
        parser.error("--end-year must be greater than --start-year")
    output_path = args.output.expanduser() if args.output else None
    pitches = get_batter_pitcher_outcome(
        args.start_year, args.end_year, output_path,
        regular_season_only=not args.all_game_types,
    )
    print(f"Saved {len(pitches):,} pitches")


if __name__ == "__main__":
    main()
