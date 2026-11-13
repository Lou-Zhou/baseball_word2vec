"""Download pitch-level Statcast data and save it as Parquet."""

import argparse
from pathlib import Path

from pybaseball import statcast


def get_batter_pitcher_outcome(start_year, end_year, output_path=None):
    """Save and return pitches from start_year inclusive to end_year exclusive.
    For example, (2024, 2025) downloads the 2024 calendar year.
    By default, save to this project's data/pitches_2024.parquet (or
    pitches_START_LAST.parquet for multiple years). Supply output_path to
    choose another location; parent directories are created automatically.
    """
    pitches = statcast(
        start_dt=f"{start_year}-01-01", end_dt=f"{end_year - 1}-12-31"
    )
    columns = [
        "game_pk", "game_date", "at_bat_number", "pitch_number",
        "batter", "pitcher", "balls", "strikes", "description", "events",
        "pitch_type",
    ]
    pitches = pitches.loc[:, columns].copy()
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
    parser.add_argument("--output", type=Path, required=True, help="Destination Parquet file")
    args = parser.parse_args(argv)
    if args.end_year <= args.start_year:
        parser.error("--end-year must be greater than --start-year")
    output_path = args.output.expanduser()
    pitches = get_batter_pitcher_outcome(args.start_year, args.end_year, output_path)
    print(f"Saved {len(pitches):,} pitches to {output_path.resolve()}")


if __name__ == "__main__":
    main()
