"""Project folders, data-path resolution and the Baseball Savant leaderboard."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
SAVANT_DIR = DATA_DIR / "savant"


def portable_path(path):
    """Path relative to the project root when inside it (so configs survive moving the repo)."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_data_path(path, run_dir=None):
    """Find a data file listed in a run's config, whatever folder the script runs from."""
    path = Path(path)
    candidates = [path, ROOT / path]
    if run_dir is not None:
        candidates.append(Path(run_dir).resolve().parents[1] / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {path}; tried {[str(c) for c in candidates]}")


def run_config(run_dir):
    return json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))


def run_data_paths(run_dir):
    return [resolve_data_path(p, run_dir) for p in run_config(run_dir)["data"]]


def run_seasons(run_dir):
    dates = pd.concat([pd.read_parquet(p, columns=["game_date"]) for p in run_data_paths(run_dir)]).game_date
    return sorted(int(year) for year in pd.to_datetime(dates).dt.year.unique())


def savant_table(season):
    """Raw Savant expected-stats leaderboard for one season (all batters), cached as CSV."""
    SAVANT_DIR.mkdir(parents=True, exist_ok=True)
    path = SAVANT_DIR / f"expected_stats_{season}.csv"
    if not path.exists():
        from pybaseball import statcast_batter_expected_stats
        statcast_batter_expected_stats(season, minPA=1).to_csv(path, index=False)
    return pd.read_csv(path)


def savant_woba(season):
    """Season PA, wOBA and xwOBA per batter, indexed by mlbam_id."""
    table = savant_table(season).rename(columns={"player_id": "mlbam_id", "est_woba": "xwoba"})
    return table.set_index("mlbam_id").loc[:, ["pa", "woba", "xwoba"]]


def savant_names(season):
    """{mlbam_id: "First Last"} from the Savant leaderboard."""
    table = savant_table(season)
    names = {}
    for pid, name in zip(table.player_id, table["last_name, first_name"]):
        last, _, first = str(name).partition(", ")
        names[int(pid)] = f"{first} {last}".strip()
    return names
