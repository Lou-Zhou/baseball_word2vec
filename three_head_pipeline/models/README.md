# Trained models and batter embeddings (2022–2026)

Ready-to-use output of `src/sequential.py`. There's no need to download data or train
anything to use these.

All five seasons share **one embedding space**: dimension 5 means the same thing in 2022
and 2026, so a batter's vectors can be compared, and projected, across years.

- 2022 was trained in full.
- Each later season fitted fresh player vectors into the previous season's frozen network,
  then fine-tuned everything gently.
- Every season's vectors come only from that season's pitches.

## What's here

```
sequential/
  batter_embeddings_by_season.csv   every batter-season: mlbam_id, train_pitches, dim_0..dim_31, season
  report.md                         year-to-year continuity results
  pair_metrics.csv, batter_pairs.csv, season_metrics.csv
  2022/ ... 2026/                   one folder per season:
    model.pt                        the trained three-head model
    batter_embeddings.csv           batter vectors (32 dims), with training pitch counts
    pitcher_embeddings.csv          pitcher vectors
    batter_value_readout.csv        each batter's season wOBA readout vs actual
    config.json, value_readout_metrics.csv, training_history.csv
```

- **Who has a vector:** each batter and pitcher with at least 100 training pitches that season.
- **Noise:** vectors from fewer pitches are noisier.
- **IDs:** `mlbam_id` is the MLB Advanced Media player ID, the one Baseball Savant uses.

## Turn embeddings into wOBA

Run from the `three_head_pipeline` folder, after `pip install -r requirements.txt`.

**A season's own embeddings:**
```bash
python src/embeddings_to_woba.py --run models/sequential/2026 --embeddings models/sequential/batter_embeddings_by_season.csv --season 2026
```

**Projected embeddings for a future season.** Make a CSV with `mlbam_id` and `dim_0` …
`dim_31` in this same space, then decode it with the latest season's model:
```bash
python src/embeddings_to_woba.py --run models/sequential/2026 --embeddings projected_2027.csv --season 2027
```

This writes `woba_<season>.csv` next to the embeddings file, with each batter's name and
predicted wOBA. When the season has been played, it also includes the actual wOBA and
xwOBA.

**Internet access is needed the first time a season is used.** Names, actual wOBA, and the
calibration all come from the Baseball Savant leaderboard, which is then cached in
`data/savant/`.

## Other scripts that work on these folders

```bash
python src/season_woba.py --run models/sequential/2025                      # season wOBA table with names
python src/batter_neighbors.py --run models/sequential/2025                 # needs data/pitches_2025.parquet
python src/neighbor_figure.py --run models/sequential/2025 --batter "Arraez"  # needs data/pitches_2025.parquet
```

The last two also compute stats from the raw pitches, so download that season first:
```bash
python src/load_data.py --season 2025
```

## Settings used

These come from the Optuna search on 2024 season wOBA (`src/tune.py`):
- 32-dimensional embeddings
- outcome head 64 → 32
- contact head 64 → 32
- linear value head on actual wOBA
- value weight 1.47, contact weight 0.62
- dropout 0.27
- learning rate 0.0049
- weight decay 4.5e-6 on the network, 9.8e-4 on the embeddings
- batch size 4096, seed 8

Each season's `config.json` has the full list.
