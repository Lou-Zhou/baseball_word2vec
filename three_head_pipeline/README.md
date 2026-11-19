# Batter embeddings from pitch-level Statcast data

Learns a 16-dimensional vector for every MLB batter and pitcher in a season by training a
neural network to predict what happens on each pitch. The batter vectors are meant to be
used downstream, for example to project a batter's embedding into next season and read
his wOBA off the projected vector, so the model is built so that the vectors encode
offensive value directly.

## The model

One network, three heads, all sharing the same batter and pitcher embeddings:

| Head | Inputs | Predicts | Trained on |
|---|---|---|---|
| Outcome | batter + pitcher vectors, count, 21 pitch features | 10 pitch outcomes (ball, called strike, whiff, foul, HBP, out in play, 1B, 2B, 3B, HR) | every pitch (cross-entropy) |
| Contact | same as above | exit velocity and launch angle | balls in play (MSE) |
| Value | batter + pitcher vectors **only** | the plate appearance's expected wOBA | PA-ending pitches (MSE) |

- **Pitch features**: release speed, position and extension; spin rate and axis; movement;
  location, including height relative to the batter's strike zone; speed and movement
  relative to the pitcher's own fastball; handedness flags. Pitch type is not an input,
  and pitches are not mirrored by handedness.
- **Value target**: Statcast's `estimated_woba_using_speedangle` on balls in play, 0 on
  strikeouts, and the wOBA weight for walks and HBP. A batter's average target is his
  xwOBA, so the target is on the wOBA scale without batted-ball luck.
  `--value-target woba` uses actual results instead.
- **Why a value head**: pitch-outcome loss is dominated by balls, strikes and fouls, so
  embeddings trained on it alone barely encode a batter's production (season wOBA r ≈ 0.45).
  The value head sees nothing but the two vectors, which forces production into the
  batter embedding.
- **Players** with fewer than 100 training pitches share one fixed "unknown player" vector.
- **Every season is self-contained**: one model per season, trained only on that season's
  pitches.

## Setup

```bash
pip install -r requirements.txt
```

A GPU is used automatically when PyTorch can see one; a season trains in a few minutes on a GPU.

## Run everything

```bash
python src/run_pipeline.py --season 2024
python src/run_pipeline.py --season 2024 2025 2026        # each season separately
python src/run_pipeline.py --season 2024 --value-weight 3  # extra options are passed to train.py
```

For each season this runs, in order:

| Step | Script | Output |
|---|---|---|
| 1. Download Statcast (skipped if the file exists) | `src/load_data.py` | `data/pitches_<season>.parquet` |
| 2. Train and evaluate the model | `src/train.py` | `runs/<season>/report.md`, embeddings, `model.pt` |
| 3. wOBA via the outcome head and the count chain | `src/woba_readout.py` | `runs/<season>/woba_readout/` |
| 4. wOBA from the embedding alone (cross-validated MLP) | `src/embedding_woba.py` | `runs/<season>/embedding_woba/` |
| 5. Nearest-neighbor explorer | `src/batter_neighbors.py` | `runs/<season>/batter_neighbors.html` |

Every script also runs on its own; `python src/<script>.py --help` lists its options.
Season wOBA and xwOBA come from the Baseball Savant expected-stats leaderboard and are
cached in `data/savant/`.

## Evaluation outputs

- **`runs/<season>/report.md`**: the three heads on held-out games (a random 5% for
  validation, 5% for test).
  - Outcome: log loss, Brier score, calibration, and the batter's decisions (swing,
    whiff, called strike, hit on contact).
  - Contact: exit-velocity and launch-angle error, plus how much worse they get without
    the batter's or pitcher's vector.
  - Value: per-PA error, and the **season readout** value_head(batter vector, mean pitcher
    vector) against Savant season wOBA and xwOBA.
- **`woba_readout/`**: scores each batter vector against 2,000 real reference pitches and
  the mean pitcher, then runs the exact ball-strike count chain to plate-appearance
  outcomes. This gives wOBA, K%, BB% and HR% for every batter.
- **`embedding_woba/`**: a small MLP from the raw 16 dimensions to wOBA and xwOBA, scored
  out of fold. It tests whether the information is in the vectors, independent of the
  trained heads.
- **`batter_neighbors.html`**: search a batter and list the closest batters (cosine
  similarity), with K%, BB%, whiff%, chase%, exit velocity, launch angle, hard-hit%, wOBA
  and xwOBA beside each. Open it in any browser.

## Results (2024, default settings, 286 batters with at least 300 PA)

| How a batter's season is read from his embedding | vs. wOBA: RMSE / r | vs. xwOBA: RMSE / r |
|---|---|---|
| League mean (no information) | .0349 / — | .0340 / — |
| Value head readout, calibrated | **.0224 / 0.77** | **.0151 / 0.90** |
| Outcome head + count chain, calibrated | .0252 / 0.69 | .0216 / 0.77 |
| MLP on the 16 dimensions (out of fold) | .0261 / 0.66 | .0192 / 0.83 |
| *Reference: the batter's own xwOBA* | *.0182 / 0.87* | — |

"Calibrated" means a two-parameter a + b·x fit, estimated out of fold.

- **Outcome head**: test log loss 1.110, calibration error 0.008.
- **Contact head**: the batter vector improves exit velocity by about 0.47 mph and launch angle by about 0.45°.
- **Count-chain readout**: K% r 0.76, HR% r 0.79.

With the value head turned off (`--value-weight 0`), the same season reaches only
wOBA r 0.54 through the count chain and 0.44 through the MLP.

**Caveat:** the vectors are trained on about 90% of the same season's plate appearances.
These numbers measure how well a season is *encoded*, not a forecast. On held-out plate
appearances, the value readout performs about as well as the batter's own average over
his training PAs. The real test is a projected embedding predicting next season.

## Layout

```
src/
  run_pipeline.py     one command for the whole pipeline
  load_data.py        Statcast download (pybaseball)
  features.py         cleaning, outcome labels, pitch features, game-level split
  model.py            ThreeHeadModel, target scalers, loss, checkpoint loading
  train.py            training loop and evaluation report
  metrics.py          shared metrics, plots, report tables
  woba_readout.py     count-chain wOBA / K% / BB% / HR% readout
  embedding_woba.py   cross-validated MLP from embedding to wOBA / xwOBA
  batter_neighbors.py nearest-neighbor HTML explorer
  paths.py            folders, config paths, Savant leaderboard cache
```

`data/` and `runs/` are created on first run and are git-ignored.
