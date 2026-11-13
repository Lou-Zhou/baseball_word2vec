# baseball_word2vec

Pitch-by-pitch player embeddings predicting the next pitch's outcome from
**batter ID, pitcher ID, and the pre-pitch count**. Swing/take is not an input.
This is the proposed pitch-level extension, rather than a reproduction of the
paper's plate-appearance model.

```text
Batter ID -> 16-dimensional embedding --+
Pitcher ID -> 16-dimensional embedding -+-> concatenate (44)
Count -> 12-dimensional one-hot -------+        |
                                        Linear(44, 64), ReLU
                                        Dropout(0.1)
                                        Linear(64, 32), ReLU
                                        Linear(32, 25)
```

Training uses cross-entropy on logits; prediction applies softmax. The 25
classes include ball, called strike, swinging strike, foul, hit by pitch, and
20 individual in-play outcomes listed in `IN_PLAY_OUTCOMES` in `pitch_model.py`.
These cover singles, doubles, triples, home runs, outs, errors, fielder's
choices, sacrifices, and interference results. Player vectors are shared across
all of a player's pitches. There is
no sequence memory in this version, and current-pitch type, location, and
velocity are not inputs.

## Setup and data

Run commands from this directory. If using the workspace's existing environment,
replace `python` with `../.venv/bin/python`.

```sh
python -m pip install -r requirements.txt
```

Prepare a Statcast Parquet file that includes `game_date`, `batter`, `pitcher`, `balls`,
`strikes`, `description`, and `events`. Retain `game_pk`, `at_bat_number`, and
`pitch_number` to identify games, plate appearances, and pitches. For example,
from a notebook in `code/` (this downloads a full season):

```python
from load_data import get_batter_pitcher_outcome

# End year is exclusive. Saves to the project's data/pitches_2024.parquet.
pitches = get_batter_pitcher_outcome(2024, 2025)
```

The loader returns the DataFrame as well as saving it. Pass
`output_path="../data/custom.parquet"` to choose a different location.
Parquet support uses the `pyarrow` dependency in `requirements.txt`.

To download from the command line and select the output path:

```sh
python code/load_data.py \
  --start-year 2024 \
  --end-year 2025 \
  --output data/pitches_2024.parquet
```

The end year is exclusive, and missing output directories are created automatically.

Use `description` as the target source, except for `hit_into_play`, where the
target is the specific result in `events` (for example, `single` or `field_out`).
Event names follow the [MLB event vocabulary](https://statsapi.mlb.com/api/v1/eventTypes).
Non-contact pitches may have missing `events` and are retained. In-play pitches
with missing or unsupported events are excluded and reported under
`unsupported_in_play_events`. A walk or strikeout event does not override the
description-based pitch label. `balls` and `strikes` are pre-pitch counts.
See the [Statcast field definitions](https://baseballsavant.mlb.com/csv-docs).

Preprocessing groups `blocked_ball` with ball, `swinging_strike_blocked` with
swinging strike, and `foul_tip` with foul. This last grouping is a coarse contact
category: foul tips and ordinary fouls have different count-transition rules,
so these labels alone are not suitable for a plate-appearance simulator.
Bunt descriptions, pitchouts, automatic calls, and other unmapped descriptions
are excluded and reported. Invalid IDs, dates, or counts are excluded too.
Fair bunts described as `hit_into_play` use their sacrifice or other event;
the excluded bunt descriptions are misses, fouls, and foul tips on bunt attempts.

Checkpoints trained with the previous six-class labels must be retrained for
the expanded outcome set.

## Train

```sh
python code/train_pitch_model.py \
  --parquet data/pitches_2024.parquet \
  --validation-start 2024-07-01 \
  --test-start 2024-09-01 \
  --output artifacts/pitch_model.pt
```

Games before July 1 train the model; July and August select the best epoch;
September onward is the test set. Splitting by game date keeps games and plate
appearances together. Each split must contain data. The defaults are 20 maximum
epochs, batch size 1024, AdamW at 0.001, and early stopping after three epochs
without improved validation log loss. Use `--help` for other settings.

Training, validation, and test evaluation show tqdm batch progress bars with
running loss (and evaluation accuracy). Timestamped logs go to the console and
`artifacts/pitch_model.log` by default, alongside the checkpoint. Logs include
data cleaning, training settings, epoch metrics and timing, early stopping,
final metrics, and output paths. Use `--log-file logs/training.log` for a custom
log path, `--log-level DEBUG` to change verbosity, or `--no-progress` to hide
the bars. For notebook calls, `train_model(..., progress=False)` hides bars;
configure Python logging to display the module's INFO messages.

Only training players enter the ID mappings. Unseen players receive a fixed
zero embedding; this is a fallback, not an estimated vector for the new player.
Test metrics include counts of pitches involving unseen players.

The command saves the best model, player mappings, class order, training
settings, cleaning report, and metrics in the `.pt` checkpoint. It also writes
an adjacent `.metrics.json` with test log loss, accuracy, and a multinomial
logistic regression baseline under `multinomial_logistic_regression_test`.
The baseline uses sparse one-hot encodings of batter, pitcher, and the 12
pre-pitch counts, fitted only on the training split. Unknown players contribute
zero to their categorical feature group. It uses L2 regularization and the
`lbfgs` solver; control regularization with `--baseline-c` (default 1.0) and
the iteration limit with `--baseline-max-iter` (default 1000). See the
[scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
Outcome classes absent from training have zero predicted probability; log loss
clips probabilities at 1e-15 and the metrics report `unseen_outcome_pitches`.
The baseline requires at least two training outcome classes. Its settings are
saved in the checkpoint's `baseline_config`. Lower log loss is better;
improvement over that baseline must be measured on actual data.

## Bayesian hyperparameter tuning

Optuna's [TPE sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
searches for the lowest validation log loss. Run from this directory:

```sh
python code/tune_pitch_model.py \
  --parquet data/pitches_2024.parquet \
  --validation-start 2024-07-01 \
  --test-start 2024-09-01 \
  --trials 30 \
  --epochs 20 \
  --output artifacts/tuned_pitch_model.pt
```

The search space is:

| Parameter | Values |
| --- | --- |
| Embedding dimensions | 8, 9, 16, 32, 64 |
| Learning rate | 0.0001–0.01, logarithmic scale |
| Dropout | 0–0.5 |
| Batch size | 256, 512, 1024, 2048 |
| AdamW weight decay | 0.000001–0.01, logarithmic scale |

The first five trials sample randomly; subsequent trials use TPE's learned
distributions. `--startup-trials` changes that number. Each trial shares the
normal training loop, uses early stopping, and reports its best validation
loss after each epoch. A median pruner can stop unpromising trials after a
three-epoch warmup once enough startup trials have completed.

Only training and validation tensors enter search trials. The selected
configuration is then trained again with the same seed and validation-based
early stopping. The test set and logistic regression baseline are evaluated
once after selection. Training is not refit on the combined train/validation
data. The neural network's hidden layers remain 64 and 32 units; baseline
regularization is controlled separately with the existing baseline flags.

Alongside the final `.pt` checkpoint and `.metrics.json`, tuning writes:

- `.optuna.sqlite3`: persistent study and trial history.
- `.tuning.json`: selected parameters, validation score, and final metrics.
- `.trials.parquet`: trial parameters, scores, and states.
- `.tuning.log`: search and training logs when using the CLI.

Rerun the same command to resume; `--trials` means additional trials on each
invocation. Reusing a study requires matching data, split dates, epoch budget,
patience, seed, device, and search space. Use a new `--study-db` or `--study-name`
when those change. `--timeout` limits search time in seconds (checked between
trials); final training runs afterward. `--no-progress` disables the progress
bars. A normal training run also accepts `--weight-decay` directly.

## Predict and inspect embeddings

From a notebook in `code/`:

```python
import pandas as pd
from pitch_model import load_checkpoint, predict_pitches

model, player_maps = load_checkpoint("../artifacts/pitch_model.pt")
matchup = pd.DataFrame({
    "batter": [545361], "pitcher": [543037], "balls": [1], "strikes": [2]
})
probabilities = predict_pitches(model, matchup, player_maps)
print(probabilities)

# Rows correspond to encoded IDs, not MLB IDs; row zero is the unknown player.
batter_vectors = model.batter_embedding.weight.detach().numpy()
pitcher_vectors = model.pitcher_embedding.weight.detach().numpy()
encoded_id = player_maps["batter"].get(545361, 0)
vector = batter_vectors[encoded_id]
```

To train directly from a notebook, import `train_model` from `train_pitch_model`
and call it with the raw DataFrame, split dates, and checkpoint path:

```python
from train_pitch_model import train_model

model, player_maps, metrics = train_model(
    pitches,
    validation_start="2024-07-01",
    test_start="2024-09-01",
    output="../artifacts/pitch_model.pt",
)
```

## Visualize with torchview

The requirements include `torchview` and the Graphviz Python package.
Rendering also requires the Graphviz `dot` executable (on macOS:
`brew install graphviz`). See the [torchview documentation](https://github.com/mert-kurttutan/torchview).

```sh
python code/visualize_model.py
```

This creates `artifacts/pitch_model_architecture.svg` and `.png`, showing the
embedding branches, count one-hot operation, concatenation, and dense layers
with tensor shapes. No training or data download is needed. Default vocabulary
sizes are illustrative (1,000 batters and 1,000 pitchers). To use the actual
dimensions from a trained checkpoint or make a horizontal diagram:

```sh
python code/visualize_model.py --checkpoint artifacts/pitch_model.pt --direction LR
```

From a notebook in `code/`, display the graph inline:

```python
from visualize_model import draw_architecture

model_graph = draw_architecture()       # Or draw_architecture(model).
model_graph.visual_graph
```

The model outputs logits, so softmax and training loss are not part of the
traced architecture. The example inputs are encoded integer indices, rather
than raw MLB IDs. The visualization traces a copy of the model on CPU in
evaluation mode.

## Offline checks

```sh
python -m unittest discover -s tests -v
```

These checks cover pitch filtering, chronological splits, training-only ID
encoding, learning player vectors, the logistic regression baseline, and checkpoint inference.
Tuning checks cover study resumption, validation-only selection, final test
evaluation, and pruning.
They use synthetic data and do not establish performance on MLB data.
