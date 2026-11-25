"""Tune the three-head model with Optuna to minimize season wOBA projection error.

    python src/tune.py --data data/pitches_2024.parquet --trials 50
    python src/tune.py --data data/pitches_2024.parquet --target-season 2025 --trials 50   # forecast target

Each trial trains a model on --data with sampled hyperparameters, then projects
every batter's season wOBA with the value-head readout used in train.py:
value_head(batter vector, pitch-weighted mean pitcher vector), calibrated
a + b x readout out of fold. The objective is the RMSE of that projection
against Baseball Savant season wOBA (or xwOBA with --target xwoba) for batters
with >= --min-pa PA.

--target-season picks which season's wOBA is the target:
  same season as --data (default)  how well the embedding encodes the season it
      was trained on. Caution: the vectors are trained on ~90% of those PAs, so
      this objective rewards memorizing each batter's season average.
  the following season              a no-aging forecast: this year's embedding
      predicting next year's wOBA. Only the Savant leaderboard is needed for the
      target season. Batters need >= --min-pa in both seasons. This is the
      objective that matches projecting next year's wOBA.

The study is stored in SQLite (default runs/tuning/<study>.db), so it can be
stopped and resumed by running the same command again. Trials that are
clearly worse than the median at the same epoch are pruned. Outputs in
runs/tuning/<study>/: trials.csv, best_params.json, best_command.txt (a train.py
command with the best settings), and plots if plotly is installed.
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from features import PITCH_INPUTS, PitchPreprocessor, prepare_pitches, random_game_split
from metrics import calibrated_out_of_fold, summary_metrics
from model import CONTACT_TARGETS, VALUE_COLUMNS, ContactScaler, ThreeHeadModel, ValueScaler
from paths import RUNS_DIR, portable_path, savant_woba
from train import encode_split, predict, train_model

# Layer options are stored as strings so Optuna can treat them as categories.
HIDDEN_CHOICES = {"64-32": [64, 32], "128-64": [128, 64], "256-128": [256, 128], "256-128-64": [256, 128, 64]}
CONTACT_CHOICES = {"32": [32], "64-32": [64, 32], "128-64": [128, 64]}
VALUE_CHOICES = {"linear": [], "16": [16], "32": [32], "64-32": [64, 32]}


# --------------------------------------------------------------------------
# Search space: edit this function to change what is tuned and the ranges.
# --------------------------------------------------------------------------

def sample_params(trial):
    return {
        "embedding_dim": trial.suggest_categorical("embedding_dim", [8, 12, 16, 24, 32]),
        "hidden": trial.suggest_categorical("hidden", list(HIDDEN_CHOICES)),
        "contact_hidden": trial.suggest_categorical("contact_hidden", list(CONTACT_CHOICES)),
        "value_hidden": trial.suggest_categorical("value_hidden", list(VALUE_CHOICES)),
        "value_target": trial.suggest_categorical("value_target", ["xwoba", "woba"]),
        "value_weight": trial.suggest_float("value_weight", 0.3, 10.0, log=True),
        "contact_weight": trial.suggest_float("contact_weight", 0.1, 3.0, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "lr": trial.suggest_float("lr", 5e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "embedding_weight_decay": trial.suggest_float("embedding_weight_decay", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [1024, 2048, 4096]),
    }


def train_flags(params):
    """train.py command-line flags for a parameter set."""
    flags = []
    for key, value in params.items():
        flag = "--" + key.replace("_", "-")
        if key == "hidden":
            flags += [flag, *map(str, HIDDEN_CHOICES[value])]
        elif key == "contact_hidden":
            flags += [flag, *map(str, CONTACT_CHOICES[value])]
        elif key == "value_hidden":
            flags += [flag, *map(str, VALUE_CHOICES[value])]
        else:
            flags += [flag, f"{value:.6g}" if isinstance(value, float) else str(value)]
    return flags


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------

class SeasonTarget:
    """Batters scored by the objective and their target-season wOBA (or xwOBA)."""

    def __init__(self, train_season, target_season, target, min_pa):
        table = savant_woba(target_season)
        table = table.loc[(table.pa >= min_pa) & table[target].notna()]
        if target_season != train_season:
            earlier = savant_woba(train_season)
            table = table.loc[table.index.isin(earlier.index[earlier.pa >= min_pa])]
        self.ids = table.index.to_numpy()
        self.actual = table[target].to_numpy()


@torch.inference_mode()
def projection_rmse(model, preprocessor, value_scaler, pitcher_weights, target, folds, seed):
    """Calibrated (out of fold) value-head readout vs the target season. Returns (rmse, r, batters)."""
    model.eval()
    device = model.batter_embedding.weight.device
    pitcher_ids = torch.as_tensor(list(pitcher_weights.index), device=device)
    weights = torch.as_tensor(pitcher_weights.to_numpy(), dtype=torch.float32, device=device)
    mean_pitcher = (model.pitcher_embedding(pitcher_ids) * weights[:, None]).sum(0)
    batter_map = preprocessor.player_maps["batter"]
    known = np.array([int(i) in batter_map for i in target.ids])
    index = torch.as_tensor([batter_map[int(i)] for i in target.ids[known]], device=device)
    vectors = model.batter_embedding(index)
    raw = value_scaler.decode(model.value_from_vectors(vectors, mean_pitcher.expand_as(vectors)).cpu().numpy())
    actual = target.actual[known]
    calibrated = calibrated_out_of_fold(raw.astype(np.float64), actual, folds, seed)
    rmse = float(np.sqrt(((calibrated - actual) ** 2).mean()))
    return rmse, float(np.corrcoef(raw, actual)[0, 1]), int(known.sum())


class Objective:
    def __init__(self, args, device):
        self.args, self.device = args, device
        raw = pd.concat([pd.read_parquet(p) for p in args.data], ignore_index=True)
        missing = (set(CONTACT_TARGETS) | set(VALUE_COLUMNS)) - set(raw.columns)
        if missing:
            raise SystemExit(f"Data is missing {sorted(missing)}; download it with load_data.py")
        pitches, _ = prepare_pitches(raw)
        for column in list(CONTACT_TARGETS) + list(VALUE_COLUMNS):
            pitches[column] = pd.to_numeric(pitches[column], errors="coerce").astype("float64")
        seasons = sorted(int(s) for s in pitches.game_date.dt.year.unique())
        if len(seasons) != 1:
            raise SystemExit(f"--data must hold one season (found {seasons})")
        self.train_season = seasons[0]
        self.target_season = args.target_season or self.train_season
        self.target = SeasonTarget(self.train_season, self.target_season, args.target, args.min_pa)
        print(f"Training on {self.train_season}; objective = RMSE vs {self.target_season} season {args.target} "
              f"for {len(self.target.ids)} batters with >= {args.min_pa} PA")

        # Data, split and preprocessing do not depend on the tuned parameters: do them once.
        self.train, self.valid, _ = random_game_split(pitches, args.valid_fraction, args.test_fraction, args.seed)
        self.preprocessor = PitchPreprocessor.fit(self.train, min_pitches=args.min_pitches)
        self.contact_scaler = ContactScaler.fit(self.train, CONTACT_TARGETS)
        self.encoded = {}  # value target kind -> (value scaler, train tensors, valid tensors)
        pitcher_ids = pd.Series(self.preprocessor.player_maps["pitcher"])
        counts = self.train.pitcher.value_counts().reindex(pitcher_ids.index).to_numpy(dtype=np.float64)
        self.pitcher_weights = pd.Series(counts / counts.sum(), index=pitcher_ids.to_numpy())

    def data_for(self, kind):
        if kind not in self.encoded:
            scaler = ValueScaler.fit(self.train, kind)
            self.encoded[kind] = (scaler,
                                  encode_split(self.train, self.preprocessor, self.contact_scaler, scaler, self.device),
                                  encode_split(self.valid, self.preprocessor, self.contact_scaler, scaler, self.device))
        return self.encoded[kind]

    def __call__(self, trial):
        import optuna

        params = sample_params(trial)
        print(f"\nTrial {trial.number}: {params}")
        torch.manual_seed(self.args.seed + trial.number)
        value_scaler, train_data, valid_data = self.data_for(params["value_target"])
        model = ThreeHeadModel(
            num_batters=len(self.preprocessor.player_maps["batter"]),
            num_pitchers=len(self.preprocessor.player_maps["pitcher"]),
            num_pitch_features=len(PITCH_INPUTS), embedding_dim=params["embedding_dim"],
            hidden_sizes=tuple(HIDDEN_CHOICES[params["hidden"]]),
            contact_hidden=tuple(CONTACT_CHOICES[params["contact_hidden"]]),
            value_hidden=tuple(VALUE_CHOICES[params["value_hidden"]]), dropout=params["dropout"],
        ).to(self.device)
        settings = SimpleNamespace(epochs=self.args.epochs, patience=self.args.patience, **{
            k: params[k] for k in ("lr", "weight_decay", "embedding_weight_decay", "batch_size",
                                   "contact_weight", "value_weight")})

        def report(epoch, current):
            rmse, _, _ = projection_rmse(current, self.preprocessor, value_scaler, self.pitcher_weights,
                                         self.target, self.args.folds, self.args.seed)
            current.train()
            trial.report(rmse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        start = time.time()
        train_model(model, train_data, valid_data, settings, self.device, on_epoch=report)
        rmse, r, batters = projection_rmse(model, self.preprocessor, value_scaler, self.pitcher_weights,
                                           self.target, self.args.folds, self.args.seed)
        # Guardrail metric: does the outcome head still predict pitches well?
        proba = predict(model, valid_data)[0]
        log_loss = summary_metrics(proba, self.valid.target.to_numpy())["log_loss"]
        trial.set_user_attr("pearson_r", r)
        trial.set_user_attr("batters", batters)
        trial.set_user_attr("valid_outcome_log_loss", log_loss)
        trial.set_user_attr("minutes", (time.time() - start) / 60)
        print(f"Trial {trial.number}: RMSE {rmse:.4f}, r {r:.3f}, outcome log loss {log_loss:.4f}")
        return rmse


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True, help="One season of pitches (load_data.py)")
    parser.add_argument("--target-season", type=int, default=None,
                        help="Season whose wOBA is the target (default: the training season)")
    parser.add_argument("--target", choices=["woba", "xwoba"], default="woba")
    parser.add_argument("--min-pa", type=int, default=300)
    parser.add_argument("--trials", type=int, default=50, help="Trials to add in this call")
    parser.add_argument("--timeout", type=float, default=None, help="Stop starting new trials after this many hours")
    parser.add_argument("--study-name", default=None, help="Default: tune_<train season>_to_<target season>_<target>")
    parser.add_argument("--storage", default=None, help="Optuna storage URL (default: SQLite in runs/tuning/)")
    parser.add_argument("--no-pruning", action="store_true")
    fixed = parser.add_argument_group("fixed settings (not tuned)")
    fixed.add_argument("--epochs", type=int, default=30)
    fixed.add_argument("--patience", type=int, default=3)
    fixed.add_argument("--valid-fraction", type=float, default=0.05)
    fixed.add_argument("--test-fraction", type=float, default=0.05)
    fixed.add_argument("--min-pitches", type=int, default=100)
    fixed.add_argument("--folds", type=int, default=5, help="Folds for the readout's out-of-fold calibration")
    fixed.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    fixed.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import optuna

    device = torch.device(args.device)
    objective = Objective(args, device)
    name = args.study_name or f"tune_{objective.train_season}_to_{objective.target_season}_{args.target}"
    output_dir = RUNS_DIR / "tuning" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{(RUNS_DIR / 'tuning' / f'{name}.db').as_posix()}"
    pruner = (optuna.pruners.NopPruner() if args.no_pruning
              else optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))
    study = optuna.create_study(study_name=name, storage=storage, load_if_exists=True, direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed), pruner=pruner)
    print(f"Study {name} ({storage}); {len(study.trials)} earlier trials")
    study.optimize(objective, n_trials=args.trials,
                   timeout=args.timeout * 3600 if args.timeout else None, gc_after_trial=True)

    trials = study.trials_dataframe()
    trials.to_csv(output_dir / "trials.csv", index=False)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials yet.")
        return
    best = study.best_trial
    summary = {"study": name, "train_data": [portable_path(p) for p in args.data],
               "train_season": objective.train_season, "target_season": objective.target_season,
               "target": args.target, "min_pa": args.min_pa, "best_trial": best.number,
               "rmse": best.value, **best.user_attrs, "params": best.params}
    (output_dir / "best_params.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    command = ["python", "src/train.py", "--data", *[portable_path(p) for p in args.data],
               "--output-dir", f"runs/{name}_best", *train_flags(best.params)]
    (output_dir / "best_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")

    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        plot_optimization_history(study).write_html(output_dir / "history.html")
        if len(completed) >= 2:
            plot_param_importances(study).write_html(output_dir / "importances.html")
    except ImportError:
        pass  # plots need plotly

    print(f"\nBest trial {best.number}: RMSE {best.value:.4f} "
          f"(r {best.user_attrs.get('pearson_r', float('nan')):.3f}, "
          f"outcome log loss {best.user_attrs.get('valid_outcome_log_loss', float('nan')):.4f})")
    print(json.dumps(best.params, indent=2))
    print(f"\nRetrain with the best settings:\n  {' '.join(command)}")
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
