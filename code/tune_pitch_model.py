"""Tune pitch-model hyperparameters with Optuna's Bayesian TPE sampler."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from tqdm.contrib.logging import logging_redirect_tqdm

from pitch_model import OUTCOMES, chronological_split, fit_player_maps, make_dataset, prepare_pitches
from train_pitch_model import LOGGER as TRAIN_LOGGER
from train_pitch_model import configure_logging, fit_model, train_model


LOGGER = logging.getLogger(__name__)
SEARCH_SPACE = {
    "embedding_dim": [8, 9, 16, 32, 64],
    "learning_rate": [1e-4, 1e-2],
    "dropout": [0.0, 0.5],
    "batch_size": [256, 512, 1024, 2048],
    "weight_decay": [1e-6, 1e-2],
}


def tune_model(raw, validation_start, test_start, output, n_trials=30, epochs=20,
               patience=3, seed=42, device="cpu", progress=True, study_db=None,
               study_name="pitch_outcome", startup_trials=5, timeout=None,
               baseline_c=1.0, baseline_max_iter=1000):
    """Select parameters on validation loss, then run final training and testing.

    Trial training shares the normal trainer's fit_model implementation, but
    receives only training and validation tensors. The test set and logistic
    regression baseline are evaluated once, in the final train_model call.
    Existing SQLite studies resume only when data and objective settings match.
    n_trials is the number of additional trials for this invocation.
    """
    if min(n_trials, epochs, patience, startup_trials) < 1:
        raise ValueError("Trials, epochs, patience, and startup_trials must be positive")
    if timeout is not None and (not np.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be finite and positive")
    if not np.isfinite(baseline_c) or baseline_c <= 0 or baseline_max_iter < 1:
        raise ValueError("Baseline C and max_iter must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    study_db = Path(study_db) if study_db is not None else output.with_suffix(".optuna.sqlite3")
    study_db.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Preparing data for tuning: %s raw pitches", f"{len(raw):,}")
    frame, cleaning = prepare_pitches(raw)
    train, validation, test = chronological_split(frame, validation_start, test_start)
    if train.target.nunique() < 2:
        raise ValueError("Final logistic regression baseline requires at least two training outcomes")
    player_maps = fit_player_maps(train)
    train_dataset = make_dataset(train, player_maps)
    validation_dataset = make_dataset(validation, player_maps)
    LOGGER.info("Tuning split sizes: train=%d validation=%d held-out test=%d",
                len(train), len(validation), len(test))

    fingerprint_columns = ["game_date", "batter", "pitcher", "balls", "strikes", "target"]
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(
        frame[fingerprint_columns], index=False
    ).to_numpy().tobytes()).hexdigest()
    run_config = {
        "data_sha256": fingerprint,
        "validation_start": str(pd.Timestamp(validation_start).normalize().date()),
        "test_start": str(pd.Timestamp(test_start).normalize().date()),
        "epochs": epochs, "patience": patience, "seed": seed, "device": str(device),
        "outcomes": list(OUTCOMES), "search_space": SEARCH_SPACE,
        "objective_version": 1,
    }
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup_trials),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=startup_trials, n_warmup_steps=3),
        storage=f"sqlite:///{study_db.resolve().as_posix()}",
        study_name=study_name, load_if_exists=True,
    )
    existing_config = study.user_attrs.get("run_config")
    if existing_config is not None and existing_config != run_config:
        raise ValueError("Study data or objective settings changed; use a new --study-db or --study-name")
    if existing_config is None:
        if study.trials:
            raise ValueError("Existing study has no matching configuration; use a new study")
        study.set_user_attr("run_config", run_config)

    def objective(trial):
        parameters = {
            "embedding_dim": trial.suggest_categorical("embedding_dim", SEARCH_SPACE["embedding_dim"]),
            "learning_rate": trial.suggest_float("learning_rate", *SEARCH_SPACE["learning_rate"], log=True),
            "dropout": trial.suggest_float("dropout", *SEARCH_SPACE["dropout"]),
            "batch_size": trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"]),
            "weight_decay": trial.suggest_float("weight_decay", *SEARCH_SPACE["weight_decay"], log=True),
        }
        LOGGER.info("Trial %d parameters: %s", trial.number, json.dumps(parameters))

        def report_epoch(epoch, best_loss):
            trial.report(best_loss, step=epoch)
            if trial.should_prune():
                LOGGER.info("Pruning trial %d at epoch %d", trial.number, epoch)
                raise optuna.TrialPruned()

        _, history, best_epoch = fit_model(
            train_dataset, validation_dataset,
            {"num_batters": len(player_maps["batter"]),
             "num_pitchers": len(player_maps["pitcher"]),
             "embedding_dim": parameters["embedding_dim"], "dropout": parameters["dropout"]},
            epochs=epochs, batch_size=parameters["batch_size"],
            learning_rate=parameters["learning_rate"], weight_decay=parameters["weight_decay"],
            patience=patience, seed=seed, device=device, progress=progress,
            epoch_callback=report_epoch, run_name=f"Trial {trial.number}",
        )
        trial.set_user_attr("best_epoch", best_epoch)
        value = history[best_epoch - 1]["validation_log_loss"]
        LOGGER.info("Trial %d best validation log loss: %.6f", trial.number, value)
        return value

    LOGGER.info("Starting %d additional trials with TPE (%d startup trials); study=%s",
                n_trials, startup_trials, study_db.resolve())
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1,
                   gc_after_trial=True, show_progress_bar=progress)
    trials_path = output.with_suffix(".trials.parquet")
    study.trials_dataframe().to_parquet(trials_path, index=False)
    complete = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    if not complete:
        raise RuntimeError("No tuning trial completed; increase the budget or adjust the settings")
    best = study.best_trial
    summary = {
        "study_name": study.study_name, "study_db": str(study_db.resolve()),
        "best_trial": best.number, "best_validation_log_loss": best.value,
        "best_params": best.params, "total_trials": len(study.trials),
        "completed_trials": len(complete), "run_config": run_config,
        "cleaning": cleaning, "trials_path": str(trials_path.resolve()),
        "checkpoint_path": str(output.resolve()),
    }
    summary_path = output.with_suffix(".tuning.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("Selected trial %d: validation_log_loss=%.6f parameters=%s",
                best.number, best.value, json.dumps(best.params))
    LOGGER.info("Running final training and held-out evaluation with the selected parameters")
    _, _, metrics = train_model(
        raw, validation_start, test_start, output, epochs=epochs, patience=patience,
        seed=seed, device=device, progress=progress,
        baseline_c=baseline_c, baseline_max_iter=baseline_max_iter, **best.params,
    )
    summary["final_metrics"] = metrics
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("Saved tuning summary to %s and trials to %s", summary_path.resolve(), trials_path.resolve())
    return study, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tuned_pitch_model.pt"))
    parser.add_argument("--trials", dest="n_trials", type=int, default=30,
                        help="Additional trials to run (default: 30)")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs per trial and final fit")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--startup-trials", type=int, default=5,
                        help="Random startup trials before TPE sampling (default: 5)")
    parser.add_argument("--timeout", type=float, help="Search time budget in seconds, excluding final training")
    parser.add_argument("--study-db", type=Path, help="SQLite study file; defaults beside the output model")
    parser.add_argument("--study-name", default="pitch_outcome")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--baseline-c", type=float, default=1.0)
    parser.add_argument("--baseline-max-iter", type=int, default=1000)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    args = vars(parser.parse_args())
    log_file = args.pop("log_file") or args["output"].with_suffix(".tuning.log")
    log_level = args.pop("log_level")
    optuna.logging.disable_default_handler()
    loggers = [LOGGER, TRAIN_LOGGER, logging.getLogger("optuna")]
    configure_logging(log_file, log_level, loggers)
    with logging_redirect_tqdm(loggers=loggers):
        try:
            parquet = args.pop("parquet")
            LOGGER.info("Reading tuning data from %s", parquet.resolve())
            tune_model(pd.read_parquet(parquet, engine="pyarrow"), **args)
        except Exception:
            LOGGER.exception("Hyperparameter tuning failed")
            raise


if __name__ == "__main__":
    main()
