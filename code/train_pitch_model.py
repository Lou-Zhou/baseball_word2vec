"""Train the unconditional pitch-outcome model on a local Statcast Parquet file."""

import argparse
import copy
import json
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from pitch_model import (
    DESCRIPTION_TO_OUTCOME, IN_PLAY_OUTCOMES, OUTCOMES, PitchOutcomeModel,
    chronological_split, fit_player_maps,
    make_dataset, prepare_pitches,
)


LOGGER = logging.getLogger(__name__)


def configure_logging(log_file, level="INFO", loggers=None):
    """Configure console/file logging for CLI entry points, not imports."""
    loggers = [LOGGER] if loggers is None else loggers
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(formatter)
    for logger in loggers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False


@torch.inference_mode()
def evaluate(model, loader, device, progress=True, description="Evaluation"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with tqdm(loader, desc=description, unit="batch", dynamic_ncols=True,
              leave=False, disable=not progress) as batches:
        for batter, pitcher, count, target in batches:
            batter, pitcher, count, target = (
                item.to(device) for item in (batter, pitcher, count, target)
            )
            logits = model(batter, pitcher, count)
            total_loss += nn.functional.cross_entropy(logits, target, reduction="sum").item()
            correct += (logits.argmax(dim=-1) == target).sum().item()
            total += len(target)
            batches.set_postfix(log_loss=f"{total_loss / total:.4f}",
                                accuracy=f"{correct / total:.3f}", refresh=False)
    return {"log_loss": total_loss / total, "accuracy": correct / total}


def logistic_regression_baseline(train, test, c=1.0, max_iter=1000, batch_size=4096):
    """Fit multinomial logistic regression on training-only categorical inputs.

    Batter, pitcher, and the 12-valued count are sparse one-hot features.
    Unknown players contribute zero to their feature group. Outcome columns
    are aligned to OUTCOMES; classes absent from training have zero probability
    and are clipped to 1e-15 only when calculating log loss.
    """
    if not np.isfinite(c) or c <= 0 or min(max_iter, batch_size) < 1:
        raise ValueError("Baseline C, max_iter, and batch_size must be positive")
    if train.empty or test.empty:
        raise ValueError("The logistic regression baseline requires nonempty train and test data")
    if train.target.nunique() < 2:
        raise ValueError("The logistic regression baseline requires at least two training outcomes")

    def categorical_inputs(frame):
        return np.column_stack([
            frame.batter.to_numpy(), frame.pitcher.to_numpy(),
            (frame.balls * 3 + frame.strikes).to_numpy(),
        ])

    started = perf_counter()
    LOGGER.info("Fitting multinomial logistic regression baseline: C=%g max_iter=%d", c, max_iter)
    encoder = OneHotEncoder(
        categories=[np.sort(train.batter.unique()), np.sort(train.pitcher.unique()), np.arange(12)],
        handle_unknown="ignore", sparse_output=True, dtype=np.float64,
    )
    inputs = encoder.fit_transform(categorical_inputs(train))
    # lbfgs optimizes the multinomial loss when there are >= 3 classes;
    # the two-class case uses equivalent binary logistic regression.
    classifier = LogisticRegression(solver="lbfgs", C=c, max_iter=max_iter)
    classifier.fit(inputs, train.target.to_numpy())
    if classifier.n_iter_.max() >= max_iter:
        LOGGER.warning("Logistic regression reached its iteration limit; consider --baseline-max-iter")

    total_loss, correct, unseen_outcomes = 0.0, 0, 0
    for start in range(0, len(test), batch_size):
        batch = test.iloc[start:start + batch_size]
        probabilities = np.zeros((len(batch), len(OUTCOMES)), dtype=float)
        probabilities[:, classifier.classes_] = classifier.predict_proba(
            encoder.transform(categorical_inputs(batch))
        )
        targets = batch.target.to_numpy()
        true_probabilities = probabilities[np.arange(len(batch)), targets]
        total_loss -= np.log(np.clip(true_probabilities, 1e-15, 1.0)).sum()
        correct += int((probabilities.argmax(axis=1) == targets).sum())
        unseen_outcomes += int((~np.isin(targets, classifier.classes_)).sum())
    LOGGER.info("Logistic regression baseline completed: features=%d iterations=%d elapsed=%.1fs",
                inputs.shape[1], classifier.n_iter_.max(), perf_counter() - started)
    return {
        "log_loss": float(total_loss / len(test)),
        "accuracy": float(correct / len(test)),
        "unseen_outcome_pitches": unseen_outcomes,
    }


def fit_model(train_dataset, validation_dataset, model_config, epochs=20,
              batch_size=1024, learning_rate=1e-3, weight_decay=1e-4,
              patience=3, seed=42, device="cpu", progress=True,
              epoch_callback=None, run_name="Train"):
    """Fit on training data and select weights using validation data only.

    An optional callback(epoch, best_validation_loss) may stop a tuning trial
    by raising an exception. No test data or baseline is accessed here.
    """
    embedding_dim = model_config["embedding_dim"]
    dropout = model_config["dropout"]
    if min(epochs, batch_size, embedding_dim, patience) < 1 or learning_rate <= 0:
        raise ValueError("Epochs, batch size, embedding size, patience, and learning rate must be positive")
    if not np.isfinite(learning_rate) or not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Learning rate must be finite and weight decay must be finite and nonnegative")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if not len(train_dataset) or not len(validation_dataset):
        raise ValueError("Training and validation datasets must be nonempty")
    torch.manual_seed(seed)
    model = PitchOutcomeModel(**model_config).to(device)
    LOGGER.debug("Model architecture:\n%s", model)
    loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=index == 0)
        for index, dataset in enumerate((train_dataset, validation_dataset))
    ]
    LOGGER.info(
        "%s on %s: epochs=%d batch_size=%d lr=%g weight_decay=%g embedding_dim=%d "
        "dropout=%g patience=%d seed=%d batters=%d pitchers=%d outcomes=%d",
        run_name, device, epochs, batch_size, learning_rate, weight_decay, embedding_dim, dropout,
        patience, seed, model_config["num_batters"], model_config["num_pitchers"], len(OUTCOMES),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history, best_loss, best_state, stale_epochs = [], float("inf"), None, 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        epoch_started = perf_counter()
        model.train()
        total_loss, seen = 0.0, 0
        with tqdm(loaders[0], desc=f"{run_name} {epoch:02d}/{epochs}", unit="batch",
                  dynamic_ncols=True, leave=False, disable=not progress) as batches:
            for batter, pitcher, count, target in batches:
                batter, pitcher, count, target = (
                    item.to(device) for item in (batter, pitcher, count, target)
                )
                optimizer.zero_grad()
                loss = criterion(model(batter, pitcher, count), target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(target)
                seen += len(target)
                batches.set_postfix(log_loss=f"{total_loss / seen:.4f}", refresh=False)
        scores = evaluate(model, loaders[1], device, progress, f"Validation {epoch:02d}/{epochs}")
        history.append({"epoch": epoch, "train_log_loss": total_loss / len(train_dataset),
                        "validation_log_loss": scores["log_loss"]})
        LOGGER.info(
            "%s epoch %02d/%d: train_log_loss=%.4f validation_log_loss=%.4f "
            "validation_accuracy=%.4f elapsed=%.1fs",
            run_name, epoch, epochs, total_loss / len(train_dataset), scores["log_loss"],
            scores["accuracy"], perf_counter() - epoch_started,
        )
        if scores["log_loss"] < best_loss:
            best_loss = scores["log_loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch, stale_epochs = epoch, 0
            LOGGER.info("New best validation log loss: %.4f (epoch %d)", best_loss, epoch)
        else:
            stale_epochs += 1
        if epoch_callback is not None:
            epoch_callback(epoch, best_loss)
        if stale_epochs >= patience:
            LOGGER.info("Early stopping after %d epochs without improvement; best epoch=%d",
                        stale_epochs, best_epoch)
            break
    if best_state is None:
        raise RuntimeError("Training failed to produce a finite validation loss")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def train_model(raw, validation_start, test_start, output, epochs=20,
                batch_size=1024, learning_rate=1e-3, embedding_dim=16,
                dropout=0.1, patience=3, seed=42, device="cpu", progress=True,
                baseline_c=1.0, baseline_max_iter=1000, weight_decay=1e-4):
    if not np.isfinite(baseline_c) or baseline_c <= 0 or baseline_max_iter < 1:
        raise ValueError("Baseline C and max_iter must be positive")
    training_started = perf_counter()
    LOGGER.info("Preparing %s raw pitches", f"{len(raw):,}")
    frame, cleaning = prepare_pitches(raw)
    train, validation, test = chronological_split(frame, validation_start, test_start)
    player_maps = fit_player_maps(train)
    model_config = {
        "num_batters": len(player_maps["batter"]),
        "num_pitchers": len(player_maps["pitcher"]),
        "embedding_dim": embedding_dim,
        "dropout": dropout,
    }
    LOGGER.info("Data summary: %s", json.dumps({"cleaning": cleaning, "split_sizes": {
        "train": len(train), "validation": len(validation), "test": len(test)
    }}))
    model, history, best_epoch = fit_model(
        make_dataset(train, player_maps), make_dataset(validation, player_maps),
        model_config, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        weight_decay=weight_decay, patience=patience, seed=seed, device=device, progress=progress,
    )
    validation_loader = DataLoader(make_dataset(validation, player_maps), batch_size=batch_size)
    test_loader = DataLoader(make_dataset(test, player_maps), batch_size=batch_size)
    LOGGER.info("Evaluating best model from epoch %d", best_epoch)
    metrics = {
        "best_epoch": best_epoch,
        "validation": evaluate(model, validation_loader, device, progress, "Best validation"),
        "test": evaluate(model, test_loader, device, progress, "Test"),
        "multinomial_logistic_regression_test": logistic_regression_baseline(
            train, test, c=baseline_c, max_iter=baseline_max_iter
        ),
        "test_unknown_players": {
            role: int((~test[role].isin(player_maps[role])).sum())
            for role in ("batter", "pitcher")
        },
    }
    checkpoint = {
        "model_config": model_config,
        "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
        "player_maps": player_maps,
        "outcomes": list(OUTCOMES),
        "description_mapping": DESCRIPTION_TO_OUTCOME,
        "in_play_outcomes": list(IN_PLAY_OUTCOMES),
        "baseline_config": {"type": "multinomial_logistic_regression", "solver": "lbfgs",
                            "C": baseline_c, "max_iter": baseline_max_iter,
                            "features": ["batter", "pitcher", "count"]},
        "validation_start": str(pd.Timestamp(validation_start).date()),
        "test_start": str(pd.Timestamp(test_start).date()),
        "training_config": {"epochs": epochs, "batch_size": batch_size,
                            "learning_rate": learning_rate, "patience": patience,
                            "weight_decay": weight_decay,
                            "seed": seed, "device": str(device)},
        "cleaning": cleaning,
        "history": history,
        "metrics": metrics,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    LOGGER.info("Final metrics: %s", json.dumps(metrics))
    LOGGER.info("Saved model to %s", output.resolve())
    LOGGER.info("Saved metrics to %s", output.with_suffix(".metrics.json").resolve())
    LOGGER.info("Training completed in %.1fs", perf_counter() - training_started)
    model.eval()
    return model, player_maps, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True, help="Statcast Parquet file from load_data.py")
    parser.add_argument("--validation-start", required=True, help="First validation game date, YYYY-MM-DD")
    parser.add_argument("--test-start", required=True, help="First test game date, YYYY-MM-DD")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pitch_model.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu, mps, cuda")
    parser.add_argument("--baseline-c", type=float, default=1.0,
                        help="Logistic regression inverse regularization strength (default: 1)")
    parser.add_argument("--baseline-max-iter", type=int, default=1000,
                        help="Maximum logistic regression iterations (default: 1000)")
    parser.add_argument("--log-file", type=Path, help="Log path; default is OUTPUT with a .log suffix")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="Disable tqdm bars while keeping logs")
    args = vars(parser.parse_args())
    log_file = args.pop("log_file") or args["output"].with_suffix(".log")
    log_level = args.pop("log_level")
    configure_logging(log_file, log_level)
    with logging_redirect_tqdm(loggers=[LOGGER]):
        try:
            parquet_path = args.pop("parquet")
            LOGGER.info("Logging to %s", log_file.resolve())
            LOGGER.info("Reading Parquet data from %s", parquet_path.resolve())
            raw = pd.read_parquet(parquet_path, engine="pyarrow")
            train_model(raw, **args)
        except Exception:
            LOGGER.exception("Training failed")
            raise


if __name__ == "__main__":
    main()
