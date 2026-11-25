"""Train the three-head pitch model on one season and write an evaluation report.

    python src/train.py --data data/pitches_2024.parquet --output-dir runs/2024

Head 1: pitch outcome (10 classes, every pitch).
Head 2: exit velocity + launch angle (balls in play).
Head 3: plate-appearance wOBA value from [batter, pitcher] vectors only (see model.py).

The split holds out a random 5% / 5% of games for validation / test. Training
uses AdamW with early stopping on the validation loss.

Outputs (--output-dir, default runs/<timestamp>):
  report.md                  start here
  metrics.csv                head 1: log loss, Brier, accuracy, calibration, decision log losses
  contact_metrics.csv        head 2: EV / LA errors, baselines and batter / pitcher ablations
  batter_contact_check.csv   per-batter average EV / LA on held-out balls in play
  value_metrics.csv          head 3: per-PA errors, baselines and ablations
  batter_value_readout.csv   value_head(batter, mean pitcher) vs Savant season wOBA / xwOBA
  value_readout_metrics.csv  RMSE / r of that readout
  batter_embeddings.csv, pitcher_embeddings.csv, model.pt, config.json,
  training_history.csv, loss_curves.png, calibration.csv, calibration.png, outcome_rates.csv
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from features import CONTINUOUS_FEATURES, OUTCOMES, PITCH_INPUTS, PitchPreprocessor, prepare_pitches, random_game_split
from metrics import (calibrated_out_of_fold, calibration_table, errors, make_plots, markdown_table,
                     mean_out_of_fold, rate_table, summary_metrics)
from model import CONTACT_TARGETS, VALUE_COLUMNS, ContactScaler, ThreeHeadModel, ValueScaler, three_head_loss
from paths import RUNS_DIR, portable_path, savant_woba

LOSS_NAMES = ("total", "outcome_ce", "contact_mse", "value_mse")
UNITS = {"launch_speed": "mph", "launch_angle": "deg"}


# --------------------------------------------------------------------------
# Training and prediction
# --------------------------------------------------------------------------

def encode_split(frame, preprocessor, contact_scaler, value_scaler, device):
    """[batter, pitcher, count, pitch, target, contact, contact_mask, value, value_mask]"""
    batter, pitcher, count, pitch = preprocessor.encode(frame)
    contact, contact_mask = contact_scaler.encode(frame)
    value, value_mask = value_scaler.encode(frame)
    target = torch.from_numpy(frame.target.to_numpy())
    return [t.to(device) for t in (batter, pitcher, count, pitch, target, contact, contact_mask, value, value_mask)]


def batch_loss(model, b, args):
    logits, contact, value = model(*b[:4])
    return three_head_loss(logits, contact, value, b[4], b[5], b[6], b[7], b[8],
                           args.contact_weight, args.value_weight)


@torch.inference_mode()
def predict(model, data, batch_size=16384, drop=None):
    """(outcome probabilities, standardized contact, standardized value); drop zeroes 'batter' or 'pitcher'."""
    model.eval()
    batter, pitcher, count, pitch = data[:4]
    if drop == "batter":
        batter = torch.zeros_like(batter)
    elif drop == "pitcher":
        pitcher = torch.zeros_like(pitcher)
    out = [[], [], []]
    for start in range(0, len(batter), batch_size):
        part = slice(start, start + batch_size)
        logits, contact, value = model(batter[part], pitcher[part], count[part], pitch[part])
        for store, x in zip(out, (logits.softmax(-1), contact, value)):
            store.append(x.cpu())
    return tuple(torch.cat(o).numpy().astype("float64") for o in out)


@torch.inference_mode()
def evaluate_loss(model, data, args, batch_size=16384):
    """Per-row averages: CE over pitches, contact MSE over balls in play, value MSE over PAs."""
    model.eval()
    sums, counts = np.zeros(3), np.zeros(3)
    for start in range(0, len(data[0]), batch_size):
        b = [t[start:start + batch_size] for t in data]
        _, ce, c_mse, v_mse = batch_loss(model, b, args)
        n = np.array([len(b[4]), int(b[6].sum()), int(b[8].sum())])
        sums += np.array([ce.item(), c_mse.item(), v_mse.item()]) * n
        counts += n
    ce, c_mse, v_mse = sums / np.maximum(counts, 1)
    return ce + args.contact_weight * c_mse + args.value_weight * v_mse, ce, c_mse, v_mse


def train_model(model, train, valid, args, device, on_epoch=None):
    """AdamW with early stopping on validation total loss; restores the best epoch.

    on_epoch(epoch, model), if given, runs after every epoch (tune.py uses it to prune trials).
    Only parameters with requires_grad are trained. args.head_lr_scale (default 1) multiplies the
    learning rate of the heads relative to the embeddings (sequential.py fine-tunes heads slowly).
    """
    embedding_params = [p for n, p in model.named_parameters() if "embedding" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "embedding" not in n and p.requires_grad]
    head_lr = args.lr * getattr(args, "head_lr_scale", 1.0)
    groups = [{"params": embedding_params, "weight_decay": args.embedding_weight_decay}]
    if other_params:
        groups.append({"params": other_params, "weight_decay": args.weight_decay, "lr": head_lr})
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    history, best_loss, best_state, stale = [], float("inf"), None, 0
    n = len(train[0])
    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()
        sums = np.zeros(4)
        for index in torch.randperm(n, device=device).split(args.batch_size):
            loss, ce, c_mse, v_mse = batch_loss(model, [t[index] for t in train], args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sums += np.array([loss.item(), ce.item(), c_mse.item(), v_mse.item()]) * len(index)
        train_losses = sums / n
        valid_losses = evaluate_loss(model, valid, args)
        scheduler.step(valid_losses[0])
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in zip(LOSS_NAMES, train_losses)})
        row.update({f"valid_{k}": v for k, v in zip(LOSS_NAMES, valid_losses)})
        row.update({"lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - start_time})
        history.append(row)
        print(f"  epoch {epoch:3d}  valid total {valid_losses[0]:.4f}  outcome CE {valid_losses[1]:.4f}  "
              f"contact MSE {valid_losses[2]:.4f}  value MSE {valid_losses[3]:.4f}  ({row['seconds']:.0f}s)")
        if valid_losses[0] < best_loss - 1e-4:
            best_loss, stale = valid_losses[0], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"  early stop; best valid total {best_loss:.4f}")
                break
        if on_epoch is not None:
            on_epoch(epoch, model)
    model.load_state_dict(best_state)
    return history


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def ablated_predictions(model, data):
    return {name: predict(model, data, drop=drop)
            for name, drop in (("model", None), ("no_batter_vector", "batter"), ("no_pitcher_vector", "pitcher"))}


def contact_metrics(frame, data, preds, scaler, split):
    """EV / LA errors on balls in play; *_vector_gain = how much worse without that player's vector."""
    mask = data[6].cpu().numpy()
    actual = frame.loc[mask, scaler.targets].to_numpy(dtype=np.float64)
    predicted = {name: scaler.decode(p[1][mask]) for name, p in preds.items()}
    rows = []
    for k, target in enumerate(scaler.targets):
        rmse = {name: float(np.sqrt(((p[:, k] - actual[:, k]) ** 2).mean())) for name, p in predicted.items()}
        rows.append({
            "split": split, "target": target, "units": UNITS.get(target, ""), "balls_in_play": int(mask.sum()),
            "rmse_model": rmse["model"],
            "rmse_train_mean": float(np.sqrt(((scaler.means[target] - actual[:, k]) ** 2).mean())),
            "batter_vector_gain": rmse["no_batter_vector"] - rmse["model"],
            "pitcher_vector_gain": rmse["no_pitcher_vector"] - rmse["model"],
            "pearson_r": float(np.corrcoef(predicted["model"][:, k], actual[:, k])[0, 1]),
        })
    return rows, predicted["model"], actual


def batter_contact_check(batters, predicted, actual, scaler, min_balls):
    """Per-batter average EV / LA on held-out balls in play: does the model rank batters' contact correctly?"""
    table = pd.DataFrame({"mlbam_id": batters})
    for k, target in enumerate(scaler.targets):
        table[f"actual_{target}"] = actual[:, k]
        table[f"predicted_{target}"] = predicted[:, k]
    grouped = table.groupby("mlbam_id")
    per_batter = grouped.mean()
    per_batter.insert(0, "balls_in_play", grouped.size())
    per_batter = per_batter.loc[per_batter.balls_in_play >= min_balls]
    summary = [{"target": t, "batters": len(per_batter),
                "correlation": float(per_batter[f"predicted_{t}"].corr(per_batter[f"actual_{t}"]))}
               for t in scaler.targets]
    return per_batter.reset_index(), summary


def value_metrics(data, preds, scaler, split):
    """Per-PA errors in wOBA units. Single PAs are mostly noise, so gains here are small by nature."""
    mask = data[8].cpu().numpy()
    actual = scaler.decode(data[7].cpu().numpy()[mask])
    predicted = {name: scaler.decode(p[2][mask]) for name, p in preds.items()}
    rmse = {name: float(np.sqrt(((p - actual) ** 2).mean())) for name, p in predicted.items()}
    return {
        "split": split, "target": scaler.kind, "plate_appearances": int(mask.sum()),
        "rmse_model": rmse["model"], "rmse_train_mean": float(np.sqrt(((scaler.mean - actual) ** 2).mean())),
        "batter_vector_gain": rmse["no_batter_vector"] - rmse["model"],
        "pitcher_vector_gain": rmse["no_pitcher_vector"] - rmse["model"],
        "pearson_r": float(np.corrcoef(predicted["model"], actual)[0, 1]),
        "prediction_sd": float(predicted["model"].std()),
    }


@torch.inference_mode()
def value_readout(model, preprocessor, scaler, train, season, min_pa, folds, seed):
    """value_head(batter vector, pitch-weighted mean pitcher vector) vs Savant season wOBA / xwOBA."""
    model.eval()
    batter_ids = pd.Series(preprocessor.player_maps["batter"]).sort_values()
    pitcher_ids = pd.Series(preprocessor.player_maps["pitcher"]).sort_values()
    weights = train.pitcher.value_counts().reindex(pitcher_ids.index).to_numpy(dtype=np.float64)
    pitcher_vectors = model.pitcher_embedding.weight[torch.as_tensor(pitcher_ids.to_numpy())]
    mean_pitcher = (pitcher_vectors * torch.as_tensor(weights / weights.sum(),
                                                      dtype=pitcher_vectors.dtype)[:, None]).sum(0)
    batter_vectors = model.batter_embedding.weight[torch.as_tensor(batter_ids.to_numpy())]
    raw = scaler.decode(model.value_from_vectors(batter_vectors, mean_pitcher.expand_as(batter_vectors)).numpy())
    everyone = pd.DataFrame({"mlbam_id": batter_ids.index, "readout": raw.astype(np.float64)}).set_index("mlbam_id")
    everyone = everyone.join(savant_woba(season), how="left")
    everyone["qualified"] = (everyone.pa >= min_pa) & everyone.woba.notna() & everyone.xwoba.notna()
    table = everyone.loc[everyone.qualified].copy()
    rows = []
    for target in ("woba", "xwoba"):
        actual = table[target].to_numpy()
        calibrated = calibrated_out_of_fold(table.readout.to_numpy(), actual, folds, seed)
        # Qualified batters: out-of-fold calibration (scored below). Everyone else: the line fit on
        # all qualified batters, since none of them are in that fit.
        b, a = np.polyfit(table.readout.to_numpy(), actual, 1)
        column = f"readout_calibrated_{target}"
        everyone[column] = a + b * everyone.readout
        everyone.loc[everyone.qualified, column] = calibrated
        for name, pred in (("league_mean (out of fold)", mean_out_of_fold(actual, folds, seed)),
                           ("readout_raw", table.readout.to_numpy()),
                           ("readout_calibrated (out of fold)", calibrated)):
            rows.append({"target": target, "prediction": name, "batters": len(table), **errors(pred, actual)})
    return everyone.reset_index(), pd.DataFrame(rows)


def save_embeddings(model, preprocessor, train, output_dir):
    for role, embedding in (("batter", model.batter_embedding), ("pitcher", model.pitcher_embedding)):
        ids = pd.Series(preprocessor.player_maps[role]).sort_values()
        vectors = embedding.weight.detach().cpu().numpy()[ids.to_numpy()]
        table = pd.DataFrame(vectors, columns=[f"dim_{i}" for i in range(vectors.shape[1])])
        table.insert(0, "train_pitches", train[role].value_counts().reindex(ids.index).to_numpy())
        table.insert(0, "mlbam_id", ids.index)
        table.to_csv(output_dir / f"{role}_embeddings.csv", index=False)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def add_model_arguments(parser):
    """Split, model and training options shared by train.py and sequential.py."""
    parser.add_argument("--valid-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--min-pitches", type=int, default=100,
                        help="Training pitches a player needs for his own vector (others share ID 0)")
    model_args = parser.add_argument_group("model")
    model_args.add_argument("--embedding-dim", type=int, default=16)
    model_args.add_argument("--hidden", type=int, nargs="+", default=[128, 64], help="Outcome head layers")
    model_args.add_argument("--contact-hidden", type=int, nargs="+", default=[64, 32], help="Contact head layers")
    model_args.add_argument("--contact-inputs", choices=["all", "players"], default="all",
                            help="all: contact head also sees count and pitch inputs; players: vectors only")
    model_args.add_argument("--contact-weight", type=float, default=1.0, help="Weight on the contact MSE")
    model_args.add_argument("--value-hidden", type=int, nargs="*", default=[32],
                            help="Value head layers (none = linear in the two vectors)")
    model_args.add_argument("--value-target", choices=["xwoba", "woba"], default="xwoba")
    model_args.add_argument("--value-weight", type=float, default=1.0,
                            help="Weight on the value MSE (0 turns the value head off)")
    model_args.add_argument("--dropout", type=float, default=0.1)
    train_args = parser.add_argument_group("training")
    train_args.add_argument("--epochs", type=int, default=30)
    train_args.add_argument("--patience", type=int, default=3)
    train_args.add_argument("--batch-size", type=int, default=2048)
    train_args.add_argument("--lr", type=float, default=2e-3)
    train_args.add_argument("--weight-decay", type=float, default=1e-4)
    train_args.add_argument("--embedding-weight-decay", type=float, default=1e-3)
    train_args.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_args.add_argument("--seed", type=int, default=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True, help="Parquet file(s) from load_data.py")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: runs/<timestamp>")
    add_model_arguments(parser)
    eval_args = parser.add_argument_group("evaluation")
    eval_args.add_argument("--min-batter-balls", type=int, default=15,
                           help="Held-out balls in play a batter needs for the per-batter contact check")
    eval_args.add_argument("--min-pa", type=int, default=300, help="Minimum PA for the season wOBA readout")
    eval_args.add_argument("--folds", type=int, default=5, help="Folds for the readout's out-of-fold calibration")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir or RUNS_DIR / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to {output_dir}")

    # ---- data ----
    raw = pd.concat([pd.read_parquet(path) for path in args.data], ignore_index=True)
    missing = (set(CONTACT_TARGETS) | set(VALUE_COLUMNS)) - set(raw.columns)
    if missing:
        parser.error(f"Data is missing {sorted(missing)}; download it with load_data.py")
    pitches, clean_report = prepare_pitches(raw)
    for column in list(CONTACT_TARGETS) + list(VALUE_COLUMNS):
        pitches[column] = pd.to_numeric(pitches[column], errors="coerce").astype("float64")
    train, valid, test = random_game_split(pitches, args.valid_fraction, args.test_fraction, args.seed)
    splits = {"train": train, "valid": valid, "test": test}
    seasons = sorted(int(s) for s in pitches.game_date.dt.year.unique())

    preprocessor = PitchPreprocessor.fit(train, min_pitches=args.min_pitches)
    contact_scaler = ContactScaler.fit(train, CONTACT_TARGETS)
    value_scaler = ValueScaler.fit(train, args.value_target)
    encoded = {name: encode_split(part, preprocessor, contact_scaler, value_scaler, device)
               for name, part in splits.items()}
    for name, part in splits.items():
        print(f"  {name:5s} {len(part):>9,} pitches, {int(encoded[name][6].sum()):>7,} balls in play, "
              f"{int(encoded[name][8].sum()):>7,} scored PAs")
    num_batters = len(preprocessor.player_maps["batter"])
    num_pitchers = len(preprocessor.player_maps["pitcher"])

    # ---- train ----
    model_config = dict(
        num_batters=num_batters, num_pitchers=num_pitchers, num_pitch_features=len(PITCH_INPUTS),
        embedding_dim=args.embedding_dim, hidden_sizes=tuple(args.hidden), contact_hidden=tuple(args.contact_hidden),
        num_contact_targets=len(CONTACT_TARGETS), contact_inputs=args.contact_inputs,
        value_hidden=tuple(args.value_hidden), dropout=args.dropout,
    )
    model = ThreeHeadModel(**model_config).to(device)
    print(f"Training (contact weight {args.contact_weight}, value weight {args.value_weight}, "
          f"value target {args.value_target}) on {device}...")
    history = train_model(model, encoded["train"], encoded["valid"], args, device)
    torch.save({
        "model_config": model_config,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "preprocessor": preprocessor.state_dict(),
        "contact_scaler": contact_scaler.state_dict(),
        "value_scaler": value_scaler.state_dict(),
        "outcomes": list(OUTCOMES),
    }, output_dir / "model.pt")

    # ---- evaluation on validation and test ----
    metrics, calibration, rates, contact_rows, value_rows, held_out = [], [], [], [], [], []
    for split in ("valid", "test"):
        preds = ablated_predictions(model, encoded[split])
        y = splits[split].target.to_numpy()
        metrics.append({"split": split, **summary_metrics(preds["model"][0], y)})
        calibration += calibration_table(preds["model"][0], y, split)
        rates += rate_table(preds["model"][0], y, split)
        rows, predicted, actual = contact_metrics(splits[split], encoded[split], preds, contact_scaler, split)
        contact_rows += rows
        held_out.append((splits[split].batter.to_numpy()[encoded[split][6].cpu().numpy()], predicted, actual))
        value_rows.append(value_metrics(encoded[split], preds, value_scaler, split))
    metrics, contact, value = pd.DataFrame(metrics), pd.DataFrame(contact_rows), pd.DataFrame(value_rows)
    # Validation + test pooled: a 5% test split alone has too few balls in play per batter.
    per_batter, batter_summary = batter_contact_check(
        np.concatenate([h[0] for h in held_out]), np.concatenate([h[1] for h in held_out]),
        np.concatenate([h[2] for h in held_out]), contact_scaler, args.min_batter_balls)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    contact.to_csv(output_dir / "contact_metrics.csv", index=False)
    per_batter.to_csv(output_dir / "batter_contact_check.csv", index=False)
    value.to_csv(output_dir / "value_metrics.csv", index=False)
    pd.DataFrame(calibration).to_csv(output_dir / "calibration.csv", index=False)
    pd.DataFrame(rates).to_csv(output_dir / "outcome_rates.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    make_plots(history, calibration, output_dir)

    # ---- season readout and embeddings ----
    model.cpu()
    readout, readout_text = None, ["Skipped: the data spans several seasons."]
    if len(seasons) == 1:
        per_batter_value, readout = value_readout(model, preprocessor, value_scaler, train, seasons[0],
                                                  args.min_pa, args.folds, args.seed)
        per_batter_value.to_csv(output_dir / "batter_value_readout.csv", index=False)
        readout.to_csv(output_dir / "value_readout_metrics.csv", index=False)
        readout_text = [markdown_table(readout)]
    save_embeddings(model, preprocessor, train, output_dir)

    # ---- report ----
    headline = metrics.loc[:, ["split", "rows", "log_loss", "brier", "accuracy", "top_label_ece"]]
    decisions = metrics.loc[:, ["split", "swing_log_loss", "whiff_given_swing_log_loss",
                                "called_strike_given_take_log_loss", "hit_given_in_play_log_loss"]]
    report = [
        "# Three-head model report", "",
        f"Data: {', '.join(portable_path(p) for p in args.data)} (seasons {', '.join(map(str, seasons))})",
        f"Train {len(train):,} / valid {len(valid):,} / test {len(test):,} pitches; random "
        f"{args.valid_fraction:.0%} / {args.test_fraction:.0%} of games held out for validation / test.",
        f"Player vectors ({args.embedding_dim} dims) for {num_batters} batters and {num_pitchers} pitchers "
        f"with at least {args.min_pitches} training pitches; {len(history)} epochs.",
        f"Loss = outcome CE + {args.contact_weight} x contact MSE + {args.value_weight} x value MSE "
        f"(value target: {args.value_target}).", "",
        "## How to read this", "",
        "- **Head 1 (pitch outcome)**: log loss is the main number; ECE near 0 means calibrated. The "
        "decision log losses split it into swing, whiff, called strike and hit-on-contact.",
        "- **Head 2 (EV / LA)**: `batter_vector_gain` is how much worse the prediction gets when the "
        "batter's vector is replaced by the unknown-player vector: the contact signal in the embedding.",
        "- **Head 3 (PA value)**: single PAs are mostly noise, so `rmse_model` barely beats "
        "`rmse_train_mean`; `prediction_sd` shows how much batter-to-batter spread it learned.",
        "- **Season readout**: value_head(batter vector, mean pitcher vector) against Savant season wOBA / "
        "xwOBA. `readout_calibrated` is a + b x readout fit out of fold. Note the vectors were trained on "
        "most of these plate appearances, so this measures how well the embedding encodes the season, "
        "not a forecast.", "",
        "## Head 1: pitch outcome", "", markdown_table(headline), "", markdown_table(decisions), "",
        "## Head 2: exit velocity and launch angle (balls in play)", "", markdown_table(contact), "",
        f"Per-batter average EV / LA on held-out balls in play (batters with >= {args.min_batter_balls}):", "",
        markdown_table(pd.DataFrame(batter_summary)), "",
        "## Head 3: plate-appearance value", "", markdown_table(value), "",
        f"## Season readout (batters with >= {args.min_pa} PA)", "", *readout_text, "",
        "## Data cleaning", "", "```json", json.dumps(clean_report, indent=2), "```",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(
        {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         "data": [portable_path(p) for p in args.data], "output_dir": portable_path(output_dir),
         "seasons": seasons, "continuous_features": list(CONTINUOUS_FEATURES)}, indent=2), encoding="utf-8")

    print("\n" + headline.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n" + contact.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n" + value.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if readout is not None:
        print("\n" + readout.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nFull report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
