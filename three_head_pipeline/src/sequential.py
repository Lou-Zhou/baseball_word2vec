"""Train one embedding space across seasons: base season, then frozen-then-fine-tune for each later one.

    python src/sequential.py --seasons 2022 2023 2024 2025 2026 --output-dir runs/sequential <train.py model flags>

Separately trained season models put their vectors in unrelated coordinates.
Here every season shares one network, so a batter's 2023 vector and 2024 vector
mean the same thing dimension by dimension:

  base season (first in --seasons)   full training, like train.py
  each later season, starting from the previous season's heads:
    phase 1  heads frozen; fresh player vectors are fit into the existing space
             (up to --frozen-epochs, early stopping on validation loss)
    phase 2  everything fine-tuned, heads at --head-lr-scale x the vectors' learning
             rate (up to --finetune-epochs). Kept only if it improves validation loss.

Player vectors use only their own season's pitches. Pitch-feature scaling and the
contact / value target scalers are fixed to the base season, so the heads always
see inputs and targets on the same scale. Each season's fastball references and
player IDs are its own.

Outputs (--output-dir):
  report.md                         start here
  pair_metrics.csv                  year-to-year batter continuity, phase-1 and final vectors
  season_metrics.csv                per season: outcome log loss, wOBA readout, cross-decoding drift
  batter_embeddings_by_season.csv   final vectors for every batter-season (mlbam_id, season, ...)
  batter_pairs.csv                  per returning batter: cosine, re-identification rank, readouts
  <season>/                         model.pt, embeddings, batter_value_readout.csv, config.json, history
Each <season>/ folder works with woba_readout.py, batter_neighbors.py and neighbor_figure.py.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from features import OUTCOMES, PITCH_INPUTS, PitchPreprocessor, prepare_pitches, random_game_split
from load_data import season_path
from metrics import markdown_table, summary_metrics
from model import CONTACT_TARGETS, VALUE_COLUMNS, ContactScaler, ThreeHeadModel, ValueScaler
from paths import RUNS_DIR, portable_path, savant_woba
from train import add_model_arguments, encode_split, evaluate_loss, predict, save_embeddings, train_model, value_readout


# --------------------------------------------------------------------------
# Per-season data and models
# --------------------------------------------------------------------------

def load_season(path, args, base=None):
    """Clean, split and encode one season. Later seasons reuse the base season's scaling and target scalers."""
    raw = pd.read_parquet(path)
    missing = (set(CONTACT_TARGETS) | set(VALUE_COLUMNS)) - set(raw.columns)
    if missing:
        raise SystemExit(f"{path} is missing {sorted(missing)}; download it with load_data.py")
    pitches, _ = prepare_pitches(raw)
    for column in list(CONTACT_TARGETS) + list(VALUE_COLUMNS):
        pitches[column] = pd.to_numeric(pitches[column], errors="coerce").astype("float64")
    seasons = pitches.game_date.dt.year.unique()
    if len(seasons) != 1:
        raise SystemExit(f"{path} must hold one season (found {sorted(seasons)})")
    train, valid, test = random_game_split(pitches, args.valid_fraction, args.test_fraction, args.seed)
    preprocessor = PitchPreprocessor.fit(train, min_pitches=args.min_pitches)
    if base is None:
        contact_scaler = ContactScaler.fit(train, CONTACT_TARGETS)
        value_scaler = ValueScaler.fit(train, args.value_target)
    else:
        preprocessor.medians, preprocessor.means, preprocessor.stds = (
            base.preprocessor.medians, base.preprocessor.means, base.preprocessor.stds)
        contact_scaler, value_scaler = base.contact_scaler, base.value_scaler
    device = torch.device(args.device)
    encoded = {name: encode_split(part, preprocessor, contact_scaler, value_scaler, device)
               for name, part in (("train", train), ("valid", valid), ("test", test))}
    return SimpleNamespace(season=int(seasons[0]), path=path, train=train, valid=valid, test=test,
                           preprocessor=preprocessor, contact_scaler=contact_scaler,
                           value_scaler=value_scaler, encoded=encoded)


def model_config(args, data):
    return dict(
        num_batters=len(data.preprocessor.player_maps["batter"]),
        num_pitchers=len(data.preprocessor.player_maps["pitcher"]),
        num_pitch_features=len(PITCH_INPUTS), embedding_dim=args.embedding_dim, hidden_sizes=tuple(args.hidden),
        contact_hidden=tuple(args.contact_hidden), num_contact_targets=len(CONTACT_TARGETS),
        contact_inputs=args.contact_inputs, value_hidden=tuple(args.value_hidden), dropout=args.dropout,
    )


def model_from_previous(previous, config, init_std):
    """New season's model: the previous season's heads, fresh player vectors near the average player."""
    model = ThreeHeadModel(**config)
    heads = {k: v for k, v in previous.state_dict().items() if "embedding" not in k}
    model.load_state_dict(heads, strict=False)
    with torch.no_grad():
        for embedding in (model.batter_embedding, model.pitcher_embedding):
            embedding.weight.normal_(0.0, init_std)
            embedding.weight[0].zero_()
    return model


def set_heads_trainable(model, trainable):
    for name, parameter in model.named_parameters():
        if "embedding" not in name:
            parameter.requires_grad_(trainable)


def settings(args, epochs, head_lr_scale=1.0):
    return SimpleNamespace(epochs=epochs, patience=args.patience, batch_size=args.batch_size, lr=args.lr,
                           weight_decay=args.weight_decay, embedding_weight_decay=args.embedding_weight_decay,
                           contact_weight=args.contact_weight, value_weight=args.value_weight,
                           head_lr_scale=head_lr_scale)


@torch.inference_mode()
def raw_readout(model, data, batter_vectors=None, batter_ids=None):
    """Value-head readout (wOBA scale) against this season's pitch-weighted mean pitcher, per batter."""
    model.eval()
    device = model.batter_embedding.weight.device
    pitcher_ids = pd.Series(data.preprocessor.player_maps["pitcher"]).sort_values()
    weights = data.train.pitcher.value_counts().reindex(pitcher_ids.index).to_numpy(dtype=np.float64)
    pitchers = model.pitcher_embedding(torch.as_tensor(pitcher_ids.to_numpy(), device=device))
    mean_pitcher = (pitchers * torch.as_tensor(weights / weights.sum(), dtype=pitchers.dtype, device=device)[:, None]).sum(0)
    if batter_vectors is None:
        ids = pd.Series(data.preprocessor.player_maps["batter"]).sort_values()
        batter_vectors = model.batter_embedding(torch.as_tensor(ids.to_numpy(), device=device))
        batter_ids = ids.index.to_numpy()
    else:
        batter_vectors = torch.as_tensor(batter_vectors, dtype=torch.float32, device=device)
    raw = model.value_from_vectors(batter_vectors, mean_pitcher.expand_as(batter_vectors)).cpu().numpy()
    return pd.Series(data.value_scaler.decode(raw), index=batter_ids)


def embedding_table(model, data):
    ids = pd.Series(data.preprocessor.player_maps["batter"]).sort_values()
    vectors = model.batter_embedding.weight.detach().cpu().numpy()[ids.to_numpy()]
    table = pd.DataFrame(vectors, columns=[f"dim_{i}" for i in range(vectors.shape[1])])
    table.insert(0, "train_pitches", data.train.batter.value_counts().reindex(ids.index).to_numpy())
    table.insert(0, "mlbam_id", ids.index)
    return table


def save_season(model, data, args, output_dir, history, extra):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.cpu()
    torch.save({
        "model_config": model_config(args, data),
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "preprocessor": data.preprocessor.state_dict(),
        "contact_scaler": data.contact_scaler.state_dict(),
        "value_scaler": data.value_scaler.state_dict(),
        "outcomes": list(OUTCOMES),
    }, output_dir / "model.pt")
    save_embeddings(model, data.preprocessor, data.train, output_dir)
    per_batter, readout = value_readout(model, data.preprocessor, data.value_scaler, data.train, data.season,
                                        args.min_pa, args.folds, args.seed)
    per_batter.to_csv(output_dir / "batter_value_readout.csv", index=False)
    readout.to_csv(output_dir / "value_readout_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "config.json").write_text(json.dumps({
        **{k: v for k, v in vars(args).items() if k not in ("data", "output_dir")},
        "data": [portable_path(data.path)], "seasons": [data.season], "output_dir": portable_path(output_dir),
        **extra}, indent=2, default=str), encoding="utf-8")
    model.to(torch.device(args.device))
    calibrated = readout.set_index(["target", "prediction"]).loc[("woba", "readout_calibrated (out of fold)")]
    return {"readout_woba_rmse": calibrated.rmse, "readout_woba_r": calibrated.pearson_r}


def test_log_loss(model, data):
    return summary_metrics(predict(model, data.encoded["test"])[0], data.test.target.to_numpy())["log_loss"]


# --------------------------------------------------------------------------
# Year-to-year continuity
# --------------------------------------------------------------------------

def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def continuity(x, y, seed):
    """x, y: (batters x dims) for the same batters in two seasons, same coordinates."""
    dims = [float(np.corrcoef(x[:, d], y[:, d])[0, 1]) for d in range(x.shape[1])]
    xc, yc = unit(x - x.mean(0)), unit(y - y.mean(0))
    sims = xc @ yc.T
    own = np.diag(sims)
    shuffled = float((xc * yc[np.random.default_rng(seed).permutation(len(yc))]).sum(1).mean())
    rank = (sims > own[:, None]).sum(1) + 1  # 1 = own next-season vector is the closest
    return {"mean_dimension_r": float(np.mean(dims)), "min_dimension_r": float(np.min(dims)),
            "mean_cosine": float(own.mean()), "shuffled_cosine": shuffled,
            "reidentified_top1": float((rank == 1).mean()), "reidentified_top5": float((rank <= 5).mean()),
            "median_rank_percentile": float(np.median(rank) / len(rank))}, own, rank


def pair_report(seasons, final, frozen, readouts, args):
    """Continuity for each consecutive pair (and base -> each season, to catch compounding drift)."""
    pairs = [(a, b) for a, b in zip(seasons, seasons[1:])] + [(seasons[0], b) for b in seasons[2:]]
    rows, batter_rows = [], []
    for a, b in pairs:
        variants = [("final", final[b])] + ([("phase1 (frozen heads)", frozen[b])] if b in frozen and (a, b) in
                                            list(zip(seasons, seasons[1:])) else [])
        for name, table_b in variants:
            ea = final[a].loc[final[a].train_pitches >= args.pair_min_pitches].set_index("mlbam_id")
            eb = table_b.loc[table_b.train_pitches >= args.pair_min_pitches].set_index("mlbam_id")
            ids = ea.index.intersection(eb.index)
            dims = [c for c in ea.columns if c.startswith("dim_")]
            stats, own, rank = continuity(ea.loc[ids, dims].to_numpy(), eb.loc[ids, dims].to_numpy(), args.seed)
            row = {"pair": f"{a}->{b}", "vectors": name, "batters": len(ids), **stats}
            if name == "final":
                sa, sb = savant_woba(a), savant_woba(b)
                both = [i for i in ids if i in sa.index and i in sb.index
                        and sa.pa.get(i, 0) >= args.min_pa and sb.pa.get(i, 0) >= args.min_pa]
                ra, rb = readouts[a].reindex(both), readouts[b].reindex(both)
                row.update({
                    "woba_batters": len(both),
                    "readout_woba_r": float(np.corrcoef(ra, rb)[0, 1]),
                    "actual_woba_r": float(np.corrcoef(sa.woba.reindex(both), sb.woba.reindex(both))[0, 1]),
                    "actual_xwoba_r": float(np.corrcoef(sa.xwoba.reindex(both), sb.xwoba.reindex(both))[0, 1]),
                    "readout_a_vs_actual_woba_b_r": float(np.corrcoef(ra, sb.woba.reindex(both))[0, 1]),
                })
                batter_rows.append(pd.DataFrame({"pair": f"{a}->{b}", "mlbam_id": ids, "cosine": own,
                                                 "reidentify_rank": rank,
                                                 f"readout_a": readouts[a].reindex(ids).to_numpy(),
                                                 f"readout_b": readouts[b].reindex(ids).to_numpy()}))
            rows.append(row)
    return pd.DataFrame(rows), pd.concat(batter_rows, ignore_index=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", type=int, nargs="+", required=True, help="In order; the first is the base")
    parser.add_argument("--data", type=Path, nargs="+", default=None,
                        help="One file per season, same order (default data/pitches_<season>.parquet)")
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "sequential")
    add_model_arguments(parser)
    seq = parser.add_argument_group("sequential training")
    seq.add_argument("--frozen-epochs", type=int, default=10, help="Phase 1 (heads frozen) epoch limit")
    seq.add_argument("--finetune-epochs", type=int, default=10, help="Phase 2 (full fine-tune) epoch limit")
    seq.add_argument("--head-lr-scale", type=float, default=0.1,
                     help="Heads' learning rate in phase 2, as a multiple of --lr (1 = same as the vectors)")
    seq.add_argument("--init-std", type=float, default=0.01, help="Std of the fresh player vectors")
    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument("--min-pa", type=int, default=300, help="PA for the wOBA comparisons")
    evaluation.add_argument("--folds", type=int, default=5)
    evaluation.add_argument("--pair-min-pitches", type=int, default=1000,
                            help="Training pitches a batter needs in both seasons for the continuity metrics")
    args = parser.parse_args(argv)
    if len(args.seasons) < 2:
        parser.error("Give at least two seasons")
    paths = args.data or [season_path(s) for s in args.seasons]
    if len(paths) != len(args.seasons):
        parser.error("--data needs one file per season")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final, frozen, readouts, season_rows = {}, {}, {}, []
    base = previous = previous_data = None

    for season, path in zip(args.seasons, paths):
        print(f"\n===== {season} =====")
        data = load_season(path, args, base)
        if data.season != season:
            raise SystemExit(f"{path} holds {data.season}, not {season}")
        config = model_config(args, data)
        print(f"  {len(data.train):,} training pitches, {config['num_batters']} batters, "
              f"{config['num_pitchers']} pitchers")
        row = {"season": season, "batters": config["num_batters"]}
        if previous is None:
            print("  base season: full training")
            model = ThreeHeadModel(**config).to(device)
            history = train_model(model, data.encoded["train"], data.encoded["valid"],
                                  settings(args, args.epochs), device)
            row.update({"phase1_epochs": None, "phase2_epochs": len(history), "finetune_kept": None})
        else:
            model = model_from_previous(previous.cpu(), config, args.init_std).to(device)
            print("  phase 1: heads frozen, fitting player vectors")
            set_heads_trainable(model, False)
            history = train_model(model, data.encoded["train"], data.encoded["valid"],
                                  settings(args, args.frozen_epochs), device)
            phase1_loss = evaluate_loss(model, data.encoded["valid"], args)[0]
            phase1_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            frozen[season] = embedding_table(model, data)
            print("  phase 2: fine-tuning everything")
            set_heads_trainable(model, True)
            history2 = train_model(model, data.encoded["train"], data.encoded["valid"],
                                   settings(args, args.finetune_epochs, args.head_lr_scale), device)
            phase2_loss = evaluate_loss(model, data.encoded["valid"], args)[0]
            kept = phase2_loss < phase1_loss
            if not kept:
                model.load_state_dict(phase1_state)
                print("  phase 2 did not improve validation loss; keeping phase 1")
            for h in history2:
                h["phase"] = 2
            for h in history:
                h["phase"] = 1
            history = history + history2
            # Drift check: last season's vectors read out through last season's heads vs this season's heads.
            prev_vectors = embedding_table(previous, previous_data)
            dims = [c for c in prev_vectors.columns if c.startswith("dim_")]
            old = raw_readout(previous.to(device), previous_data)
            new = raw_readout(model, data, prev_vectors[dims].to_numpy(), prev_vectors.mlbam_id.to_numpy())
            row.update({"phase1_epochs": sum(h["phase"] == 1 for h in history),
                        "phase2_epochs": sum(h["phase"] == 2 for h in history), "finetune_kept": bool(kept),
                        "phase1_valid_loss": float(phase1_loss), "phase2_valid_loss": float(phase2_loss),
                        "cross_decode_r": float(np.corrcoef(old, new.reindex(old.index))[0, 1]),
                        "cross_decode_mean_abs_diff": float((old - new.reindex(old.index)).abs().mean())})
        row["test_log_loss"] = test_log_loss(model, data)
        readouts[season] = raw_readout(model, data)
        final[season] = embedding_table(model, data)
        row.update(save_season(model, data, args, args.output_dir / str(season), history,
                               {"sequential_base_season": args.seasons[0]}))
        season_rows.append(row)
        print(f"  test log loss {row['test_log_loss']:.4f}; wOBA readout RMSE {row['readout_woba_rmse']:.4f} "
              f"(r {row['readout_woba_r']:.3f})")
        if base is None:
            base = data
        previous, previous_data = model, data

    # ---- reports ----
    seasons = list(args.seasons)
    pairs, batter_pairs = pair_report(seasons, final, frozen, readouts, args)
    season_table = pd.DataFrame(season_rows)
    pairs.to_csv(args.output_dir / "pair_metrics.csv", index=False)
    batter_pairs.to_csv(args.output_dir / "batter_pairs.csv", index=False)
    season_table.to_csv(args.output_dir / "season_metrics.csv", index=False)
    pd.concat([t.assign(season=s) for s, t in final.items()]).to_csv(
        args.output_dir / "batter_embeddings_by_season.csv", index=False)

    consecutive = pairs.loc[pairs.pair.isin([f"{a}->{b}" for a, b in zip(seasons, seasons[1:])])]
    continuity_cols = ["pair", "vectors", "batters", "mean_dimension_r", "mean_cosine", "shuffled_cosine",
                       "reidentified_top1", "reidentified_top5", "median_rank_percentile"]
    woba_cols = ["pair", "woba_batters", "readout_woba_r", "actual_woba_r", "actual_xwoba_r",
                 "readout_a_vs_actual_woba_b_r"]
    base_drift = pairs.loc[(pairs.vectors == "final") & pairs.pair.str.startswith(f"{seasons[0]}->"),
                           ["pair", "batters", "mean_dimension_r", "mean_cosine", "reidentified_top1"]]
    report = [
        "# Sequential embeddings: frozen, then fine-tuned", "",
        f"Seasons {', '.join(map(str, seasons))}; base season {seasons[0]} trained fully. Each later season "
        f"starts from the previous season's heads: phase 1 fits fresh player vectors with the heads frozen "
        f"(up to {args.frozen_epochs} epochs), phase 2 fine-tunes everything with the heads at "
        f"{args.head_lr_scale} x the learning rate (up to {args.finetune_epochs} epochs).",
        f"Continuity uses batters with >= {args.pair_min_pitches} training pitches in both seasons; the wOBA "
        f"columns use batters with >= {args.min_pa} PA in both.", "",
        "## How to read this", "",
        "- **mean_dimension_r**: correlation of each embedding dimension with itself the next season, averaged.",
        "- **mean_cosine** vs **shuffled_cosine**: how close a batter's two vectors are, against random pairs.",
        "- **reidentified_top1 / top5**: share of batters whose own next-season vector is the closest (top 5) "
        "of all returning batters. Chance is about 1 / batters.",
        "- **phase1 rows** are the vectors before the heads moved: if the final rows are much lower, "
        "fine-tuning drifted the space.",
        "- **readout_woba_r**: the value-head wOBA readout year to year, next to actual wOBA and xwOBA. "
        "**readout_a_vs_actual_woba_b_r** is a no-aging forecast: this year's readout vs next year's wOBA.",
        "- **cross_decode_r** (season table): last season's vectors read out through last season's heads vs "
        "this season's heads. Near 1 = the heads still decode the old vectors the same way.", "",
        "## Year-to-year batter continuity", "", markdown_table(consecutive[continuity_cols]), "",
        "## Year-to-year wOBA", "",
        markdown_table(consecutive.loc[consecutive.vectors == "final", woba_cols]), "",
        f"## Drift from the base season ({seasons[0]} -> each later season)", "", markdown_table(base_drift), "",
        "## Seasons", "", markdown_table(season_table), "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n" + consecutive[continuity_cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n" + consecutive.loc[consecutive.vectors == "final", woba_cols].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n" + base_drift.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n" + season_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nReport: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
