"""Player embeddings predicting pitch results and individual in-play events."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset


IN_PLAY_OUTCOMES = (
    "single", "double", "triple", "home_run",
    "field_out", "force_out", "grounded_into_double_play", "double_play",
    "triple_play", "field_error", "fielders_choice", "fielders_choice_out",
    "sac_fly", "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play",
    "other_out", "batter_interference", "catcher_interf", "fan_interference",
)
OUTCOMES = (
    "ball", "called_strike", "swinging_strike", "foul", "hit_by_pitch",
    *IN_PLAY_OUTCOMES,
)
DESCRIPTION_TO_OUTCOME = {
    "ball": "ball",
    "blocked_ball": "ball",
    "called_strike": "called_strike",
    "swinging_strike": "swinging_strike",
    "swinging_strike_blocked": "swinging_strike",
    "foul": "foul",
    "foul_tip": "foul",
    "hit_by_pitch": "hit_by_pitch",
}

FEATURES = ("batter", "pitcher", "balls", "strikes")


class PitchOutcomeModel(nn.Module):
    """Two player lookups + 12-count one-hot -> 64 -> 32 -> outcome logits.

    ID zero is a fixed zero vector for players absent from the training set.
    CrossEntropyLoss consumes logits; apply softmax only for probabilities.
    """

    def __init__(self, num_batters, num_pitchers, embedding_dim=16, dropout=0.1):
        super().__init__()
        self.batter_embedding = nn.Embedding(
            num_batters + 1, embedding_dim, padding_idx=0
        )
        self.pitcher_embedding = nn.Embedding(
            num_pitchers + 1, embedding_dim, padding_idx=0
        )
        self.network = nn.Sequential(
            nn.Linear(2 * embedding_dim + 12, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(OUTCOMES)),
        )

    def forward(self, batter, pitcher, count):
        count_one_hot = F.one_hot(count, num_classes=12).to(
            dtype=self.batter_embedding.weight.dtype
        )
        inputs = torch.cat(
            [self.batter_embedding(batter), self.pitcher_embedding(pitcher),
             count_one_hot], dim=-1
        )
        return self.network(inputs)


def numeric_features(frame):
    """Coerce features and identify valid integer IDs and pre-pitch counts."""
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    numeric = frame.loc[:, list(FEATURES)].apply(pd.to_numeric, errors="coerce")
    valid = (numeric.notna() & np.isfinite(numeric) & numeric.mod(1).eq(0)).all(axis=1)
    valid &= numeric.batter.gt(0) & numeric.pitcher.gt(0)
    valid &= numeric.balls.between(0, 3) & numeric.strikes.between(0, 2)
    return numeric, valid


def prepare_pitches(raw):
    """Use descriptions for pitch results and events for hit_into_play rows.

    Bunt descriptions, pitchouts, automatic calls, and unknown descriptions
    are excluded. Foul tips share the coarse foul class; this model does not
    attempt to simulate count transitions. Missing events are allowed for
    non-contact pitches. In-play rows need a supported event; otherwise they
    are excluded and reported separately from unsupported descriptions.
    """
    missing = {"game_date", "description", "events"} - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = raw.copy()
    numeric, valid_features = numeric_features(frame)
    dates = pd.to_datetime(frame.game_date, errors="coerce").dt.normalize()
    outcomes = frame.description.map(DESCRIPTION_TO_OUTCOME)
    in_play = frame.description.eq("hit_into_play").fillna(False)
    outcomes.loc[in_play] = frame.loc[in_play, "events"].where(
        frame.loc[in_play, "events"].isin(IN_PLAY_OUTCOMES)
    )
    valid = valid_features & dates.notna() & outcomes.notna()
    report = {
        "input_rows": len(frame),
        "retained_rows": int(valid.sum()),
        "excluded_rows": int((~valid).sum()),
        "unsupported_descriptions": {
            str(key): int(value)
            for key, value in frame.loc[outcomes.isna() & ~in_play, "description"]
            .fillna("<missing>").value_counts().items()
        },
        "unsupported_in_play_events": {
            str(key): int(value)
            for key, value in frame.loc[in_play & outcomes.isna(), "events"]
            .fillna("<missing>").value_counts().items()
        },
        "invalid_features_or_dates": int((~valid_features | dates.isna()).sum()),
    }
    frame = frame.loc[valid].copy()
    frame[list(FEATURES)] = numeric.loc[valid].astype("int64")
    frame["game_date"] = dates.loc[valid]
    frame["target"] = outcomes.loc[valid].map(
        {name: index for index, name in enumerate(OUTCOMES)}
    ).astype("int64")
    return frame.sort_values("game_date", kind="stable"), report


def chronological_split(frame, validation_start, test_start):
    """Partition by game date; each boundary belongs to the later split."""
    validation_start = pd.Timestamp(validation_start).normalize()
    test_start = pd.Timestamp(test_start).normalize()
    if not validation_start < test_start:
        raise ValueError("validation_start must be earlier than test_start")
    if "game_pk" in frame and frame.groupby("game_pk").game_date.nunique().gt(1).any():
        raise ValueError("A game has multiple game_date values; fix dates before splitting")
    dates = frame.game_date
    splits = (
        frame.loc[dates < validation_start].copy(),
        frame.loc[(dates >= validation_start) & (dates < test_start)].copy(),
        frame.loc[dates >= test_start].copy(),
    )
    if any(part.empty for part in splits):
        raise ValueError("Train, validation, and test splits must each contain pitches")
    return splits


def fit_player_maps(train):
    """Fit encoders on training players only, reserving zero for unknown IDs."""
    return {
        role: {int(player): index for index, player in
               enumerate(sorted(train[role].unique()), start=1)}
        for role in ("batter", "pitcher")
    }


def encode_features(frame, player_maps):
    numeric, valid = numeric_features(frame)
    if not valid.all():
        raise ValueError("Features require positive player IDs and integer counts (0-3, 0-2)")
    numeric = numeric.astype("int64")
    players = [
        torch.from_numpy(
            numeric[role].map(player_maps[role]).fillna(0).to_numpy(dtype=np.int64, copy=True)
        ) for role in ("batter", "pitcher")
    ]
    count = torch.from_numpy(
        (numeric.balls * 3 + numeric.strikes).to_numpy(dtype=np.int64, copy=True)
    )
    return (*players, count)


def make_dataset(frame, player_maps):
    return TensorDataset(
        *encode_features(frame, player_maps),
        torch.from_numpy(frame.target.to_numpy(dtype=np.int64, copy=True)),
    )


def load_checkpoint(path):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = PitchOutcomeModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["player_maps"]


@torch.inference_mode()
def predict_pitches(model, frame, player_maps, batch_size=4096):
    """Return probabilities in input row order; no outcome columns are needed."""
    features = encode_features(frame, player_maps)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        batches = [
            model(*(feature[start:start + batch_size].to(device) for feature in features))
            .softmax(dim=-1).cpu()
            for start in range(0, len(frame), batch_size)
        ]
    finally:
        model.train(was_training)
    return pd.DataFrame(torch.cat(batches).numpy(), index=frame.index, columns=OUTCOMES)
