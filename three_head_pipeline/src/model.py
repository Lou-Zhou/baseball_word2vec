"""Three-head pitch model: shared batter and pitcher embeddings feeding three heads.

  outcome head  [batter, pitcher, count one-hot, pitch inputs] -> 10 pitch outcomes   (every pitch)
  contact head  same inputs (or players only)                   -> exit velo, launch angle (balls in play)
  value head    [batter, pitcher] only                          -> PA wOBA value     (PA-ending pitches)

The value head sees nothing but the two player vectors, so whatever it knows
about a batter's production must live in his embedding. Reading a batter's
value out is value_head(batter vector, mean pitcher vector).

Value targets (ValueScaler kind):
  xwoba  balls in play use Statcast's estimated_woba_using_speedangle (falls
         back to woba_value if missing); strikeouts 0; walks/HBP their weight.
         A batter's mean target is his xwOBA: value-scaled, luck removed.
  woba   Statcast woba_value: actual results, batted-ball luck included.
Only PAs with woba_denom == 1 are scored (no IBB, sac bunts, interference).

Embedding ID 0 is a fixed zero vector shared by unknown or rare players.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from features import OUTCOME_INDEX, OUTCOMES, PITCH_INPUTS, PitchPreprocessor

CONTACT_TARGETS = ("launch_speed", "launch_angle")
IN_PLAY_CLASSES = ("out_in_play", "single", "double", "triple", "home_run")
VALUE_COLUMNS = ("woba_value", "woba_denom", "estimated_woba_using_speedangle")


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------

def contact_mask(frame, targets=CONTACT_TARGETS):
    """Balls in play with every contact target measured."""
    in_play = frame.target.isin([OUTCOME_INDEX[c] for c in IN_PLAY_CLASSES])
    measured = frame.loc[:, list(targets)].apply(np.isfinite).all(axis=1)
    return (in_play & measured).to_numpy()


class ContactScaler:
    """Standardizes exit velocity and launch angle with training-split balls in play."""

    def __init__(self, targets, means, stds):
        self.targets, self.means, self.stds = list(targets), means, stds

    @classmethod
    def fit(cls, train, targets=CONTACT_TARGETS):
        rows = train.loc[contact_mask(train, targets), list(targets)]
        return cls(targets, {t: float(rows[t].mean()) for t in targets},
                   {t: float(rows[t].std()) for t in targets})

    def encode(self, frame):
        """(standardized targets with 0 where unmeasured, mask of rows to score)."""
        mask = contact_mask(frame, self.targets)
        values = frame.loc[:, self.targets].to_numpy(dtype=np.float64)
        values = (values - [self.means[t] for t in self.targets]) / [self.stds[t] for t in self.targets]
        values[~mask] = 0.0
        return torch.from_numpy(values.astype(np.float32)), torch.from_numpy(mask)

    def decode(self, values):
        """Standardized predictions (rows x targets) back to mph / degrees."""
        return values * [self.stds[t] for t in self.targets] + [self.means[t] for t in self.targets]

    def state_dict(self):
        return {"targets": self.targets, "means": self.means, "stds": self.stds}

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)


def value_target(frame, kind="xwoba"):
    """(per-pitch wOBA value of the PA ending on that pitch, mask of scored PAs)."""
    woba = frame.woba_value.astype("float64")
    if kind == "xwoba":
        estimated = frame.estimated_woba_using_speedangle.astype("float64")
        in_play = frame.description.eq("hit_into_play") & estimated.notna()
        value = woba.where(~in_play, estimated)
    elif kind == "woba":
        value = woba
    else:
        raise ValueError("kind must be 'xwoba' or 'woba'")
    mask = frame.woba_denom.astype("float64").eq(1) & value.notna()
    return value.fillna(0.0).to_numpy(), mask.to_numpy()


class ValueScaler:
    """Standardizes the PA value target with training-split PAs."""

    def __init__(self, kind, mean, std):
        self.kind, self.mean, self.std = kind, mean, std

    @classmethod
    def fit(cls, train, kind="xwoba"):
        value, mask = value_target(train, kind)
        return cls(kind, float(value[mask].mean()), float(value[mask].std()))

    def encode(self, frame):
        value, mask = value_target(frame, self.kind)
        value = np.where(mask, (value - self.mean) / self.std, 0.0)
        return torch.from_numpy(value.astype(np.float32)), torch.from_numpy(mask)

    def decode(self, values):
        return values * self.std + self.mean

    def state_dict(self):
        return {"kind": self.kind, "mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)


# --------------------------------------------------------------------------
# Model: edit this class to customize the architecture.
# --------------------------------------------------------------------------

def mlp(width, hidden, out, dropout):
    layers = []
    for size in hidden:
        layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
        width = size
    layers.append(nn.Linear(width, out))
    return nn.Sequential(*layers)


class ThreeHeadModel(nn.Module):
    def __init__(self, num_batters, num_pitchers, num_pitch_features=len(PITCH_INPUTS), embedding_dim=16,
                 hidden_sizes=(128, 64), contact_hidden=(64, 32), num_contact_targets=len(CONTACT_TARGETS),
                 contact_inputs="all", value_hidden=(32,), dropout=0.1):
        super().__init__()
        if contact_inputs not in ("all", "players"):
            raise ValueError("contact_inputs must be 'all' or 'players'")
        self.contact_inputs = contact_inputs
        self.batter_embedding = nn.Embedding(num_batters + 1, embedding_dim, padding_idx=0)
        self.pitcher_embedding = nn.Embedding(num_pitchers + 1, embedding_dim, padding_idx=0)
        full_width = 2 * embedding_dim + 12 + num_pitch_features
        self.outcome_head = mlp(full_width, hidden_sizes, len(OUTCOMES), dropout)
        contact_width = full_width if contact_inputs == "all" else 2 * embedding_dim
        self.contact_head = mlp(contact_width, contact_hidden, num_contact_targets, dropout)
        self.value_head = mlp(2 * embedding_dim, value_hidden, 1, dropout)

    def outcome_logits_from_vectors(self, batter_vec, pitcher_vec, count, pitch):
        inputs = [batter_vec, pitcher_vec, F.one_hot(count, num_classes=12).to(pitch.dtype), pitch]
        return self.outcome_head(torch.cat(inputs, dim=-1))

    def value_from_vectors(self, batter_vec, pitcher_vec):
        return self.value_head(torch.cat([batter_vec, pitcher_vec], dim=-1)).squeeze(-1)

    def forward(self, batter, pitcher, count, pitch):
        b, p = self.batter_embedding(batter), self.pitcher_embedding(pitcher)
        full = torch.cat([b, p, F.one_hot(count, num_classes=12).to(pitch.dtype), pitch], dim=-1)
        contact_input = full if self.contact_inputs == "all" else torch.cat([b, p], dim=-1)
        return self.outcome_head(full), self.contact_head(contact_input), self.value_from_vectors(b, p)


def three_head_loss(logits, contact, value, target, contact_target, contact_mask_, value_target_,
                    value_mask, contact_weight, value_weight):
    """CE (every pitch) + contact_weight x MSE (balls in play) + value_weight x MSE (PA-ending pitches).

    Both MSEs are on standardized targets. Returns (total, ce, contact_mse, value_mse).
    """
    ce = F.cross_entropy(logits, target)
    zero = logits.new_zeros(())
    c_mse = F.mse_loss(contact[contact_mask_], contact_target[contact_mask_]) if contact_mask_.any() else zero
    v_mse = F.mse_loss(value[value_mask], value_target_[value_mask]) if value_mask.any() else zero
    return ce + contact_weight * c_mse + value_weight * v_mse, ce, c_mse, v_mse


def load_checkpoint(path):
    """(model, preprocessor, contact scaler, value scaler) from a run's model.pt."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    config = dict(checkpoint["model_config"])
    for key in ("hidden_sizes", "contact_hidden", "value_hidden"):
        config[key] = tuple(config[key])
    model = ThreeHeadModel(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return (model, PitchPreprocessor.from_state_dict(checkpoint["preprocessor"]),
            ContactScaler.from_state_dict(checkpoint["contact_scaler"]),
            ValueScaler.from_state_dict(checkpoint["value_scaler"]))
