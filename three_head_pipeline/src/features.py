"""Pitch cleaning, outcome labels, model inputs and the train/validation/test split.

prepare_pitches(raw) -> random_game_split -> PitchPreprocessor.fit(train) -> encode(frame)
"""

import numpy as np
import pandas as pd
import torch

OUTCOMES = (
    "ball", "called_strike", "swinging_strike", "foul", "hit_by_pitch",
    "out_in_play", "single", "double", "triple", "home_run",
)
OUTCOME_INDEX = {name: index for index, name in enumerate(OUTCOMES)}
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
# Errors and fielder's choices count as outs: the batter did not earn a hit.
# Interference events are rare and not about the pitch, so they are excluded.
IN_PLAY_EVENT_TO_OUTCOME = {
    "single": "single",
    "double": "double",
    "triple": "triple",
    "home_run": "home_run",
    **{event: "out_in_play" for event in (
        "field_out", "force_out", "grounded_into_double_play", "double_play",
        "triple_play", "field_error", "fielders_choice", "fielders_choice_out",
        "sac_fly", "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play",
        "other_out",
    )},
}

ID_FEATURES = ("batter", "pitcher", "balls", "strikes")
HAND_FEATURES = ("stand", "p_throws")
# Rows missing any of these are dropped.
REQUIRED_PITCH_FEATURES = (
    "release_speed", "release_pos_x", "release_pos_z",
    "pfx_x", "pfx_z", "plate_x", "plate_z", "sz_top", "sz_bot",
)
# Missing values are imputed with the training median and flagged.
OPTIONAL_PITCH_FEATURES = ("release_spin_rate", "spin_axis", "release_extension")

FASTBALL_TYPES = ("FF", "SI")
MIN_FASTBALLS = 20

CONTINUOUS_FEATURES = (
    "release_speed", "release_pos_x", "release_pos_z", "release_extension",
    "release_spin_rate", "spin_axis_sin", "spin_axis_cos",
    "pfx_x", "pfx_z",
    "plate_x", "plate_z", "plate_z_in_zone",
    "speed_minus_fastball", "pfx_x_minus_fastball", "pfx_z_minus_fastball",
)
BINARY_FEATURES = (
    "batter_bats_right", "pitcher_throws_right", "same_handed",
    "release_spin_rate_missing", "spin_axis_missing", "release_extension_missing",
)
PITCH_INPUTS = CONTINUOUS_FEATURES + BINARY_FEATURES


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def numeric_features(frame):
    """Coerce ID/count columns and identify valid integer IDs and pre-pitch counts."""
    missing = set(ID_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    numeric = frame.loc[:, list(ID_FEATURES)].apply(pd.to_numeric, errors="coerce")
    valid = (numeric.notna() & np.isfinite(numeric) & numeric.mod(1).eq(0)).all(axis=1)
    valid &= numeric.batter.gt(0) & numeric.pitcher.gt(0)
    valid &= numeric.balls.between(0, 3) & numeric.strikes.between(0, 2)
    return numeric, valid


def pitch_features_valid(frame):
    """Rows with usable handedness, required pitch measurements and strike zone."""
    missing = set(HAND_FEATURES + REQUIRED_PITCH_FEATURES + OPTIONAL_PITCH_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    required = frame.loc[:, list(REQUIRED_PITCH_FEATURES)].apply(pd.to_numeric, errors="coerce")
    valid = (required.notna() & np.isfinite(required)).all(axis=1)
    valid &= required.sz_top.gt(required.sz_bot)
    for column in HAND_FEATURES:
        valid &= frame[column].isin(("L", "R"))
    return valid


def prepare_pitches(raw):
    """Clean raw Statcast rows into model-ready pitches with an ``outcome`` and ``target`` column.

    Pitch results come from ``description``; hit_into_play rows use ``events``.
    Bunt descriptions, pitchouts, automatic calls, interference and unknown
    descriptions are excluded, as are rows missing required pitch measurements
    and non-regular-season games. Foul tips share the foul class. Other columns
    (contact and value targets) are kept. Returns (frame sorted by date, report).
    """
    missing = {"game_date", "description", "events"} - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = raw.copy()
    numeric, valid_ids = numeric_features(frame)
    valid_pitch = pitch_features_valid(frame)
    regular = frame.game_type.eq("R") if "game_type" in frame else pd.Series(True, index=frame.index)
    dates = pd.to_datetime(frame.game_date, errors="coerce").dt.normalize()
    outcomes = frame.description.map(DESCRIPTION_TO_OUTCOME)
    in_play = frame.description.eq("hit_into_play").fillna(False)
    outcomes.loc[in_play] = frame.loc[in_play, "events"].map(IN_PLAY_EVENT_TO_OUTCOME)
    valid = valid_ids & valid_pitch & regular & dates.notna() & outcomes.notna()
    report = {
        "input_rows": len(frame),
        "retained_rows": int(valid.sum()),
        "excluded_rows": int((~valid).sum()),
        "not_regular_season": int((~regular).sum()),
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
        "invalid_ids_counts_or_dates": int((~valid_ids | dates.isna()).sum()),
        "missing_pitch_measurements": int((~valid_pitch).sum()),
    }
    frame = frame.loc[valid].copy()
    frame[list(ID_FEATURES)] = numeric.loc[valid].astype("int64")
    for column in REQUIRED_PITCH_FEATURES + OPTIONAL_PITCH_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame["game_date"] = dates.loc[valid]
    frame["outcome"] = outcomes.loc[valid]
    frame["target"] = frame.outcome.map(OUTCOME_INDEX).astype("int64")
    return frame.sort_values("game_date", kind="stable"), report


def random_game_split(frame, valid_fraction, test_fraction, seed):
    """Hold out whole games at random from across the season.

    Splitting by game keeps pitches from the same at-bat together, so the
    held-out pitches are not near-copies of training pitches.
    """
    games = frame.game_pk.drop_duplicates().sample(frac=1.0, random_state=seed).to_numpy()
    n_valid, n_test = int(len(games) * valid_fraction), int(len(games) * test_fraction)
    valid_games, test_games = set(games[:n_valid]), set(games[n_valid:n_valid + n_test])
    in_valid, in_test = frame.game_pk.isin(valid_games), frame.game_pk.isin(test_games)
    splits = frame.loc[~in_valid & ~in_test].copy(), frame.loc[in_valid].copy(), frame.loc[in_test].copy()
    if any(part.empty for part in splits):
        raise ValueError("Train, validation, and test splits must each contain pitches")
    return splits


# --------------------------------------------------------------------------
# Model inputs
# --------------------------------------------------------------------------

def fastball_reference(train):
    """Average fastball speed and movement per pitcher, with per-hand fallbacks."""
    fastballs = train.loc[train.pitch_type.isin(FASTBALL_TYPES)]
    columns = ["release_speed", "pfx_x", "pfx_z"]
    by_hand = fastballs.groupby("p_throws")[columns].mean()
    grouped = fastballs.groupby("pitcher")[columns]
    per_pitcher = grouped.mean().loc[grouped.size() >= MIN_FASTBALLS]
    return {
        "pitcher": {int(p): [float(v) for v in row] for p, row in zip(per_pitcher.index, per_pitcher.to_numpy())},
        "hand": {str(h): [float(v) for v in row] for h, row in zip(by_hand.index, by_hand.to_numpy())},
    }


def derive_pitch_features(frame, fastball_ref):
    """Model-input columns before imputation and scaling (NaN where missing)."""
    out = pd.DataFrame(index=frame.index)
    for column in ("release_speed", "release_pos_x", "release_pos_z", "release_extension",
                   "release_spin_rate", "pfx_x", "pfx_z", "plate_x", "plate_z"):
        out[column] = frame[column].astype("float64")
    radians = np.deg2rad(frame.spin_axis.astype("float64"))
    out["spin_axis_sin"] = np.sin(radians)
    out["spin_axis_cos"] = np.cos(radians)
    # 0 = bottom of this batter's zone, 1 = top.
    out["plate_z_in_zone"] = (frame.plate_z - frame.sz_bot) / (frame.sz_top - frame.sz_bot)

    hand_ref = frame.p_throws.map(fastball_ref["hand"])
    pitcher_ref = frame.pitcher.map(fastball_ref["pitcher"])
    reference = pitcher_ref.where(pitcher_ref.notna(), hand_ref)
    reference = np.array([r if isinstance(r, list) else [np.nan] * 3 for r in reference],
                         dtype="float64").reshape(-1, 3)
    out["speed_minus_fastball"] = out.release_speed.to_numpy() - reference[:, 0]
    out["pfx_x_minus_fastball"] = out.pfx_x.to_numpy() - reference[:, 1]
    out["pfx_z_minus_fastball"] = out.pfx_z.to_numpy() - reference[:, 2]

    out["batter_bats_right"] = frame.stand.eq("R").astype("float64")
    out["pitcher_throws_right"] = frame.p_throws.eq("R").astype("float64")
    out["same_handed"] = frame.stand.eq(frame.p_throws).astype("float64")
    for column in OPTIONAL_PITCH_FEATURES:
        out[f"{column}_missing"] = frame[column].isna().astype("float64")
    return out.loc[:, list(PITCH_INPUTS)]


class PitchPreprocessor:
    """Everything learned from the training split: player IDs and feature scaling.

    Players with fewer than ``min_pitches`` training pitches share ID zero with
    players never seen in training.
    """

    def __init__(self, player_maps, fastball_ref, medians, means, stds, min_pitches):
        self.player_maps = player_maps
        self.fastball_ref = fastball_ref
        self.medians = medians
        self.means = means
        self.stds = stds
        self.min_pitches = min_pitches

    @classmethod
    def fit(cls, train, min_pitches=100):
        player_maps = {}
        for role in ("batter", "pitcher"):
            counts = train[role].value_counts()
            kept = sorted(int(player) for player in counts.index[counts >= min_pitches])
            player_maps[role] = {player: index for index, player in enumerate(kept, start=1)}
        fastball_ref = fastball_reference(train)
        raw = derive_pitch_features(train, fastball_ref)
        medians = raw.loc[:, list(CONTINUOUS_FEATURES)].median()
        filled = raw.loc[:, list(CONTINUOUS_FEATURES)].fillna(medians)
        stds = filled.std().replace(0, 1.0)
        return cls(
            player_maps, fastball_ref,
            {k: float(v) for k, v in medians.items()},
            {k: float(v) for k, v in filled.mean().items()},
            {k: float(v) for k, v in stds.items()},
            min_pitches,
        )

    def pitch_matrix(self, frame):
        """Imputed, standardized pitch inputs as a float32 array (rows x PITCH_INPUTS)."""
        raw = derive_pitch_features(frame, self.fastball_ref)
        continuous = raw.loc[:, list(CONTINUOUS_FEATURES)].fillna(pd.Series(self.medians))
        continuous = (continuous - pd.Series(self.means)) / pd.Series(self.stds)
        return np.concatenate(
            [continuous.to_numpy(dtype=np.float32), raw.loc[:, list(BINARY_FEATURES)].to_numpy(dtype=np.float32)],
            axis=1,
        )

    def encode(self, frame):
        """Return (batter_ids, pitcher_ids, count_index, pitch_matrix) tensors."""
        numeric, valid = numeric_features(frame)
        if not valid.all():
            raise ValueError("Features require positive player IDs and integer counts (0-3, 0-2)")
        if not pitch_features_valid(frame).all():
            raise ValueError("Rows are missing required pitch measurements; run prepare_pitches first")
        numeric = numeric.astype("int64")
        players = [
            torch.from_numpy(
                numeric[role].map(self.player_maps[role]).fillna(0).to_numpy(dtype=np.int64, copy=True)
            ) for role in ("batter", "pitcher")
        ]
        count = torch.from_numpy((numeric.balls * 3 + numeric.strikes).to_numpy(dtype=np.int64, copy=True))
        frame = frame.assign(pitcher=numeric.pitcher)
        pitch = torch.from_numpy(self.pitch_matrix(frame))
        return (*players, count, pitch)

    def state_dict(self):
        return {
            "player_maps": self.player_maps, "fastball_ref": self.fastball_ref,
            "medians": self.medians, "means": self.means, "stds": self.stds,
            "min_pitches": self.min_pitches,
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)
