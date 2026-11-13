"""Offline checks for data leakage, learning, and saved-model inference."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from pitch_model import (
    IN_PLAY_OUTCOMES, OUTCOMES, PitchOutcomeModel, chronological_split, encode_features,
    fit_player_maps, load_checkpoint, predict_pitches, prepare_pitches,
)
from train_pitch_model import logistic_regression_baseline, train_model


def sample_pitches():
    descriptions = ["ball", "called_strike", "swinging_strike", "foul",
                    "hit_into_play", "hit_by_pitch"]
    rows = []
    for day in range(1, 7):
        for pitch in range(24):
            rows.append({
                "game_pk": day, "game_date": f"2024-06-{day:02d}",
                "at_bat_number": pitch // 4 + 1, "pitch_number": pitch % 4 + 1,
                "batter": 999 if day >= 5 else 10 + pitch % 3,
                "pitcher": 20 + pitch % 2, "balls": pitch % 4,
                "strikes": pitch % 3, "description": descriptions[pitch % 6],
                "events": "single" if descriptions[pitch % 6] == "hit_into_play" else None,
            })
    return pd.DataFrame(rows)


class PitchModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cleaning_retains_nonterminal_pitches_and_reports_exclusions(self):
        raw = sample_pitches().iloc[:9].copy()
        raw["description"] = [
            "ball", "blocked_ball", "swinging_strike_blocked", "foul_tip",
            "automatic_ball", "foul_bunt", "unknown", "hit_into_play", "ball",
        ]
        raw.loc[7, "balls"] = 4
        raw.loc[8, "game_date"] = "invalid"
        clean, report = prepare_pitches(raw)
        self.assertEqual(clean.target.tolist(), [0, 0, 2, 3])
        self.assertTrue(clean.events.isna().all())
        self.assertEqual(report["excluded_rows"], 5)
        self.assertEqual(report["invalid_features_or_dates"], 2)
        self.assertEqual(report["unsupported_descriptions"]["automatic_ball"], 1)

    def test_time_split_and_training_only_player_maps(self):
        clean, _ = prepare_pitches(sample_pitches())
        train, validation, test = chronological_split(clean, "2024-06-03", "2024-06-05")
        self.assertLess(train.game_date.max(), validation.game_date.min())
        self.assertLess(validation.game_date.max(), test.game_date.min())
        self.assertTrue(set(train.game_pk).isdisjoint(test.game_pk))
        maps = fit_player_maps(train)
        self.assertNotIn(999, maps["batter"])
        batter, _, count = encode_features(test, maps)
        self.assertTrue(batter.eq(0).all())
        self.assertEqual(set(count.tolist()), set(range(12)))
        with self.assertRaises(ValueError):
            chronological_split(clean, "2024-06-05", "2024-06-03")
        with self.assertRaises(ValueError):
            chronological_split(clean, "2024-01-01", "2024-06-03")

    def test_in_play_events_become_individual_outcomes(self):
        raw = sample_pitches().iloc[:len(IN_PLAY_OUTCOMES)].copy()
        raw["description"] = "hit_into_play"
        raw["events"] = list(IN_PLAY_OUTCOMES)
        clean, report = prepare_pitches(raw)
        labels = [OUTCOMES[index] for index in clean.target]
        self.assertEqual(labels, list(IN_PLAY_OUTCOMES))
        self.assertNotIn("in_play", OUTCOMES)
        self.assertEqual(report["excluded_rows"], 0)

    def test_invalid_in_play_events_are_reported_and_noncontact_events_are_ignored(self):
        raw = sample_pitches().iloc[:8].copy()
        raw["description"] = ["hit_into_play"] * 5 + ["ball", "called_strike", "swinging_strike"]
        raw["events"] = [
            "single", None, "walk", "unrecognized_event", "home_run",
            "walk", "strikeout", "strikeout",
        ]
        clean, report = prepare_pitches(raw)
        self.assertEqual([OUTCOMES[index] for index in clean.target], [
            "single", "home_run", "ball", "called_strike", "swinging_strike",
        ])
        self.assertEqual(report["excluded_rows"], 3)
        self.assertEqual(report["unsupported_descriptions"], {})
        self.assertEqual(report["unsupported_in_play_events"], {
            "<missing>": 1, "walk": 1, "unrecognized_event": 1,
        })
        with self.assertRaisesRegex(ValueError, "events"):
            prepare_pitches(raw.drop(columns="events"))

    def test_learning_updates_player_vectors_and_keeps_unknown_zero(self):
        torch.manual_seed(42)
        model = PitchOutcomeModel(2, 1, dropout=0)
        batters = torch.tensor([1, 2] * 16)
        pitchers = torch.ones(32, dtype=torch.long)
        counts = torch.zeros(32, dtype=torch.long)
        targets = batters - 1
        criterion = torch.nn.CrossEntropyLoss()
        initial_loss = criterion(model(batters, pitchers, counts), targets).item()
        initial_embedding = model.batter_embedding.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        for _ in range(40):
            optimizer.zero_grad()
            loss = criterion(model(batters, pitchers, counts), targets)
            loss.backward()
            optimizer.step()
        final_loss = criterion(model(batters, pitchers, counts), targets).item()
        self.assertLess(final_loss, initial_loss * 0.1)
        self.assertFalse(torch.equal(initial_embedding[1:], model.batter_embedding.weight[1:]))
        self.assertTrue(model.batter_embedding.weight[0].eq(0).all())
        self.assertTrue(model.pitcher_embedding.weight[0].eq(0).all())

    def test_train_save_load_predict_without_outcome_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pitch_model.pt"
            with contextlib.redirect_stdout(io.StringIO()):
                model, maps, metrics = train_model(
                    sample_pitches(), "2024-06-03", "2024-06-05", output,
                    epochs=2, batch_size=32, progress=False,
                )
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".metrics.json").exists())
            self.assertTrue(np.isfinite(metrics["test"]["log_loss"]))
            self.assertTrue(np.isfinite(metrics["multinomial_logistic_regression_test"]["log_loss"]))
            self.assertNotIn("count_baseline_test", metrics)
            self.assertEqual(metrics["test_unknown_players"]["batter"], 48)
            loaded_model, loaded_maps = load_checkpoint(output)
            inputs = sample_pitches().loc[[120, 1, 42], ["batter", "pitcher", "balls", "strikes"]]
            expected = predict_pitches(model, inputs, maps)
            actual = predict_pitches(loaded_model, inputs, loaded_maps, batch_size=2)
            self.assertEqual(actual.index.tolist(), [120, 1, 42])
            self.assertEqual(actual.columns.tolist(), list(OUTCOMES))
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-6)
            with self.assertRaises(ValueError):
                predict_pitches(loaded_model, inputs.assign(strikes=3), loaded_maps)

    def test_logistic_baseline_learns_player_effects_without_test_label_leakage(self):
        train = pd.DataFrame({
            "batter": [11, 22, 33] * 60, "pitcher": [100] * 180,
            "balls": [0] * 180, "strikes": [0] * 180,
            "target": [0, 2, 5] * 60,
        })
        test = train.iloc[:3].copy()
        correct = logistic_regression_baseline(train, test, batch_size=2)
        incorrect = logistic_regression_baseline(train, test.assign(target=[2, 5, 0]))
        self.assertEqual(correct["accuracy"], 1)
        self.assertEqual(incorrect["accuracy"], 0)
        self.assertLess(correct["log_loss"], np.log(3))
        self.assertGreater(incorrect["log_loss"], correct["log_loss"])

    def test_logistic_baseline_handles_unseen_players_and_outcomes(self):
        train, _ = prepare_pitches(sample_pitches())
        test = train.iloc[:1].assign(batter=12345, pitcher=67890, target=len(OUTCOMES) - 1)
        result = logistic_regression_baseline(train, test)
        self.assertEqual(result["accuracy"], 0)
        self.assertEqual(result["unseen_outcome_pitches"], 1)
        self.assertAlmostEqual(result["log_loss"], -np.log(1e-15))
        with self.assertRaisesRegex(ValueError, "two training outcomes"):
            logistic_regression_baseline(train.assign(target=0), test)


if __name__ == "__main__":
    unittest.main()
