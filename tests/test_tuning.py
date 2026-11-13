"""Offline tuning checks: held-out evaluation, persistence, and pruning."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import optuna
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from pitch_model import load_checkpoint
from test_pitch_model import sample_pitches
import train_pitch_model
import tune_pitch_model


class TuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def test_tuning_resumes_and_evaluates_test_only_in_final_training(self):
        raw = sample_pitches()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tuned.pt"
            options = dict(
                validation_start="2024-06-03", test_start="2024-06-05", output=output,
                n_trials=2, epochs=1, startup_trials=1, progress=False,
            )
            with patch.object(tune_pitch_model, "fit_model", wraps=train_pitch_model.fit_model) as fits, \
                 patch.object(train_pitch_model, "evaluate", wraps=train_pitch_model.evaluate) as evaluations, \
                 patch.object(train_pitch_model, "logistic_regression_baseline",
                              wraps=train_pitch_model.logistic_regression_baseline) as baseline:
                study, summary = tune_pitch_model.tune_model(raw, **options)
                self.assertIsInstance(study.sampler, optuna.samplers.TPESampler)
                self.assertEqual(len(study.trials), 2)
                self.assertEqual(baseline.call_count, 1)
                self.assertEqual(sum(call.args[-1] == "Test" for call in evaluations.call_args_list), 1)
                self.assertEqual(fits.call_count, 2)
                # Held-out batters in this fixture encode as zero. Neither
                # dataset handed to a search trial contains those batters.
                for call in fits.call_args_list:
                    for dataset in call.args[:2]:
                        self.assertTrue(dataset.tensors[0].gt(0).all())
                model, _ = load_checkpoint(output)
                self.assertEqual(model.batter_embedding.embedding_dim, summary["best_params"]["embedding_dim"])
                self.assertAlmostEqual(summary["best_validation_log_loss"],
                                       summary["final_metrics"]["validation"]["log_loss"], places=6)
                options["n_trials"] = 1
                resumed, _ = tune_pitch_model.tune_model(raw, **options)
                self.assertEqual(len(resumed.trials), 3)
                self.assertEqual(baseline.call_count, 2)
                with self.assertRaisesRegex(ValueError, "settings changed"):
                    tune_pitch_model.tune_model(raw, seed=43, **options)
            self.assertTrue(output.with_suffix(".optuna.sqlite3").exists())
            self.assertTrue(output.with_suffix(".tuning.json").exists())
            self.assertEqual(len(pd.read_parquet(output.with_suffix(".trials.parquet"))), 3)

    def test_pruned_trials_do_not_run_final_training(self):
        pruner = optuna.pruners.ThresholdPruner(upper=0.0, n_warmup_steps=0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pruned.pt"
            with patch.object(tune_pitch_model.optuna.pruners, "MedianPruner", return_value=pruner), \
                 patch.object(tune_pitch_model, "train_model") as final_training:
                with self.assertRaisesRegex(RuntimeError, "No tuning trial completed"):
                    tune_pitch_model.tune_model(
                        sample_pitches(), "2024-06-03", "2024-06-05", output,
                        n_trials=1, epochs=2, startup_trials=1, progress=False,
                    )
                final_training.assert_not_called()
            study = optuna.load_study(
                study_name="pitch_outcome",
                storage=f"sqlite:///{output.with_suffix('.optuna.sqlite3').resolve().as_posix()}",
            )
            self.assertEqual(study.trials[0].state, optuna.trial.TrialState.PRUNED)
            self.assertIn(1, study.trials[0].intermediate_values)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
