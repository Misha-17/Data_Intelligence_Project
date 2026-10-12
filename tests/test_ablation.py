"""Independent checks for feature ablation controls and preserved partitions."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from armd.ablation import (FEATURE_SETS, LABS_VITALS, NUMERIC, PRIOR_HISTORY,
                           _availability_frame, _restore_training_roles,
                           _verify_full_predictions, run_ablation)
from armd.evaluate import (FEATURES, ModelArtifact, _feature_frame, _logit,
                           _make_preprocessor, _raw_probability)


def example_inputs():
    rng = np.random.default_rng(6)
    rows = []
    for patient in range(100):
        for drug in range(3):
            row = {
                "patient_id": f"p{patient}", "culture_id": f"c{patient}", "organism": "organism",
                "candidate_antibiotic": f"drug{drug}", "age_group": str(patient % 4),
                "gender_code": "A" if patient % 2 else "B", "culture_type": "urine",
                "ward_ip": int(patient % 2 == 0), "ward_op": 0, "ward_er": None, "ward_icu": 0,
                "label": (patient + drug) % 2,
                "split": "train" if patient < 60 else "validation" if patient < 80 else "test",
                "culture_time": "2020-01-01",
            }
            row.update({name: float(rng.normal()) if rng.random() > .3 else np.nan for name in NUMERIC})
            rows.append(row)
    roles = pd.DataFrame({"patient_id": [f"p{patient}" for patient in range(60)],
                          "training_role": ["proper_fit" if patient < 48 else "calibration" for patient in range(60)]})
    return pd.DataFrame(rows), roles


def fitted_example(frame, roles):
    fit, cal, _ = _restore_training_roles(frame[frame["split"] != "test"], roles)
    preprocessor = _make_preprocessor(NUMERIC)
    x_fit = preprocessor.fit_transform(_feature_frame(fit, NUMERIC))
    x_cal = preprocessor.transform(_feature_frame(cal, NUMERIC))
    estimator = LGBMClassifier(n_estimators=10, min_child_samples=5, num_leaves=3,
                               random_state=42, n_jobs=1, verbosity=-1,
                               deterministic=True, force_col_wise=True)
    estimator.fit(x_fit, fit["label"], sample_weight=compute_sample_weight("balanced", fit["label"]))
    calibrator = LogisticRegression(C=1e6, max_iter=1000, random_state=42)
    calibrator.fit(_logit(_raw_probability(estimator, x_cal)), cal["label"])
    return ModelArtifact(FEATURES + NUMERIC, preprocessor, estimator, calibrator, .5, "lightgbm_balanced")


class AblationTests(unittest.TestCase):
    def test_control_erases_values_but_preserves_missingness_and_original_data(self):
        frame, _ = example_inputs()
        original = frame.copy(deep=True)
        control = _availability_frame(frame, NUMERIC)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(control[FEATURES], original[FEATURES])
        np.testing.assert_array_equal(control[NUMERIC].isna(), original[NUMERIC].isna())
        for name in NUMERIC:
            self.assertEqual(set(control[name].dropna().unique()), {0.0})
        changed = frame.copy()
        changed[NUMERIC] *= 10000
        pd.testing.assert_frame_equal(_availability_frame(changed, NUMERIC), control)
        self.assertFalse(set(LABS_VITALS) & set(PRIOR_HISTORY))
        self.assertEqual(set(LABS_VITALS + PRIOR_HISTORY), set(NUMERIC))

    def test_saved_roles_are_restored_exactly_and_contamination_is_rejected(self):
        frame, roles = example_inputs()
        development = frame[frame["split"] != "test"]
        fit, cal, validation = _restore_training_roles(development, roles.sample(frac=1, random_state=9))
        self.assertEqual(set(fit["patient_id"]), set(roles.loc[roles.training_role == "proper_fit", "patient_id"]))
        self.assertEqual(set(cal["patient_id"]), set(roles.loc[roles.training_role == "calibration", "patient_id"]))
        self.assertFalse(set(validation["patient_id"]) & set(roles["patient_id"]))
        for invalid_roles in [roles.iloc[:-1], pd.concat([roles, roles.iloc[:1]])]:
            with self.assertRaises(ValueError):
                _restore_training_roles(development, invalid_roles)
        with self.assertRaisesRegex(ValueError, "test rows are forbidden"):
            _restore_training_roles(frame, roles)
        contaminated = development.copy()
        contaminated.loc[0, "split"] = "validation"
        with self.assertRaisesRegex(ValueError, "Patient overlap"):
            _restore_training_roles(contaminated, roles)

    def test_frozen_prediction_check_aligns_keys_and_rejects_changed_probabilities(self):
        frame, roles = example_inputs()
        full = fitted_example(frame, roles)
        validation = frame[frame["split"] == "validation"].reset_index(drop=True)
        recorded = validation.copy()
        recorded["p_s_lightgbm_balanced"] = full.predict_proba(validation)[:, 1]
        result = _verify_full_predictions(validation, recorded.sample(frac=1, random_state=3), full)
        np.testing.assert_array_equal(result, recorded["p_s_lightgbm_balanced"])
        recorded.loc[0, "p_s_lightgbm_balanced"] += 1e-8
        with self.assertRaisesRegex(ValueError, "not exactly"):
            _verify_full_predictions(validation, recorded, full)
        recorded = validation.copy()
        recorded["p_s_lightgbm_balanced"] = result
        recorded.loc[0, "label"] = 1 - recorded.loc[0, "label"]
        with self.assertRaisesRegex(ValueError, "labels have changed"):
            _verify_full_predictions(validation, recorded, full)

    def test_real_small_pipeline_never_loads_test_and_metrics_match_sklearn(self):
        frame, roles = example_inputs()
        full = fitted_example(frame, roles)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "results" / "models"
            (models / "models").mkdir(parents=True)
            joblib.dump(full, models / "models" / "lightgbm_balanced.joblib")
            (models / "frozen_selection.json").write_text(json.dumps({"selected_model": "lightgbm_balanced"}))
            roles.to_csv(models / "training_patient_roles.csv", index=False)
            recorded = frame[frame["split"].isin(["validation", "test"])].copy()
            recorded["p_s_lightgbm_balanced"] = full.predict_proba(recorded)[:, 1]
            # Invalid test labels and probabilities expose accidental inclusion.
            frame.loc[frame["split"] == "test", "label"] = 99
            recorded.loc[recorded["split"] == "test", "p_s_lightgbm_balanced"] = np.nan
            frame.to_parquet(root / "results" / "analytical.parquet", index=False)
            recorded.to_parquet(models / "predictions.parquet", index=False)
            original_parquet_reader = pd.read_parquet
            with patch("armd.ablation.pd.read_parquet", wraps=original_parquet_reader) as reader:
                metadata = run_ablation(root, seed=142, bootstrap_repeats=5)
            self.assertEqual(reader.call_count, 2)
            self.assertEqual(reader.call_args_list[0].kwargs["filters"], [("split", "in", ["train", "validation"])])
            self.assertEqual(reader.call_args_list[1].kwargs["filters"], [("split", "==", "validation")])
            self.assertFalse(metadata["test_rows_loaded"])
            self.assertTrue(metadata["full_predictions_exactly_match_original"])
            self.assertEqual(metadata["hyperparameters"]["random_state"], 42)
            self.assertEqual(metadata["bootstrap"]["seed"], 142)
            output = root / "results" / "advanced"
            predictions = pd.read_parquet(output / "ablation_predictions.parquet")
            self.assertEqual(set(predictions["split"]), {"validation"})
            metrics = pd.read_csv(output / "ablation_metrics.csv")
            indexed = metrics.set_index(["model", "metric"])["estimate"]
            for name in FEATURE_SETS:
                probability = predictions[f"p_s_{name}"]
                expected_ap = np.mean([average_precision_score(1 - group.label, 1 - group[f"p_s_{name}"])
                                       for _, group in predictions.groupby("candidate_antibiotic")])
                expected_auc = np.mean([roc_auc_score(group.label, group[f"p_s_{name}"])
                                        for _, group in predictions.groupby("candidate_antibiotic")])
                self.assertAlmostEqual(indexed[name, "antibiotic_macro_auprc_r"], expected_ap, places=12)
                self.assertAlmostEqual(indexed[name, "antibiotic_macro_auroc"], expected_auc, places=12)
                self.assertAlmostEqual(indexed[name, "brier"], brier_score_loss(predictions.label, probability), places=12)
            differences = pd.read_csv(output / "ablation_paired_differences.csv")
            for row in differences.itertuples():
                self.assertAlmostEqual(row.difference_a_minus_b, indexed[row.model_a, row.metric] - indexed[row.model_b, row.metric], places=12)
            self.assertEqual(set(metrics["requested_replicates"]), {5})


if __name__ == "__main__":
    unittest.main()
