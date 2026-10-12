"""Independent checks for probabilities, clustered uncertainty and leakage guards."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, log_loss, precision_score,
                             recall_score, roc_auc_score)

from armd.evaluate import (FEATURES, _BinaryMetricCache, _MetricCache,
                           _cluster_bootstrap, _feature_frame, _make_preprocessor,
                           _numeric_features, _select_threshold, _training_partition,
                           _validate_input, run_evaluation)


def example_data(patients=60):
    rng = np.random.default_rng(6)
    rows = []
    for patient in range(patients):
        split = "train" if patient < patients * .6 else (
            "validation" if patient < patients * .8 else "test")
        for candidate in ["ampicillin", "cefepime"]:
            rows.append({"patient_id": f"p{patient}", "culture_id": f"c{patient}",
                         "organism": "e_coli" if patient % 2 else "k_pneumoniae",
                         "candidate_antibiotic": candidate,
                         "age_group": "adult" if patient % 3 else None,
                         "gender_code": "A" if patient % 2 else "B", "culture_type": "urine",
                         "ward_ip": patient % 2, "ward_op": 0, "ward_er": None,
                         "ward_icu": patient % 5 == 0,
                         "lab_median_wbc_14d": float(patient + 1) if patient % 4 else np.nan,
                         "label": int(rng.random() < (.75 if candidate == "cefepime" else .45)),
                         "split": split, "culture_time": "2020-01-01", "dataset": "ARMD"})
    return pd.DataFrame(rows)


class MetricTests(unittest.TestCase):
    def test_cached_metrics_match_sklearn_with_ties_weights_and_endpoints(self):
        rng = np.random.default_rng(71)
        y = rng.integers(0, 2, 173)
        p = rng.choice([0., .05, .25, .5, .7, .95, 1.], len(y))
        weight_cases = [np.ones(len(y)), rng.integers(0, 5, len(y)), rng.uniform(.01, 5, len(y))]
        for weights in weight_cases:
            with self.subTest(weights=weights[:4]):
                actual = _BinaryMetricCache(y, p, .7).metrics(weights)
                pred = p >= .7
                expected = {
                    "auroc_s": roc_auc_score(y, p, sample_weight=weights),
                    "auprc_s": average_precision_score(y, p, sample_weight=weights),
                    "auprc_r": average_precision_score(1 - y, 1 - p, sample_weight=weights),
                    "balanced_accuracy": balanced_accuracy_score(y, pred, sample_weight=weights),
                    "precision_r": precision_score(y, pred, pos_label=0, sample_weight=weights, zero_division=0),
                    "recall_r": recall_score(y, pred, pos_label=0, sample_weight=weights),
                    "brier": brier_score_loss(y, p, sample_weight=weights),
                    "log_loss": log_loss(y, p, sample_weight=weights),
                }
                for metric, expected_value in expected.items():
                    self.assertAlmostEqual(actual[metric], expected_value, places=12, msg=metric)

    def test_zero_weight_score_groups_and_single_class(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        p = np.array([.1, .2, .3, .5, .8, .9])
        weights = np.array([0, 3, 0, 1, 0, 2])
        values = _BinaryMetricCache(y, p, .5).metrics(weights)
        expanded = np.repeat(np.arange(len(y)), weights)
        repeated = _BinaryMetricCache(y[expanded], p[expanded], .5).metrics()
        for name in values:
            self.assertAlmostEqual(values[name], repeated[name], places=12, msg=name)
        self.assertTrue(np.isnan(_BinaryMetricCache(y, p, .5).metrics(np.zeros(6))["auroc_s"]))
        one_class = _BinaryMetricCache(np.ones(6), p, .5).metrics()
        self.assertTrue(np.isnan(one_class["auroc_s"]))
        self.assertTrue(np.isnan(one_class["auprc_r"]))
        self.assertEqual(one_class["n_r"], 0)

    def test_macro_metrics_are_unweighted_across_evaluable_antibiotics(self):
        frame = pd.DataFrame({"label": [0, 1, 0, 0, 1, 1, 1],
                              "candidate_antibiotic": ["A", "A", "B", "B", "B", "B", "C"]})
        p = np.array([.2, .8, .7, .1, .6, .9, .3])
        actual = _MetricCache(frame, p, .5).metrics()
        expected = np.mean([average_precision_score(1 - frame.loc[idx, "label"], 1 - p[idx])
                            for idx in [np.array([0, 1]), np.array([2, 3, 4, 5])]])
        self.assertAlmostEqual(actual["antibiotic_macro_auprc_r"], expected)
        self.assertEqual(actual["antibiotics_evaluable"], 2)
        self.assertEqual(actual["antibiotics_present"], 3)

    def test_threshold_maximizes_validation_balanced_accuracy(self):
        y = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        p = np.array([.1, .1, .3, .3, .5, .5, .9, .9])
        candidates = np.unique(np.r_[p, .5])
        scores = np.array([balanced_accuracy_score(y, p >= threshold) for threshold in candidates])
        best = candidates[np.isclose(scores, scores.max(), atol=1e-12, rtol=0)]
        self.assertEqual(_select_threshold(y, p), best[np.argmin(abs(best - .5))])

    def test_patient_bootstrap_equals_explicit_resampled_rows(self):
        frame = pd.DataFrame({"patient_id": ["a", "a", "b", "c", "c", "c", "d", "d"],
                              "label": [0, 1, 1, 0, 0, 1, 1, 0],
                              "candidate_antibiotic": ["A", "B", "A", "A", "B", "B", "A", "B"]})
        p = np.array([.1, .8, .7, .3, .5, .7, .9, .2])
        caches = {"first": _MetricCache(frame, p, .5), "second": _MetricCache(frame, 1 - p, .5)}
        _, differences, reps = _cluster_bootstrap(frame, caches, "first", 41, repeats=4)
        rng = np.random.default_rng(41)
        codes, names = pd.factorize(frame["patient_id"], sort=True)
        for replicate in range(4):
            draws = rng.integers(0, len(names), size=len(names))
            index = np.concatenate([np.flatnonzero(codes == code) for code in draws])
            expected = _MetricCache(frame.iloc[index], p[index], .5).metrics()
            row = reps[(reps["replicate"] == replicate) & (reps["model"] == "first")].iloc[0]
            for metric in ["brier", "auprc_r", "antibiotic_macro_auprc_r"]:
                self.assertAlmostEqual(row[metric], expected[metric], places=12)
        self.assertTrue((differences["model_a"] == "first").all())

    def test_invalid_probability_or_weights_rejected(self):
        with self.assertRaises(ValueError):
            _BinaryMetricCache(np.array([0, 1]), np.array([.5, np.nan]), .5)
        with self.assertRaises(ValueError):
            _BinaryMetricCache(np.array([0, 1]), np.array([.5, .8]), .5).metrics(np.array([-1, 1]))


class LeakageTests(unittest.TestCase):
    def test_patient_disjoint_calibration_and_input_guards(self):
        frame = example_data()
        _validate_input(frame)
        train = frame[frame["split"] == "train"]
        fit, calibration = _training_partition(train, 42)
        self.assertFalse(set(fit["patient_id"]) & set(calibration["patient_id"]))
        self.assertEqual(set(fit["patient_id"]) | set(calibration["patient_id"]), set(train["patient_id"]))
        broken = frame.copy()
        broken.loc[0, "split"] = "test"
        with self.assertRaisesRegex(ValueError, "Patient overlap"):
            _validate_input(broken)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _validate_input(pd.concat([frame, frame.iloc[:1]], ignore_index=True))

    def test_numeric_preprocessing_only_learns_from_fit_and_keeps_missing_flags(self):
        fit = example_data().iloc[:4].copy()
        fit["measured"] = [1., 3., np.nan, 5.]
        fit["unavailable"] = np.nan
        fit["previously_complete"] = [2., 4., 6., 8.]
        numeric = ["measured", "unavailable", "previously_complete"]
        preprocessor = _make_preprocessor(numeric)
        preprocessor.fit(_feature_frame(fit, numeric))
        imputer = preprocessor.named_transformers_["numeric"].named_steps["impute"]
        np.testing.assert_allclose(imputer.statistics_, [3., 0., 5.])
        evaluation = fit.iloc[:2].copy()
        evaluation["measured"] = [999999., np.nan]
        evaluation["previously_complete"] = [0., np.nan]
        evaluation["age_group"] = "previously_unseen"
        transformed = preprocessor.transform(_feature_frame(evaluation, numeric)).toarray()
        names = preprocessor.get_feature_names_out().tolist()
        self.assertEqual(transformed[0, names.index("missing__missingindicator_previously_complete")], 0)
        self.assertEqual(transformed[1, names.index("missing__missingindicator_previously_complete")], 1)
        np.testing.assert_allclose(imputer.statistics_, [3., 0., 5.])
        self.assertFalse(any("patient_id" in name or "organism" in name or "label" in name for name in names))

    def test_only_audited_features_are_selected(self):
        frame = example_data()
        frame["future_lab"] = 4.
        selected = _numeric_features(frame)
        self.assertNotIn("future_lab", selected)
        self.assertNotIn("organism", selected)
        self.assertNotIn("label", selected)


class EndToEndTests(unittest.TestCase):
    def test_small_pipeline_freezes_selection_and_reloads_predictions(self):
        frame = example_data(100)
        with tempfile.TemporaryDirectory() as directory:
            # The preprocessing test is independent of whether support-table
            # metadata has already been generated in this checkout.
            with patch("armd.evaluate._numeric_features", return_value=["lab_median_wbc_14d"]):
                summary = run_evaluation(frame, directory, seed=17, bootstrap_repeats=3)
            output = Path(directory)
            selection = json.loads((output / "frozen_selection.json").read_text())
            self.assertFalse(selection["test_used_for_selection"])
            self.assertIn("antibiotic_macro_auprc_r", selection["selection_metric"])
            self.assertEqual(summary["numeric_features"], ["lab_median_wbc_14d"])
            comparison = pd.read_csv(output / "model_comparison.csv")
            validation = comparison[(comparison["split"] == "validation") &
                                    ~comparison["model"].str.contains("baseline")]
            expected = validation.sort_values(["antibiotic_macro_auprc_r", "brier", "model"],
                                               ascending=[False, True, True]).iloc[0]["model"]
            self.assertEqual(selection["selected_model"], expected)
            artifact = joblib.load(output / "models" / f"{expected}.joblib")
            test = frame[frame["split"] == "test"].reset_index(drop=True)
            predictions = pd.read_parquet(output / "predictions.parquet")
            recorded = predictions[predictions["split"] == "test"][f"p_s_{expected}"].to_numpy()
            np.testing.assert_allclose(artifact.predict_proba(test)[:, 1], recorded)
            importance = pd.read_csv(output / "validation_permutation_importance.csv")
            self.assertEqual(set(importance["split"]), {"validation"})
            self.assertEqual(set(importance["feature"]), set(FEATURES + ["lab_median_wbc_14d"]))
            groups = pd.read_csv(output / "subgroup_metrics.csv")
            self.assertIn("organism", set(groups["group_field"]))
            self.assertTrue((output / "figures" / "calibration_test.png").exists())


if __name__ == "__main__":
    unittest.main()
