"""Check threshold ties, patient clustering and frozen-source diagnostics."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from armd.diagnostics import (_bin_codes, _calibration_group, _cluster_sums,
    _freeze_thresholds, _operating_values, _organism_slices, _patient_draws,
    _tradeoff_curve, _validate_sources, _validation_edges, run_diagnostics)


def test_threshold_targets_keep_tied_scores_together_and_apply_strict_rule():
    # Tied resistant/susceptible scores must be flagged together, including 0/1.
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    p = np.array([0., 0., .2, .2, .7, .7, 1., 1.])
    frame = pd.DataFrame({"label": y})
    points, curve = _freeze_thresholds(frame, p, .7, targets=(.5, .7, 1.))
    assert points[0]["threshold_s"] == .7
    assert _operating_values(y, p, .7)["n_flagged"] == 4
    for point in points[1:]:
        flagged = p < point["threshold_s"]
        assert np.mean(flagged[y == 0]) >= point["target_recall_r"]
        # The immediately stricter exact score boundary cannot meet the target.
        strict_threshold = np.nextafter(point["threshold_s"], -np.inf)
        assert np.mean((p < strict_threshold)[y == 0]) < point["target_recall_r"]
        for score in np.unique(p):
            assert len(np.unique(flagged[p == score])) == 1
    assert points[-1]["threshold_s"] > 1
    assert curve.n_flagged.tolist() == [0, 2, 4, 6, 8]
    for row in curve.itertuples(index=False):
        actual = _operating_values(y, p, row.threshold_s)
        assert actual["n_flagged"] == row.n_flagged
        assert actual["true_r"] == row.true_r
    assert np.isnan(curve.precision_r.iloc[0])
    with pytest.raises(ValueError, match="resistant examples"):
        _freeze_thresholds(pd.DataFrame({"label": [1, 1]}), np.array([.1, .8]), .5)


def test_validation_bins_collapse_ties_and_cover_probability_endpoints():
    edges = _validation_edges(np.array([.25] * 6), n_bins=5)
    np.testing.assert_array_equal(edges, [0., .25, 1.])
    np.testing.assert_array_equal(_bin_codes(np.array([0., .2, .25, 1.]), edges), [0, 0, 1, 1])
    with pytest.raises(ValueError):
        _validation_edges(np.array([np.nan]))


def test_patient_weighting_matches_explicit_whole_patient_resampling():
    patients = pd.Series(["b", "a", "a", "c", "c", "c"])
    codes, draws = _patient_draws(patients, repeats=7, seed=55)
    columns = np.column_stack([np.arange(6), np.ones(6)])
    actual = _cluster_sums(codes, draws, columns)
    rng = np.random.default_rng(55)
    for rep in range(7):
        sampled = rng.integers(0, 3, 3)
        index = np.concatenate([np.flatnonzero(codes == patient) for patient in sampled])
        np.testing.assert_allclose(actual[rep], columns[index].sum(axis=0))
        assert draws[rep].sum() == 3


def test_calibration_gap_cluster_intervals_match_explicit_resampled_rows():
    frame = pd.DataFrame({"patient_id": ["a", "a", "b", "c", "c", "c"],
                          "label": [0, 1, 1, 0, 1, 0]})
    p = np.array([.7, .7, .1, .8, .1, .5])
    codes, draws = _patient_draws(frame.patient_id, repeats=20, seed=71)
    rows, summary, histogram, samples = _calibration_group(
        frame, p, np.array([0., .4, 1.]), codes, draws,
        {"scope": "overall", "candidate_antibiotic": "__ALL__", "probability": "calibrated"})
    explicit_gaps = []
    for counts in draws:
        idx = np.repeat(np.arange(len(frame)), counts[codes])
        explicit_gaps.append(np.mean(p[idx] - (1 - frame.label.to_numpy()[idx])))
    np.testing.assert_allclose(samples["calibration_gap"], explicit_gaps)
    np.testing.assert_allclose([summary["calibration_gap_ci_low"], summary["calibration_gap_ci_high"]],
                               np.quantile(explicit_gaps, [.025, .975]))
    assert sum(row["n"] for row in rows) == len(frame)
    assert sum(row["n"] for row in histogram) == len(frame)
    assert summary["brier"] == pytest.approx(brier_score_loss(1 - frame.label, p))


def _example_sources():
    rows = []
    for patient in range(36):
        split = "train" if patient < 12 else "validation" if patient < 24 else "test"
        for drug_index, drug in enumerate(["A", "B"]):
            rows.append({"patient_id": f"p{patient:02}", "culture_id": f"c{patient:02}",
                         "organism": "organism_a" if patient % 3 else "organism_b",
                         "candidate_antibiotic": drug, "split": split,
                         "label": int((patient + drug_index) % 4 > 0)})
    analytical = pd.DataFrame(rows)
    predictions = analytical[analytical.split != "train"].copy().reset_index(drop=True)
    rng = np.random.default_rng(12)
    predictions["p_s_frozen"] = .1 + .8 * rng.random(len(predictions))
    predictions["p_s_raw_frozen"] = .05 + .7 * rng.random(len(predictions))
    return analytical, predictions


def test_source_audit_rejects_label_mismatch_and_patient_leakage():
    analytical, predictions = _example_sources()
    _validate_sources(analytical, predictions, "frozen", .6)
    wrong = predictions.copy()
    wrong.loc[0, "label"] = 1 - wrong.loc[0, "label"]
    with pytest.raises(ValueError, match="match analytical"):
        _validate_sources(analytical, wrong, "frozen", .6)
    wrong = analytical.copy()
    wrong.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="patient overlap"):
        _validate_sources(wrong, predictions, "frozen", .6)


def test_organism_selection_uses_training_events_and_never_imputes_untested_labels():
    train = pd.DataFrame({"patient_id": ["a", "a", "b", "c"], "culture_id": ["1", "1", "2", "3"],
                          "organism": ["rare", "rare", "common", "common"],
                          "candidate_antibiotic": ["A", "B", "A", "A"]})
    test = pd.DataFrame({"patient_id": ["d", "e"], "culture_id": ["4", "5"],
                         "organism": ["common", "common"], "candidate_antibiotic": ["A", "A"],
                         "label": [0, 1]})
    slices, selected = _organism_slices(train, test, np.array([.8, .7]), .75, top_n=1)
    assert selected == [{"organism": "common", "training_organism_events": 2}]
    missing = slices[slices.candidate_antibiotic == "B"].iloc[0]
    assert missing.n == missing.n_r == 0
    assert missing.observed_panel_coverage == 0
    assert np.isnan(missing.prevalence_r)
    assert np.isnan(missing.recall_r)
    assert not missing.outcome_observed


def test_end_to_end_frozen_choices_precede_test_metrics_and_ignore_test_mutations(tmp_path):
    analytical, predictions = _example_sources()
    model_dir = tmp_path / "results" / "models"
    model_dir.mkdir(parents=True)
    analytical.to_parquet(tmp_path / "results" / "analytical.parquet", index=False)
    predictions.to_parquet(model_dir / "predictions.parquet", index=False)
    selection = {"selected_model": "frozen", "thresholds_s": {"frozen": .6},
                 "test_used_for_selection": False, "selection_split": "validation"}
    (model_dir / "frozen_selection.json").write_text(json.dumps(selection))
    out = tmp_path / "advanced"
    from armd.diagnostics import _drug_performance

    def check_frozen(*args, **kwargs):
        assert (out / "fixed_thresholds_validation.json").exists()
        assert (out / "fixed_calibration_bins_validation.json").exists()
        return _drug_performance(*args, **kwargs)

    with patch("armd.diagnostics._drug_performance", side_effect=check_frozen):
        summary = run_diagnostics(tmp_path, out, repeats=12)
    assert summary["model_refit"] is False
    assert summary["main_estimates_changed"] is False
    perf = pd.read_csv(out / "drug_performance.csv")
    test = predictions[predictions.split == "test"]
    for drug in ["A", "B"]:
        frame = test[test.candidate_antibiotic == drug]
        row = perf[perf.candidate_antibiotic == drug].iloc[0]
        assert row.auroc == pytest.approx(roc_auc_score(frame.label, frame.p_s_frozen))
        assert row.ap_r == pytest.approx(average_precision_score(1 - frame.label, 1 - frame.p_s_frozen))
        assert row.brier == pytest.approx(brier_score_loss(frame.label, frame.p_s_frozen))
    original_points = json.loads((out / "fixed_thresholds_validation.json").read_text())["operating_points"]
    original_bins = json.loads((out / "fixed_calibration_bins_validation.json").read_text())
    # Change all test outcomes and scores, leaving validation entirely unchanged.
    analytical.loc[analytical.split == "test", "label"] = 1 - analytical.loc[analytical.split == "test", "label"]
    predictions.loc[predictions.split == "test", "label"] = 1 - predictions.loc[predictions.split == "test", "label"]
    predictions.loc[predictions.split == "test", "p_s_frozen"] = .99
    analytical.to_parquet(tmp_path / "results" / "analytical.parquet", index=False)
    predictions.to_parquet(model_dir / "predictions.parquet", index=False)
    run_diagnostics(tmp_path, out, repeats=12)
    assert original_points == json.loads((out / "fixed_thresholds_validation.json").read_text())["operating_points"]
    assert original_bins == json.loads((out / "fixed_calibration_bins_validation.json").read_text())
