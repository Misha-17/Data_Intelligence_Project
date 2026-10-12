"""Explore frozen ARMD predictions without refitting.

Predict resistance when p(susceptible) < threshold_s. Set new cutoffs and bins
on validation before test evaluation. This is exploratory follow-up because
the original test results were already reviewed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from armd.evaluate import _BinaryMetricCache, _json_clean

KEY = ["patient_id", "culture_id", "organism", "candidate_antibiotic"]
PERFORMANCE = ["auroc", "ap_r", "recall_r", "precision_r", "brier",
               "prevalence_r", "flagged_fraction", "ap_lift", "ap_above_chance"]
OPERATING = ["recall_r", "precision_r", "flagged_fraction",
             "false_alerts_per_1000", "missed_r_per_1000"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_clean(value), indent=2, allow_nan=False), encoding="utf-8")


def _safe_divide(a: Any, b: Any) -> np.ndarray:
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def _intervals(values: dict[str, np.ndarray]) -> dict[str, float | int]:
    result = {}
    for name, array in values.items():
        finite = np.asarray(array)[np.isfinite(array)]
        limits = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]
        result.update({name + "_ci_low": float(limits[0]),
                       name + "_ci_high": float(limits[1]),
                       name + "_valid_replicates": len(finite)})
    return result


def _patient_draws(patients: pd.Series, repeats: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw patients, retaining all their culture/drug rows at each draw."""
    codes, unique = pd.factorize(patients, sort=True)
    if (codes < 0).any() or not len(unique):
        raise ValueError("Nonempty data with observed patient IDs are required.")
    rng = np.random.default_rng(seed)
    draws = np.empty((repeats, len(unique)), dtype=np.int32)
    for rep in range(repeats):
        draws[rep] = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
    return codes, draws


def _cluster_sums(codes: np.ndarray, draws: np.ndarray, columns: np.ndarray) -> np.ndarray:
    columns = np.asarray(columns, float)
    if columns.ndim == 1:
        columns = columns[:, None]
    patient_sums = np.zeros((draws.shape[1], columns.shape[1]))
    np.add.at(patient_sums, codes, columns)
    return draws @ patient_sums


def _performance(cache: _BinaryMetricCache, weights: np.ndarray | None = None) -> dict[str, float]:
    values = cache.metrics(weights)
    prevalence = 1 - values["prevalence_s"]
    # Undefined precision stays missing when the model flags no rows.
    precision = values["precision_r"] if values["predicted_r_fraction"] > 0 else np.nan
    return {"auroc": values["auroc_s"], "ap_r": values["auprc_r"],
            "recall_r": values["recall_r"], "precision_r": precision,
            "brier": values["brier"], "prevalence_r": prevalence,
            "flagged_fraction": values["predicted_r_fraction"],
            "ap_lift": float(_safe_divide(values["auprc_r"], prevalence)),
            "ap_above_chance": float(_safe_divide(values["auprc_r"] - prevalence, 1 - prevalence))}


def _drug_performance(test: pd.DataFrame, p_s: np.ndarray, threshold_s: float,
                      codes: np.ndarray, draws: np.ndarray) -> pd.DataFrame:
    rows = []
    for drug in sorted(test.candidate_antibiotic.unique()):
        idx = np.flatnonzero(test.candidate_antibiotic.to_numpy() == drug)
        subset = test.iloc[idx]
        y = subset.label.to_numpy(int)
        cache = _BinaryMetricCache(y, p_s[idx], threshold_s)
        samples = {metric: [] for metric in PERFORMANCE}
        for draw in draws:
            values = _performance(cache, draw[codes[idx]])
            for metric in PERFORMANCE:
                samples[metric].append(values[metric])
        rows.append({"candidate_antibiotic": drug, "n": len(idx), "n_r": int((y == 0).sum()),
                     "n_s": int(y.sum()), "patients": subset.patient_id.nunique(),
                     "patients_with_r": subset.loc[y == 0, "patient_id"].nunique(),
                     "patients_with_s": subset.loc[y == 1, "patient_id"].nunique(),
                     "n_flagged": int((p_s[idx] < threshold_s).sum()),
                     "false_alerts": int(((y == 1) & (p_s[idx] < threshold_s)).sum()),
                     "missed_r": int(((y == 0) & (p_s[idx] >= threshold_s)).sum()),
                     "threshold_s": threshold_s, "sparse": len(idx) < 100 or min((y == 0).sum(), y.sum()) < 20,
                     **_performance(cache), **_intervals(samples)})
    return pd.DataFrame(rows)


def _validation_edges(probabilities: np.ndarray, n_bins: int = 5) -> np.ndarray:
    p = np.asarray(probabilities, float)
    if not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Calibration binning requires observed validation probabilities in [0, 1].")
    if n_bins < 1:
        raise ValueError("At least one calibration bin is required.")
    internal = np.quantile(p, np.arange(1, n_bins) / n_bins)
    return np.r_[0., np.unique(internal[(internal > 0) & (internal < 1)]), 1.]


def _bin_codes(probabilities: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges[1:-1], probabilities, side="right")


def _calibration_group(test: pd.DataFrame, p_r: np.ndarray, edges: np.ndarray,
                       codes: np.ndarray, draws: np.ndarray, keys: dict[str, Any]
                       ) -> tuple[list[dict], dict, list[dict], dict[str, np.ndarray]]:
    r = (test.label.to_numpy(int) == 0).astype(float)
    bins = _bin_codes(p_r, edges)
    rows, bin_samples = [], []
    for b in range(len(edges) - 1):
        mask = bins == b
        n, n_r = int(mask.sum()), int(r[mask].sum())
        samples = _cluster_sums(codes[mask], draws,
                                np.column_stack([np.ones(n), r[mask], p_r[mask]]))
        n_sample, r_sample, pred_sample = samples.T
        obs = _safe_divide(r_sample, n_sample)
        pred = _safe_divide(pred_sample, n_sample)
        rows.append({**keys, "bin": b + 1, "edge_left": edges[b], "edge_right": edges[b + 1],
                     "n": n, "n_r": n_r, "patients": test.loc[mask, "patient_id"].nunique(),
                     "mean_predicted_r": float(p_r[mask].mean()) if n else np.nan,
                     "observed_r": float(r[mask].mean()) if n else np.nan,
                     "calibration_gap": float((p_r[mask] - r[mask]).mean()) if n else np.nan,
                     "sparse": n < 100 or n_r < 20,
                     **_intervals({"mean_predicted_r": pred, "observed_r": obs,
                                   "calibration_gap": pred - obs})})
        bin_samples.append((n_sample, np.abs(pred_sample - r_sample)))
    overall = _cluster_sums(codes, draws,
                            np.column_stack([np.ones(len(r)), r, p_r, (p_r - r) ** 2]))
    n_sample, r_sample, pred_sample, sq_sample = overall.T
    samples = {"brier": _safe_divide(sq_sample, n_sample),
               "mean_predicted_r": _safe_divide(pred_sample, n_sample),
               "prevalence_r": _safe_divide(r_sample, n_sample),
               "calibration_gap": _safe_divide(pred_sample - r_sample, n_sample),
               "ece_validation_bins": _safe_divide(np.sum([v[1] for v in bin_samples], axis=0), n_sample)}
    summary = {**keys, "n": len(r), "n_r": int(r.sum()), "patients": test.patient_id.nunique(),
               "brier": float(np.mean((p_r - r) ** 2)), "mean_predicted_r": float(p_r.mean()),
               "prevalence_r": float(r.mean()), "calibration_gap": float((p_r - r).mean()),
               "ece_validation_bins": sum(abs(v["calibration_gap"]) * v["n"] for v in rows if v["n"]) / len(r),
               **_intervals(samples)}
    hist_counts, hist_edges = np.histogram(p_r, bins=np.linspace(0, 1, 21))
    histogram = [{**keys, "bin": b + 1, "edge_left": hist_edges[b], "edge_right": hist_edges[b + 1],
                  "n": int(count)} for b, count in enumerate(hist_counts)]
    return rows, summary, histogram, samples


def _tradeoff_curve(y: np.ndarray, p_s: np.ndarray) -> pd.DataFrame:
    """Exact validation score cutoffs, including tied scores in one operation."""
    y, p_s = np.asarray(y, int), np.asarray(p_s, float)
    if len(y) != len(p_s) or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Threshold selection requires equal-length binary labels and scores.")
    if not np.isfinite(p_s).all() or ((p_s < 0) | (p_s > 1)).any():
        raise ValueError("Threshold selection requires probabilities in [0, 1].")
    order = np.argsort(p_s, kind="stable")
    sorted_scores, sorted_r = p_s[order], (y[order] == 0).astype(int)
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y) - 1]
    flagged = np.r_[0, ends + 1]
    true_r = np.r_[0, np.cumsum(sorted_r)[ends]]
    thresholds = np.r_[0., np.nextafter(sorted_scores[ends], np.inf)]
    n_r = int(sorted_r.sum())
    return pd.DataFrame({"threshold_s": thresholds, "threshold_r": 1 - thresholds,
                         "n_flagged": flagged, "true_r": true_r,
                         "recall_r": _safe_divide(true_r, n_r),
                         "precision_r": _safe_divide(true_r, flagged),
                         "flagged_fraction": flagged / len(y),
                         "false_alerts_per_1000": 1000 * (flagged - true_r) / len(y),
                         "missed_r_per_1000": 1000 * (n_r - true_r) / len(y)})


def _freeze_thresholds(validation: pd.DataFrame, p_s: np.ndarray,
                       original_threshold_s: float, targets: tuple[float, ...] = (.7, .8, .9)
                       ) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    curve = _tradeoff_curve(validation.label.to_numpy(int), p_s)
    if not (validation.label == 0).any():
        raise ValueError("Validation requires resistant examples to select recall targets.")
    points = [{"operating_point": "original", "target_recall_r": None,
               "threshold_s": original_threshold_s, "threshold_r": 1 - original_threshold_s,
               "selection": "original frozen validation-balanced-accuracy threshold"}]
    for target in targets:
        if not 0 < target <= 1:
            raise ValueError("Resistance recall targets must be in (0, 1].")
        eligible = curve[curve.recall_r >= target]
        chosen = eligible.iloc[0]
        points.append({"operating_point": f"recall_{round(target * 100)}", "target_recall_r": target,
                       "threshold_s": float(chosen.threshold_s), "threshold_r": float(chosen.threshold_r),
                       "selection": "highest resistance cutoff achieving validation recall target, ties kept together"})
    return points, curve


def _operating_values(y: np.ndarray, p_s: np.ndarray, threshold_s: float,
                       codes: np.ndarray | None = None, draws: np.ndarray | None = None) -> dict[str, Any]:
    r, flagged = np.asarray(y) == 0, np.asarray(p_s) < threshold_s
    columns = np.column_stack([np.ones(len(y)), r, flagged, r & flagged, ~r & flagged, r & ~flagged])
    sums = columns.sum(axis=0)
    names = OPERATING

    def metrics(values: np.ndarray) -> list[np.ndarray]:
        n, n_r, n_flag, tp, fp, fn = values.T
        return [_safe_divide(tp, n_r), _safe_divide(tp, n_flag), _safe_divide(n_flag, n),
                1000 * _safe_divide(fp, n), 1000 * _safe_divide(fn, n)]

    values = {name: float(value) for name, value in zip(names, metrics(sums))}
    values.update(n=len(y), n_r=int(r.sum()), n_flagged=int(flagged.sum()),
                  true_r=int((r & flagged).sum()), false_alerts=int((~r & flagged).sum()),
                  missed_r=int((r & ~flagged).sum()))
    if draws is not None:
        values.update(_intervals(dict(zip(names, metrics(_cluster_sums(codes, draws, columns))))))
    return values


def _organism_slices(train: pd.DataFrame, test: pd.DataFrame, p_s: np.ndarray,
                     threshold_s: float, top_n: int = 6) -> tuple[pd.DataFrame, list[dict]]:
    """Describe recorded panel coverage, never fill in unobserved drug outcomes."""
    event_key = ["patient_id", "culture_id", "organism"]
    training_events = train[event_key].drop_duplicates()
    counts = training_events.groupby("organism", observed=True).size().rename("training_organism_events")
    selected = (counts.reset_index().sort_values(["training_organism_events", "organism"],
                ascending=[False, True]).head(top_n))
    rows = []
    # The panel is selected from training records, not test prevalence/performance.
    drugs = sorted(train.candidate_antibiotic.unique())
    for organism, training_n in selected.itertuples(index=False, name=None):
        organism_mask = test.organism.to_numpy() == organism
        denominator = len(test.loc[organism_mask, event_key].drop_duplicates())
        for drug in drugs:
            mask = organism_mask & (test.candidate_antibiotic.to_numpy() == drug)
            subset = test.loc[mask]
            n = len(subset)
            y = subset.label.to_numpy(int)
            n_r = int((y == 0).sum())
            missed = int(((y == 0) & (p_s[mask] >= threshold_s)).sum())
            rows.append({"organism": organism, "candidate_antibiotic": drug,
                         "training_organism_events": int(training_n), "test_organism_events": denominator,
                         "observed_drug_events": len(subset[event_key].drop_duplicates()),
                         "observed_panel_coverage": n / denominator if denominator else np.nan,
                         "n": n, "n_r": n_r, "n_s": n - n_r,
                         "patients": subset.patient_id.nunique(), "missed_r": missed,
                         "prevalence_r": n_r / n if n else np.nan,
                         "missed_r_fraction": missed / n_r if n_r else np.nan,
                         "recall_r": (n_r - missed) / n_r if n_r else np.nan,
                         "supported_error_slice": n >= 50 and n_r >= 20,
                         "outcome_observed": bool(n),
                         "coverage_denominator": "organism-events with >=1 retained panel S/R label"})
    return pd.DataFrame(rows), selected.to_dict(orient="records")


def _validate_sources(analytical: pd.DataFrame, predictions: pd.DataFrame,
                      model: str, threshold_s: float) -> None:
    for name, frame in [("analytical", analytical), ("predictions", predictions)]:
        required = set(KEY + ["label", "split"])
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} lacks required record keys, label or split.")
        if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
            raise ValueError(f"{name} has missing or duplicate record keys.")
        if not frame.label.isin([0, 1]).all():
            raise ValueError(f"{name} includes nonbinary labels.")
        if frame.groupby("patient_id", observed=True).split.nunique().max() != 1:
            raise ValueError(f"{name} includes patient overlap across splits.")
    if set(analytical.split.unique()) != {"train", "validation", "test"}:
        raise ValueError("Analytical data must include the original train, validation and test splits.")
    if set(predictions.split.unique()) != {"validation", "test"}:
        raise ValueError("Predictions must contain validation and test only.")
    expected = analytical[analytical.split.isin(["validation", "test"])][KEY + ["label", "split"]]
    check = expected.merge(predictions[KEY + ["label", "split"]], on=KEY, how="outer",
                           validate="one_to_one", suffixes=("_data", "_pred"), indicator=True)
    if not (check._merge == "both").all() or not (check.label_data == check.label_pred).all() or not (check.split_data == check.split_pred).all():
        raise ValueError("Saved predictions do not match analytical keys, labels and splits exactly.")
    for column in [f"p_s_{model}", f"p_s_raw_{model}"]:
        if column not in predictions:
            raise ValueError(f"Frozen prediction column missing: {column}")
        p = predictions[column].to_numpy(float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Saved model probabilities must be finite and within [0, 1].")
    predicted_label = f"predicted_label_{model}"
    if predicted_label in predictions and not np.array_equal(
            predictions[predicted_label].to_numpy(), (predictions[f"p_s_{model}"].to_numpy() >= threshold_s).astype(int)):
        raise ValueError("Saved labels differ from the frozen model threshold.")


def run_diagnostics(root: str | Path, out_dir: str | Path, seed: int = 142, repeats: int = 300) -> dict[str, Any]:
    """Export diagnostics from audited frozen predictions; preserve source files."""
    if repeats < 2:
        raise ValueError("At least two patient bootstrap replicates are required.")
    root, out = Path(root).resolve(), Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_dir = root / "results" / "models"
    source_paths = [root / "results" / "analytical.parquet", model_dir / "predictions.parquet",
                    model_dir / "frozen_selection.json"]
    source_hashes = {str(path.relative_to(root)).replace("\\", "/"): _sha256(path) for path in source_paths}
    selection = json.loads(source_paths[2].read_text(encoding="utf-8"))
    if selection.get("test_used_for_selection") is not False or selection.get("selection_split") != "validation":
        raise ValueError("Original model selection must be frozen using validation only.")
    model = selection["selected_model"]
    threshold = float(selection["thresholds_s"][model])
    analytical, predictions = (pd.read_parquet(path) for path in source_paths[:2])
    _validate_sources(analytical, predictions, model, threshold)
    train = analytical[analytical.split == "train"].reset_index(drop=True)
    validation = predictions[predictions.split == "validation"].reset_index(drop=True)
    pcol, rawcol = f"p_s_{model}", f"p_s_raw_{model}"
    points, curve = _freeze_thresholds(validation, validation[pcol].to_numpy(float), threshold)
    bin_specs = []
    for probability, column in [("calibrated", pcol), ("raw", rawcol)]:
        edges = _validation_edges(1 - validation[column].to_numpy(float), n_bins=10)
        bin_specs.append({"scope": "overall", "candidate_antibiotic": "__ALL__", "probability": probability,
                          "edges": edges.tolist(), "requested_bins": 10})
    for drug in sorted(train.candidate_antibiotic.unique()):
        values = validation.loc[validation.candidate_antibiotic == drug, pcol].to_numpy(float)
        if not len(values):
            raise ValueError(f"No validation records to set calibration bins for {drug}.")
        bin_specs.append({"scope": "drug", "candidate_antibiotic": drug, "probability": "calibrated",
                          "edges": _validation_edges(1 - values, n_bins=5).tolist(), "requested_bins": 5})
    # Save all new choices before computing any advanced held-out test metrics.
    _write_json(out / "fixed_thresholds_validation.json", {
        "selected_model": model, "exploratory": True, "selection_split": "validation",
        "test_used_for_threshold_selection": False, "source_sha256": source_hashes,
        "decision_rule": "resistant iff p_s < threshold_s; threshold_r = 1-threshold_s is a display value",
        "tie_rule": "include every tied validation score via nextafter(score_s, +inf)",
        "operating_points": points,
    })
    _write_json(out / "fixed_calibration_bins_validation.json", {
        "selection_split": "validation", "test_used_for_bin_selection": False,
        "closure": "[left, right), final bin includes 1", "bins": bin_specs})
    curve.to_csv(out / "validation_tradeoff_curve.csv", index=False)

    test = predictions[predictions.split == "test"].reset_index(drop=True)
    p_s = test[pcol].to_numpy(float)
    codes, draws = _patient_draws(test.patient_id, repeats, seed)
    drug_performance = _drug_performance(test, p_s, threshold, codes, draws)
    drug_performance.to_csv(out / "drug_performance.csv", index=False)
    bin_rows, summaries, histogram_rows, overall_samples = [], [], [], {}
    for spec in bin_specs:
        mask = np.ones(len(test), bool) if spec["scope"] == "overall" else test.candidate_antibiotic.to_numpy() == spec["candidate_antibiotic"]
        keys = {key: spec[key] for key in ["scope", "candidate_antibiotic", "probability"]}
        column = pcol if spec["probability"] == "calibrated" else rawcol
        bins, summary, histogram, samples = _calibration_group(
            test.loc[mask].reset_index(drop=True), 1 - test.loc[mask, column].to_numpy(float),
            np.asarray(spec["edges"]), codes[mask], draws, keys)
        bin_rows.extend(bins)
        summaries.append(summary)
        histogram_rows.extend(histogram)
        if spec["scope"] == "overall":
            overall_samples[spec["probability"]] = samples
    pd.DataFrame(bin_rows).to_csv(out / "calibration_bins.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "calibration_summary.csv", index=False)
    pd.DataFrame(histogram_rows).to_csv(out / "probability_histogram.csv", index=False)
    comparison = []
    overall = {row["probability"]: row for row in summaries if row["scope"] == "overall"}
    for metric in ["brier", "ece_validation_bins", "calibration_gap"]:
        samples = overall_samples["raw"][metric] - overall_samples["calibrated"][metric]
        comparison.append({"metric": metric, "raw": overall["raw"][metric],
                           "calibrated": overall["calibrated"][metric],
                           "raw_minus_calibrated": overall["raw"][metric] - overall["calibrated"][metric],
                           **_intervals({"raw_minus_calibrated": samples})})
    pd.DataFrame(comparison).to_csv(out / "calibration_comparison.csv", index=False)
    operating_rows = []
    for split, frame in [("validation", validation), ("test", test)]:
        for point in points:
            operating_rows.append({**point, "split": split, "patients": frame.patient_id.nunique(),
                **_operating_values(frame.label.to_numpy(int), frame[pcol].to_numpy(float), point["threshold_s"],
                                    codes if split == "test" else None, draws if split == "test" else None)})
    operating = pd.DataFrame(operating_rows)
    operating.to_csv(out / "threshold_operating_points.csv", index=False)
    slices, selected_organisms = _organism_slices(train, test, p_s, threshold)
    slices.to_csv(out / "organism_drug_slices.csv", index=False)
    metadata = {
        "analysis_status": "exploratory follow-up; the frozen test set was examined in the original report",
        "selected_model": model, "original_threshold_s": threshold,
        "model_refit": False, "main_estimates_changed": False,
        "prediction_target": "S=1; R=0; figures and calibration use P(R)=1-P(S)",
        "seed": seed, "bootstrap_replicates": repeats,
        "confidence_intervals": "pointwise 95% percentile intervals from shared resampling of test patients with all rows retained; conditional on the fixed fitted model, thresholds and bin boundaries; not simultaneous or model-training uncertainty",
        "cluster_resampling": "same test-patient draws used across all drugs, bins, operating points and paired calibration comparisons",
        "source_sha256": source_hashes, "source_contract": {
            "analytical": "results/analytical.parquet: one patient x culture x organism x candidate antibiotic; patient-disjoint train/validation/test",
            "predictions": f"results/models/predictions.parquet: exact validation/test keys, labels and splits; {pcol} and {rawcol}",
            "selection": "results/models/frozen_selection.json: selected_model, thresholds_s and validation-only selection audit"},
        "missing_metric_policy": "undefined metrics are missing, including precision when no rows are flagged; CIs use finite replicates and report their count",
        "calibration": "10 overall bins, 5 per-drug bins from validation quantiles with duplicate edges collapsed; bins fixed before test evaluation; no test-fitted calibrator",
        "calibration_sparse_flag": "n<100 or resistant count<20; zero-resistant bins may have degenerate bootstrap intervals and must not imply certainty",
        "boundary_intervals": "bootstrap intervals can collapse to 0 or 1 when all observed cases share a binary outcome; this does not imply certainty for future patients",
        "calibration_comparison": "raw minus already calibrated probabilities from the same frozen model; the original calibrator was fit on a patient-disjoint subset of training",
        "ap_comparison_caution": "resistance AP depends on prevalence and case mix; AP/prevalence and (AP-prevalence)/(1-prevalence) are descriptive and do not establish cross-drug superiority",
        "operating_points": "prespecified validation resistance recall targets 70%, 80%, 90% plus the original threshold; counts are retrospective classification consequences per 1000 retained candidate rows, not patient-level clinical decisions or treatment outcomes",
        "organism_selection": "top six organisms by unique organism-event count in training only, with alphabetical tie breaking",
        "selected_organisms": selected_organisms,
        "organism_slice_support": "error-rate displays require n>=50 and n_r>=20; coverage includes all selected organism x drug combinations and missing outcomes are never filled in",
        "coverage_denominator": "test organism-events with at least one retained observed S/R outcome in the provisional antibiotic panel; this is retained analytical coverage, not all clinical susceptibility testing",
        "site_validation": "unavailable; no hospital identifier exists in supplied source tables",
        "test_rows": len(test), "test_patients": test.patient_id.nunique(),
        "test_resistant": int((test.label == 0).sum()), "outputs": sorted(path.name for path in out.glob("*.csv") if path.name in {
            "drug_performance.csv", "calibration_bins.csv", "calibration_summary.csv", "probability_histogram.csv",
            "calibration_comparison.csv", "threshold_operating_points.csv", "validation_tradeoff_curve.csv", "organism_drug_slices.csv"}),
    }
    _write_json(out / "diagnostics_metadata.json", metadata)
    for path in source_paths:
        key = str(path.relative_to(root)).replace("\\", "/")
        if _sha256(path) != source_hashes[key]:
            raise RuntimeError(f"Frozen source changed during diagnostics: {path}")
    return metadata
