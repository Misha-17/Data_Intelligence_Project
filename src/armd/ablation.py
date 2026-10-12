"""Compare feature groups on validation without loading test data.

Keep the full model frozen. Reduced models share its fitting and calibration
patients, settings and training recipe. Results are exploratory because these
validation patients already informed model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

from .evaluate import (
    FEATURES, ModelArtifact, _MetricCache, _cluster_bootstrap, _feature_frame,
    _json_clean, _logit, _make_preprocessor, _raw_probability,
)


LABS_VITALS = [
    "lab_median_wbc_14d", "lab_median_creatinine_14d",
    "vital_median_heart_rate_48h", "vital_median_temperature_48h",
]
PRIOR_HISTORY = [
    "prior_abx_days_ago", "prior_abx_class_count_90d",
    "prior_candidate_resistance_days_ago",
]
NUMERIC = LABS_VITALS + PRIOR_HISTORY
KEY = ["patient_id", "culture_id", "organism", "candidate_antibiotic"]
METRICS = ["antibiotic_macro_auprc_r", "antibiotic_macro_auroc", "brier"]
FEATURE_SETS = {
    "full": {"label": "All features", "numeric": NUMERIC},
    "core_only": {"label": "Core features only", "numeric": []},
    "core_labs_vitals": {"label": "Core + labs and vital signs", "numeric": LABS_VITALS},
    "core_prior_history": {"label": "Core + prior history", "numeric": PRIOR_HISTORY},
    "core_availability_only": {"label": "Core + availability only", "numeric": NUMERIC},
}


def _availability_frame(df: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    """Replace observed magnitudes with zero; keep missing values as NaN.

    Zero is a control value, not a replacement for missing clinical data.
    Missingness indicators retain record availability.
    """
    result = df.copy()
    for name in numeric:
        values = pd.to_numeric(result[name], errors="raise")
        result[name] = np.where(np.isfinite(values), 0.0, np.nan)
    return result


def _restore_training_roles(
    development: pd.DataFrame, roles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Restore the saved patient partition, rejecting missing or extra roles."""
    required = set(KEY + FEATURES + NUMERIC + ["label", "split"])
    if required - set(development):
        raise ValueError(f"Missing analytical columns: {sorted(required - set(development))}")
    if set(development["split"].unique()) != {"train", "validation"}:
        raise ValueError("Ablations accept train and validation only; test rows are forbidden.")
    if development["patient_id"].isna().any() or development["label"].isna().any():
        raise ValueError("Patient identifiers and labels must be observed.")
    if not development["label"].isin([0, 1]).all():
        raise ValueError("Ablations require susceptible=1 and resistant=0.")
    if development.duplicated(KEY).any():
        raise ValueError("Duplicate analytical keys would change the comparison grain.")
    if development.groupby("patient_id", observed=True)["split"].nunique().max() != 1:
        raise ValueError("Patient overlap between train and validation is forbidden.")
    if not {"patient_id", "training_role"}.issubset(roles):
        raise ValueError("Saved training roles need patient_id and training_role.")
    if roles["patient_id"].isna().any() or roles["patient_id"].duplicated().any():
        raise ValueError("Saved training roles must have one row per observed patient.")
    if set(roles["training_role"].unique()) != {"proper_fit", "calibration"}:
        raise ValueError("Saved training roles must be proper_fit and calibration only.")
    train = development[development["split"] == "train"]
    if set(roles["patient_id"]) != set(train["patient_id"]):
        raise ValueError("Saved training roles do not exactly cover training patients.")
    mapped = train["patient_id"].map(roles.set_index("patient_id")["training_role"])
    fit = train[mapped == "proper_fit"].reset_index(drop=True)
    cal = train[mapped == "calibration"].reset_index(drop=True)
    validation = development[development["split"] == "validation"].reset_index(drop=True)
    for name, frame in [("proper_fit", fit), ("calibration", cal), ("validation", validation)]:
        if frame["label"].nunique() != 2:
            raise ValueError(f"{name} must contain both outcome classes.")
    return fit, cal, validation


def _verify_full_predictions(validation: pd.DataFrame, recorded: pd.DataFrame,
                             full: ModelArtifact) -> np.ndarray:
    """Require unchanged labels, keys and exact frozen full-model predictions."""
    if not recorded["split"].eq("validation").all():
        raise ValueError("Only original validation predictions may be loaded.")
    probability_name = f"p_s_{full.model_name}"
    columns = KEY + ["label", probability_name]
    if len(recorded) != len(validation) or recorded.duplicated(KEY).any():
        raise ValueError("Original validation prediction keys differ from the analytical data.")
    joined = validation[KEY + ["label"]].merge(
        recorded[columns], on=KEY, how="left", validate="one_to_one",
        suffixes=("", "_recorded"), sort=False, indicator=True,
    )
    if not joined["_merge"].eq("both").all() or not joined["label"].eq(joined["label_recorded"]).all():
        raise ValueError("Original validation keys or outcome labels have changed.")
    probability = full.predict_proba(validation)[:, 1]
    if not np.array_equal(probability, joined[probability_name].to_numpy(float)):
        raise ValueError("Full-model validation predictions are not exactly the frozen predictions.")
    return probability


def _fit_variant(fit: pd.DataFrame, cal: pd.DataFrame, validation: pd.DataFrame,
                 full: ModelArtifact, name: str, numeric: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    if name == "core_availability_only":
        fit, cal, validation = [_availability_frame(frame, numeric) for frame in (fit, cal, validation)]
    preprocessor = _make_preprocessor(numeric)
    encoded_fit = preprocessor.fit_transform(_feature_frame(fit, numeric))
    encoded_cal = preprocessor.transform(_feature_frame(cal, numeric))
    estimator = clone(full.estimator)
    y_fit = fit["label"].to_numpy(int)
    estimator.fit(encoded_fit, y_fit, sample_weight=compute_sample_weight("balanced", y_fit))
    calibrator = clone(full.calibrator)
    calibrator.fit(_logit(_raw_probability(estimator, encoded_cal)), cal["label"].to_numpy(int))
    artifact = ModelArtifact(FEATURES + numeric, preprocessor, estimator, calibrator, 0.5, name)
    return artifact.predict_proba(validation)[:, 1], {
        "encoded_features": int(encoded_fit.shape[1]),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "fit_seconds": time.perf_counter() - started,
    }


def run_ablation(root: str | Path | None = None, output: str | Path | None = None,
                 seed: int = 142, bootstrap_repeats: int = 500) -> dict[str, Any]:
    """Write exploratory validation-only contrasts without modifying originals."""
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be positive.")
    started = time.perf_counter()
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    output = Path(output) if output is not None else root / "results" / "advanced"
    original = root / "results" / "models"
    selection = json.loads((original / "frozen_selection.json").read_text(encoding="utf-8"))
    if selection["selected_model"] != "lightgbm_balanced":
        raise ValueError("This analysis is specified for the original selected balanced LightGBM model.")
    full = joblib.load(original / "models" / "lightgbm_balanced.joblib")
    if not isinstance(full.estimator, LGBMClassifier) or full.calibrator is None:
        raise ValueError("The original full model must be a calibrated LightGBM model.")
    if full.feature_names != FEATURES + NUMERIC:
        raise ValueError("Original full-model features differ from the specified 15-feature analysis.")
    development = pd.read_parquet(root / "results" / "analytical.parquet",
                                  filters=[("split", "in", ["train", "validation"])])
    roles = pd.read_csv(original / "training_patient_roles.csv")
    fit, cal, validation = _restore_training_roles(development, roles)
    recorded = pd.read_parquet(original / "predictions.parquet", filters=[("split", "==", "validation")])
    probabilities = {"full": _verify_full_predictions(validation, recorded, full)}
    details = {}
    for name, spec in FEATURE_SETS.items():
        if name == "full":
            details[name] = {"status": "original saved model; no refitting"}
            continue
        probabilities[name], details[name] = _fit_variant(fit, cal, validation, full, name, spec["numeric"])
    caches = {name: _MetricCache(validation, probability, .5) for name, probability in probabilities.items()}
    intervals, differences, replicates = _cluster_bootstrap(
        validation, caches, "full", seed=seed, repeats=bootstrap_repeats,
    )
    intervals = intervals[intervals["metric"].isin(METRICS)].copy()
    intervals["feature_set"] = intervals["model"].map({name: spec["label"] for name, spec in FEATURE_SETS.items()})
    intervals["split"] = "validation"
    intervals["exploratory"] = True
    intervals["rows"] = len(validation)
    intervals["patients"] = validation["patient_id"].nunique()
    intervals["direction"] = np.where(intervals["metric"].eq("brier"), "lower_is_better", "higher_is_better")
    differences = differences[differences["metric"].isin(METRICS)].copy()
    differences["split"] = "validation"
    differences["exploratory"] = True
    differences["requested_replicates"] = bootstrap_repeats
    prediction_columns = KEY + ["label", "split"] + (["culture_time"] if "culture_time" in validation else [])
    predictions = validation[prediction_columns].copy()
    for name, probability in probabilities.items():
        predictions[f"p_s_{name}"] = probability
    supports = []
    for antibiotic, frame in validation.groupby("candidate_antibiotic", observed=True, sort=True):
        supports.append({"antibiotic": antibiotic, "rows": len(frame), "patients": frame["patient_id"].nunique(),
                         "resistant": int(frame["label"].eq(0).sum()), "susceptible": int(frame["label"].eq(1).sum())})
    metadata = {
        "analysis": "Feature-group ablation", "evaluation_split": "validation", "exploratory": True,
        "original_model": full.model_name, "full_predictions_exactly_match_original": True,
        "test_rows_loaded": False, "test_used_for_fitting_or_selection": False,
        "purpose": "Check the incremental predictive value of feature groups and availability patterns.",
        "primary_metric": "antibiotic_macro_auprc_r", "metrics": METRICS,
        "macro_definition": "Equal-weight mean across candidate antibiotics with both classes in the evaluated sample; AUPRC is average precision.",
        "brier_definition": "Mean squared susceptibility probability error across all validation rows; lower is better.",
        "paired_difference": "full minus ablation; positive AP/AUROC and negative Brier favor full",
        "core_features": FEATURES, "labs_vitals_features": LABS_VITALS, "prior_history_features": PRIOR_HISTORY,
        "feature_sets": {name: {**spec, "source_feature_count": len(FEATURES) + len(spec["numeric"]),
                               "availability_control": name == "core_availability_only", **details[name]}
                         for name, spec in FEATURE_SETS.items()},
        "availability_control": "For seven numeric support features, observed values become an arbitrary constant zero and absent values remain NaN. Their missing indicators carry availability only. Core features stay unchanged. This derived control makes no claim that missing clinical information is zero.",
        "partitions": {name: {"rows": len(frame), "patients": frame["patient_id"].nunique(),
                              "resistant": int(frame["label"].eq(0).sum()), "susceptible": int(frame["label"].eq(1).sum())}
                       for name, frame in [("proper_fit", fit), ("calibration", cal), ("validation", validation)]},
        "training_roles_source": "results/models/training_patient_roles.csv",
        "preprocessing": "Original helper: proper-fit-only median imputation, scaling and one-hot encoding; separate missing indicators for all support features.",
        "class_weights": "Balanced weights computed on unchanged proper-fit training rows only.",
        "calibration": "Original unweighted sigmoid recipe fitted on the same held-out training patients; no validation recalibration.",
        "hyperparameters": full.estimator.get_params(), "calibrator_parameters": full.calibrator.get_params(),
        "hyperparameters_or_thresholds_tuned": False,
        "bootstrap": {"unit": "patient", "replicates": bootstrap_repeats, "seed": seed,
                      "interval": "95% percentile", "paired": True,
                      "scope": "Fixed fitted models; intervals exclude model-training and prior model-selection uncertainty."},
        "validation_support_by_antibiotic": supports,
        "full_probability_sha256": hashlib.sha256(probabilities["full"].astype("<f8").tobytes()).hexdigest(),
        "limitations": [
            "These validation patients already informed original model and threshold selection; results are exploratory, not a new independent performance estimate.",
            "Feature groups can overlap in information; differences do not isolate causal effects or identify treatment effects.",
            "Availability patterns may reflect documentation or care processes and may not transfer to another setting.",
            "Each feature set uses one unchanged training recipe, so results do not compare each feature set's optimally tuned model.",
            "The original frozen model and test results remain unchanged; no new winning model is selected.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(output / "ablation_metrics.csv", index=False)
    differences.to_csv(output / "ablation_paired_differences.csv", index=False)
    predictions.to_parquet(output / "ablation_predictions.parquet", index=False)
    replicates[["replicate", "model"] + METRICS].to_csv(output / "ablation_bootstrap_replicates.csv", index=False)
    metadata = _json_clean(metadata)
    (output / "ablation_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=142, help="Bootstrap seed; model seeds remain fixed to the original.")
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    args = parser.parse_args()
    metadata = run_ablation(args.root, args.output, args.seed, args.bootstrap_repeats)
    print(json.dumps({"validation_rows": metadata["partitions"]["validation"]["rows"],
                      "test_rows_loaded": metadata["test_rows_loaded"], "elapsed_seconds": metadata["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
