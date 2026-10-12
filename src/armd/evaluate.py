"""Fit and evaluate patient-disjoint ARMD models.

Probabilities denote susceptible=1; resistant=0. Keep the supplied splits,
and divide training patients again for model fitting and calibration.
"""

from __future__ import annotations

import json
import platform
import time
import warnings
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


FEATURES = [
    "candidate_antibiotic", "age_group", "gender_code", "culture_type",
    "ward_ip", "ward_op", "ward_er", "ward_icu",
]
BOOTSTRAP_METRICS = [
    "auroc_s", "auprc_s", "auprc_r", "class_macro_auprc",
    "antibiotic_macro_auroc", "antibiotic_macro_auprc_s",
    "antibiotic_macro_auprc_r", "antibiotic_macro_class_auprc",
    "balanced_accuracy", "precision_r", "recall_r", "brier", "log_loss",
]


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _numeric_features(df: pd.DataFrame) -> list[str]:
    """Only explicitly audited support features are allowed into a model."""
    spec_path = Path(__file__).with_name("support_feature_spec.json")
    if not spec_path.exists():
        return []
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    names = spec["numeric_features"]
    if len(names) != len(set(names)) or any(name in FEATURES for name in names):
        raise ValueError("Support feature specification contains duplicate feature names.")
    forbidden = {"patient_id", "culture_id", "organism", "label", "split", "culture_time",
                 "susceptibility", "result_time", "site", "dataset"}
    if forbidden.intersection(names):
        raise ValueError("Support feature specification includes a forbidden predictor.")
    return [name for name in names if name in df]


def _feature_frame(df: pd.DataFrame, numeric_features: list[str] | None = None) -> pd.DataFrame:
    """Keep unobserved values distinct from a recorded zero, without fitted stats."""
    frame = df[FEATURES].copy()
    # Avoid treating 0 and 0.0 as different categories after nullable CSV imports.
    for column in [c for c in FEATURES if c.startswith("ward_")]:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & (values.isna() | ~values.isin([0, 1]))
        if invalid.any():
            raise ValueError(f"{column} must contain 0, 1 or missing values.")
        frame[column] = values.astype("Int64")
    frame = frame.astype("string").fillna("__MISSING__").astype(object)
    for name in numeric_features or []:
        values = pd.to_numeric(df[name], errors="coerce")
        if (df[name].notna() & values.isna()).any():
            raise ValueError(f"Audited numeric feature {name} contains nonnumeric values.")
        frame[name] = values.replace([np.inf, -np.inf], np.nan).astype(float)
    return frame


def _make_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    transformations = [("categorical", OneHotEncoder(handle_unknown="ignore",
                         sparse_output=True, dtype=np.float64), FEATURES)]
    if numeric_features:
        transformations.extend([
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler(with_mean=False)),
            ]), numeric_features),
            # Include indicators even when missingness first appears after fitting.
            ("missing", MissingIndicator(features="all", sparse=True), numeric_features),
        ])
    return ColumnTransformer(transformations, remainder="drop", sparse_threshold=1.0)


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)).reshape(-1, 1)


def _raw_probability(estimator: Any, encoded: Any) -> np.ndarray:
    with warnings.catch_warnings():
        # LightGBM adds synthetic names; the fitted transformer fixes column order.
        warnings.filterwarnings("ignore", category=UserWarning,
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")
        return estimator.predict_proba(encoded)[:, 1]


@dataclass
class ModelArtifact:
    """A self-contained fitted model; .predict_proba accepts unprocessed records."""

    feature_names: list[str]
    preprocessor: Any
    estimator: Any
    calibrator: Any
    threshold_s: float
    model_name: str
    baseline_prevalence_s: float | None = None
    candidate_prevalence_s: dict[str, float] | None = None

    def predict_proba(self, df: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        if self.estimator is None:
            p = np.full(len(df), self.baseline_prevalence_s, dtype=float)
            if self.candidate_prevalence_s is not None:
                p = (df["candidate_antibiotic"].astype(str)
                     .map(self.candidate_prevalence_s)
                     .fillna(self.baseline_prevalence_s).to_numpy(float))
        else:
            numeric_features = [name for name in self.feature_names if name not in FEATURES]
            encoded = self.preprocessor.transform(_feature_frame(df, numeric_features))
            p = _raw_probability(self.estimator, encoded)
            if calibrated and self.calibrator is not None:
                p = self.calibrator.predict_proba(_logit(p))[:, 1]
        return np.column_stack([1 - p, p])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df)[:, 1] >= self.threshold_s).astype(np.int8)


class _BinaryMetricCache:
    """Cache score ordering so the patient bootstrap does not sort repeatedly."""

    def __init__(self, y: np.ndarray, p: np.ndarray, threshold: float):
        self.y = np.asarray(y, dtype=float)
        self.p = np.asarray(p, dtype=float)
        if self.y.ndim != 1 or self.p.shape != self.y.shape:
            raise ValueError("Labels and probabilities must be equal-length one-dimensional arrays.")
        if not np.isin(self.y, [0, 1]).all():
            raise ValueError("Metrics require binary labels.")
        if not np.isfinite(self.p).all() or ((self.p < 0) | (self.p > 1)).any():
            raise ValueError("Metrics require finite probabilities between zero and one.")
        self.pred = self.p >= threshold
        self.order = np.argsort(-self.p, kind="stable")
        sorted_p = self.p[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(sorted_p)), len(y) - 1]
        self.brier = (self.p - self.y) ** 2
        epsilon = np.finfo(float).eps
        clip = np.clip(self.p, epsilon, 1 - epsilon)
        self.loss = -self.y * np.log(clip) - (1 - self.y) * np.log(1 - clip)

    def metrics(self, weights: np.ndarray | None = None) -> dict[str, float]:
        w = np.ones(len(self.y)) if weights is None else np.asarray(weights, float)
        if w.shape != self.y.shape or not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("Metric weights must be finite, nonnegative and match the rows.")
        total = w.sum()
        if total <= 0:
            return {k: np.nan for k in [
                "n", "n_s", "n_r", "prevalence_s", "auroc_s", "auprc_s",
                "auprc_r", "class_macro_auprc", "balanced_accuracy", "precision_r",
                "recall_r", "f1_r", "brier", "log_loss", "predicted_r_fraction",
            ]}
        positive = np.dot(w, self.y)
        negative = total - positive
        tp_s = np.dot(w, self.y * self.pred)
        tp_r = np.dot(w, (1 - self.y) * (~self.pred))
        predicted_r = np.dot(w, ~self.pred)
        recall_s = tp_s / positive if positive else np.nan
        recall_r = tp_r / negative if negative else np.nan
        precision_r = tp_r / predicted_r if predicted_r else 0.0
        f1_r = (2 * precision_r * recall_r / (precision_r + recall_r)
                if (precision_r + recall_r) > 0 else 0.0)
        auc = ap_s = ap_r = np.nan
        if positive > 0 and negative > 0:
            yw = self.y[self.order] * w[self.order]
            nw = (1 - self.y[self.order]) * w[self.order]
            tps = np.cumsum(yw)[self.ends]
            fps = np.cumsum(nw)[self.ends]
            tpr, fpr = tps / positive, fps / negative
            auc = float(np.trapezoid(np.r_[0, tpr], np.r_[0, fpr]))
            precision_s = np.divide(tps, tps + fps, out=np.zeros_like(tps),
                                    where=(tps + fps) > 0)
            ap_s = float(np.dot(np.diff(np.r_[0, tpr]), precision_s))
            group_s = np.diff(np.r_[0, tps])
            group_r = np.diff(np.r_[0, fps])
            r_tp = np.cumsum(group_r[::-1])
            r_fp = np.cumsum(group_s[::-1])
            r_precision = np.divide(r_tp, r_tp + r_fp, out=np.zeros_like(r_tp),
                                    where=(r_tp + r_fp) > 0)
            ap_r = float(np.dot(np.diff(np.r_[0, r_tp / negative]), r_precision))
        return {
            "n": float(total), "n_s": float(positive), "n_r": float(negative),
            "prevalence_s": float(positive / total), "auroc_s": auc,
            "auprc_s": ap_s, "auprc_r": ap_r,
            "class_macro_auprc": (ap_s + ap_r) / 2,
            "balanced_accuracy": (recall_s + recall_r) / 2,
            "precision_r": float(precision_r), "recall_r": float(recall_r),
            "f1_r": float(f1_r), "brier": float(np.dot(w, self.brier) / total),
            "log_loss": float(np.dot(w, self.loss) / total),
            "predicted_r_fraction": float(predicted_r / total),
        }


class _MetricCache:
    def __init__(self, df: pd.DataFrame, p: np.ndarray, threshold: float):
        y = df["label"].to_numpy(int)
        self.overall = _BinaryMetricCache(y, p, threshold)
        self.groups = []
        for antibiotic in sorted(df["candidate_antibiotic"].astype(str).unique()):
            index = np.flatnonzero(df["candidate_antibiotic"].astype(str).to_numpy() == antibiotic)
            self.groups.append((index, _BinaryMetricCache(y[index], p[index], threshold)))

    def metrics(self, weights: np.ndarray | None = None) -> dict[str, float]:
        result = self.overall.metrics(weights)
        group_metrics = [cache.metrics(None if weights is None else weights[index])
                         for index, cache in self.groups]
        mapping = {
            "antibiotic_macro_auroc": "auroc_s",
            "antibiotic_macro_auprc_s": "auprc_s",
            "antibiotic_macro_auprc_r": "auprc_r",
            "antibiotic_macro_class_auprc": "class_macro_auprc",
        }
        for name, group_name in mapping.items():
            values = np.array([m[group_name] for m in group_metrics])
            valid = np.isfinite(values)
            result[name] = float(values[valid].mean()) if valid.any() else np.nan
        result["antibiotics_evaluable"] = sum(np.isfinite(m["auroc_s"]) for m in group_metrics)
        result["antibiotics_present"] = len(group_metrics)
        return result


def _select_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    thresholds, scores = thresholds[valid], ((tpr + 1 - fpr) / 2)[valid]
    # Always include 0.5, including the constant-prevalence model.
    default_score = _BinaryMetricCache(y, p, 0.5).metrics()["balanced_accuracy"]
    thresholds = np.r_[thresholds, 0.5]
    scores = np.r_[scores, default_score]
    best = np.flatnonzero(np.isclose(scores, np.nanmax(scores), atol=1e-12, rtol=0))
    return float(thresholds[best[np.argmin(abs(thresholds[best] - 0.5))]])


def _calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    bins = np.minimum((np.clip(p, 0, 1) * 10).astype(int), 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * np.sum(bins == b)
              for b in range(10) if np.any(bins == b)) / len(y)
    result = {"ece_10_equal_width": float(ece), "mean_predicted_s": float(p.mean())}
    if len(np.unique(y)) == 2 and np.std(_logit(p)) > 1e-8:
        diagnostic = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
        diagnostic.fit(_logit(p), y)
        result.update(calibration_intercept=float(diagnostic.intercept_[0]),
                      calibration_slope=float(diagnostic.coef_[0, 0]))
    else:
        result.update(calibration_intercept=np.nan, calibration_slope=np.nan)
    return result


def _validate_input(df: pd.DataFrame) -> None:
    required = set(FEATURES + ["patient_id", "culture_id", "organism", "label", "split", "culture_time"])
    missing = sorted(required - set(df))
    if missing:
        raise ValueError(f"Analytical dataset is missing columns: {missing}")
    if df["patient_id"].isna().any() or df["label"].isna().any():
        raise ValueError("Patient identifiers and labels must be observed.")
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError("Only resistant=0 and susceptible=1 labels are accepted.")
    if set(df["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("The split column must contain train, validation and test only.")
    overlap = df.groupby("patient_id", observed=True)["split"].nunique().max()
    if overlap != 1:
        raise ValueError("Patient overlap between splits would leak patient information.")
    culture_overlap = df.groupby(["patient_id", "culture_id"], observed=True)["split"].nunique().max()
    if culture_overlap != 1:
        raise ValueError("Culture overlap between splits would leak information.")
    key = ["patient_id", "culture_id", "organism", "candidate_antibiotic"]
    if df.duplicated(key).any():
        raise ValueError("Analytical grain has duplicate patient/culture/organism/antibiotic keys.")
    for name, group in df.groupby("split", observed=True):
        if group["label"].nunique() != 2:
            raise ValueError(f"Split {name} needs both classes for evaluation.")


def _training_partition(train: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train["patient_id"].nunique() < 5:
        raise ValueError("At least five training patients are required for held-out calibration.")
    splitter = GroupShuffleSplit(n_splits=20, test_size=0.2, random_state=seed)
    for fit_idx, cal_idx in splitter.split(train, groups=train["patient_id"]):
        fit, cal = train.iloc[fit_idx].copy(), train.iloc[cal_idx].copy()
        if fit["label"].nunique() == cal["label"].nunique() == 2:
            return fit, cal
    raise ValueError("Could not create a patient-held-out training calibration set with both labels.")


def _group_results(df: pd.DataFrame, predictions: dict[str, np.ndarray],
                   thresholds: dict[str, float]) -> pd.DataFrame:
    frame = df.copy()
    year = pd.to_datetime(frame["culture_time"], errors="coerce").dt.year
    era_start = (year // 5 * 5).astype("Int64").astype("string")
    era_end = (year // 5 * 5 + 4).astype("Int64").astype("string")
    frame["calendar_era"] = era_start + "-" + era_end
    fields = ["candidate_antibiotic", "organism", "culture_type", "gender_code", "age_group", "calendar_era"]
    if "dataset" in frame:
        fields.append("dataset")
    # Site is reported only if an actual site field is supplied.
    for site_column in ["site", "site_id"]:
        if site_column in frame and frame[site_column].notna().any():
            fields.append(site_column)
    rows = []
    for field in fields:
        values = frame[field].astype("string").fillna("__MISSING__")
        for value in sorted(values.unique()):
            idx = np.flatnonzero((values == value).to_numpy())
            subset = frame.iloc[idx]
            y = subset["label"].to_numpy(int)
            for model, p in predictions.items():
                metrics = _BinaryMetricCache(y, p[idx], thresholds[model]).metrics()
                rows.append({"split": "test", "model": model, "group_field": field,
                             "group_value": value, "patients": subset["patient_id"].nunique(),
                             "small_n": len(idx) < 100 or min(np.sum(y == 0), np.sum(y == 1)) < 20,
                             "both_classes": len(np.unique(y)) == 2, **metrics})
    return pd.DataFrame(rows)


def _cluster_bootstrap(test: pd.DataFrame, caches: dict[str, _MetricCache],
                       selected: str, seed: int, repeats: int = 200
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patient_code, patients = pd.factorize(test["patient_id"], sort=True)
    n_patients = len(patients)
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(repeats):
        drawn = rng.integers(0, n_patients, size=n_patients)
        counts = np.bincount(drawn, minlength=n_patients)
        weights = counts[patient_code].astype(float)
        for model, cache in caches.items():
            metrics = cache.metrics(weights)
            rows.append({"replicate": replicate, "model": model,
                         **{k: metrics[k] for k in BOOTSTRAP_METRICS}})
    replicates = pd.DataFrame(rows)
    intervals = []
    point_estimates = {name: cache.metrics() for name, cache in caches.items()}
    for model, cache in caches.items():
        point = point_estimates[model]
        model_rows = replicates[replicates["model"] == model]
        for metric in BOOTSTRAP_METRICS:
            values = model_rows[metric].dropna()
            intervals.append({"model": model, "metric": metric, "estimate": point[metric],
                              "ci_low": values.quantile(0.025), "ci_high": values.quantile(0.975),
                              "valid_replicates": len(values), "requested_replicates": repeats,
                              "method": "patient-cluster percentile bootstrap"})
    differences = []
    # Models share exactly the same resampled patients in every paired contrast.
    selected_rows = replicates[replicates["model"] == selected].set_index("replicate")
    for other in caches:
        if other == selected:
            continue
        other_rows = replicates[replicates["model"] == other].set_index("replicate")
        for metric in BOOTSTRAP_METRICS:
            values = (selected_rows[metric] - other_rows[metric]).dropna()
            differences.append({"model_a": selected, "model_b": other, "metric": metric,
                                "difference_a_minus_b": point_estimates[selected][metric] - point_estimates[other][metric],
                                "ci_low": values.quantile(0.025), "ci_high": values.quantile(0.975),
                                "valid_replicates": len(values),
                                "direction": "lower_is_better" if metric in {"brier", "log_loss"} else "higher_is_better"})
    return pd.DataFrame(intervals), pd.DataFrame(differences), replicates


def _permutation_importance(validation: pd.DataFrame, artifact: ModelArtifact,
                            seed: int, repeats: int = 3,
                            max_rows: int = 10000) -> pd.DataFrame:
    """Validation-only exploratory importance on the original input columns."""
    sample = validation
    if len(sample) > max_rows:
        # This diagnostic sample is never used for fitting or model selection.
        sample = sample.sample(n=max_rows, random_state=seed)
    sample = sample.reset_index(drop=True)
    p = artifact.predict_proba(sample)[:, 1]
    baseline = _MetricCache(sample, p, artifact.threshold_s).metrics()["antibiotic_macro_auprc_r"]
    rng = np.random.default_rng(seed)
    rows = []
    for feature in artifact.feature_names:
        scores = []
        for repeat in range(repeats):
            shuffled = sample.copy()
            shuffled[feature] = sample[feature].to_numpy()[rng.permutation(len(sample))]
            p = artifact.predict_proba(shuffled)[:, 1]
            # Antibiotic groups always use the real candidate, including when
            # candidate itself is the permuted predictor.
            score = _MetricCache(sample, p, artifact.threshold_s).metrics()["antibiotic_macro_auprc_r"]
            scores.append(baseline - score)
        rows.append({"model": artifact.model_name, "feature": feature,
                     "split": "validation", "metric": "antibiotic_macro_auprc_r",
                     "baseline_score": baseline, "mean_score_decrease": float(np.mean(scores)),
                     "std_score_decrease": float(np.std(scores, ddof=1)) if repeats > 1 else 0.0,
                     "rows": len(sample), "repeats": repeats})
    return pd.DataFrame(rows).sort_values("mean_score_decrease", ascending=False)


def _plots(test: pd.DataFrame, predictions: dict[str, np.ndarray],
           thresholds: dict[str, float], output: Path) -> None:
    y = test["label"].to_numpy(int)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for model, p in predictions.items():
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, "o-", markersize=3, label=model.replace("_", " "))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted susceptibility probability",
           ylabel="Observed susceptible fraction", title="Test calibration (quantile bins)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "calibration_test.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for model, p in predictions.items():
        for ax, outcome, probability in [(axes[0], 1 - y, 1 - p), (axes[1], y, p)]:
            precision, recall, _ = precision_recall_curve(outcome, probability)
            ax.plot(recall, precision, linewidth=1.3, label=model.replace("_", " "))
    for ax, prevalence, title in [(axes[0], (1 - y).mean(), "Resistance"), (axes[1], y.mean(), "Susceptibility")]:
        ax.axhline(prevalence, color="gray", linestyle="--", label="Class prevalence")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision", title=f"Test {title.lower()} precision-recall")
    axes[1].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(output / "precision_recall_test.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4))
    for ax, (model, p) in zip(axes.flat, predictions.items()):
        matrix = confusion_matrix(y, (p >= thresholds[model]).astype(int), labels=[0, 1])
        ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for col in range(2):
                ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center",
                        color="white" if matrix[row, col] > matrix.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["R", "S"], yticklabels=["R", "S"],
               xlabel="Predicted", ylabel="Observed", title=model.replace("_", " "))
    for ax in list(axes.flat)[len(predictions):]:
        ax.axis("off")
    fig.suptitle("Test confusion matrices at validation-selected thresholds", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrices_test.png", dpi=180)
    plt.close(fig)


def run_evaluation(df: pd.DataFrame, out_dir: str | Path, seed: int = 42,
                   bootstrap_repeats: int = 200) -> dict[str, Any]:
    """Fit four model variants and two baselines with fixed settings.

    Only proper-fit training rows determine class weights. Validation and test
    retain their class balance. Validation drug-macro resistance AP selects the
    model, with Brier breaking ties. Freeze choices before evaluating test labels.
    """
    started = time.perf_counter()
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be at least 1.")
    _validate_input(df)
    numeric_features = _numeric_features(df)
    feature_names = FEATURES + numeric_features
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifact_dir, figures = out / "models", out / "figures"
    artifact_dir.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    partitions = {name: df[df["split"] == name].reset_index(drop=True)
                  for name in ["train", "validation", "test"]}
    fit, cal = _training_partition(partitions["train"], seed)
    fit_ids, cal_ids = set(fit["patient_id"]), set(cal["patient_id"])
    assert fit_ids.isdisjoint(cal_ids)
    train_roles = partitions["train"][["patient_id"]].drop_duplicates().copy()
    train_roles["training_role"] = np.where(train_roles["patient_id"].isin(fit_ids), "proper_fit", "calibration")
    train_roles.to_csv(out / "training_patient_roles.csv", index=False)
    preprocessor = _make_preprocessor(numeric_features)
    x_fit = preprocessor.fit_transform(_feature_frame(fit, numeric_features))
    x_cal = preprocessor.transform(_feature_frame(cal, numeric_features))
    y_fit, y_cal = fit["label"].to_numpy(int), cal["label"].to_numpy(int)
    weights = compute_sample_weight("balanced", y_fit)
    prevalence_s = float(y_fit.mean())
    candidate_prevalence = fit.groupby("candidate_antibiotic", observed=True)["label"].mean().to_dict()
    artifacts = {
        "prevalence_baseline": ModelArtifact(feature_names, None, None, None, 0.5, "prevalence_baseline", prevalence_s),
        "antibiotic_baseline": ModelArtifact(feature_names, None, None, None, 0.5, "antibiotic_baseline", prevalence_s, candidate_prevalence),
    }
    model_specs = {
        "logistic_unweighted": LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed),
        "logistic_balanced": LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed),
        "lightgbm_unweighted": LGBMClassifier(n_estimators=220, learning_rate=0.04, num_leaves=15,
            min_child_samples=50, reg_lambda=1.0, colsample_bytree=0.9, random_state=seed,
            n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True),
        "lightgbm_balanced": LGBMClassifier(n_estimators=220, learning_rate=0.04, num_leaves=15,
            min_child_samples=50, reg_lambda=1.0, colsample_bytree=0.9, random_state=seed,
            n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True),
    }
    fit_details = []
    for name, estimator in model_specs.items():
        tick = time.perf_counter()
        sample_weight = weights if name.endswith("_balanced") else None
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            estimator.fit(x_fit, y_fit, sample_weight=sample_weight)
            raw_cal = _raw_probability(estimator, x_cal)
            calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=seed)
            calibrator.fit(_logit(raw_cal), y_cal)
        artifacts[name] = ModelArtifact(feature_names, preprocessor, estimator, calibrator, 0.5, name)
        fit_details.append({"model": name, "fit_seconds": time.perf_counter() - tick,
                            "parameters": estimator.get_params(),
                            "warnings": list(dict.fromkeys(str(w.message) for w in recorded)),
                            "calibration_intercept": float(calibrator.intercept_[0]),
                            "calibration_slope": float(calibrator.coef_[0, 0])})
        encoded_names = preprocessor.get_feature_names_out()
        values = (estimator.coef_[0] if name.startswith("logistic") else estimator.feature_importances_)
        pd.DataFrame({"encoded_feature": encoded_names, "value": values,
                      "measure": "coefficient_for_S" if name.startswith("logistic") else "split_count"}) \
            .sort_values("value", key=lambda s: s.abs(), ascending=False) \
            .to_csv(out / f"feature_importance_{name}.csv", index=False)

    # Freeze all model choices before any call that evaluates test labels.
    validation = partitions["validation"]
    y_val = validation["label"].to_numpy(int)
    validation_rows, thresholds = [], {}
    for name, artifact in artifacts.items():
        p_val = artifact.predict_proba(validation)[:, 1]
        threshold = _select_threshold(y_val, p_val)
        artifact.threshold_s = thresholds[name] = threshold
        validation_rows.append({"model": name, "split": "validation", "threshold_s": threshold,
                                **_MetricCache(validation, p_val, threshold).metrics()})
    validation_metrics = pd.DataFrame(validation_rows)
    eligible = validation_metrics[~validation_metrics["model"].str.contains("baseline")]
    selected = str(eligible.sort_values(["antibiotic_macro_auprc_r", "brier", "model"],
                                       ascending=[False, True, True]).iloc[0]["model"])
    for name, artifact in artifacts.items():
        joblib.dump(artifact, artifact_dir / f"{name}.joblib", compress=3)
    (out / "frozen_selection.json").write_text(json.dumps({
        "selected_model": selected, "selection_split": "validation",
        "selection_metric": "antibiotic_macro_auprc_r; lower Brier breaks ties",
        "threshold_metric": "validation balanced accuracy; nearest 0.5 breaks ties",
        "thresholds_s": thresholds, "test_used_for_selection": False,
    }, indent=2), encoding="utf-8")
    _permutation_importance(validation, artifacts[selected], seed + 2).to_csv(
        out / "validation_permutation_importance.csv", index=False)

    overall_rows, calibration_rows, prediction_frames = [], [], []
    test_predictions, test_caches = {}, {}
    for split in ["validation", "test"]:
        subset = partitions[split]
        y = subset["label"].to_numpy(int)
        keys = [c for c in ["patient_id", "culture_id", "organism", "candidate_antibiotic", "label", "split", "culture_time"] if c in subset]
        pred_frame = subset[keys].copy()
        for name, artifact in artifacts.items():
            p = artifact.predict_proba(subset)[:, 1]
            raw = artifact.predict_proba(subset, calibrated=False)[:, 1]
            cache = _MetricCache(subset, p, thresholds[name])
            metrics = cache.metrics()
            fixed = _BinaryMetricCache(y, p, 0.5).metrics()
            overall_rows.append({"split": split, "model": name, "selected_model": name == selected,
                                 "threshold_s": thresholds[name], "patients": subset["patient_id"].nunique(),
                                 **metrics, **_calibration_metrics(y, p),
                                 "balanced_accuracy_at_0_5": fixed["balanced_accuracy"],
                                 "precision_r_at_0_5": fixed["precision_r"], "recall_r_at_0_5": fixed["recall_r"]})
            for state, probability in [("uncalibrated", raw), ("calibrated", p)]:
                diagnostic = _BinaryMetricCache(y, probability, 0.5).metrics()
                calibration_rows.append({"split": split, "model": name, "state": state,
                                         "brier": diagnostic["brier"], "log_loss": diagnostic["log_loss"],
                                         **_calibration_metrics(y, probability)})
            pred_frame[f"p_s_{name}"] = p
            pred_frame[f"p_s_raw_{name}"] = raw
            pred_frame[f"predicted_label_{name}"] = (p >= thresholds[name]).astype(np.int8)
            if split == "test":
                test_predictions[name], test_caches[name] = p, cache
        prediction_frames.append(pred_frame)
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(out / "model_comparison.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(out / "calibration_diagnostics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    groups = _group_results(partitions["test"], test_predictions, thresholds)
    groups.to_csv(out / "subgroup_metrics.csv", index=False)
    intervals, paired, replicates = _cluster_bootstrap(partitions["test"], test_caches, selected,
                                                      seed + 1, repeats=bootstrap_repeats)
    intervals.to_csv(out / "test_confidence_intervals.csv", index=False)
    paired.to_csv(out / "paired_model_differences.csv", index=False)
    replicates.to_csv(out / "bootstrap_replicates.csv", index=False)
    _plots(partitions["test"], test_predictions, thresholds, figures)
    chosen = overall[(overall["model"] == selected) & (overall["split"] == "test")].iloc[0].to_dict()
    summary = {
        "seed": seed, "features": feature_names, "categorical_features": FEATURES,
        "numeric_features": numeric_features, "positive_label": "susceptible=1", "resistant_label": 0,
        "prediction_time": "culture order; organism identity and susceptibility results excluded",
        "selected_model": selected, "selected_test_metrics": chosen,
        "selection_metric": "validation antibiotic-macro resistance average precision; lower Brier breaks ties",
        "thresholds_s": thresholds, "proper_fit_rows": len(fit), "proper_fit_patients": len(fit_ids),
        "calibration_rows": len(cal), "calibration_patients": len(cal_ids),
        "training_prevalence_s": prevalence_s, "encoded_features": int(x_fit.shape[1]),
        "class_weights_proper_fit": {str(label): float(len(y_fit) / (2 * np.sum(y_fit == label))) for label in [0, 1]},
        "calibration": "unweighted sigmoid on logit(probability), 20% training patients held out",
        "missing_values": "categorical: explicit __MISSING__ category; numeric: proper-fit median imputation with a separate missing indicator for every numeric feature; all-missing proper-fit columns use a constant zero plus the missing indicator",
        "numeric_scaling": "standard deviation learned only on proper-fit imputed values; applies to both model families",
        "all_missing_proper_fit_numeric": [name for name in numeric_features if fit[name].isna().all()],
        "unknown_categories": "one-hot all-zero at inference; encoder fitted on proper-fit training only",
        "hyperparameters": "fixed in advance; weighted and unweighted variants compared",
        "macro_definition": "unweighted mean across antibiotics with both outcome classes; AP is average precision",
        "single_class_groups": "AUROC and AP omitted; group counts and flags retained",
        "bootstrap": {"unit": "patient", "replicates": bootstrap_repeats, "interval": "95% percentile",
                      "paired": True, "scope": "fixed fitted models; does not include retraining uncertainty"},
        "small_group_flag": "fewer than 100 rows or fewer than 20 observations in either class",
        "site_validation": "reported only if an observed site column is supplied; ward flags are not sites",
        "subgroup_fields": groups["group_field"].drop_duplicates().tolist(),
        "permutation_importance": "selected model only; 3 permutations per original feature on up to 10,000 validation rows; antibiotic-macro resistance AP decrease; exploratory and not causal",
        "models": fit_details,
        "versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "lightgbm", "matplotlib", "joblib"]},
        "python": platform.python_version(), "elapsed_seconds": time.perf_counter() - started,
    }
    summary = _json_clean(summary)
    (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
