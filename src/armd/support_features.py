"""Build pre-order features without using cohort outcomes.

Lab (14-day) and vital (48-hour) windows follow the source documentation.
Individual measurement times are unavailable for independent checks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


LINK = ["patient_id", "culture_id"]
RAW_LINK = {"anon_id": "patient_id", "order_proc_id_coded": "culture_id"}
NUMERIC_FEATURES = [
    "lab_median_wbc_14d", "lab_median_creatinine_14d",
    "vital_median_heart_rate_48h", "vital_median_temperature_48h",
    "prior_abx_days_ago", "prior_abx_class_count_90d",
    "prior_candidate_resistance_days_ago",
]
CHUNKSIZE = 150_000


def _clean(series):
    values = series.astype("string").str.strip()
    return values.mask(values.str.lower().isin(["", "null", "none", "nan", "na", "n/a"]))


def _read_relevant(path, columns, events, audit):
    """Read a bounded set of columns, retaining only requested event keys."""
    record = {"source_rows_scanned": 0, "matching_event_rows": 0}
    audit[path.name] = record
    desired = list(dict.fromkeys([*RAW_LINK, *columns]))
    if not path.exists():
        record["table_unavailable"] = True
        return pd.DataFrame(columns=[*LINK, *columns])
    present = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(desired) - set(present))
    if missing:
        raise ValueError(f"{path.name}: missing support columns {missing}")
    keep_keys = pd.MultiIndex.from_frame(events[LINK])
    pieces = []
    for chunk in pd.read_csv(path, usecols=desired, dtype="string", chunksize=CHUNKSIZE):
        record["source_rows_scanned"] += len(chunk)
        chunk = chunk.rename(columns=RAW_LINK)
        for column in chunk:
            chunk[column] = _clean(chunk[column])
        if "Period_Day" in chunk:
            observed = record.setdefault("source_period_day_counts", {})
            for value, count in chunk.Period_Day.fillna("MISSING").value_counts().items():
                observed[str(value)] = observed.get(str(value), 0) + int(count)
        mask = pd.MultiIndex.from_frame(chunk[LINK]).isin(keep_keys)
        matched = chunk.loc[mask].copy()
        record["matching_event_rows"] += len(matched)
        if len(matched):
            pieces.append(matched)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=[*LINK, *columns])


def _collapse_summaries(frame, mapping, record):
    """A repeated source summary is not another raw measurement to average."""
    values = list(mapping.values())
    frame = frame.rename(columns=mapping).copy()
    invalid = 0
    for column in values:
        source = frame[column]
        numeric = pd.to_numeric(source, errors="coerce").astype("float64")
        numeric = numeric.where(np.isfinite(numeric))
        invalid += int((source.notna() & numeric.isna()).sum())
        frame[column] = numeric
    group = frame.groupby(LINK, sort=False)[values]
    unique = group.nunique(dropna=True)
    summary = group.first().mask(unique.gt(1)).reset_index()
    record.update({
        "duplicate_event_rows": int(frame.duplicated(LINK).sum()),
        "conflicting_cells_set_missing": int(unique.gt(1).to_numpy().sum()),
        "invalid_numeric_cells_set_missing": invalid,
        "aggregated_event_rows": len(summary),
    })
    return summary


def _history_rows(frame, lag_column, events, record):
    """Require an aligned event time and strictly positive documented prior lag."""
    frame = frame.merge(events, on=LINK, how="left", validate="many_to_one")
    source_time = pd.to_datetime(frame.order_time_jittered_utc, errors="coerce", utc=True)
    aligned = source_time.eq(frame.culture_time) & source_time.notna()
    lag = pd.to_numeric(frame[lag_column], errors="coerce").astype("float64")
    finite = pd.Series(np.isfinite(lag), index=frame.index)
    positive = finite & lag.gt(0)
    record.update({
        "event_timestamp_mismatch_or_missing_rows_excluded": int((~aligned).sum()),
        "same_day_or_future_rows_excluded": int((finite & lag.le(0)).sum()),
        "invalid_lag_rows_excluded": int((~finite).sum()),
        "prior_lag_convention": "positive days before culture order; zero and negative excluded",
    })
    frame = frame.loc[aligned & positive].copy()
    frame[lag_column] = lag.loc[frame.index]
    record["eligible_history_rows"] = len(frame)
    return frame


def _merge_summary(frame, summary, keys):
    """Keep analytical key dtypes even when an optional summary is empty."""
    summary = summary.copy()
    for key in keys:
        summary[key] = summary[key].astype(frame[key].dtype)
    return frame.merge(summary, on=keys, how="left", validate="many_to_one", sort=False)


def enrich_features(frame, raw_dir, audit):
    """Add seven fixed features, preserving the analytical grain.

    No labels, current organism or fitted cohort statistics are used.
    Missing history is unknown; recorded counts do not imply complete history.
    """
    raw_dir = Path(raw_dir)
    original_index = frame.index.copy()
    output = frame.copy()
    existing = set(NUMERIC_FEATURES).intersection(frame.columns)
    if existing:
        raise ValueError(f"Support features already exist: {sorted(existing)}")
    events = frame[[*LINK, "culture_time"]].drop_duplicates().copy()
    events["culture_time"] = pd.to_datetime(events.culture_time, errors="coerce", utc=True)
    if events.duplicated(LINK).any() or events[LINK].isna().any().any():
        raise ValueError("Support features require one nonmissing patient/event reference time")
    if events.culture_time.isna().any():
        raise ValueError("Support features require a valid culture time")
    profile = {"prediction_time": "culture order", "numeric_features": NUMERIC_FEATURES,
               "input_rows": len(output), "tables": {}}
    audit["support_features"] = profile
    tables = profile["tables"]

    path = raw_dir / "microbiology_cultures_labs.csv"
    mapping = {"median_wbc": NUMERIC_FEATURES[0], "median_cr": NUMERIC_FEATURES[1]}
    labs = _read_relevant(path, ["Period_Day", *mapping], events, tables)
    period = pd.to_numeric(labs.Period_Day, errors="coerce")
    tables[path.name]["undocumented_window_rows_excluded"] = int((~period.eq(14)).sum())
    labs = labs.loc[period.eq(14)].copy()
    summary = _collapse_summaries(labs, mapping, tables[path.name])
    output = _merge_summary(output, summary, LINK)

    path = raw_dir / "microbiology_cultures_vitals.csv"
    mapping = {"median_heartrate": NUMERIC_FEATURES[2], "median_temp": NUMERIC_FEATURES[3]}
    vitals = _read_relevant(path, list(mapping), events, tables)
    summary = _collapse_summaries(vitals, mapping, tables[path.name])
    output = _merge_summary(output, summary, LINK)

    path = raw_dir / "microbiology_cultures_antibiotic_class_exposure.csv"
    lag = "time_to_culturetime"
    exposure = _read_relevant(path, ["order_time_jittered_utc", "antibiotic_class", lag], events, tables)
    exposure = _history_rows(exposure, lag, events, tables[path.name])
    recency = exposure.groupby(LINK, sort=False)[lag].min().rename(NUMERIC_FEATURES[4]).reset_index()
    exposure["antibiotic_class"] = exposure.antibiotic_class.str.lower()
    recent = exposure.loc[exposure[lag].le(90)].dropna(subset=["antibiotic_class"])
    counts = recent.groupby(LINK, sort=False).antibiotic_class.nunique().rename(NUMERIC_FEATURES[5]).reset_index()
    for summary in [recency, counts]:
        output = _merge_summary(output, summary, LINK)
    tables[path.name]["events_with_observed_prior_exposure"] = len(recency)
    tables[path.name]["events_with_observed_class_in_90d"] = len(counts)

    path = raw_dir / "microbiology_cultures_microbial_resistance.csv"
    lag = "resistant_time_to_culturetime"
    resistance = _read_relevant(path, ["order_time_jittered_utc", "antibiotic", lag], events, tables)
    resistance = _history_rows(resistance, lag, events, tables[path.name])
    # Import lazily to keep build.py free to import this module during startup.
    from .build import canonical_drug
    resistance["candidate_antibiotic"] = resistance.antibiotic.map(canonical_drug)
    resistance = resistance.dropna(subset=["candidate_antibiotic"])
    keys = [*LINK, "candidate_antibiotic"]
    summary = resistance.groupby(keys, sort=False)[lag].min().rename(NUMERIC_FEATURES[6]).reset_index()
    output = _merge_summary(output, summary, keys)
    tables[path.name]["aggregated_event_candidate_rows"] = len(summary)

    if len(output) != len(frame):
        raise AssertionError("Support joins changed the analytical row count")
    for column in NUMERIC_FEATURES:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("float64")
    output.index = original_index
    if not output[frame.columns].equals(frame):
        raise AssertionError("Support joins changed original rows or ordering")
    profile["output_rows"] = len(output)
    profile["feature_nonmissing_rows"] = {name: int(output[name].notna().sum()) for name in NUMERIC_FEATURES}
    profile["timing_limitations"] = [
        "Laboratory and vital-sign measurement times are not released; pre-order windows follow the author manuscript.",
        "Historical lags follow the README convention and positive prior-history observations; exact result availability times cannot be reconstructed.",
        "No observation-coverage table is released; absent history cannot establish no prior exposure or resistance.",
        "Measurement units are not documented in this release and are retained without conversion.",
    ]
    return output
