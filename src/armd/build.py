"""Construct one patient x culture x organism x candidate antibiotic per row."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

KEY = ['patient_id', 'culture_id', 'organism', 'candidate_antibiotic']
LINK = ['patient_id', 'culture_id']
FEATURES = ['candidate_antibiotic', 'age_group', 'gender_code', 'culture_type',
            'ward_ip', 'ward_op', 'ward_er', 'ward_icu']
RAW_KEYS = {'anon_id': 'patient_id', 'order_proc_id_coded': 'culture_id'}


def clean_text(series):
    s = series.astype('string').str.strip()
    return s.mask(s.str.lower().isin(['', 'null', 'none', 'nan', 'na', 'n/a']))


def stable_fraction(value, salt):
    return int.from_bytes(hashlib.sha256((salt + '|' + str(value)).encode()).digest()[:8], 'big') / 2**64


def sample_patients(series, fraction, seed):
    lookup = {x: stable_fraction(x, f'sample:{seed}') < fraction for x in series.dropna().unique()}
    return series.map(lookup).fillna(False).astype(bool)


def patient_split(series, seed=42):
    def assign(value):
        p = stable_fraction(value, f'split:{seed}')
        return 'train' if p < .6 else ('validation' if p < .8 else 'test')
    return series.map({x: assign(x) for x in series.unique()})


def canonical_drug(value):
    if pd.isna(value):
        return pd.NA
    name = str(value).strip().lower()
    name = re.sub(r'\s*/\s*', '/', name)
    aliases = {
        'amoxicillin-clavulanate': 'amoxicillin/clavulanate',
        'amoxicillin/clavulanic acid': 'amoxicillin/clavulanate',
        'amoxicillin-clavulanic acid': 'amoxicillin/clavulanate',
        'amoxicillin clavulanate': 'amoxicillin/clavulanate',
        'piperacillin-tazobactam': 'piperacillin/tazobactam',
        'piperacillin + tazobactam': 'piperacillin/tazobactam',
        'trimethoprim-sulfamethoxazole': 'trimethoprim/sulfamethoxazole',
        'sulfamethoxazole/trimethoprim': 'trimethoprim/sulfamethoxazole',
        'trimethoprim/sulfamethoxazole (co-trimoxazole)': 'trimethoprim/sulfamethoxazole',
        'co-trimoxazole': 'trimethoprim/sulfamethoxazole',
    }
    return aliases.get(name, name)


def normalize_label(series):
    # Deliberately do not infer susceptibility from growth or drug names.
    return clean_text(series).str.lower().map({'s': 1, 'susceptible': 1, 'r': 0, 'resistant': 0})


def resolve_labels(frame):
    """Drop whole keys with ANY invalid result or more than one valid label."""
    frame = frame.copy()
    frame['label'] = normalize_label(frame['susceptibility'])
    grouped = frame.groupby(KEY, dropna=False, sort=False)['label']
    counts = grouped.agg(n_rows='size', n_valid='count', distinct='nunique').reset_index()
    counts['reason'] = np.select(
        [counts['distinct'].gt(1), counts['n_valid'].lt(counts['n_rows'])],
        ['conflicting_s_r', 'invalid_or_unknown_result'], default='retained')
    keep = counts.loc[counts.reason.eq('retained'), KEY]
    result = frame.merge(keep, on=KEY, how='inner', validate='many_to_one').drop_duplicates(KEY)
    result['label'] = result['label'].astype('int8')
    return result, counts


def collapse_support(frame, columns, audit, table_name):
    """Agreeing repeats collapse. Disagreeing nonmissing values become missing."""
    frame = frame.copy()
    frame = frame.dropna(subset=LINK)
    absent = [c for c in columns if c not in frame]
    for col in absent:
        frame[col] = pd.NA
    g = frame.groupby(LINK, sort=False, dropna=False)[columns]
    unique = g.nunique(dropna=True)
    result = g.first()
    result = result.mask(unique.gt(1))
    audit[table_name] = {
        'input_rows': len(frame), 'unique_event_keys': len(result),
        'duplicate_event_rows': int(frame.duplicated(LINK).sum()),
        'unavailable_columns': absent,
        'conflicting_cells_set_missing': int(unique.gt(1).sum().sum()),
    }
    return result.reset_index()


def read_support(path, rename, columns, patients, audit, name):
    if not path.exists():
        audit[name] = {'table_unavailable': True}
        return pd.DataFrame(columns=LINK + columns)
    pieces = []
    for chunk in pd.read_csv(path, dtype='string', chunksize=150000):
        chunk = chunk.rename(columns={**RAW_KEYS, **rename})
        if not set(LINK).issubset(chunk.columns):
            raise ValueError(f'{path.name}: required linkage columns {LINK} missing')
        for col in LINK:
            chunk[col] = clean_text(chunk[col])
        chunk = chunk[chunk.patient_id.isin(patients)].copy()
        for col in columns:
            if col in chunk:
                chunk[col] = clean_text(chunk[col])
        pieces.append(chunk)
    frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=LINK + columns)
    return collapse_support(frame, columns, audit, name)


def build_dataset(raw_dir, out_dir, config):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = {'source_dataset': config['source_dataset'], 'prediction_time': 'culture order',
             'target_list_status': config['target_list_status'], 'sample_fraction': config['sample_fraction'],
             'source_rows_scanned': 0, 'sample_rows': 0}
    if not 0 < config['sample_fraction'] <= 1:
        raise ValueError('sample_fraction must be in (0,1]')
    drugs = [canonical_drug(x) for x in config['antibiotics']]
    if len(drugs) != len(set(drugs)):
        raise ValueError('Duplicate configured candidate antibiotics')
    rename = {**RAW_KEYS, 'order_time_jittered_utc': 'culture_time',
              'culture_description': 'culture_type', 'antibiotic': 'candidate_antibiotic'}
    parts = []
    source_path = raw_dir / 'microbiology_cultures_cohort.csv'
    for chunk in pd.read_csv(source_path, dtype='string', chunksize=150000):
        audit['source_rows_scanned'] += len(chunk)
        chunk = chunk.rename(columns=rename)
        needed = set(KEY + ['culture_time', 'culture_type', 'susceptibility'])
        if not needed.issubset(chunk.columns):
            raise ValueError(f'Cohort columns missing: {sorted(needed - set(chunk.columns))}')
        for col in chunk.columns:
            chunk[col] = clean_text(chunk[col])
        chunk = chunk[sample_patients(chunk.patient_id, config['sample_fraction'], config['seed'])].copy()
        audit['sample_rows'] += len(chunk)
        parts.append(chunk)
    if not parts:
        raise ValueError('Source cohort is empty')
    frame = pd.concat(parts, ignore_index=True)
    audit['sample_patients_before_filters'] = int(frame.patient_id.nunique())
    audit['sample_cultures_before_filters'] = int(frame[LINK].drop_duplicates().shape[0])
    frame['candidate_antibiotic'] = frame.candidate_antibiotic.map(canonical_drug)
    frame['organism'] = frame.organism.str.lower().str.replace(r'\s+', ' ', regex=True)
    # Metadata agreement is checked across the entire sampled event, before drug filtering.
    event_meta = frame.groupby(LINK, dropna=False)[['culture_time', 'culture_type']].nunique(dropna=True)
    bad_meta = event_meta.loc[event_meta.gt(1).any(axis=1)].index
    bad_mask = pd.MultiIndex.from_frame(frame[LINK]).isin(bad_meta)
    audit['events_with_conflicting_metadata_excluded'] = int(len(bad_meta))
    frame = frame.loc[~bad_mask].copy()
    frame['culture_time'] = pd.to_datetime(frame.culture_time, errors='coerce', utc=True)
    valid_key = frame[KEY].notna().all(axis=1) & frame.culture_time.notna()
    audit['rows_missing_key_or_time_excluded'] = int((~valid_key).sum())
    frame = frame[valid_key].copy()
    in_target = frame.candidate_antibiotic.isin(drugs)
    audit['rows_outside_target_set_excluded'] = int((~in_target).sum())
    frame = frame[in_target].copy()
    # S/R result plus a nonmissing organism defines positive eligible cultures.
    if 'was_positive' in frame:
        neg = frame.was_positive.str.lower().isin(['false', '0', 'no'])
        audit['negative_culture_rows_excluded'] = int(neg.sum())
        frame = frame[~neg].copy()
    audit['target_rows_before_label_resolution'] = len(frame)
    frame, resolutions = resolve_labels(frame)
    resolutions.to_csv(out_dir / 'label_resolution_audit.csv', index=False)
    audit['key_label_decisions'] = resolutions.reason.value_counts().to_dict()
    audit['duplicate_target_rows_before_resolution'] = int((resolutions.n_rows - 1).clip(lower=0).sum())
    audit['agreeing_duplicate_rows_collapsed'] = int((resolutions.loc[resolutions.reason.eq('retained'), 'n_rows'] - 1).sum())
    keep_cols = KEY + ['culture_time', 'culture_type', 'label']
    if 'site_id' in frame:
        keep_cols.append('site_id')
    frame = frame[keep_cols].copy()
    patients = set(frame.patient_id)
    demo = read_support(raw_dir / 'microbiology_cultures_demographics.csv',
                        {'age': 'age_group', 'gender': 'gender_code'}, ['age_group', 'gender_code'],
                        patients, audit, 'demographics')
    wards = ['ward_ip', 'ward_op', 'ward_er', 'ward_icu']
    ward = read_support(raw_dir / 'microbiology_cultures_ward_info.csv',
                        {f'hosp_ward_{x.upper()}': f'ward_{x}' for x in ['ip', 'op', 'er', 'icu']},
                        wards, patients, audit, 'ward_info')
    n_before = len(frame)
    frame = frame.merge(demo, on=LINK, how='left', validate='many_to_one')
    frame = frame.merge(ward, on=LINK, how='left', validate='many_to_one')
    if len(frame) != n_before:
        raise AssertionError('Supporting joins changed analytical row count')
    for col in wards:
        frame[col] = frame[col].str.lower().replace({'true': '1', 'false': '0'})
        frame[col] = pd.to_numeric(frame[col], errors='coerce').where(lambda x: x.isin([0, 1]))
    frame['age_group'] = frame.age_group.str.replace('\u2013', '-', regex=False).str.replace('\u2014', '-', regex=False)
    # ARMD source inclusion is adult. Explicit under-18 source bins are still removed.
    lower_age = pd.to_numeric(frame.age_group.str.extract(r'^(\d+)', expand=False), errors='coerce')
    # Missing age must not exclude an adult-source patient.
    under18 = lower_age.lt(18).fillna(False)
    audit['explicit_under18_rows_excluded'] = int(under18.sum())
    frame = frame[~under18].copy()
    frame['dataset'] = config['source_dataset']
    # IDs are retained for grouping/audit only, never supplied as model predictors.
    frame['split'] = patient_split(frame.patient_id, config['seed'])
    if config.get('include_clinical_features', True):
        from .support_features import enrich_features
        frame = enrich_features(frame, raw_dir, audit)
    frame = frame.sort_values(KEY).reset_index(drop=True)
    if frame.empty:
        raise ValueError('No eligible rows. Check the source schema and candidate list.')
    if frame.duplicated(KEY).any():
        raise AssertionError('Duplicate analytical keys')
    if frame.groupby('patient_id').split.nunique().max() != 1:
        raise AssertionError('Patient overlap across splits')
    if not set(FEATURES).issubset(frame):
        raise AssertionError('Feature contract missing')
    frame.to_parquet(out_dir / 'analytical.parquet', index=False)
    frame.to_csv(out_dir / 'analytical.csv.gz', index=False)
    frame[['patient_id', 'split']].drop_duplicates().to_csv(out_dir / 'patient_splits.csv', index=False)
    audit['analytical_rows'] = len(frame)
    audit['join_rows_before'] = n_before
    audit['join_rows_after'] = n_before
    audit['duplicate_analytical_keys'] = int(frame.duplicated(KEY).sum())
    audit['patient_overlap_between_splits'] = 0
    from .support_features import NUMERIC_FEATURES
    audit['features'] = FEATURES + [c for c in NUMERIC_FEATURES if c in frame]
    (out_dir / 'construction_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return frame, audit


def profile_dataset(frame, out_dir):
    from .support_features import NUMERIC_FEATURES
    predictors = FEATURES + [c for c in NUMERIC_FEATURES if c in frame]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split, group in [('all', frame)] + list(frame.groupby('split')):
        depth = group.groupby(LINK + ['organism']).candidate_antibiotic.nunique()
        summaries.append({'split': split, 'rows': len(group), 'patients': group.patient_id.nunique(),
                          'culture_events': len(group[LINK].drop_duplicates()),
                          'organism_events': len(depth), 'susceptible_n': int(group.label.sum()),
                          'resistant_n': int(group.label.eq(0).sum()), 'susceptible_prevalence': float(group.label.mean()),
                          'resistant_prevalence': float(group.label.eq(0).mean()),
                          'time_min_shifted': str(group.culture_time.min()), 'time_max_shifted': str(group.culture_time.max()),
                          'candidate_depth_mean': float(depth.mean()), 'candidate_depth_median': float(depth.median()),
                          'candidate_depth_min': int(depth.min()), 'candidate_depth_max': int(depth.max()),
                          'duplicate_keys': int(group.duplicated(KEY).sum())})
    pd.DataFrame(summaries).to_csv(out_dir / 'cohort_summary.csv', index=False)
    missing = []
    for split, group in [('all', frame)] + list(frame.groupby('split')):
        for col in frame:
            missing.append({'split': split, 'feature': col, 'n': len(group), 'missing_n': int(group[col].isna().sum()),
                            'missing_fraction': float(group[col].isna().mean()),
                            'is_predictor': col in predictors})
    pd.DataFrame(missing).to_csv(out_dir / 'missingness.csv', index=False)
    imbalance = frame.groupby(['split', 'candidate_antibiotic']).label.agg(n='size', susceptible_n='sum').reset_index()
    imbalance['resistant_n'] = imbalance.n - imbalance.susceptible_n
    imbalance['resistant_prevalence'] = imbalance.resistant_n / imbalance.n
    imbalance['minority_class'] = np.where(imbalance.resistant_n <= imbalance.susceptible_n, 'resistant', 'susceptible')
    imbalance.to_csv(out_dir / 'class_balance.csv', index=False)
    frame.groupby(LINK + ['organism']).candidate_antibiotic.nunique().value_counts().sort_index().rename_axis(
        'candidate_depth').rename('organism_events').to_csv(out_dir / 'candidate_depth.csv')
    return summaries
