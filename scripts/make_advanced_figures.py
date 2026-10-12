"""Create exportable statistical figures from the frozen and exploratory results."""
from pathlib import Path
import json
import os
import sys
import argparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from armd.visual_style import setup, xgrid, save, NAVY, TEAL, ORANGE, GRAY, GRID, PALE, DRUG_LABELS, MODEL_LABELS, FEATURE_LABELS

def interval(ax, x, y, lo, hi, color=TEAL, marker='o', size=7):
    if np.isfinite(lo) and np.isfinite(hi):
        ax.hlines(y, lo, hi, color=color, lw=2, alpha=0.75)
        ax.vlines([lo, hi], y - 0.065, y + 0.065, color=color, lw=1)
    ax.plot(x, y, marker=marker, color=color, ms=size, markeredgecolor='white', markeredgewidth=0.8, zorder=4)

def register(index, stem, title, question, sources, note):
    index.append({'stem': stem, 'title': title, 'question': question, 'source_tables': sources, 'note': note, 'files': {'png': f'figures/{stem}.png'}})

def comparison(root, out, index):
    ci = pd.read_csv(root / 'results/models/test_confidence_intervals.csv')
    order = ['lightgbm_balanced', 'lightgbm_unweighted', 'logistic_balanced', 'logistic_unweighted', 'antibiotic_baseline', 'prevalence_baseline']
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 6.6), sharey=True)
    definitions = [('antibiotic_macro_auroc', 'Drug-macro AUROC', (0.45, 0.79)), ('antibiotic_macro_auprc_r', 'Drug-macro resistance AP', (0.12, 0.49)), ('brier', 'Probability error (Brier)', (0.095, 0.17))]
    for ax, (metric, title, limits) in zip(axes, definitions):
        for y, model in enumerate(order):
            row = ci[(ci.model == model) & (ci.metric == metric)].iloc[0]
            color = TEAL if model == 'lightgbm_balanced' else NAVY if 'lightgbm' in model else GRAY
            if y == 0:
                ax.axhspan(-0.38, 0.38, color=PALE, zorder=0)
            interval(ax, row.estimate, y, row.ci_low, row.ci_high, color)
        ax.set_xlim(limits)
        xgrid(ax)
        ax.set_xlabel(title)
        ax.set_yticks(range(len(order)), [MODEL_LABELS[x] for x in order])
        ax.set_ylim(len(order) - 0.5, -0.5)
    axes[0].set_ylabel('Model')
    fig.subplots_adjust(left=0.235, right=0.96, bottom=0.2, top=0.75, wspace=0.28)
    stem = '01_model_comparison'
    save(fig, out, stem)
    register(index, stem, 'Model differences, with uncertainty', 'How large are the differences between the original models?', ['../models/test_confidence_intervals.csv'], 'Original test results; fixed fitted models; pointwise intervals.')

def ablation(root, out, index):
    f = pd.read_csv(root / 'results/advanced/ablation_metrics.csv')
    order = ['core_only', 'core_labs_vitals', 'core_availability_only', 'core_prior_history', 'full']
    labels = ['Core only', 'Core + labs and vital signs', 'Core + availability patterns', 'Core + prior history', 'All features (original model)']
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.8), sharey=True)
    for ax, metric, title, limits in zip(axes, ['antibiotic_macro_auprc_r', 'brier'], ['Resistance average precision', 'Probability error (Brier)'], [(0.18, 0.45), (0.1, 0.14)]):
        for y, model in enumerate(order):
            row = f[(f.model == model) & (f.metric == metric)].iloc[0]
            color = TEAL if model in ['core_prior_history', 'full'] else ORANGE if model == 'core_availability_only' else GRAY
            if model == 'full':
                ax.axhspan(y - 0.36, y + 0.36, color=PALE)
            interval(ax, row.estimate, y, row.ci_low, row.ci_high, color)
        ax.set(xlim=limits, ylim=(4.6, -0.6), yticks=range(5), yticklabels=labels)
        ax.set_xlabel(title)
        xgrid(ax)
    axes[0].set_ylabel('Feature group')
    fig.subplots_adjust(left=0.26, right=0.96, top=0.75, bottom=0.23, wspace=0.32)
    stem = '02_feature_group_ablation'
    save(fig, out, stem)
    register(index, stem, 'What each feature group adds on validation', 'Do measured values add information beyond basic fields and recording patterns?', ['ablation_metrics.csv', 'ablation_paired_differences.csv'], 'Validation only. Similar scores do not establish equivalence or transportability.')

def per_drug(root, out, index, drugs):
    f = pd.read_csv(root / 'results/advanced/drug_performance.csv').set_index('candidate_antibiotic').loc[drugs]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 8.3), sharey=True)
    defs = [('auroc', 'Ranking (AUROC)', (0.3, 1.05)), ('ap_r', 'Resistance average precision', (0, 1.02)), ('recall_r', 'Resistance found (recall)', (-0.03, 1.03))]
    for ax, (metric, title, limits) in zip(axes, defs):
        for y, (drug, row) in enumerate(f.iterrows()):
            if y % 2 == 0:
                ax.axhspan(y - 0.42, y + 0.42, color='#F6F8F9')
            interval(ax, row[metric], y, row[metric + '_ci_low'], row[metric + '_ci_high'], TEAL)
            if metric == 'ap_r':
                ax.plot(row.prevalence_r, y, '|', color=ORANGE, ms=16, mew=2)
        ax.set(xlim=limits, ylim=(8.6, -0.65), yticks=range(9), yticklabels=[DRUG_LABELS[x] for x in drugs], xlabel=title)
        xgrid(ax)
        if metric == 'recall_r':
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].axvline(0.5, ls='--', color=GRAY, lw=1)
    axes[0].set_ylabel('Antibiotic')
    fig.subplots_adjust(left=0.3, right=0.97, top=0.78, bottom=0.22, wspace=0.24)
    stem = '03_drug_performance'
    save(fig, out, stem)
    register(index, stem, 'Performance varies substantially across antibiotics', 'Which drugs have stronger ranking and resistance detection, and how certain are those estimates?', ['drug_performance.csv'], 'Drug order follows the configured panel, not test performance.')

def calibration_overall(root, out, index):
    f = pd.read_csv(root / 'results/advanced/calibration_bins.csv')
    hist = pd.read_csv(root / 'results/advanced/probability_histogram.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7), gridspec_kw={'width_ratios': [1.12, 1]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls='--', color=GRAY, lw=1, label='Exact agreement')
    for state, color, label in [('raw', ORANGE, 'Before calibration'), ('calibrated', TEAL, 'After calibration')]:
        sub = f[(f.scope == 'overall') & (f.probability == state)].sort_values('bin')
        sub = sub[sub.n > 0]
        ax.plot(sub.mean_predicted_r, sub.observed_r, color=color, lw=1.6, alpha=0.85)
        ax.vlines(sub.mean_predicted_r, sub.observed_r_ci_low, sub.observed_r_ci_high, color=color, lw=1.5, alpha=0.55)
        ax.scatter(sub.mean_predicted_r, sub.observed_r, color=color, s=42, label=label, zorder=4, edgecolors='white')
        h = hist[(hist.scope == 'overall') & (hist.probability == state)]
        mid = (h.edge_left + h.edge_right) / 2
        axes[1].plot(mid, h.n / h.n.sum(), color=color, lw=2, label=label)
        axes[1].fill_between(mid, h.n / h.n.sum(), color=color, alpha=0.08)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted probability of resistance', ylabel='Observed resistant share')
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    axes[1].set(xlim=(0, 1), ylim=(0, None), xlabel='Predicted probability of resistance', ylabel='Share of test rows')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].grid(axis='y', color=GRID, lw=0.6)
    fig.subplots_adjust(left=0.085, right=0.965, bottom=0.2, top=0.75, wspace=0.35)
    stem = '04_overall_calibration'
    save(fig, out, stem)
    register(index, stem, 'Probability checks before and after calibration', 'Are the probabilities close to observed resistance rates, and how concentrated are they?', ['calibration_bins.csv', 'probability_histogram.csv', 'calibration_summary.csv'], 'A lower Brier score alone does not prove better calibration; use the reliability display too.')

def calibration_drugs(root, out, index, drugs):
    f = pd.read_csv(root / 'results/advanced/calibration_bins.csv')
    fig, axes = plt.subplots(3, 3, figsize=(13.6, 11.5), sharex=True, sharey=True)
    for ax, drug in zip(axes.flat, drugs):
        sub = f[(f.scope == 'drug') & (f.candidate_antibiotic == drug) & (f.probability == 'calibrated')].sort_values('bin')
        sub = sub[sub.n > 0]
        ax.plot([0, 1], [0, 1], ls='--', color=GRAY, lw=0.9)
        ax.plot(sub.mean_predicted_r, sub.observed_r, color=TEAL, lw=1, alpha=0.6)
        for _, row in sub.iterrows():
            ax.vlines(row.mean_predicted_r, row.observed_r_ci_low, row.observed_r_ci_high, color=TEAL, lw=1.25, alpha=0.65)
            ax.scatter(row.mean_predicted_r, row.observed_r, s=25 + 65 * np.sqrt(row.n / sub.n.max()), facecolor='white' if bool(row['sparse']) else TEAL, edgecolor=TEAL, lw=1.2, zorder=4)
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[0, 0.25, 0.5, 0.75, 1])
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.supxlabel('Mean predicted probability of resistance', fontsize=12)
    fig.supylabel('Observed resistant share', fontsize=12)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.856, bottom=0.14, wspace=0.19, hspace=0.28)
    stem = '05_calibration_by_drug'
    save(fig, out, stem)
    register(index, stem, 'A good pooled calibration curve can hide drug differences', 'Does probability agreement hold within each candidate drug?', ['calibration_bins.csv', 'calibration_summary.csv'], 'Open markers flag sparse bins; their estimates need more caution.')

def threshold_tradeoff(root, out, index):
    f = pd.read_csv(root / 'results/advanced/threshold_operating_points.csv')
    test = f[f['split'] == 'test'].copy()
    order = ['original', 'recall_70', 'recall_80', 'recall_90']
    field = 'operating_point' if 'operating_point' in test else 'policy'
    test = test.set_index(field).loc[order]
    labels = ['Original cutoff', '70% validation recall target', '80% validation recall target', '90% validation recall target']
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.6), sharey=True)
    for ax, metric, title, color in zip(axes, ['false_alerts_per_1000', 'missed_r_per_1000'], ['False resistance alerts', 'Resistant results missed'], [ORANGE, TEAL]):
        upper = test[metric + '_ci_high'].max() * 1.3
        for y, (_, row) in enumerate(test.iterrows()):
            ax.barh(y, row[metric], height=0.42, color=color, alpha=0.18)
            interval(ax, row[metric], y, row[metric + '_ci_low'], row[metric + '_ci_high'], color, size=7)
        ax.set(xlim=(0, upper), ylim=(3.6, -0.65), yticks=range(4), yticklabels=labels)
        ax.set_xlabel(title + ' per 1,000 test rows')
        xgrid(ax)
    axes[0].set_ylabel('Decision cutoff')
    fig.subplots_adjust(left=0.26, right=0.965, top=0.75, bottom=0.23, wspace=0.35)
    stem = '06_threshold_tradeoffs'
    save(fig, out, stem)
    register(index, stem, 'Finding more resistance brings more false alerts', 'What is the error burden when a cutoff aims to find more resistance?', ['threshold_operating_points.csv', 'fixed_thresholds_validation.json'], 'Counts refer to candidate test rows, not patients. No treatment benefit or cost assumption is made.')

def coverage(root, out, index, drugs):
    f = pd.read_csv(root / 'results/advanced/organism_drug_slices.csv')
    organisms = list(dict.fromkeys(f.organism))
    matrix = f.pivot(index='organism', columns='candidate_antibiotic', values='observed_panel_coverage').reindex(index=organisms, columns=drugs)
    fig, ax = plt.subplots(figsize=(14, 7.5))
    cmap = LinearSegmentedColormap.from_list('coverage', ['#F5F8F8', '#B5D8D9', TEAL, NAVY])
    image = ax.imshow(matrix.to_numpy(float), cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax.set(xlabel='Antibiotic', ylabel='Organism')
    ax.set_xticks(range(9), [DRUG_LABELS[d].replace(' / ', ' /\n') for d in drugs], rotation=35, ha='right')
    ax.set_yticks(range(len(organisms)), [x.capitalize() for x in organisms])
    ax.set_xticks(np.arange(-0.5, 9, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(organisms), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    bar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    bar.set_label('Retained label coverage')
    fig.subplots_adjust(left=0.23, right=0.94, top=0.77, bottom=0.31)
    stem = '07_label_coverage'
    save(fig, out, stem)
    register(index, stem, 'The available labels are uneven across organisms and drugs', 'Which organism-drug combinations are well represented in the retained data?', ['organism_drug_slices.csv'], 'The analysis cannot establish why a label is absent or estimate performance for untested combinations.')

def missingness(root, out, index):
    df = pd.read_parquet(root / 'results/analytical.parquet')
    features = list(FEATURE_LABELS)
    rows = []
    for name in features:
        keys = ['patient_id', 'culture_id'] + (['candidate_antibiotic'] if name.startswith('prior_candidate') else [])
        values = df[keys + ['split', name]]
        if values.groupby(keys, dropna=False)[name].nunique(dropna=False).max() > 1:
            raise ValueError(f'Conflicting values at missingness unit: {name}')
        frame = values.drop_duplicates(keys)
        for split, group in frame.groupby('split'):
            rows.append({'feature': name, 'split': split, 'units': len(group), 'missing_n': int(group[name].isna().sum()), 'missing_fraction': float(group[name].isna().mean()), 'unit': 'culture-candidate' if name.startswith('prior_candidate') else 'culture'})
    f = pd.DataFrame(rows)
    f.to_csv(root / 'results/advanced/missingness_by_split.csv', index=False)
    fig, ax = plt.subplots(figsize=(12, 7))
    for split, offset, color in [('train', -0.15, NAVY), ('validation', 0, ORANGE), ('test', 0.15, TEAL)]:
        sub = f[f['split'] == split].set_index('feature').loc[features]
        ax.scatter(sub.missing_fraction, np.arange(7) + offset, color=color, s=55, label=split.capitalize(), edgecolors='white', zorder=4)
    ax.set(xlim=(0, 1), ylim=(6.65, -0.65), yticks=range(7), yticklabels=[FEATURE_LABELS[n] for n in features], xlabel='Share with no usable value', ylabel='Feature')
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    xgrid(ax)
    fig.subplots_adjust(left=0.3, right=0.965, top=0.77, bottom=0.26)
    stem = '08_missingness'
    save(fig, out, stem)
    register(index, stem, 'Missing information is common in every data split', 'How much clinical information is unavailable, and does that change across splits?', ['missingness_by_split.csv'], 'Counts avoid repeatedly counting shared culture features once per organism and drug.')

def report_calibration(root, out):
    """A compact version of figure 04 sized for the five-page report."""
    f = pd.read_csv(root / 'results/advanced/calibration_bins.csv')
    fig, ax = plt.subplots(figsize=(6.4, 3.45))
    ax.plot([0, 1], [0, 1], ls='--', color=GRAY, lw=1, label='Exact agreement')
    for state, color, label in [('raw', ORANGE, 'Before calibration'), ('calibrated', TEAL, 'After calibration')]:
        sub = f[(f.scope == 'overall') & (f.probability == state) & (f.n > 0)].sort_values('bin')
        ax.plot(sub.mean_predicted_r, sub.observed_r, color=color, lw=1.6, marker='o', ms=4, label=label)
        ax.vlines(sub.mean_predicted_r, sub.observed_r_ci_low, sub.observed_r_ci_high, color=color, lw=1, alpha=0.75)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted resistance probability', ylabel='Observed resistant share')
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.97, bottom=0.16)
    save(fig, out, 'report_calibration')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    out = root / 'results/advanced/figures'
    setup()
    drugs = json.loads((root / 'config.json').read_text())['antibiotics']
    index = []
    comparison(root, out, index)
    ablation(root, out, index)
    per_drug(root, out, index, drugs)
    calibration_overall(root, out, index)
    calibration_drugs(root, out, index, drugs)
    threshold_tradeoff(root, out, index)
    coverage(root, out, index, drugs)
    missingness(root, out, index)
    report_calibration(root, out)
    notes = {
        '03_drug_performance': 'Teal dots are model estimates. Orange ticks mark resistance prevalence. Intervals are 95% patient-bootstrap intervals with 500 draws.',
        '04_overall_calibration': 'Orange is before calibration; teal is after calibration. The gray dashed line marks exact agreement.',
        '05_calibration_by_drug': 'Panels run left to right, then top to bottom, in this order: ' + ', '.join(drugs) + '. The gray dashed line marks exact agreement. Open circles flag sparse bins.',
        '08_missingness': 'Navy is training, orange is validation, and teal is test. The vertical offsets only separate the dots.',
    }
    for entry in index:
        entry['visible_text'] = 'axis names and tick labels only'
        if entry['stem'] in notes:
            entry['note'] += ' ' + notes[entry['stem']]
    (out.parent / 'figure_index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    print(f'Created {len(index)} figures and one compact calibration view in PNG format.')
if __name__ == '__main__':
    main()
