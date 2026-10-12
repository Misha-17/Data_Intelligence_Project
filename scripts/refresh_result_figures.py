"""Refresh primary, summary and advanced figures from saved results, without fitting."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import pandas as pd
import matplotlib.pyplot as plt
from armd.evaluate import _plots


# This is the insertion order used by the original evaluation artifacts.
MODEL_ORDER = [
    "prevalence_baseline", "antibiotic_baseline", "logistic_unweighted",
    "logistic_balanced", "lightgbm_unweighted", "lightgbm_balanced",
]


def refresh(root: Path) -> None:
    root = root.resolve()
    model_dir = root / "results" / "models"
    frozen = json.loads((model_dir / "frozen_selection.json").read_text(encoding="utf-8"))
    thresholds = frozen["thresholds_s"]
    if set(thresholds) != set(MODEL_ORDER):
        raise ValueError("Frozen thresholds do not match the original six model variants.")
    test = pd.read_parquet(model_dir / "predictions.parquet", filters=[("split", "==", "test")])
    if test.empty or not test["split"].eq("test").all():
        raise ValueError("Expected the saved test predictions.")
    predictions = {name: test[f"p_s_{name}"].to_numpy(float) for name in MODEL_ORDER}
    output = model_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    _plots(test, predictions, thresholds, output)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    notes = {
        'visible_text': 'axis names and tick labels only',
        'model_curve_colors': {name: colors[i % len(colors)] for i, name in enumerate(MODEL_ORDER)},
        'confusion_matrix_panel_order': MODEL_ORDER,
        'panel_order_rule': 'Left to right, then top to bottom.',
        'precision_recall_panels': ['resistant', 'susceptible'],
        'dashed_references': {'calibration': 'exact agreement', 'precision_recall': 'class prevalence'},
        'summary_drug_plot': 'Blue bars show resistance average precision; orange ticks show resistance prevalence.',
        'advanced_figure_notes': 'advanced/figure_index.json',
    }
    (root/'results/figure_notes.json').write_text(json.dumps(notes, indent=2), encoding='utf-8')
    for script, arguments in [
        ("summarize_results.py", ["--results", str(root / "results")]),
        ("make_advanced_figures.py", ["--root", str(root)]),
    ]:
        subprocess.run([sys.executable, str(root / "scripts" / script), *arguments],
                       cwd=root, check=True)
    print("Refreshed 3 primary, 3 summary, 8 advanced figures and one compact calibration view. No models were fitted.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    refresh(arguments.root)


if __name__ == "__main__":
    main()
