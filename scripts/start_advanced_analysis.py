"""Record the extension plan and protect the already reported primary results."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = [
    'results/analytical.parquet', 'results/models/predictions.parquet',
    'results/models/model_comparison.csv', 'results/models/frozen_selection.json',
    'results/models/training_patient_roles.csv', 'results/models/evaluation_summary.json',
    'results/models/models/lightgbm_balanced.joblib',
]


def main():
    out = ROOT/'results/advanced'
    out.mkdir(exist_ok=True)
    protocol = {
        'status': 'exploratory extension after the primary test results were reported',
        'main_model': 'frozen selected weighted LightGBM; no new selection or tuning on test',
        'random_seed': 142, 'patient_bootstrap_replicates': 500,
        'questions': [
            {'question':'Which feature groups add useful prediction information?',
             'design':'Fixed LightGBM ablations on the existing proper-fit/calibration patients; score only validation. Compare core, core plus labs/vitals, core plus history, full, and core plus availability patterns.'},
            {'question':'For which drugs are ranking, resistance detection and calibration reliable?',
             'design':'Frozen test predictions; drug-level patient-cluster intervals, resistance AP with prevalence baselines, validation-defined reliability bins with sparse flags.'},
            {'question':'What happens when screening aims to find more resistance?',
             'design':'Choose 70%, 80%, 90% resistance-recall targets on validation; freeze thresholds, then report test recall, false alerts and missed resistance per 1000 candidate rows. Retain original threshold as reference.'},
            {'question':'Where does incomplete label coverage limit interpretation?',
             'design':'Top six organisms by training frequency, fixed candidate order; report retained S/R label coverage among included organism-events. Missing labels never become outcomes.'},
        ],
        'uncertainty':'95% pointwise patient-cluster percentile intervals, fixed models; no refitting or simultaneous coverage claim',
        'presentation':'Consistent drug order, resistance probabilities, readable direct labels, restrained colors; PNG, SVG and PDF figures',
        'limits':['No causal interpretation of feature ablations or availability patterns.',
                  'No clinical utility, treatment-benefit or external-site claim.',
                  'No calendar trend analysis because dates are shifted per patient.',
                  'Validation ablations and post hoc test diagnostics need independent confirmation.'],
        'protected_sha256': {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in PROTECTED},
    }
    path = out/'analysis_protocol.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['protected_sha256'] != protocol['protected_sha256']:
            raise RuntimeError('Primary results changed since this extension began')
    else:
        path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print('Advanced protocol recorded; primary results fingerprinted.')


if __name__ == '__main__':
    main()
