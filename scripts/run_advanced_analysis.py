"""Reproduce the exploratory extension without changing the primary fitted models.

Run after the original pipeline has completed. The default recomputes the
validation ablations and fixed-test diagnostics with 500 patient bootstrap draws.
Use --presentation-only to reuse completed numerical results when changing layout.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))


def run(script, *arguments):
    subprocess.run([sys.executable, str(ROOT/'scripts'/script), *arguments],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--presentation-only', action='store_true',
                        help='Reuse saved results; regenerate figures, workbooks and checks.')
    parser.add_argument('--include-reports', action='store_true',
                        help='Also create and check local reports, which are excluded from Git.')
    args = parser.parse_args()
    run('start_advanced_analysis.py')
    if not args.presentation_only:
        from armd.ablation import run_ablation
        from armd.diagnostics import run_diagnostics
        print('Validation feature-group comparisons...', flush=True)
        run_ablation(ROOT, ROOT/'results/advanced', seed=142, bootstrap_repeats=500)
        print('Frozen-model test diagnostics...', flush=True)
        run_diagnostics(ROOT, ROOT/'results/advanced', seed=142, repeats=500)
    run('refresh_result_figures.py')
    report_args = ['--include-reports'] if args.include_reports else []
    run('make_advanced_package.py', *report_args)
    if args.include_reports:
        run('make_report.py', '--results', 'results', '--output', 'report')
    run('validate_package.py', *report_args)
    run('validate_advanced.py', '--require-present', *report_args)
    run('package_submission.py', *report_args)
    print('Advanced figures, workbooks, checks and submission archive are complete.')


if __name__ == '__main__':
    main()
