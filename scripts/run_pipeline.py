"""Build the analytical data and run model evaluation."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_inputs(raw):
    manifest = json.loads((raw / 'dryad_manifest.json').read_text(encoding='utf-8'))
    verified = []
    for item in manifest['_embedded']['stash:files']:
        path = raw / item['path']
        if path.exists():
            if path.stat().st_size != item['size'] or sha256(path) != item['digest']:
                raise ValueError('Source checksum mismatch: ' + path.name)
            verified.append({'file': path.name, 'bytes': item['size'], 'sha256': item['digest']})
    return verified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'data' / 'raw')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--stage', choices=['build', 'evaluate', 'all'], default='all')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    start = time.perf_counter()
    import pandas as pd
    if args.stage in ['build', 'all']:
        print('Verifying publisher checksums...', flush=True)
        verified = verify_inputs(args.raw_dir)
        (args.output / 'source_checksums.json').write_text(json.dumps(verified, indent=2), encoding='utf-8')
        print('Building analytical cohort...', flush=True)
        from armd.build import build_dataset, profile_dataset
        frame, audit = build_dataset(args.raw_dir, args.output, config)
        profile_dataset(frame, args.output)
        print(f'Built {len(frame):,} rows for {frame.patient_id.nunique():,} patients.', flush=True)
    else:
        frame = pd.read_parquet(args.output / 'analytical.parquet')
    if args.stage in ['evaluate', 'all']:
        print('Fitting models and exporting results...', flush=True)
        from armd.evaluate import run_evaluation
        summary = run_evaluation(frame, args.output / 'models', seed=config['seed'],
                                 bootstrap_repeats=config.get('bootstrap_replicates', 200))
        print('Selected on validation: ' + summary['selected_model'], flush=True)
    sources = sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'scripts').glob('*.py'))
    provenance = {'config': config, 'python': platform.python_version(), 'stage': args.stage,
                  'elapsed_seconds': time.perf_counter() - start,
                  'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
                  'config_sha256': sha256(args.config), 'source_release': '2025-10-22, Dryad version 403223',
                  'data_type': 'real de-identified public ARMD; no synthetic observations'}
    (args.output / f'run_manifest_{args.stage}.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f'Completed {args.stage} in {time.perf_counter()-start:.1f} seconds.', flush=True)


if __name__ == '__main__':
    main()
