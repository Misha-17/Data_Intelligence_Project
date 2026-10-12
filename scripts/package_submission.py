"""Package the completed work without duplicating the multi-gigabyte source data."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    validation = json.loads((ROOT/'results/package_validation.json').read_text())
    if validation['status'] != 'passed':
        raise ValueError('Validate final outputs before packaging')
    advanced_path = ROOT/'results/advanced/validation.json'
    advanced_validation = None
    if (ROOT/'results/advanced/analysis_protocol.json').exists():
        if not advanced_path.is_file():
            raise ValueError('Validate the advanced extension before packaging')
        advanced_validation = json.loads(advanced_path.read_text())
        if advanced_validation['status'] != 'passed' or not advanced_validation.get('require_present'):
            raise ValueError('Run final advanced validation with --require-present before packaging')
    files = []
    for directory in ['src', 'scripts', 'tests', 'docs', 'results', 'report']:
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix != '.pyc'
                     and p.name != 'report_preview.png')
    files.extend(ROOT/name for name in ['README.md','README.txt','.gitignore','.gitattributes','config.json','requirements.txt','requirements-lock.txt'])
    files.extend(ROOT/'data/raw'/name for name in ['dryad_manifest.json','verified_downloads.json','README.md'])
    files = sorted(set(files))
    manifest = {'source_data_note':'Raw publisher CSVs are not duplicated in this archive. Keep the supplied files in data/raw.',
                'analysis_data_note':'The cleaned analytical dataset, held-out predictions and fitted models are included.',
                'validation':validation, 'advanced_validation':advanced_validation, 'files':[]}
    for path in files:
        manifest['files'].append({'path':path.relative_to(ROOT).as_posix(), 'bytes':path.stat().st_size,
                                  'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    destination = ROOT/'submission'
    destination.mkdir(exist_ok=True)
    target = destination/'ARMD_project_package.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files:
            archive.write(path,path.relative_to(ROOT).as_posix())
        archive.writestr('package_manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError('Archive CRC failure: '+bad)
    print(f'Created {target} ({target.stat().st_size/1e6:.1f} MB; {len(files)} files plus manifest)')


if __name__ == '__main__':
    main()
