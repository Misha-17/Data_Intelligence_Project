"""Safely unpack publisher CSVs from the user-provided nested ZIP archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    out = ROOT / 'data' / 'raw'
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'dryad_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = {x['path']: x for x in manifest['_embedded']['stash:files']}
    records = []
    with zipfile.ZipFile(args.archive) as outer:
        for item in outer.infolist():
            if not item.filename.endswith('.zip') or item.filename.startswith('__MACOSX/'):
                continue
            with tempfile.TemporaryFile(dir=out) as temporary:
                with outer.open(item) as stream:
                    shutil.copyfileobj(stream, temporary, length=1024 * 1024)
                temporary.seek(0)
                with zipfile.ZipFile(temporary) as inner:
                    for source in inner.infolist():
                        name = Path(source.filename).name
                        if source.is_dir() or name not in expected or source.filename.startswith('__MACOSX/'):
                            continue
                        # Write only publisher-allowlisted basenames under the explicit output directory.
                        destination = out / name
                        if destination.resolve().parent != out.resolve():
                            raise ValueError('Unsafe path')
                        info = expected[name]
                        if source.file_size != info['size']:
                            raise ValueError('Unexpected file size: ' + name)
                        partial = destination.with_suffix(destination.suffix + '.part')
                        h = hashlib.sha256()
                        with inner.open(source) as stream, partial.open('wb') as target:
                            for block in iter(lambda: stream.read(1024 * 1024), b''):
                                h.update(block)
                                target.write(block)
                        if h.hexdigest() != info['digest']:
                            raise ValueError('Publisher checksum mismatch: ' + name)
                        partial.replace(destination)
                        records.append({'file': name, 'bytes': source.file_size, 'sha256': h.hexdigest(),
                                        'verified_against_dryad_version': '403223'})
                        print('Verified: ' + name, flush=True)
    (out / 'verified_downloads.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Finished: {len(records)} verified source files.', flush=True)


if __name__ == '__main__':
    main()
