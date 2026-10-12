"""Download the pinned, public ARMD core tables and verify publisher SHA-256."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://datadryad.org'
VERSION = '403223'
NAMES = {'microbiology_cultures_cohort.csv',
         'microbiology_cultures_demographics.csv',
         'microbiology_cultures_ward_info.csv', 'README.md'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'data' / 'raw')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_url = BASE + '/api/v2/versions/' + VERSION + '/files'
    with urllib.request.urlopen(manifest_url, timeout=60) as response:
        manifest = json.load(response)
    (args.out / 'dryad_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    selected = [x for x in manifest['_embedded']['stash:files'] if x['path'] in NAMES]
    if len(selected) != len(NAMES):
        raise RuntimeError('Pinned version has missing source files')

    def download(item):
        target = args.out / item['path']
        if target.exists() and digest(target) == item['digest']:
            print('Verified cached: ' + target.name, flush=True)
            return
        # API download routes require a Dryad token. The public web route is
        # the anonymous endpoint linked from the dataset landing page.
        file_id = item['_links']['self']['href'].rsplit('/', 1)[-1]
        url = BASE + '/downloads/file_stream/' + file_id
        partial = target.with_suffix(target.suffix + '.part')
        print('Downloading: ' + target.name, flush=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'ARMD-course-reproduction/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as out:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                out.write(block)
        if partial.stat().st_size != item['size'] or digest(partial) != item['digest']:
            raise RuntimeError('Publisher checksum mismatch: ' + target.name)
        partial.replace(target)
        print('Verified downloaded: ' + target.name, flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(download, selected))
    print('All selected source files verified.', flush=True)


if __name__ == '__main__':
    main()
