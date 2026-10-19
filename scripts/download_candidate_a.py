"""Preserve the original public Mendeley v1 deposit, verifying publisher hashes."""
import hashlib
import json
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'raw' / 'mendeley_rgn2d2jb7f_v1'
OUT.mkdir(parents=True, exist_ok=True)
url = 'https://data.mendeley.com/public-api/datasets/rgn2d2jb7f'
payload = subprocess.check_output(['curl', '-fLsS', '--retry', '3', '--max-time', '60', url])
metadata = json.loads(payload)
assert metadata['version'] == 1, 'Unexpected dataset version; stop rather than change source.'
meta_path = OUT / 'repository_metadata.json'
if not meta_path.exists():
    meta_path.write_bytes(payload)
records = []
for item in metadata['files']:
    name = item['filename']
    assert Path(name).name == name
    target = OUT / name
    expected = item['content_details']['sha256_hash']
    if not target.exists():
        partial = OUT / (name + '.partial')
        print('Downloading', name, item['size'], 'bytes', flush=True)
        if item['size'] > 100_000_000:
            chunk_size = 8 * 1024 * 1024
            parts = OUT / '.download_parts'
            parts.mkdir(exist_ok=True)
            def fetch(start):
                end = min(start + chunk_size, item['size']) - 1
                piece = parts / f'{start:012d}.part'
                if not piece.exists() or piece.stat().st_size != end - start + 1:
                    subprocess.run(['curl', '-fLsS', '--retry', '4', '--connect-timeout', '30', '--max-time', '600', '--range', f'{start}-{end}', '-o', str(piece), item['content_details']['download_url']], check=True)
                assert piece.stat().st_size == end - start + 1
                print('Archive chunk complete', start, end, flush=True)
                return piece
            with ThreadPoolExecutor(max_workers=24) as pool:
                pieces = list(pool.map(fetch, range(0, item['size'], chunk_size)))
            with partial.open('wb') as dest:
                for piece in pieces:
                    dest.write(piece.read_bytes())
        else:
            subprocess.run(['curl', '-fLsS', '--retry', '3', '--connect-timeout', '30', '--max-time', '120', '-o', str(partial), item['content_details']['download_url']], check=True)
        with partial.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        assert digest == expected, (name, digest, expected)
        assert partial.stat().st_size == item['size']
        partial.rename(target)
    with target.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    assert digest == expected, (name, digest, expected)
    records.append({'file': name, 'bytes': target.stat().st_size, 'sha256': digest, 'publisher_hash_matches': True})
    print('Verified', name, flush=True)
audit = ROOT / 'outputs' / 'audit'
audit.mkdir(parents=True, exist_ok=True)
(audit / 'download_verification.json').write_text(json.dumps(records, indent=2) + '\n')
