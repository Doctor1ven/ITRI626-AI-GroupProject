"""Audit an unmodified ZIP without making splits or asserting inferred tuber IDs."""
import collections
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import threading
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/mendeley_rgn2d2jb7f_v1'
OUT = ROOT / 'outputs/audit'
OUT.mkdir(parents=True, exist_ok=True)
ZIP = RAW / 'potato_desease_dataset.zip'
N = 32
DCT = np.cos(np.pi / N * (np.arange(N) + .5)[None, :] * np.arange(N)[:, None])
LOCAL = threading.local()

def write_csv(name, records, fields=None):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)

def hash_bits(bits):
    return int.from_bytes(np.packbits(bits.ravel()).tobytes(), 'big')

def inspect(name):
    if not hasattr(LOCAL, 'archive'):
        LOCAL.archive = zipfile.ZipFile(ZIP)
    raw = LOCAL.archive.read(name)  # Also verifies member CRC.
    category = PurePosixPath(name).parent.as_posix()
    variety, condition = category.split('__')
    selected = variety != 'Rumana' and condition != 'Fully_Rotten'
    record = {
        'archive_member': name, 'filename': PurePosixPath(name).name,
        'source_category': category, 'variety': variety, 'source_condition': condition,
        'primary_candidate': selected,
        'proposed_class': condition.replace('DryRot_', 'DryRot-') if selected else '',
        'exclusion_reason': '' if selected else ('Rumana/scab excluded' if variety == 'Rumana' else 'Fully-Rotten excluded'),
        'file_bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
        'decode_ok': False, 'width': '', 'height': '', 'format': '', 'mode': '',
        'pixel_sha256': '', 'phash64': '', 'dhash64': '',
        'exif': '', 'filename_capture_time': '',
        'tuber_id': '', 'group_identity_status': 'UNKNOWN; stable identity not retained by publisher',
        'orientation': 'UNKNOWN', 'background': 'UNKNOWN',
        'visual_review_status': 'not individually reviewed', 'decode_error': '',
    }
    match = re.search(r'IMG_?(\d{8})_?(\d{6})', record['filename'])
    if match:
        try:
            record['filename_capture_time'] = datetime.strptime(''.join(match.groups()), '%Y%m%d%H%M%S').isoformat()
        except ValueError:
            pass
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            record.update(decode_ok=True, width=image.width, height=image.height, format=image.format, mode=image.mode)
            record['exif'] = json.dumps({str(k): str(v) for k, v in image.getexif().items()})
            rgb = image.convert('RGB')
            record['pixel_sha256'] = hashlib.sha256(rgb.tobytes()).hexdigest()
            small = np.asarray(rgb.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
            low = (DCT @ small @ DCT.T)[:8, :8]
            record['phash64'] = f'{hash_bits(low > np.median(low.ravel()[1:])):016x}'
            small = np.asarray(rgb.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
            record['dhash64'] = f'{hash_bits(small[:, 1:] > small[:, :-1]):016x}'
    except Exception as exc:
        record['decode_error'] = repr(exc)
    return record

with zipfile.ZipFile(ZIP) as archive:
    infos = archive.infolist()
    entries = [{'member': x.filename, 'directory': x.is_dir(), 'bytes': x.file_size, 'compressed_bytes': x.compress_size, 'crc32': f'{x.CRC:08x}'} for x in infos]
    names = sorted(x.filename for x in infos if not x.is_dir())
    assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
write_csv('archive_entries.csv', entries)
with ThreadPoolExecutor(max_workers=6) as pool:
    records = list(pool.map(inspect, names))
write_csv('image_manifest.csv', records)
print('Decoded', sum(r['decode_ok'] for r in records), '/', len(records), flush=True)

categories = collections.Counter(r['source_category'] for r in records)
write_csv('category_counts.csv', [{'category': k, 'images': v} for k, v in sorted(categories.items())])
cross = collections.Counter((r['variety'], r['proposed_class']) for r in records if r['primary_candidate'])
write_csv('primary_counts_by_variety.csv', [{'variety': k[0], 'class': k[1], 'images': v} for k, v in sorted(cross.items())])

duplicate_counts = {}
for key in ['sha256', 'pixel_sha256']:
    groups = collections.defaultdict(list)
    for r in records:
        if r[key]:
            groups[r[key]].append(r)
    duplicates = [{'hash': digest, 'members': json.dumps([r['archive_member'] for r in group]), 'count': len(group), 'cross_category': len({r['source_category'] for r in group}) > 1} for digest, group in groups.items() if len(group) > 1]
    write_csv(key + '_duplicates.csv', duplicates, ['hash', 'members', 'count', 'cross_category'])
    duplicate_counts[key] = {'groups': len(duplicates), 'images_in_groups': sum(r['count'] for r in duplicates), 'redundant_images': sum(r['count'] - 1 for r in duplicates), 'cross_category_groups': sum(r['cross_category'] for r in duplicates)}

valid = [r for r in records if r['decode_ok']]
hashes = np.array([int(r['phash64'], 16) for r in valid], dtype=np.uint64)
dhashes = np.array([int(r['dhash64'], 16) for r in valid], dtype=np.uint64)
popcount = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
def distances(values):
    return popcount[values.view(np.uint8).reshape(-1, 8)].sum(axis=1)
pairs = []
for i, r in enumerate(valid[:-1]):
    dist = distances(hashes[i+1:] ^ hashes[i])
    for relative in np.flatnonzero(dist <= 6):
        j = i + 1 + int(relative)
        s = valid[j]
        if r['pixel_sha256'] == s['pixel_sha256']:
            continue
        dh = int(distances(np.array([dhashes[i] ^ dhashes[j]], dtype=np.uint64))[0])
        if dh > 12:
            continue
        pairs.append({'image_a': r['archive_member'], 'image_b': s['archive_member'], 'phash_hamming': int(dist[relative]), 'dhash_hamming': dh, 'cross_category': r['source_category'] != s['source_category'], 'status': 'candidate only; not a confirmed tuber identity'})
pairs.sort(key=lambda r: (r['phash_hamming'], r['dhash_hamming'], r['image_a']))
write_csv('near_duplicate_candidates.csv', pairs, ['image_a', 'image_b', 'phash_hamming', 'dhash_hamming', 'cross_category', 'status'])
print('Near-duplicate candidate pairs:', len(pairs), flush=True)

def load_font(size):
    for candidate in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()

font = load_font(12)
largefont = load_font(18)
sheets = OUT / 'contact_sheets'
sheets.mkdir(exist_ok=True)
review = []
def thumb(archive, name, size=(224, 224)):
    with Image.open(io.BytesIO(archive.read(name))) as image:
        return ImageOps.contain(image.convert('RGB'), size)
with zipfile.ZipFile(ZIP) as archive:
    for category in sorted(categories):
        subset = [r for r in valid if r['source_category'] == category]
        rng = np.random.default_rng(42)
        chosen = sorted(rng.choice(len(subset), size=min(12, len(subset)), replace=False).tolist())
        canvas = Image.new('RGB', (960, 850), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), category, fill='black', font=largefont)
        for slot, index in enumerate(chosen):
            r = subset[index]
            x, y = (slot % 4) * 240, 42 + (slot // 4) * 268
            canvas.paste(thumb(archive, r['archive_member']), (x, y))
            draw.text((x+2, y+226), r['filename'], fill='black', font=font)
            review.append({'archive_member': r['archive_member'], 'sheet': category + '.jpg', 'slot': slot, 'selection': 'seed 42 random sample, up to 12/category', 'external_surface': 'pending', 'symptom_visibility': 'pending', 'notes': ''})
        canvas.save(sheets / (category + '.jpg'), quality=90)
    for page in range(min(4, (len(pairs) + 5) // 6)):
        canvas = Image.new('RGB', (1200, 850), 'white')
        draw = ImageDraw.Draw(canvas)
        for slot, pair in enumerate(pairs[page*6:(page+1)*6]):
            x, y = (slot % 3)*400, (slot // 3)*425
            for side, key in enumerate(['image_a', 'image_b']):
                canvas.paste(thumb(archive, pair[key], (190, 240)), (x+side*200, y))
                path = PurePosixPath(pair[key])
                draw.text((x+side*200, y+242), path.parent.name[:25], fill='black', font=font)
                draw.text((x+side*200, y+263), path.name[:25], fill='black', font=font)
            draw.text((x, y+290), f'pHash={pair["phash_hamming"]}; dHash={pair["dhash_hamming"]}', fill='black', font=font)
        canvas.save(sheets / f'near_pairs_{page+1}.jpg', quality=90)
write_csv('visual_review_sample.csv', review)

with (RAW / 'potato_dataset_class_counts.csv').open(newline='') as stream:
    published = {r['class_name']: int(r['image_count']) for r in csv.DictReader(stream)}
summary = {
    'images': len(records), 'archive_entries': len(entries), 'categories': dict(sorted(categories.items())),
    'published_counts_match': dict(categories) == published,
    'decoded_images': sum(r['decode_ok'] for r in records),
    'formats_dimensions_modes': dict(collections.Counter(f"{r['format']} {r['width']}x{r['height']} {r['mode']}" for r in records)),
    'exif_records_with_metadata': sum(r['exif'] not in ['', '{}'] for r in records),
    'timestamp_names': sum(bool(r['filename_capture_time']) for r in records),
    'primary_classes': dict(collections.Counter(r['proposed_class'] for r in records if r['primary_candidate'])),
    'primary_images': sum(r['primary_candidate'] for r in records),
    'duplicates': duplicate_counts, 'near_duplicate_candidate_pairs': len(pairs),
    'near_duplicate_cross_category_pairs': sum(r['cross_category'] for r in pairs),
    'visual_sample_images': len(review), 'independent_tubers': 'UNKNOWN',
    'grouping_conclusive': False, 'final_split_created': False,
    'near_duplicate_method': '64-bit grayscale DCT pHash distance <=6 plus dHash <=12; same decoded-pixel images excluded. Unrotated full-frame screening only; not complete identity recovery.',
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
