"""Supplement the automated inventory with explicit audit observations, no relabelling."""
import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/audit'
with (OUT / 'image_manifest.csv').open() as stream:
    records = list(csv.DictReader(stream))
for record in records:
    filename = record['filename']
    record['capture_sequence_candidate'] = filename.split('_BURST')[0] if '_BURST' in filename else ''
    record['capture_sequence_scope'] = 'burst prefix only; not a complete tuber group' if '_BURST' in filename else ''
fields = list(records[0])
with (OUT / 'image_manifest.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(records)

dates = collections.Counter((r['variety'], r['source_condition'], r['filename_capture_time'][:10]) for r in records)
with (OUT / 'capture_dates_by_category.csv').open('w', newline='') as stream:
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['variety', 'condition', 'filename_date_not_verified_exif', 'images'])
    for key, count in sorted(dates.items()):
        writer.writerow([*key, count])

groups = collections.defaultdict(list)
for record in records:
    if record['capture_sequence_candidate']:
        groups[record['capture_sequence_candidate']].append(record)
with (OUT / 'burst_sequence_candidates.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['capture_prefix', 'images', 'category_count', 'members', 'status'], lineterminator='\n')
    writer.writeheader()
    for key, group in sorted(groups.items()):
        writer.writerow({'capture_prefix': key, 'images': len(group), 'category_count': len({r['source_category'] for r in group}), 'members': json.dumps([r['archive_member'] for r in group]), 'status': 'capture-sequence candidate; not a complete physical-tuber group'})

with (OUT / 'visual_review_sample.csv').open() as stream:
    review = list(csv.DictReader(stream))
for item in review:
    category = item['archive_member'].split('/')[0]
    item['external_surface'] = 'tuber exterior or exposed damaged/decayed tissue observed; no leaf-only image in reviewed sample'
    item['symptom_visibility'] = 'not independently adjudicated at image level'
    item['notes'] = 'Contact-sheet review completed by assistant; non-expert appearance screening, not diagnosis or severity ground-truth validation.'
    if 'DryRot_' in category:
        item['notes'] += ' Severity overlap and view-dependent visibility require expert review.'
    elif category.endswith('__Healthy'):
        item['notes'] += ' Healthy is a publisher condition label, not proof of absence of all defects or sprouting.'
with (OUT / 'visual_review_sample.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(review[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(review)
reviewed = {r['archive_member'] for r in review}
for record in records:
    if record['archive_member'] in reviewed:
        record['visual_review_status'] = 'reviewed in 216-image stratified-by-category sample; labels not independently validated'
with (OUT / 'image_manifest.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(records)

summary_path = OUT / 'summary.json'
summary = json.loads(summary_path.read_text())
summary.update({
    'burst_images': sum(len(group) for group in groups.values()),
    'burst_timestamp_prefixes': len(groups),
    'multi_image_burst_prefixes': sum(len(group) > 1 for group in groups.values()),
    'cross_category_burst_prefixes': sum(len({r['source_category'] for r in group}) > 1 for group in groups.values()),
    'all_image_content_manually_verified': False,
    'visual_review_scope': '216 random category-stratified images (12/category), 24 strongest perceptual candidate pairs, plus both cross-category burst pairs at full resolution. Not an exhaustive manual image audit.',
    'grouping_decision': 'No complete trustworthy mapping from images to physical tubers; no final split created.',
})
summary_path.write_text(json.dumps(summary, indent=2) + '\n')

conflicts = [
    {'capture_prefix': 'IMG20251108123556', 'categories': ['BARI_Alu_25_Asterix__DryRot_Mild', 'BARI_Alu_25_Asterix__DryRot_Moderate'], 'observation': 'Same capture-second burst prefix and visually near-identical tuber view in two severity folders. Label conflict needs publisher/expert resolution.'},
    {'capture_prefix': 'IMG20251108124413', 'categories': ['BARI_Alu_25_Asterix__DryRot_Severe', 'BARI_Alu_25_Asterix__Fully_Rotten'], 'observation': 'Same capture-second burst prefix and visually near-identical view across Severe and excluded Fully-Rotten. No files changed.'},
]
(OUT / 'visually_reviewed_burst_conflicts.json').write_text(json.dumps(conflicts, indent=2) + '\n')
print('Audit supplements saved; original data unchanged.')
