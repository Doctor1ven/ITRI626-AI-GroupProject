# Candidate A dataset verification

Audit date: 16 October 2026 (Africa/Johannesburg).

## Decision

The original release is downloaded and technically readable, but it is **not
cleared for a tuber-independent four-class experiment**. Reliable complete
physical-tuber grouping is unavailable, and targeted review identified
near-identical burst photographs with conflicting category labels. No final
split, relabelling, removal, or dataset training was performed.

## Provenance and preservation

- Original repository: https://data.mendeley.com/datasets/rgn2d2jb7f/1
- DOI: 10.17632/rgn2d2jb7f.1
- Creators: Samiul Islam Saif, Mst. Fahmida Tasnim Saba, M Shamim Kaiser.
- Licence: CC BY 4.0; preserve attribution, licence link, and disclosure of changes.
- All four deposited files were downloaded and matched the publisher's SHA-256
  hashes and byte sizes. See `download_verification.json`.
- The archive's actual deposited filename is `potato_desease_dataset.zip`.
  The publisher README/CSVs call it `potato_disease_dataset.zip`. This spelling
  discrepancy was retained, not corrected in the original files.
- The ZIP is 540,742,370 bytes, SHA-256
  `0d6ff2f249993a404d3b533b0247063e0f442e0c6267c24b23d7463645492662`.
- Raw data remain inside `data/raw/mendeley_rgn2d2jb7f_v1/`. Images were audited
  directly from the archive without extracting or modifying source images.

## Exact structure and categories

The archive contains 18 top-level directories and 9,106 files, with no extra
metadata file inside it: 9,124 ZIP entries including directories. Category names
encode variety and condition with a double underscore. All exact member paths,
file sizes, CRCs and filenames are in `archive_entries.csv`; the per-image audit
is in `image_manifest.csv`.

```text
potato_desease_dataset.zip
├── BARI_Alu_25_Asterix__DryRot_Mild/       142
├── BARI_Alu_25_Asterix__DryRot_Moderate/   544
├── BARI_Alu_25_Asterix__DryRot_Severe/    1081
├── BARI_Alu_25_Asterix__Fully_Rotten/     102
├── BARI_Alu_25_Asterix__Healthy/          510
├── BARI_Alu_7_Diamant__DryRot_Mild/        77
├── BARI_Alu_7_Diamant__DryRot_Moderate/   330
├── BARI_Alu_7_Diamant__DryRot_Severe/     885
├── BARI_Alu_7_Diamant__Fully_Rotten/      141
├── BARI_Alu_7_Diamant__Healthy/           514
├── Lal_Pakri__DryRot_Mild/                61
├── Lal_Pakri__DryRot_Moderate/           254
├── Lal_Pakri__DryRot_Severe/            1292
├── Lal_Pakri__Healthy/                  647
├── Rumana__Healthy/                     580
├── Rumana__Scab_Mild/                   205
├── Rumana__Scab_Moderate/               742
└── Rumana__Scab_Severe/                 999
```

Every member decoded successfully and passed ZIP CRC verification. Every image
is JPEG, 512 x 512, RGB. Zero retained EXIF records were found. No byte-identical
or decoded-pixel-identical images were found. Counts agree exactly with the
publisher's class-count CSV.

Filename examples include `IMG20251108134927.jpg`,
`IMG_20260722_135419.jpg`, and
`IMG20251108123556_BURST000_COVER.jpg`. All filenames contain parseable camera
timestamps; none has a verified complete tuber/view/background coding scheme.
There are zero repeated basenames across the release. Filename timestamps are
not independently verified acquisition metadata because EXIF was stripped.

## Approved four-class candidate

| Variety | Healthy | DryRot-Mild | DryRot-Moderate | DryRot-Severe | Total |
|---|---:|---:|---:|---:|---:|
| Asterix | 510 | 142 | 544 | 1,081 | 2,277 |
| Diamant | 514 | 77 | 330 | 885 | 1,806 |
| Lal Pakri | 647 | 61 | 254 | 1,292 | 2,254 |
| Total | 1,671 | 280 | 1,128 | 3,258 | 6,337 |

This is manifest selection only, not a physical split or cleaned-data claim.
Rumana's 2,526 images and the 243 Fully-Rotten images are excluded from the
primary candidate but preserved in raw data. The largest-to-smallest class ratio
is 11.64:1; percentages are approximately Healthy 26.37%, Mild 4.42%, Moderate
17.80%, Severe 51.41%. Independent support could be much smaller than image
support and is UNKNOWN.

The publisher defines Fully-Rotten as advanced decay with collapse or
disintegration, contrasted with Severe tubers remaining structurally
recognisable. Mild/Moderate/Severe lack a quantitative lesion-area grading rubric
in the deposit. Labels are visual phenotypes, not laboratory diagnoses.

## Physical identity and leakage

The publisher README explicitly states that stable tuber identifiers were not
retained in filenames and that random image-level splits cannot be interpreted
as confirmed tuber-independent evaluation.

Consequently:

- Physical-tuber count for every class/variety: **UNKNOWN**.
- Number of images per independent tuber: **UNKNOWN**.
- Complete linkage of five orientations: **UNKNOWN / not recoverable reliably**.
- Complete linkage of two backgrounds: **UNKNOWN / not recoverable reliably**.
- Confirmed exhaustive tuber-level mapping: **not established**.
- No ten-images-per-tuber assumption was used; incomplete groups and retained
  views cannot be inferred from the stated collection protocol.

There are 18 filenames with BURST suffixes, covering 11 capture-time prefixes;
seven prefixes have two images. These provide partial capture-sequence evidence,
not a complete physical-tuber identity system. Two prefixes cross categories:

1. `IMG20251108123556`: BURST000_COVER in DryRot_Mild and BURST001 in
   DryRot_Moderate. Full-resolution review shows a near-identical tuber/view.
2. `IMG20251108124413`: BURST000_COVER in DryRot_Severe and BURST002 in
   Fully_Rotten. Full-resolution review again shows a near-identical view.

See `burst_sequence_candidates.csv`, `visually_reviewed_burst_conflicts.json`,
and `contact_sheets/critical_burst_conflicts.png`. These are strong apparent label
conflicts; no category was changed or independently adjudicated as correct.

All-image perceptual screening used 64-bit grayscale DCT pHash distance <=6 and
dHash distance <=12. It returned 4,141 candidate pairs, 1,915 spanning source
category folders (including different varieties). These are NOT 4,141 confirmed
duplicates, tuber identities, or label errors. Review of 24 strongest pairs found
both likely repeated views and obvious false matches driven by silhouette,
framing, padding and background. Rotation, different surfaces, and background
changes can also hide genuine repeats. In particular, a pHash distance of zero
does not imply identical pixels. Both conflicting burst pairs were flagged by
the screening; their burst filenames supplied additional evidence beyond the
perceptual scores.

Similarity clusters or fixed timestamp windows must not be used as verified
physical groups. Merging all connected candidates would risk joining unrelated
tubers while still missing other views of the same tuber.

## Content and appearance review

All 9,106 images received technical integrity checks. A deterministic seed-42
sample of 12 images from each category (216 images total) was visually reviewed,
along with 24 strongest candidate pairs and both cross-category burst pairs.
The review is appearance screening by the assistant, not expert pathological
diagnosis. It is NOT an exhaustive visual certification of all 9,106 images.

Observed sample characteristics:

- Images show potato tubers, including exposed damaged/decayed tissue and severely
  degraded tubers; no leaf-only images were observed in this sample.
- Both paper and concrete backgrounds occur. Tuber scale, illumination, colour
  cast, shadow, dirt, framing and padding vary substantially. Some tubers occupy
  a small part of the frame, making 224-pixel downsampling a potential problem.
- Many rot-labelled samples show cavities, dark lesions, wrinkling, exposed
  damaged tissue, or mould-like material. This does not establish a pathogen.
- Some labelled views have subtle or unclear lesions. For example,
  Asterix Mild `IMG_20260722_132238.jpg` and Lal Pakri Mild
  `IMG20260724165326.jpg` require expert view-specific checking rather than
  assuming a lesion is clearly visible.
- Severity appearances overlap. Full-image appearance alone cannot certify that
  every label is correct or the grade boundaries consistent across varieties.
- Healthy examples can contain sprouts or minor surface marks; this category
  should not be presented as universal market quality or food-safety approval.
- Some Rumana samples include additional background tubers, even though Rumana
  is excluded from the primary experiment.

Whether ALL images show external-only surfaces and whether EVERY rot-labelled
view contains visible symptoms: **not established**. A comprehensive content/
expert label review remains outstanding. No images were excluded based on this
screening.

## Pooling assessment

All three proposed varieties have all four categories, so the label mapping is
structurally possible. Pooling is not yet cleared scientifically:

1. Variety-by-class proportions differ substantially (see the count matrix).
2. Group identity remains unresolved.
3. The source labels have apparent conflicts and undocumented grade boundaries.
4. Background and capture conditions can correlate with labels. All 510 Asterix
   Healthy filenames and 514 Diamant Healthy filenames encode 2025-11-11, while
   their diseased folders span other dates including 2026-07-22. All 647 Lal
   Pakri Healthy filenames encode 2026-07-24, while its diseased folders also
   include 2026-07-22. These are filename-derived associations, not verified EXIF
   dates. See `capture_dates_by_category.csv`.
5. Sampled Asterix/Diamant Healthy images are consistently framed on white paper,
   whereas disease samples also include concrete, dirt and different scales.
   Dataset-wide background proportions have NOT been manually established.

Cropping and training-only augmentation may reduce some shortcuts but cannot
repair missing identities or prove independence. Binary merging may reduce
severity-boundary problems but does not solve leakage, provenance, or date/
background confounding.

## Recommendation and next gate

Keep the four-class design provisional; do not create the final split or train.
The next step should be obtaining from the creators a physical-tuber-to-filename
mapping (including all views/backgrounds), plus clarification of severity grading
and the two burst conflicts. No contact was made on the user's behalf.

If identities cannot be recovered, discuss an alternative with documented
independent samples or newly acquired data with the lecturer. An explicitly
limited image-level benchmark would require a deliberate change to the current
leakage-safe study design; it cannot be described as tuber-independent.

## Reproduction of this preparation audit

```bash
python scripts/download_candidate_a.py
.venv-potato/bin/python scripts/audit_candidate_a.py
.venv-potato/bin/python scripts/summarize_dataset_audit.py
```

## Subsequent experiment update

After this initial audit, the project proceeded with a four-class appearance/severity experiment using the Asterix, Diamant, and Lal Pakri categories. A later best-effort leakage-control pass quarantined eight contradictory or inferred-related captures, grouped likely related captures, and froze seed-42 train/validation/test manifests for 6,329 images. The three requested model families were trained and evaluated on those same partitions. This later work does not establish physical-tuber identities or remove the uncertainty about independence; see `outputs/experiments/COMPLETION_REPORT.md` for the exact split, results, and remaining risks. The original audit findings about unverified identities and severity labels remain applicable.

The summarisation script records the specific manual contact-sheet observations
made during this audit; re-running it does not perform a new human review.
Source images remain unchanged. No final experimental metrics exist.
