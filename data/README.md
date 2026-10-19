# Dataset setup

## Source and licence

The project uses the public *Potato disease dataset*, version 1, by Samiul Islam Saif, Mst. Fahmida Tasnim Saba, and M. Shamim Kaiser. Original source: [Mendeley Data](https://data.mendeley.com/datasets/rgn2d2jb7f/1), DOI [10.17632/rgn2d2jb7f.1](https://doi.org/10.17632/rgn2d2jb7f.1). The publisher lists the licence as CC BY 4.0. Retain attribution, link to the licence, and disclose changes. The raw data are intentionally excluded from this repository.

## Task and selected classes

The task is image-only, four-class classification of visible tuber appearance/severity labels:

- `Healthy`
- `DryRot-Mild`
- `DryRot-Moderate`
- `DryRot-Severe`

Images come from the Asterix, Diamant, and Lal Pakri categories. Rumana, Scab, and Fully-Rotten categories are excluded. These are publisher-provided visual labels, not laboratory-confirmed diagnoses. See `../outputs/audit/DATASET_VERIFICATION_REPORT.md` and `../outputs/experiments/COMPLETION_REPORT.md` for verified counts, cleaning decisions, and limitations.

## Obtain and prepare the data

From the repository root, use the included downloader. It reads the original Mendeley metadata and verifies the publisher-provided file hashes:

```bash
python scripts/download_candidate_a.py
```

The archive is expected at:

```text
data/raw/mendeley_rgn2d2jb7f_v1/potato_desease_dataset.zip
```

The filename spelling `desease` matches the deposited archive. The preparation program checks the archive SHA-256 against the verified source audit. It uses the frozen manifests already in this repository; if the split artifacts are absent in a new checkout, it reconstructs the inferred grouping and split with seed 42.

```bash
PYTHONPATH=src python -m potato.prepare
PYTHONPATH=src python -m potato.prepare --verify
```

Preparation extracts only selected images into `data/processed/candidate_a/images/`, which is ignored by Git. Frozen split membership is stored in `data/manifests/candidate_a/{train,validation,test}.csv`; `all.csv` gives the full selected membership, and `exclusions.csv` documents every quarantined image. `freeze.json` records hashes for the split and audit files. The `inferred_group` field identifies captures inferred to be related; it is not a verified physical-tuber identifier.

After download and preparation, train from the repository root:

```bash
PYTHONPATH=src python -m potato.train
```

Never tune on the held-out test split. Checkpoints, raw dataset files, and extracted images remain local and are not redistributed through GitHub.
