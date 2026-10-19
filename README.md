# Potato Tuber Dry-Rot Appearance and Severity Classification

This project compares three deep-learning approaches for classifying photographs of potato tubers by visible condition: a custom convolutional neural network (CNN), EfficientNetV2-S, and Vision Transformer ViT-B/16. The goal is an image-classification study of publisher-labelled visual categories, not laboratory diagnosis or food-safety certification.

## Classification task

The input is a colour image of a potato tuber. The model predicts one of four labels:

1. Healthy
2. DryRot-Mild
3. DryRot-Moderate
4. DryRot-Severe

The labels are the source dataset's appearance/severity categories. The publisher does not provide a quantitative lesion grading rubric or verified physical-tuber IDs.

## Dataset

The source is the public *Potato disease dataset*, version 1, from [Mendeley Data](https://data.mendeley.com/datasets/rgn2d2jb7f/1), DOI [10.17632/rgn2d2jb7f.1](https://doi.org/10.17632/rgn2d2jb7f.1). It is released under CC BY 4.0. Attribute the creators and source, link the licence, and disclose any changes. The raw archive and extracted photographs are excluded from GitHub. Dataset retrieval and local preparation are documented in [`data/README.md`](data/README.md).

The selected Asterix, Diamant, and Lal Pakri categories contained 6,337 images before leakage-control exclusions. Eight contradictory/inferred-related captures were quarantined, leaving 6,329 images. The frozen split has 4,706 training, 760 validation, and 863 test images. Counts by class and partition are in [`outputs/experiments/COMPLETION_REPORT.md`](outputs/experiments/COMPLETION_REPORT.md).

The split groups likely related captures using filenames, capture times, and conservative image-similarity evidence. These are inferred groups, not confirmed physical-tuber IDs. Some weaker perceptual candidates remain across partitions; the results must not be described as verified tuber-independent performance. Details are in the completion report and the source audit.

## Models and comparison

- **Custom CNN:** baseline trained from scratch.
- **EfficientNetV2-S:** ImageNet1K_V1 pretrained; classifier-head training followed by full fine-tuning.
- **ViT-B/16:** ImageNet1K_V1 pretrained; classifier-head training followed by full fine-tuning.

All three use the same frozen training, validation, and test manifests, image classes, core transforms, and test metrics. Checkpoint selection uses validation Macro-F1. Class weights are computed from the training partition only. The test set is held out for the final model comparison.

The recorded run achieved test Macro-F1 of 0.592 for the custom CNN, 0.654 for EfficientNetV2-S, and 0.700 for ViT-B/16. Full metrics, per-class results, confusion matrices, and error discussion are in `outputs/experiments/`. These figures represent one run on one fixed split.

## Repository layout

```text
configs/                       Experiment configuration
data/README.md                 Dataset source, licence, and setup
data/manifests/candidate_a/    Frozen split membership and exclusions
notebooks/                     Results notebook with saved aggregate outputs
outputs/audit/                 Dataset integrity and grouping audit
outputs/experiments/           Metrics, histories, reports, and aggregate figures
scripts/                       Download, audit, and artifact-building scripts
src/potato/                    Preparation, data loading, models, and training
requirements.txt               Python dependencies
```

Raw archives, extracted images, local environments, pretrained weights, and trained checkpoints remain outside version control. After setup, raw images are placed under `data/raw/` and extracted into `data/processed/`.

## Environment and dependencies

The completed runs used Python 3.12.14, PyTorch 2.13.0+cu126, torchvision 0.28.0+cu126, NumPy 2.5.2, Pillow 12.3.0, and an NVIDIA RTX 2070 SUPER with 8 GiB VRAM. GPU training requires a compatible NVIDIA driver and a PyTorch build that supports the available CUDA runtime. CPU use is possible but may be substantially slower, especially for ViT-B/16.

Create an environment from the repository root:

```bash
python3 -m venv .venv-potato
source .venv-potato/bin/activate
python -m pip install --upgrade pip
```

Install the PyTorch/torchvision build appropriate for your platform and CUDA version using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

The dataset downloader also requires `curl`. Notebook use requires Jupyter/IPython; `nbformat` is included for rebuilding the notebook.

## Dataset download and preparation

Download the original publisher files and verify their hashes:

```bash
python scripts/download_candidate_a.py
```

Prepare the selected images and preserve the frozen split membership:

```bash
PYTHONPATH=src python -m potato.prepare
PYTHONPATH=src python -m potato.prepare --verify
```

When the frozen manifests are present, the preparation command verifies them and materialises any missing selected images from the verified archive without changing split membership. On a fresh project without manifests, it recreates the inferred grouping and seed-42 split. See [`data/README.md`](data/README.md) for details.

## Training and evaluation

From the repository root, with the environment active and the dataset prepared:

```bash
PYTHONPATH=src python -m potato.train
```

The command trains the custom CNN and both pretrained models, selects checkpoints using validation Macro-F1, then evaluates each selected checkpoint on the held-out test set. Settings are in [`configs/experiment.json`](configs/experiment.json). Existing artifacts under `outputs/experiments/` will be replaced by a new run, so copy them elsewhere first if they need to be preserved. Pretrained weights are downloaded from the official torchvision/PyTorch URLs if not already cached.

Rebuild plots and the notebook from saved histories, metrics, and predictions without running test inference again:

```bash
python scripts/make_final_artifacts.py
python scripts/build_results_notebook.py
```

## Results and reproducibility

- [`outputs/experiments/COMPLETION_REPORT.md`](outputs/experiments/COMPLETION_REPORT.md): dataset counts, split, training setup, results, error analysis, rubric status, and remaining coursework items.
- [`outputs/experiments/three_model_comparison.csv`](outputs/experiments/three_model_comparison.csv): aggregate test comparison.
- `outputs/experiments/<model>/`: saved run configuration, training history, test metrics, predictions, and best checkpoint (checkpoints are ignored by Git).
- `outputs/experiments/*confusion_matrix*.png`: raw and normalised confusion matrices.
- `outputs/experiments/*loss.png` and `*accuracy.png` / `*macro_f1.png`: training and validation curves.
- [`notebooks/experiment_results.ipynb`](notebooks/experiment_results.ipynb): saved aggregate outputs for review.
- `data/manifests/candidate_a/freeze.json`: manifest hashes, seed, and group identity caveat.

The run records include random seed, software versions, GPU and driver details, model weights, optimiser, scheduler, loss, class weights, transforms, batch size, stopping settings, and run time. Use the exact saved manifests for model comparisons; do not tune on the test set.

## Coursework submission

The code, README, dataset instructions, split files, results notebook, metrics, and figures are prepared in this repository. The final student-authored PDF report and presentation slides remain to be completed. The group should discuss the unresolved physical-sample identity, remaining candidate similarity pairs, class imbalance, and visual-label limitations honestly in those submissions.
