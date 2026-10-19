# Candidate A experiment completion report

## Task and data

Image-only four-class classification of the publisher-labelled appearance/severity categories: Healthy, DryRot-Mild, DryRot-Moderate and DryRot-Severe. Source: Mendeley Data, DOI 10.17632/rgn2d2jb7f.1, CC BY 4.0. Images are 512 x 512 JPEGs; model input is 224 x 224. The dataset labels are visual grades, not laboratory-confirmed diagnoses.

The requested variety subset contained 6337 images. 8 images were quarantined during the pre-split leakage/conflict pass, leaving 6329 images. Independent physical tuber count remains UNKNOWN.

| Class | Before leakage-control pass | Usable after exclusions | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| Healthy | 1671 | 1671 | 1425 | 56 | 190 |
| DryRot-Mild | 280 | 277 | 185 | 51 | 41 |
| DryRot-Moderate | 1128 | 1127 | 789 | 173 | 165 |
| DryRot-Severe | 3258 | 3254 | 2307 | 480 | 467 |

Split totals: 4706 train, 760 validation, 863 test. Seed 42. Membership and file hashes are frozen in `data/manifests/candidate_a/freeze.json`.

## Leakage controls and limits

Images were grouped using capture-name/time windows and conservative perceptual similarity corroborated by timestamp or decoded-pixel similarity. Two visually reviewed contradictory burst sequences and highly similar conflicting-label captures were quarantined with their inferred components. Group allocations were selected by a seed-42 search over 1,000 grouped 70/15/15 candidate splits, minimising relative class-count deviation. The largest inferred group contains 856 images. These are inferred related-capture groups, not verified potato identities. Physical tuber IDs and complete linkage remain UNKNOWN. The audit also records 608 weaker perceptual candidate pairs across partitions; they were not merged because the evidence was insufficient and would risk over-grouping. Therefore results must not be described as confirmed tuber-independent performance.

## Training and reproducibility

All models used the same frozen split, class mapping, RGB conversion, resize to 256 then 224 crop, ImageNet normalization, training-only random resized crop (scale 0.82-1.0), horizontal/vertical flips and random rotation up to 10 degrees. The CNN was trained from scratch; EfficientNetV2-S and ViT-B/16 used torchvision ImageNet1K_V1 weights, first training the classification head for 3 epochs and then fine-tuning the full model. Batch size 8; AdamW; ReduceLROnPlateau (factor 0.5, patience 2); weighted cross-entropy; class weights derived from training counts only; seed 42; early stopping patience 5 on validation Macro-F1; fine-tuning cap 20 epochs. CUDA mixed precision was enabled on the RTX 2070 SUPER (8 GiB).

Versions recorded in each `run.json`; PyTorch 2.13.0+cu126 and torchvision 0.28.0+cu126. Class weights (Healthy, Mild, Moderate, Severe): 0.8256, 6.3595, 1.4911, 0.5100.

## Test results

| Model | Epochs run | Best checkpoint epoch | Test accuracy | Macro precision | Macro recall | Macro-F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Custom CNN | 20 | 19 | 0.771 | 0.623 | 0.581 | 0.592 | 0.768 |
| EfficientNetV2-S | 23 | 23 | 0.771 | 0.662 | 0.677 | 0.654 | 0.794 |
| ViT-B/16 | 14 | 9 | 0.826 | 0.698 | 0.706 | 0.700 | 0.833 |

ViT-B/16 had the highest test Macro-F1 (0.700) and accuracy (0.826) in this run. This is a single fixed split and one run per model, so the difference is descriptive rather than a statistical claim.

Mild dry rot was the hardest class: support was 41 images and recall was low for the custom CNN (0.073), higher for EfficientNetV2-S (0.439), and 0.390 for ViT-B/16. Mild was most often confused with Moderate. Moderate and Severe also overlapped, especially Moderate predicted as Severe. Healthy recall was 0.800, 0.805 and 0.921 respectively. These errors align with class imbalance and the absence of a quantitative, independently adjudicated severity rubric.

### Per-class test metrics

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Custom CNN | Healthy | 1.000 | 0.800 | 0.889 | 190 |
| Custom CNN | DryRot-Mild | 0.188 | 0.073 | 0.105 | 41 |
| Custom CNN | DryRot-Moderate | 0.458 | 0.558 | 0.503 | 165 |
| Custom CNN | DryRot-Severe | 0.846 | 0.895 | 0.870 | 467 |
| EfficientNetV2-S | Healthy | 1.000 | 0.805 | 0.892 | 190 |
| EfficientNetV2-S | DryRot-Mild | 0.200 | 0.439 | 0.275 | 41 |
| EfficientNetV2-S | DryRot-Moderate | 0.528 | 0.630 | 0.575 | 165 |
| EfficientNetV2-S | DryRot-Severe | 0.922 | 0.835 | 0.876 | 467 |
| ViT-B/16 | Healthy | 1.000 | 0.921 | 0.959 | 190 |
| ViT-B/16 | DryRot-Mild | 0.286 | 0.390 | 0.330 | 41 |
| ViT-B/16 | DryRot-Moderate | 0.594 | 0.612 | 0.603 | 165 |
| ViT-B/16 | DryRot-Severe | 0.911 | 0.901 | 0.906 | 467 |

## Figures and examples

The outputs folder contains training/validation loss curves, validation Macro-F1 and accuracy curves, raw and row-normalised confusion matrices for each model, and eight correct plus eight incorrect test examples per model. The CSV comparison table, prediction records, checkpoints, run configurations, and histories are saved alongside them.

## Rubric audit

| Assignment requirement | Status in this project |
|---|---|
| Clear task and practical value | Implemented: four-class visible tuber appearance/severity classification; describe as image classification support, not diagnosis. |
| Dataset source, access, licence, classes and counts | Documented in this report, README, and source audit; CC BY 4.0 attribution/change disclosure required. |
| Cleaning, labels, duplicates and imbalance | 8 contradictory captures quarantined with reasons; 0 corrupt source images and 0 exact duplicate groups in the source audit; class weighting computed from training only. |
| Three distinct model families and transfer learning | Implemented: custom CNN, EfficientNetV2-S, ViT-B/16; same four classes and same partitions. |
| Leakage-safe split | Best-effort inferred capture grouping is frozen and reproducible. Physical IDs remain UNKNOWN, 608 weak perceptual candidate pairs remain across partitions, and the very large inferred groups distort the target ratios. Full physical-sample independence is not verified; report this as a material limitation. |
| Training information and reproducibility | Config, weights, hyperparameters, software/hardware versions, histories, checkpoints, split manifests, and run times are saved. |
| Evaluation and error interpretation | Implemented: accuracy, macro precision/recall/F1, weighted F1, per-class metrics, raw/normalised confusion matrices, curves, and correct/incorrect examples. |
| Final report PDF (20 marks) | Not yet prepared. Use the completion report and artifacts to write the concise final 10-15 page PDF; include references and candid limitations. |
| Oral presentation slides (10 marks) | Not yet prepared. All group members must present and answer questions. |
| Notebook, source files, README, split files | Notebook with visible outputs, Python scripts, README, and frozen split files are present. |

## Remaining assignment deliverables

The final 10-15 page PDF report and oral-presentation slides still need to be prepared and reviewed by the group. The project has an output-visible results notebook, source scripts, a README, and frozen split manifests. All group members must understand the dataset limitations, model choices, errors, and results. Cite the dataset, pretrained models, PyTorch/torchvision and other technical sources in the final report. The dataset licence requires attribution and disclosure of changes.

## References

Islam Saif, S., Tasnim Saba, M. F., and Kaiser, M. S. Mendeley Data, Potato disease dataset, version 1. https://doi.org/10.17632/rgn2d2jb7f.1. CC BY 4.0.

PyTorch/torchvision pretrained model weights: EfficientNetV2-S ImageNet1K_V1 and ViT-B/16 ImageNet1K_V1, downloaded from the official PyTorch model-weight URLs recorded in each run.json.
