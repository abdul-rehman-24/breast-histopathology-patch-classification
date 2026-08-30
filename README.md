# Breast Cancer Histopathology Classification — CNN vs Foundation Model Fusion (V1.0)

## Problem Statement
Breast cancer diagnosis from histopathology images requires expert pathologist review, which is time-consuming
and subject to inter-observer variability. This project investigates whether combining a task-specific CNN with
a general-purpose vision foundation model improves automated benign/malignant classification on the BreakHis
histopathology image dataset.

## Research Question
Do CNN and Foundation Model (DINOv2) representations provide complementary information when fused, and does
patient-aware aggregation or ensembling provide additional benefit?

## Dataset
- BreakHis breast cancer histopathology dataset: 7,909 images from 82 patients
- Images span benign (2,480) and malignant (5,429) tumor classes at four magnifications (40X/100X/200X/400X)
- Source: Kaggle - ambarish/breakhis

## Leakage-Safe Splitting Strategy
- Patient/group-wise 70/15/15 split (57/12/13 patients)
- Automated leakage checker confirmed zero patient overlap and zero duplicate images across splits

## Models
| Model | Role | Frozen? |
|---|---|---|
| EfficientNet-B0 | CNN baseline, fine-tuned | No |
| DINOv2 ViT-S/14 | Foundation model, feature extractor | Yes |
| Fusion classifier | Combines both representations | Only fusion head trained |
| Attention aggregator | Patient-level aggregation of fused vectors | Only attention head trained |

## Architecture
See figures/final_architecture_diagram.png

## Five Experiments
1. CNN Baseline
2. Foundation-Only Baseline
3. CNN + Foundation Fusion
4. Fusion + Attention/Grouped Aggregation
5. Validation-Weighted Ensemble

## Results (Test Set, n=1,395 images unless noted)

| Experiment | AUROC | AUPRC | F1 | Sensitivity | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| CNN Baseline | 0.9062 | 0.9238 | 0.8578 | 0.7990 | 0.8541 | 0.8158 |
| Foundation-Only | 0.9054 | 0.9326 | 0.9011 | 0.8825 | 0.8259 | 0.8652 |
| **CNN + Foundation Fusion** | **0.9087** | 0.9269 | **0.9255** | **0.9227** | 0.8376 | **0.8968** |
| Attention/Grouped (n=13 groups) | 0.7778 | 0.8412 | 0.9474 | 1.0000 | 0.7500 | 0.9231 |
| Weighted Ensemble | 0.9054 | 0.9326 | 0.9011 | 0.8825 | 0.8259 | 0.8652 |

Recommended model: CNN + Foundation Fusion — most balanced/consistent performer.

## Explainability
- Grad-CAM applied to EfficientNet-B0 backbone on 8 representative test images
- Grad-CAM explains the CNN pathway only; DINOv2 branch not directly compatible (documented limitation)
- Attention weights from grouped model visualized for one test patient

## Robustness
| Condition | AUROC | Change |
|---|---:|---:|
| Clean (baseline) | 0.9062 | — |
| Gaussian Noise (σ=0.05) | 0.9032 | -0.0030 |
| Gaussian Blur (k=5) | 0.8713 | -0.0349 |

## Limitations
- Patch/image-level pipeline using patient/group identifiers — NOT a true Whole-Slide Image (WSI) pipeline
- AUROC gains from fusion are modest, not dramatic
- Attention/grouped results based on only 13 test-set patient groups — exploratory, not conclusive
- No clinical validation or diagnostic deployment claims are made

## Reproducibility
See docs/REPRODUCIBILITY.md

## Storage Strategy
- GitHub: code, configs, splits, metrics, experiment records, small figures
- Google Drive: model checkpoints, feature/embedding caches, full-resolution visualizations

## Phase 2 Direction
True WSI-level modeling with real slide-level coordinates is a planned Phase 2 extension.

## Citation
BreakHis dataset via Kaggle: https://www.kaggle.com/datasets/ambarish/breakhis
