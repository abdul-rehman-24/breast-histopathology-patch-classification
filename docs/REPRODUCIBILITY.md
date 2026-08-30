# Reproducibility Guide

## Fresh Colab Session — What to Run

### RUN AGAIN (every fresh session)
1. Mount Google Drive
2. Clone/pull this GitHub repository
3. Re-download the BreakHis dataset via kagglehub (fast, ~30s, ~4GB, not stored in Drive)
4. Load split manifests from splits/*.csv

### LOAD (from persistent storage, do not regenerate)
- configs/config.yaml - project configuration
- splits/train.csv, val.csv, test.csv - patient-wise split (GitHub)
- Checkpoints (Google Drive):
  - checkpoints/cnn_baseline/best.pt
  - checkpoints/foundation_only/best.pt
  - checkpoints/fusion_model/best.pt
  - checkpoints/attention_grouped/best.pt
- Feature/embedding caches (Google Drive):
  - embeddings/dinov2_vits14/train_val_test_embeddings.pt
  - embeddings/cnn_efficientnet_b0/train_val_test_cnn_features.pt
  - embeddings/fused_vectors/train_val_test_fused.pt

### DO NOT RERUN (unless intentionally reproducing that specific experiment)
- Patient-wise split generation (splits/*.csv already finalized and leakage-checked)
- Leakage check (already PASS)
- CNN training (10 epochs already completed, best checkpoint saved)
- DINOv2 embedding extraction (all 7,909 images already cached)
- CNN feature extraction (all 7,909 images already cached)
- Fusion / Foundation-only / Attention model training (checkpoints already saved)
- Ensemble weight search (weights already finalized in results/ensemble_weights.json)

## Resume Behavior
All expensive operations were implemented with restartable caching:
- Training loops save a latest.pt checkpoint every epoch (optimizer + scheduler state included) and a
  separate best.pt checkpoint whenever validation AUROC improves. If a session disconnects, rerunning the
  training cell automatically resumes from latest.pt.
- Feature/embedding extraction checks which image paths are already present in the cache file and only
  processes the remaining ones - safe to interrupt and rerun at any time.

## Environment
- Platform: Google Colab (free tier, Tesla T4 GPU)
- Key libraries: PyTorch, torchvision, scikit-learn, pandas, kagglehub
- Dataset: BreakHis via Kaggle (ambarish/breakhis)

## Verifying Reproducibility
An evaluation audit (results/evaluation_audit.json) was performed in Day 6: CNN test metrics were
recomputed from the saved checkpoint and matched the originally reported Day-2 values exactly
(AUROC, F1, Accuracy all matched to 4 decimal places).
