"""Re-evaluates all trained models on the test set using cached features (fast, no retraining).
Usage:
    from scripts.evaluate_all import evaluate_all_models
    results = evaluate_all_models(everything, device)
"""
import torch
from pathology.evaluation.metrics import compute_full_metrics
from pathology.data.dataset import LABEL_MAP


def evaluate_all_models(everything, device="cuda", batch_size=128):
    cnn_model = everything["cnn_model"]
    fusion_model = everything["fusion_model"]
    fo_model = everything["foundation_only_model"]
    cnn_cache = everything["test_cnn"]
    emb_cache = everything["test_emb"]

    labels = []
    for l in cnn_cache["label"]:
        labels.append(LABEL_MAP[l])
    n = len(labels)

    cnn_probs = []
    fo_probs = []
    fusion_probs = []

    start = 0
    while start < n:
        end = min(start + batch_size, n)
        cnn_feat = torch.stack(cnn_cache["feature"][start:end]).to(device)
        found_feat = torch.stack(emb_cache["embedding"][start:end]).to(device)
        with torch.no_grad():
            cnn_out = cnn_model.classifier(cnn_feat)
            cnn_batch_probs = torch.softmax(cnn_out, dim=1)[:, 1].cpu().numpy()
            cnn_probs.extend(cnn_batch_probs)

            fo_out = fo_model(found_feat)
            fo_batch_probs = torch.softmax(fo_out, dim=1)[:, 1].cpu().numpy()
            fo_probs.extend(fo_batch_probs)

            fusion_out = fusion_model(cnn_feat, found_feat)
            fusion_batch_probs = torch.softmax(fusion_out, dim=1)[:, 1].cpu().numpy()
            fusion_probs.extend(fusion_batch_probs)

        start = end

    results = {}
    results["cnn"] = compute_full_metrics(labels, cnn_probs)
    results["foundation_only"] = compute_full_metrics(labels, fo_probs)
    results["fusion"] = compute_full_metrics(labels, fusion_probs)

    for name in results:
        m = results[name]
        line = name + ": AUROC=" + str(round(m["auroc"], 4)) + " F1=" + str(round(m["f1"], 4)) + " Acc=" + str(round(m["accuracy"], 4))
        print(line)

    return results
