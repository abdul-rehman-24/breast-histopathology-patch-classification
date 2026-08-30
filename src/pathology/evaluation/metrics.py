"""Shared metric computation used across all experiments."""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score,
                               accuracy_score, balanced_accuracy_score, confusion_matrix)


def compute_full_metrics(labels, probs, threshold=0.5):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs > threshold).astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    result = {
        "auroc": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else None,
        "auprc": float(average_precision_score(labels, probs)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "sensitivity_recall": float(tp / (tp + fn)) if (tp + fn) > 0 else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else None,
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(labels)),
    }
    return result


def plot_confusion_matrix(cm, title, save_path, cmap="Blues"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Malignant"])
    ax.set_yticklabels(["Benign", "Malignant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center")
    plt.colorbar(im)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
