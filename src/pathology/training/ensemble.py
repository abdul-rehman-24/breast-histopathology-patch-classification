"""Validation-only weighted ensemble search."""
import numpy as np
from sklearn.metrics import roc_auc_score


def search_ensemble_weights(val_probs_dict, val_labels, step=0.1):
    """
    val_probs_dict: {"cnn": np.array, "foundation_only": np.array, "fusion": np.array}
    Returns (best_weights: dict, best_val_auroc: float). Weights sum to 1, searched on validation ONLY.
    """
    keys = list(val_probs_dict.keys())
    assert len(keys) == 3, "Expects exactly 3 model probability arrays"
    k1, k2, k3 = keys

    best_weights, best_auroc = None, 0.0
    grid = np.arange(0, 1.01, step)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < 0 or w3 > 1:
                continue
            ensemble_prob = w1 * val_probs_dict[k1] + w2 * val_probs_dict[k2] + w3 * val_probs_dict[k3]
            auroc = roc_auc_score(val_labels, ensemble_prob)
            if auroc > best_auroc:
                best_auroc = auroc
                best_weights = {k1: round(w1, 2), k2: round(w2, 2), k3: round(w3, 2)}
    return best_weights, best_auroc


def apply_ensemble(test_probs_dict, weights):
    """Applies finalized weights to test-set probabilities. Call ONLY after weights are finalized on validation."""
    result = np.zeros_like(next(iter(test_probs_dict.values())))
    for k, w in weights.items():
        result += w * test_probs_dict[k]
    return result
