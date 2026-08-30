"""Visualization of learned attention weights from the grouped/attention aggregation model.
Note: reflects model-internal weighting, not confirmed biological/clinical importance."""
import numpy as np


def get_group_attention(attn_model, instances, device):
    """Returns attention weights (numpy array) for one patient group's instances."""
    instances = instances.to(device)
    import torch
    with torch.no_grad():
        _, attn_weights = attn_model(instances)
    return attn_weights.squeeze().cpu().numpy()


def top_k_attended(filepaths, attn_weights_np, k=5):
    top_idx = np.argsort(attn_weights_np)[-k:][::-1]
    return [{"file": filepaths[i].split("/")[-1], "weight": float(attn_weights_np[i])} for i in top_idx]


def plot_attention_distribution(attn_weights_np, group_id, save_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(attn_weights_np)), sorted(attn_weights_np, reverse=True))
    ax.set_xlabel("Image rank (sorted by attention weight)")
    ax.set_ylabel("Attention weight")
    ax.set_title(f"Attention weight distribution — Group {group_id}")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
