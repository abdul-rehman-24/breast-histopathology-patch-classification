"""Loads all trained checkpoints + feature caches. Run after setup_environment.py.
Usage:
    from scripts.load_all_checkpoints import load_everything
    everything = load_everything(ctx, device)
"""
import torch
from pathology.models.cnn import build_efficientnet_b0
from pathology.models.fusion import FusionModel, FoundationOnlyModel
from pathology.models.attention import AttentionAggregator


def load_everything(ctx, device="cuda"):
    drive_root = ctx["drive_root"]

    cnn_model = build_efficientnet_b0().to(device)
    cnn_ckpt_path = drive_root + "/checkpoints/cnn_baseline/best.pt"
    cnn_ckpt = torch.load(cnn_ckpt_path, map_location=device, weights_only=False)
    cnn_model.load_state_dict(cnn_ckpt["model_state"])
    cnn_model.eval()

    fusion_model = FusionModel().to(device)
    fusion_ckpt_path = drive_root + "/checkpoints/fusion_model/best.pt"
    fusion_ckpt = torch.load(fusion_ckpt_path, map_location=device, weights_only=False)
    fusion_model.load_state_dict(fusion_ckpt["model_state"])
    fusion_model.eval()

    fo_model = FoundationOnlyModel().to(device)
    fo_ckpt_path = drive_root + "/checkpoints/foundation_only/best.pt"
    fo_ckpt = torch.load(fo_ckpt_path, map_location=device, weights_only=False)
    fo_model.load_state_dict(fo_ckpt["model_state"])
    fo_model.eval()

    attn_model = AttentionAggregator().to(device)
    attn_ckpt_path = drive_root + "/checkpoints/attention_grouped/best.pt"
    attn_ckpt = torch.load(attn_ckpt_path, map_location=device, weights_only=False)
    attn_model.load_state_dict(attn_ckpt["model_state"])
    attn_model.eval()

    caches = {}
    splits = ["train", "val", "test"]
    for split in splits:
        cnn_path = drive_root + "/embeddings/cnn_efficientnet_b0/" + split + "_cnn_features.pt"
        emb_path = drive_root + "/embeddings/dinov2_vits14/" + split + "_embeddings.pt"
        fused_path = drive_root + "/embeddings/fused_vectors/" + split + "_fused.pt"
        caches[split + "_cnn"] = torch.load(cnn_path, weights_only=False)
        caches[split + "_emb"] = torch.load(emb_path, weights_only=False)
        caches[split + "_fused"] = torch.load(fused_path, weights_only=False)

    print("Loaded: CNN, Fusion, Foundation-only, Attention checkpoints + all feature caches")

    result = {}
    result["cnn_model"] = cnn_model
    result["fusion_model"] = fusion_model
    result["foundation_only_model"] = fo_model
    result["attn_model"] = attn_model
    for key in caches:
        result[key] = caches[key]
    return result
