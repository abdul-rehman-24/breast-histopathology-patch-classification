"""DINOv2 foundation model loader (frozen feature extractor)."""
import torch

def load_dinov2(model_name="dinov2_vits14", device="cuda", freeze=True):
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    model.eval()
    return model.to(device)
