"""Test that the FusionModel produces the expected tensor shapes end-to-end."""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathology.models.fusion import FusionModel, ProjectionHead


def test_fusion_model_output_shape():
    model = FusionModel(cnn_dim=1280, foundation_dim=384, proj_dim=256, num_classes=2)
    batch_size = 4
    cnn_feat = torch.randn(batch_size, 1280)
    foundation_feat = torch.randn(batch_size, 384)

    out = model(cnn_feat, foundation_feat)
    assert out.shape == (batch_size, 2)

    out2, fused = model(cnn_feat, foundation_feat, return_fused=True)
    assert fused.shape == (batch_size, 512)


def test_projection_head_shape():
    proj = ProjectionHead(in_dim=1280, out_dim=256)
    x = torch.randn(4, 1280)
    out = proj(x)
    assert out.shape == (4, 256)
