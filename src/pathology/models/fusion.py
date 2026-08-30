"""Projection heads and the CNN+Foundation fusion model."""
import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(0.3))
    def forward(self, x): return self.proj(x)

class FusionModel(nn.Module):
    def __init__(self, cnn_dim=1280, foundation_dim=384, proj_dim=256, num_classes=2):
        super().__init__()
        self.cnn_projection = ProjectionHead(cnn_dim, proj_dim)
        self.foundation_projection = ProjectionHead(foundation_dim, proj_dim)
        self.classifier = nn.Sequential(nn.Linear(proj_dim*2, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))
    def forward(self, cnn_feat, foundation_feat, return_fused=False):
        proj_cnn = self.cnn_projection(cnn_feat)
        proj_found = self.foundation_projection(foundation_feat)
        fused = torch.cat([proj_cnn, proj_found], dim=1)
        out = self.classifier(fused)
        return (out, fused) if return_fused else out

class FoundationOnlyModel(nn.Module):
    def __init__(self, in_dim=384, proj_dim=256, num_classes=2):
        super().__init__()
        self.projection = ProjectionHead(in_dim, proj_dim)
        self.classifier = nn.Sequential(nn.Linear(proj_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))
    def forward(self, x): return self.classifier(self.projection(x))
