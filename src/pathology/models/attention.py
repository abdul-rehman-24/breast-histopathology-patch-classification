"""Gated attention aggregator for patient/group-level aggregation (NOT true WSI-MIL)."""
import torch
import torch.nn as nn

class AttentionAggregator(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_classes=2):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))
    def forward(self, instances):
        A_V = self.attention_V(instances)
        A_U = self.attention_U(instances)
        attn_scores = self.attention_w(A_V * A_U)
        attn_weights = torch.softmax(attn_scores, dim=0)
        group_repr = torch.sum(attn_weights * instances, dim=0, keepdim=True)
        out = self.classifier(group_repr)
        return out, attn_weights
