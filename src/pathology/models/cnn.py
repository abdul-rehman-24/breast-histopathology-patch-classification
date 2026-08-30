"""EfficientNet-B0 CNN baseline + feature extractor wrapper."""
import torch.nn as nn
import torchvision.models as models

def build_efficientnet_b0(num_classes=2, pretrained=True):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

class CNNFeatureExtractor(nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        self.features = trained_model.features
        self.avgpool = trained_model.avgpool
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)
