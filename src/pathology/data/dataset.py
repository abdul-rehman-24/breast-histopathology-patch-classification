"""Dataset classes for BreakHis raw images and cached feature vectors."""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

LABEL_MAP = {"benign": 0, "malignant": 1}
LABEL_MAP_INV = {0: "benign", 1: "malignant"}

IMAGE_TRANSFORM = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
TRAIN_TRANSFORM = T.Compose([
    T.Resize((224, 224)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def resolve_path(path, data_root_fix=None):
    if data_root_fix:
        old_root, new_root = data_root_fix
        return path.replace(old_root, new_root)
    return path

class BreakHisDataset(Dataset):
    def __init__(self, df, transform=None, data_root_fix=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform or IMAGE_TRANSFORM
        self.data_root_fix = data_root_fix
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = resolve_path(row["filepath"], self.data_root_fix)
        img = Image.open(path).convert("RGB")
        return self.transform(img), LABEL_MAP[row["label"]]

class ExtractionDataset(Dataset):
    def __init__(self, df, transform=None, data_root_fix=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform or IMAGE_TRANSFORM
        self.data_root_fix = data_root_fix
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = resolve_path(row["filepath"], self.data_root_fix)
        img = Image.open(path).convert("RGB")
        return self.transform(img), row["filepath"], row["group_id"], row["label"]

class CachedFeatureDataset(Dataset):
    def __init__(self, cache):
        self.filepaths = cache["filepath"]
        self.features = cache.get("feature") or cache.get("embedding")
        self.labels = [LABEL_MAP[l] for l in cache["label"]]
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

class FusedFeatureDataset(Dataset):
    def __init__(self, cnn_cache, emb_cache):
        emb_lookup = {p: i for i, p in enumerate(emb_cache["filepath"])}
        self.cnn_features = cnn_cache["feature"]
        self.foundation_features = []
        self.labels = []
        for i, path in enumerate(cnn_cache["filepath"]):
            j = emb_lookup[path]
            self.foundation_features.append(emb_cache["embedding"][j])
            self.labels.append(LABEL_MAP[cnn_cache["label"][i]])
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.cnn_features[idx], self.foundation_features[idx], self.labels[idx]

def build_grouped_structure(df, cnn_cache, emb_cache):
    cnn_lookup = {p: i for i, p in enumerate(cnn_cache["filepath"])}
    emb_lookup = {p: i for i, p in enumerate(emb_cache["filepath"])}
    grouped = {}
    for _, row in df.iterrows():
        gid, path = row["group_id"], row["filepath"]
        if gid not in grouped:
            grouped[gid] = {"filepaths": [], "cnn_idx": [], "emb_idx": [], "label": row["label"]}
        grouped[gid]["filepaths"].append(path)
        grouped[gid]["cnn_idx"].append(cnn_lookup[path])
        grouped[gid]["emb_idx"].append(emb_lookup[path])
    return grouped
