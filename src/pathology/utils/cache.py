"""Restartable feature/embedding cache helpers — skip already-processed images on resume."""
import os
import torch
from torch.utils.data import DataLoader


def extract_features_restartable(df, extract_fn, cache_path, extraction_dataset_cls,
                                    transform, data_root_fix, batch_size=64, save_every=10,
                                    feature_key="feature"):
    """
    Generic restartable extraction loop. `extract_fn(batch_imgs) -> tensor` does the model forward pass.
    Saves/resumes a dict cache: {filepath, group_id, label, <feature_key>}.
    """
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=False)
        done_paths = set(cache["filepath"])
    else:
        cache = {"filepath": [], "group_id": [], "label": [], feature_key: []}
        done_paths = set()

    remaining_df = df[~df["filepath"].isin(done_paths)].reset_index(drop=True)
    if len(remaining_df) == 0:
        return cache

    ds = extraction_dataset_cls(remaining_df, transform=transform, data_root_fix=data_root_fix)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    batch_count = 0
    for imgs, paths, group_ids, labels in loader:
        feats = extract_fn(imgs)
        cache["filepath"].extend(list(paths))
        cache["group_id"].extend(list(group_ids))
        cache["label"].extend(list(labels))
        cache[feature_key].extend(list(feats))
        batch_count += 1
        if batch_count % save_every == 0:
            torch.save(cache, cache_path)

    torch.save(cache, cache_path)
    return cache
