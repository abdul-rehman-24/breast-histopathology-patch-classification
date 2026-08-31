"""Patient-wise split creation and leakage checking."""
import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def build_dataframe_from_directory(base_dir):
    records = []
    for dirpath, _, filenames in os.walk(base_dir):
        for fname in filenames:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(dirpath, fname)
                parts = full_path.split(os.sep)
                try:
                    mag_folder, patient_folder, tumor_type = parts[-2], parts[-3], parts[-4]
                    label = parts[-6]
                except IndexError:
                    continue
                records.append({"filepath": full_path, "filename": fname, "group_id": patient_folder,
                                 "magnification": mag_folder, "tumor_type": tumor_type, "label": label})
    return pd.DataFrame(records)

def patient_wise_split(df, seed=42, train_size=0.70, val_size=0.15):
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df["group_id"]))
    train_df, temp_df = df.iloc[train_idx], df.iloc[temp_idx]
    remaining_val_fraction = val_size / (1 - train_size)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=remaining_val_fraction, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["group_id"]))
    return train_df, temp_df.iloc[val_idx], temp_df.iloc[test_idx]

def check_leakage(train_df, val_df, test_df):
    train_groups, val_groups, test_groups = set(train_df["group_id"]), set(val_df["group_id"]), set(test_df["group_id"])
    overlap_tv, overlap_tt, overlap_vt = train_groups & val_groups, train_groups & test_groups, val_groups & test_groups
    dup_paths = pd.concat([train_df["filepath"], val_df["filepath"], test_df["filepath"]])
    dup_count = dup_paths.duplicated().sum()
    passed = len(overlap_tv) == 0 and len(overlap_tt) == 0 and len(overlap_vt) == 0 and dup_count == 0
    report = {"passed": passed, "train_val_overlap": len(overlap_tv), "train_test_overlap": len(overlap_tt),
              "val_test_overlap": len(overlap_vt), "duplicate_paths": int(dup_count)}
    return passed, report
