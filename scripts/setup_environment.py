"""Run this first in any fresh Colab/Kaggle session.
Mounts Drive, clones/pulls repo, downloads dataset, returns key paths.
Usage (in a notebook cell):
    %run scripts/setup_environment.py
"""
import os
import sys

REPO_ROOT = "/content/pathology_project"
DRIVE_ROOT = "/content/drive/MyDrive/pathology_project"

def setup():
    from google.colab import drive
    drive.mount("/content/drive")

    if not os.path.exists(REPO_ROOT):
        os.system("git clone https://github.com/abdul-rehman-24/breast-histopathology-patch-classification.git " + REPO_ROOT)
    else:
        os.system("cd " + REPO_ROOT + " && git pull")

    sys.path.insert(0, REPO_ROOT + "/src")

    os.system("pip install -q kagglehub")
    import kagglehub
    data_path = kagglehub.dataset_download("ambarish/breakhis")

    import pandas as pd
    train_df = pd.read_csv(REPO_ROOT + "/splits/train.csv")
    val_df = pd.read_csv(REPO_ROOT + "/splits/val.csv")
    test_df = pd.read_csv(REPO_ROOT + "/splits/test.csv")

    data_root_fix = None
    sample_path = train_df.iloc[0]["filepath"]
    if not os.path.exists(sample_path):
        old_root = sample_path.split("BreaKHis_v1")[0].rstrip("/")
        data_root_fix = (old_root, data_path)

    print("REPO_ROOT:", REPO_ROOT)
    print("DRIVE_ROOT:", DRIVE_ROOT)
    print("Dataset path:", data_path)
    print("Splits -> train=" + str(len(train_df)) + " val=" + str(len(val_df)) + " test=" + str(len(test_df)))

    result = {}
    result["repo_root"] = REPO_ROOT
    result["drive_root"] = DRIVE_ROOT
    result["data_path"] = data_path
    result["data_root_fix"] = data_root_fix
    result["train_df"] = train_df
    result["val_df"] = val_df
    result["test_df"] = test_df
    return result
