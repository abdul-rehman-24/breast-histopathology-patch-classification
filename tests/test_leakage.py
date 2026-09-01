"""Test that the actual saved split files satisfy patient-wise leakage safety."""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathology.data.splits import check_leakage

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_no_patient_overlap_across_splits():
    train_df = pd.read_csv(os.path.join(REPO_ROOT, "splits", "train.csv"))
    val_df = pd.read_csv(os.path.join(REPO_ROOT, "splits", "val.csv"))
    test_df = pd.read_csv(os.path.join(REPO_ROOT, "splits", "test.csv"))

    passed, report = check_leakage(train_df, val_df, test_df)

    assert passed, f"Leakage detected: {report}"
    assert report["train_val_overlap"] == 0
    assert report["train_test_overlap"] == 0
    assert report["val_test_overlap"] == 0
    assert report["duplicate_paths"] == 0
