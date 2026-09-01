"""Checkpoint save/load/resume helpers, used consistently across all training scripts.
Backward compatible: old checkpoints (without seed/config/git_commit) still load fine via .get().
"""
import os
import subprocess
import torch


def get_git_commit(repo_root=None):
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def load_latest_if_exists(path, model, optimizer=None, scheduler=None, device="cuda"):
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        return ckpt["epoch"] + 1, ckpt.get("best_val_metric", 0.0)
    return 0, 0.0


def save_latest(path, epoch, model, optimizer, scheduler, val_metric, best_val_metric,
                 seed=None, config=None, repo_root=None):
    torch.save({
        "epoch": epoch, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
        "val_metric": float(val_metric), "best_val_metric": float(max(best_val_metric, val_metric)),
        "seed": seed, "config": config, "git_commit": get_git_commit(repo_root),
    }, path)


def save_best(path, epoch, model, val_metric, seed=None, config=None, repo_root=None):
    torch.save({
        "epoch": epoch, "model_state": model.state_dict(), "val_metric": float(val_metric),
        "seed": seed, "config": config, "git_commit": get_git_commit(repo_root),
    }, path)


def load_best(path, model, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt
