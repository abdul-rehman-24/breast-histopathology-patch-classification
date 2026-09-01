"""Deterministic seeding utilities. Investigated cause of Foundation-only run variance:
torch.manual_seed() alone was set, but random/numpy/cuda seeds, DataLoader worker seeding,
and cuDNN determinism flags were NOT set -- allowing run-to-run variance despite a fixed seed."""
import os
import random
import numpy as np
import torch


def set_full_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
        deterministic_algorithms_enabled = True
    except Exception as e:
        deterministic_algorithms_enabled = False
        print(f"WARNING: torch.use_deterministic_algorithms(True) failed: {e}")
        print("Falling back to cudnn.deterministic=True only (weaker guarantee).")

    return {"seed": seed, "deterministic_algorithms": deterministic_algorithms_enabled}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_seeded_generator(seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    return g
