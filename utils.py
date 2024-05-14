import os

import torch


def load_artifacts(*args):
    artifacts = {}
    for key in args:
        artifacts[key] = torch.load(f"artifacts/{key}.pt")
    return artifacts


def save_artifacts(**kwargs):
    os.makedirs("artifacts", exist_ok=True)
    for key, value in kwargs.items():
        torch.save(value, f"artifacts/{key}.pt")
