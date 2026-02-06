"""Check if parameter initialization is the issue."""

import sys

import torch
import torch.nn as nn

sys.path.insert(0, ".")

print("Checking parameter types...")

# Method 1: What net.py does
key1 = nn.Parameter(torch.randn(3, 15, 5)) * (3 * 15) ** -0.5
print("Method 1 (net.py style):")
print(f"  Type: {type(key1)}")
print(f"  Is Parameter: {isinstance(key1, nn.Parameter)}")
print(f"  requires_grad: {key1.requires_grad}")
print(f"  grad_fn: {key1.grad_fn}")

# Method 2: Correct way
key2 = nn.Parameter(torch.randn(3, 15, 5) * (3 * 15) ** -0.5)
print("\nMethod 2 (correct way):")
print(f"  Type: {type(key2)}")
print(f"  Is Parameter: {isinstance(key2, nn.Parameter)}")
print(f"  requires_grad: {key2.requires_grad}")
print(f"  grad_fn: {key2.grad_fn}")
