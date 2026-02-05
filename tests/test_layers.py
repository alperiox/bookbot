"""
Minimal test to isolate which layer causes backward issues.
"""

import sys

import torch
import torch.nn as nn

sys.path.insert(0, ".")
from net import (
    Embedding,
    FeedForwardBlock,
    LayerNorm,
    Linear,
    MultiHeadAttention,
    ReLU,
    Tanh,
)


def test_layer(name, create_layer, create_input, num_passes=3):
    """Test a single layer with multiple backward passes."""
    print(f"\nTesting {name}...")

    layer = create_layer()
    layer.train()

    for i in range(num_passes):
        try:
            x = create_input()
            out = layer(x)
            loss = out.sum()
            loss.backward()
            # Zero grads for next pass
            for p in layer.parameters():
                if p.grad is not None:
                    p.grad = None
        except RuntimeError as e:
            print(f"  FAIL on pass {i+1}: {str(e)[:80]}...")
            return False

    print(f"  PASS ({num_passes} passes)")
    return True


# Test each layer type
results = {}

# Linear
results["Linear"] = test_layer(
    "Linear", lambda: Linear(64, 32), lambda: torch.randn(4, 64, requires_grad=True)
)

# Embedding
results["Embedding"] = test_layer(
    "Embedding", lambda: Embedding(100, 64), lambda: torch.randint(0, 100, (4, 16))
)

# LayerNorm
results["LayerNorm"] = test_layer(
    "LayerNorm",
    lambda: LayerNorm(64),
    lambda: torch.randn(4, 16, 64, requires_grad=True),
)

# ReLU
results["ReLU"] = test_layer(
    "ReLU", lambda: ReLU(), lambda: torch.randn(4, 64, requires_grad=True)
)

# Tanh
results["Tanh"] = test_layer(
    "Tanh", lambda: Tanh(), lambda: torch.randn(4, 64, requires_grad=True)
)

# MultiHeadAttention
results["MultiHeadAttention"] = test_layer(
    "MultiHeadAttention",
    lambda: MultiHeadAttention(num_heads=3, n_embd=15, head_size=5, block_size=16),
    lambda: torch.randn(4, 16, 15, requires_grad=True),
)

# FeedForwardBlock
results["FeedForwardBlock"] = test_layer(
    "FeedForwardBlock",
    lambda: FeedForwardBlock(64),
    lambda: torch.randn(4, 16, 64, requires_grad=True),
)

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
for name, passed in results.items():
    status = "PASS" if passed else "FAIL"
    print(f"  {name}: {status}")
