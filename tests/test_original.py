"""Verify original MultiHeadAttention still fails."""

import sys

import torch

sys.path.insert(0, ".")
from net import MultiHeadAttention

print("Testing ORIGINAL MultiHeadAttention from net.py...")

layer = MultiHeadAttention(num_heads=3, n_embd=15, head_size=5, block_size=16)
layer.train()

for i in range(3):
    try:
        x = torch.randn(4, 16, 15, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        for p in layer.parameters():
            if p.grad is not None:
                p.grad = None
        print(f"  Pass {i+1}: OK")
    except RuntimeError as e:
        print(f"  Pass {i+1}: FAIL - {str(e)[:80]}...")
        break
