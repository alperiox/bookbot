"""Debug what's stored in Linear.self.x"""

import sys

import torch

sys.path.insert(0, ".")
from net import MultiHeadAttention

print("Testing with debug output...")

layer = MultiHeadAttention(num_heads=3, n_embd=15, head_size=5, block_size=16)
layer.train()

for i in range(3):
    print(f"\n--- Pass {i+1} ---")
    try:
        x = torch.randn(4, 16, 15, requires_grad=True)
        print(f"Input x id: {id(x)}")

        out = layer(x)

        # Check what's stored in self.proj.x
        if hasattr(layer.proj, "x"):
            print(f"layer.proj.x id: {id(layer.proj.x)}")
            print(f"layer.proj.x.grad_fn: {layer.proj.x.grad_fn}")

        loss = out.sum()
        print(f"Loss grad_fn: {loss.grad_fn}")

        loss.backward()

        # After backward, check if grad_fn still exists
        if hasattr(layer.proj, "x"):
            print(
                f"After backward - layer.proj.x.grad_fn: {getattr(layer.proj.x, 'grad_fn', 'N/A')}"
            )

        for p in layer.parameters():
            if p.grad is not None:
                p.grad = None

        print(f"  Pass {i+1}: OK")

    except RuntimeError as e:
        print(f"  Pass {i+1}: FAIL - {str(e)[:100]}...")

        # Debug: what's in self.x at failure?
        if hasattr(layer.proj, "x"):
            print(
                f"  At failure - layer.proj.x.grad_fn: {getattr(layer.proj.x, 'grad_fn', 'N/A')}"
            )
        break
