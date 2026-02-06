"""Test the fix: move scaling inside nn.Parameter()"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from net import Linear


class FixedMultiHeadAttention(nn.Module):
    """Fixed version with correct parameter initialization."""

    def __init__(self, num_heads, n_embd, head_size, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_size = head_size

        # FIXED: Scaling INSIDE nn.Parameter
        self.key = nn.Parameter(
            torch.randn(num_heads, n_embd, head_size) * (num_heads * n_embd) ** -0.5
        )
        self.query = nn.Parameter(
            torch.randn(num_heads, n_embd, head_size) * (num_heads * n_embd) ** -0.5
        )
        self.value = nn.Parameter(
            torch.randn(num_heads, n_embd, head_size) * (num_heads * n_embd) ** -0.5
        )

        self.proj = Linear(n_embd, n_embd)
        self.register_buffer(
            "tril", torch.tril(torch.ones(num_heads, block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        x = x.unsqueeze(1)
        k = x @ self.key
        q = x @ self.query

        wei = q @ k.transpose(-2, -1)
        wei = wei.masked_fill(self.tril[:, :T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        v = x @ self.value
        out = wei @ v
        out = out.transpose(1, 2)
        out = out.reshape(out.size(0), out.size(1), self.n_embd)
        out = self.proj(out)
        return out


print("Testing FIXED MultiHeadAttention (scaling inside nn.Parameter)...")
layer = FixedMultiHeadAttention(num_heads=3, n_embd=15, head_size=5, block_size=16)

# Verify parameters are correct
print("\nParameter check:")
print(f"  self.key is nn.Parameter: {isinstance(layer.key, nn.Parameter)}")
print(f"  self.key.grad_fn: {layer.key.grad_fn}")

layer.train()
for i in range(5):  # Test 5 passes
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

print("\n✓ FIX VERIFIED!" if i == 4 else "\n✗ Fix didn't work")
