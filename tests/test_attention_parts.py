"""
Test MultiHeadAttention step by step to find exact cause.
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from net import Linear


class MinimalAttention(nn.Module):
    """Stripped down attention to isolate the issue."""

    def __init__(self, num_heads, n_embd, head_size, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_size = head_size

        # Test 1: Are the Parameters themselves the issue?
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


class AttentionNoProj(nn.Module):
    """Attention without the final Linear projection."""

    def __init__(self, num_heads, n_embd, head_size, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_size = head_size

        self.key = nn.Parameter(
            torch.randn(num_heads, n_embd, head_size) * (num_heads * n_embd) ** -0.5
        )
        self.query = nn.Parameter(
            torch.randn(num_heads, n_embd, head_size) * (num_heads * n_embd) ** -0.5
        )
        self.value = nn.Parameter(
            torch.randn(num_heads, n_embd, head_size) * (num_heads * n_embd) ** -0.5
        )

        # NO proj layer!
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

        # NO proj!
        return out


class AttentionWithStdLinear(nn.Module):
    """Attention with PyTorch's standard nn.Linear instead of custom."""

    def __init__(self, num_heads, n_embd, head_size, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_size = head_size

        self.key = nn.Parameter(
            torch.randn(num_heads, n_embd, head_size) * (num_heads * n_embd) ** -0.5
        )
        self.query = nn.Parameter(
            torch.randn(num_heads, n_embd, head_size) * (num_heads * n_embd) ** -0.5
        )
        self.value = nn.Parameter(
            torch.randn(num_heads, n_embd, head_size) * (num_heads * n_embd) ** -0.5
        )

        # Use PyTorch's nn.Linear instead of custom
        self.proj = nn.Linear(n_embd, n_embd)
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


def test_layer(name, layer, num_passes=3):
    """Test a layer with multiple backward passes."""
    print(f"\nTesting {name}...")
    layer.train()

    for i in range(num_passes):
        try:
            x = torch.randn(4, 16, 15, requires_grad=True)
            out = layer(x)
            loss = out.sum()
            loss.backward()
            for p in layer.parameters():
                if p.grad is not None:
                    p.grad = None
        except RuntimeError as e:
            print(f"  FAIL on pass {i+1}: {str(e)[:80]}...")
            return False

    print(f"  PASS ({num_passes} passes)")
    return True


# Run tests
print("=" * 60)
print("ISOLATING MULTIHEADATTENTION ISSUE")
print("=" * 60)

# Test 1: Full attention with custom Linear (should FAIL)
attn1 = MinimalAttention(num_heads=3, n_embd=15, head_size=5, block_size=16)
test_layer("MinimalAttention (with custom Linear)", attn1)

# Test 2: Attention WITHOUT proj layer
attn2 = AttentionNoProj(num_heads=3, n_embd=15, head_size=5, block_size=16)
test_layer("AttentionNoProj (no Linear at all)", attn2)

# Test 3: Attention with PyTorch's nn.Linear
attn3 = AttentionWithStdLinear(num_heads=3, n_embd=15, head_size=5, block_size=16)
test_layer("AttentionWithStdLinear (PyTorch nn.Linear)", attn3)

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
