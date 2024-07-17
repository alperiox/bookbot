import torch
from torch.nn import functional as F

from models import Sequential


class Head:
    """one head of self-attention"""

    def __init__(self, n_in, n_head, context_length):
        self.head_size = n_head
        self.key = Linear(n_in, n_head, bias=False)  # (n_in, n_head)
        self.query = Linear(n_in, n_head, bias=False)  # (n_in, n_head)
        self.value = Linear(n_in, n_head, bias=False)  # (n_in, n_head)

        self.tril = torch.tril(torch.ones(context_length, context_length))

    def __call__(self, x):
        B, T, C = x.shape

        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        wei = q @ k.transpose(-2, -1)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # why :T, :T (?)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)

        self.out = wei @ v  # (B, T, hs)
        return self.out


class MultiHeadAttention:
    """multi-head self-attention that'll be used in GPT implementation"""

    def __init__(self, num_head, n_in, head_size, context_length):
        self.head_size = head_size
        self.num_head = num_head

        self.heads = [Head(n_in, head_size, context_length)]
        self.proj = Linear(n_in, head_size)

    def __call__(self, x):
        out = [h(x) for h in self.heads]  # list of (B, T, head_size)
        self.out = torch.concat(out, -1)  # (B, T, head_size * num_heads)
        self.out = self.proj(self.out)
        return self.out


class FeedForwardBlock:
    def __init__(self, n_hidden):
        self.net = Sequential(
            [
                # provide a larger dimensional space for network to process the connection between tokens
                Linear(n_hidden, n_hidden * 4),
                ReLU(),
                Linear(n_hidden * 4, n_hidden),
            ]
        )

    def __call__(self, x):
        self.out = self.net(x)
        return self.out


class ReLU:
    def __call__(self, x):
        return (x > 0) * x


class DecoderTransformerBlock:
    def __init__(self, n_head, n_hidden, context_length):
        self.head_size = n_hidden // n_head
        self.self_attn = MultiHeadAttention(
            n_head, n_hidden, self.head_size, context_length
        )
        self.ffwd_net = FeedForwardBlock(n_hidden)
        self.ln1 = LayerNorm(n_hidden)
        self.ln2 = LayerNorm(n_hidden)

    def __call__(self, x):
        # plus the residual connections
        x = x + self.self_attn(self.ln1(x))
        x = x + self.ffwd_net(self.ln2(x))
        self.out = x

        return self.out


class Linear:
    def __init__(self, n_in, n_out, bias=True):
        self.weight = torch.randn(n_in, n_out) / n_in**0.5

        self.has_bias = bias
        if self.has_bias:
            self.bias = torch.zeros(n_out)

    def __call__(self, x):
        self.x = x
        if self.has_bias:
            self.out = x @ self.weight + self.bias
        else:
            self.out = x @ self.weight
        return self.out

    def parameters(self):
        if self.has_bias:
            return [self.weight, self.bias]
        else:
            return [self.weight]


class Tanh:
    def __call__(self, x):
        self.x = x
        self.out = F.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, n_vocab, n_embed):
        self.weight = torch.randn(n_vocab, n_embed)

    def __call__(self, x):
        self.x = x
        self.out = self.weight[x]
        return self.out

    def parameters(self):
        return [self.weight]


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        # parameters to be trained with backprop
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)  # layers mean
        xvar = x.var(1, keepdim=True)  # layers var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters to be trained with backprop
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers, trained with a running `momentum update`
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(dim, keepdim=True)  # batch mean
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    self.running_mean * (1 - self.momentum) + xmean * self.momentum
                )
                self.running_var = (
                    self.running_var * (1 - self.momentum) + xvar * self.momentum
                )
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class LinearBlock:
    def __init__(self, n_in, n_out):
        self.linear = Linear(n_in, n_out)
        self.bn = BatchNorm1d(n_out)
        self.tanh = Tanh()

    def __call__(self, x):
        self.x = x
        self.out = self.tanh(self.bn(self.linear(x)))

        return self.out

    def parameters(self):
        return self.linear.parameters() + self.bn.parameters() + self.tanh.parameters()


class Flatten:
    def __call__(self, x: torch.tensor) -> torch.tensor:
        self.out = x.view(x.size(0), -1)
        return self.out

    def parameters(self):
        return []


class FlattenConsecutive:
    def __init__(self, n):
        self.n = n  # sum n consecutive elements

    def __call__(self, x):
        B, T, C = x.shape
        self.out = x.view(B, T // self.n, C * self.n)
        if self.out.shape[1] == 1:
            self.out = self.out.squeeze(1)
        return self.out

    def parameters(self):
        return []
