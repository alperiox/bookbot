import torch
from torch.nn import functional as F


class Linear:
    def __init__(self, n_in, n_out, bias=True):
        self.weight = torch.randn(n_in, n_out) / n_in**.5

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
        return self.weight[x]

    def parameters(self):
        return [self.weight]


class BatchNorm1d:
    def __init__(self, n_in, eps=1e-5, momentum=0.1):
        self.training = True
        self.eps = eps
        self.momentum = momentum

        self.gamma = torch.ones(n_in)
        self.bias = torch.zeros(n_in)

        self.mean = torch.zeros(n_in)
        self.var = torch.ones(n_in)

    def __call__(self, x):
        self.x = x  # (n_samples, n_in)

        if self.training:
            mean = x.mean(dim=0)  #
            var = x.var(dim=0)

            self.mean = self.momentum * self.mean + (1 - self.momentum) * mean
            self.var = self.momentum * self.var + (1 - self.momentum) * var
        else:
            mean = self.mean
            var = self.var

        self.x_hat = (x - mean) / torch.sqrt(var + self.eps)
        self.out = self.gamma * self.x_hat + self.bias

        return self.out

    def parameters(self):
        return [self.gamma, self.bias]


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


class Model:
    def __init__(self, vocab_size, block_size, n_embed, n_hidden, n_layers=4):
        self.layers = []
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.embedding = Embedding(vocab_size, n_embed)

        self.layers.append(Linear(n_embed*block_size, n_hidden))

        self.layers += [LinearBlock(n_hidden, n_hidden) for _ in range(n_layers - 2)]

        self.layers.append(Linear(n_hidden, vocab_size))

    def __call__(self, x):
        self.x = x

        x = self.embedding(x)
        x = x.view(x.size(0), -1)

        for layer in self.layers:
            x = layer(x)

        self.out = x
        return self.out
    
    def eval(self):
        for layer in self.layers:
            if isinstance(layer, BatchNorm1d):
                layer.training = False
            elif isinstance(layer, LinearBlock):
                layer.bn.training = False

    def parameters(self):
        return [self.embedding.weight] + [param for layer in self.layers for param in layer.parameters()]
