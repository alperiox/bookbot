import torch
from torch.nn import functional as F


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
        return self.weight[x]

    def parameters(self):
        return [self.weight]


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


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        self.out = x
        for layer in self.layers:
            self.out = layer(self.out)

        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class HierarchicalModel:
    def __init__(self, vocab_size, n_consecutive, n_embed, n_hidden, n_layers=3):
        self.layers = [
            Embedding(vocab_size, n_embed),
        ]

        for _ in n_layers:
            layers = [
                FlattenConsecutive(n_consecutive),
                Linear(n_embed * n_consecutive, n_hidden, bias=False),
                BatchNorm1d(n_hidden),
                Tanh(),
            ]
            self.layers += layers

        self.layers.append(Linear(n_hidden, vocab_size))

        with torch.no_grad():
            self.layers[-1].weight *= 0.1  # make the last layer less confident

        self.model = Sequential(self.layers)

    def __call__(self, x):
        self.x = x
        self.out = self.model(self.x)
        return self.out

    def eval(self):
        for layer in self.layers:
            if isinstance(layer, BatchNorm1d):
                layer.training = False

    def parameters(self):
        return self.model.parameters()


class Model:
    def __init__(self, vocab_size, block_size, n_embed, n_hidden, n_layers=4):
        self.layers = []
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.embedding = Embedding(vocab_size, n_embed)

        self.layers.append(Linear(n_embed * block_size, n_hidden))

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
        return [self.embedding.weight] + [
            param for layer in self.layers for param in layer.parameters()
        ]
