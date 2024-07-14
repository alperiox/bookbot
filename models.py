from layers import *


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
    def __init__(
        self, vocab_size, n_consecutive, n_embed, n_hidden, block_size, n_layers=4
    ):
        self.layers = [
            Embedding(vocab_size, n_embed),
            FlattenConsecutive(n_consecutive),
            Linear(n_embed * n_consecutive, n_hidden, bias=False),
            BatchNorm1d(n_hidden),
            Tanh(),
        ]

        for _ in range(n_layers - 1):
            layers = [
                FlattenConsecutive(n_consecutive),
                Linear(n_hidden * n_consecutive, n_hidden, bias=False),
                BatchNorm1d(n_hidden),
                Tanh(),
            ]
            self.layers += layers

        self.layers.append(Linear(n_hidden, vocab_size))

        with torch.no_grad():
            self.layers[-1].weight *= 0.1  # make the last layer less confident

        self.model = Sequential(self.layers)
        self.block_size = block_size

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
