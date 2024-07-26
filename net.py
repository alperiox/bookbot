from abc import ABC, abstractmethod

import torch
from torch.nn import functional as F


def debug(func):
    """a decorator to save the object's __call__ outputs to the `out` attribute."""

    def wrapper(obj, *args, **kwargs):
        out = func(obj, *args, **kwargs)
        if hasattr(obj, "log_outputs"):
            if obj.log_outputs:
                obj.out = out
        else:
            obj.out = None
        return out

    return wrapper


# a simple metaclass that'd let us collect the defined layers and their parameters
class Meta(type):
    # now all the new subclasses will have their `__call__` methods wrapped in `debug` decorator.
    def __new__(cls, name, bases, attrs):
        if "__call__" in attrs:
            attrs["__call__"] = debug(attrs["__call__"])

        return super().__new__(cls, name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj._collect_layers()
        return obj


class CombinedMeta(Meta, ABC):
    pass


class BaseLayer(metaclass=CombinedMeta):

    @abstractmethod
    def __init__(self, log_outputs=False):
        self.out = None
        self.log_outputs = log_outputs

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def parameters(self):
        pass

    def _collect_layers(self):
        layers = []
        for attr_name, attr in vars(
            self
        ).items():  # loop through every attribute defined in the object
            # all my layers will be defined as BaseLayer, so maybe I can use that as a distinctive condition
            if isinstance(attr, BaseLayer):
                layers.append(attr)
        self._layers = layers
        return

    def to(self, device):
        # this might make me reconsider how to store the layer weights.
        pass

    def train(self):
        for layer in self._layers:
            layer.training = True

    def eval(self):
        for layer in self._layers:
            layer.training = False


class Head(BaseLayer):
    """one head of self-attention"""

    def __init__(self, n_in, n_head, context_length):
        super().__init__()
        # the head size that'll be used to map the
        # tokens to n_head-dimensional space, without additional bias.
        self.head_size = n_head
        # define the key, query and value transformations
        # these will be used to set up the affinity calculation between tokens.
        self.key = Linear(n_in, n_head, bias=False)  # (n_in, n_head)
        self.query = Linear(n_in, n_head, bias=False)  # (n_in, n_head)
        # this will be used to represent the tokens in a higher dimensional space
        # which will let model to learn more concise token representations (my take)
        self.value = Linear(n_in, n_head, bias=False)  # (n_in, n_head)
        # lower triangle matrix to just aggregate the previous context
        # as we try to predict the next token.
        self.tril = torch.tril(torch.ones(context_length, context_length))

    def __call__(self, x):
        B, T, C = x.shape
        # every token will have a key, query and value vector
        # we will then calculate the dot products of all the keys and queries
        # the higher value will mean that there's a higher affinity between those
        # two tokens.
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)
        # experimentally, one can check the dot products will
        # have the same variance around the value of head_size
        # thus scaling it down will help us preserve output variance
        wei = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)  # (B, T, T)
        # we might want to work with shorter sequences rather than defined context length, hence `self.tril[:T, :T]`.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # get the normalized affinities
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        # use the value vector and aggregate the results
        self.out = wei @ v  # (B, T, hs)
        return self.out

    def parameters(self):
        return {
            "key": self.key.parameters(),
            "query": self.query.parameters(),
            "value": self.value.parameters(),
        }


class MultiHeadAttention(BaseLayer):
    """multi-head self-attention that'll be used in GPT implementation"""

    def __init__(self, num_head, n_in, head_size, context_length):
        super().__init__()
        self.head_size = head_size
        self.num_head = num_head
        # set up the multiple heads
        self.heads = [Head(n_in, head_size, context_length) for _ in range(num_head)]
        # add the projection transformation to connect the outputs to the
        # residual pathway again
        self.proj = Linear(n_in, n_in)

    def __call__(self, x):
        # calculate the outputs of the heads
        out = [h(x) for h in self.heads]  # list of (B, T, head_size)
        # just concatenate them all
        self.out = torch.concat(out, -1)  # (B, T, head_size * num_heads)
        # apply the projection transformation
        self.out = self.proj(self.out)
        return self.out

    def parameters(self):
        params = {}
        for ix, h in enumerate(self.heads):
            params[f"head{ix}"] = h.parameters()
        params["proj"] = self.proj.parameters()
        return params


class FeedForwardBlock(BaseLayer):
    def __init__(self, n_hidden):
        super().__init__()
        # a simple feed-forward network at the end of the
        # decoder transformer architecture
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

    def parameters(self):
        return self.net.parameters()


class ReLU(BaseLayer):
    def __call__(self, x):
        return (x > 0) * x

    def parameters(self):
        return []


class DecoderTransformerBlock(BaseLayer):
    def __init__(self, num_heads, n_hidden, context_length):
        super().__init__()
        # using multiple heads will let us to provide more
        # communication channels between the tokens
        # but there's a caveat, the implementation downscales
        # the attention layers' dimensions, resulting in
        # more condensed communication channels.
        self.head_size = n_hidden // num_heads
        self.self_attn = MultiHeadAttention(
            num_heads, n_hidden, self.head_size, context_length
        )
        self.ffwd_net = FeedForwardBlock(n_hidden)
        self.ln1 = LayerNorm(n_hidden)
        self.ln2 = LayerNorm(n_hidden)

    def __call__(self, x):
        # add the residual connections and normalize given inputs
        # just as in the paper.
        x = x + self.self_attn(self.ln1(x))
        x = x + self.ffwd_net(self.ln2(x))
        self.out = x

        return self.out

    def parameters(self):
        return {
            "layernorm1": self.ln1.parameters(),
            "self_attn": self.self_attn.parameters(),
            "layernorm2": self.ln2.parameters(),
            "ffwd_net": self.ffwd_net.parameters(),
        }


class Linear(BaseLayer):
    def __init__(self, n_in, n_out, bias=True):
        super().__init__()
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
        params = {"weight": self.weight}
        if self.has_bias:
            params["bias"] = self.bias
        return params


class Tanh(BaseLayer):
    def __call__(self, x):
        self.x = x
        self.out = F.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding(BaseLayer):
    def __init__(self, n_vocab, n_embed):
        super().__init__()
        self.weight = torch.randn(n_vocab, n_embed)

    def __call__(self, x):
        self.x = x
        self.out = self.weight[x]
        return self.out

    def parameters(self):
        return {"weight": self.weight}


class LayerNorm(BaseLayer):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
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
        return {"gamma": self.gamma, "beta": self.beta}


class BatchNorm1d(BaseLayer):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
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
        return {"gamma": self.gamma, "beta": self.beta}


class LinearBlock(BaseLayer):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.linear = Linear(n_in, n_out)
        self.bn = BatchNorm1d(n_out)
        self.tanh = Tanh()

    def __call__(self, x):
        self.x = x
        self.out = self.tanh(self.bn(self.linear(x)))

        return self.out

    def parameters(self):
        return {"linear": self.linear.parameters(), "batchnorm": self.bn.parameters()}


class Flatten(BaseLayer):
    def __call__(self, x: torch.tensor) -> torch.tensor:
        self.out = x.view(x.size(0), -1)
        return self.out

    def parameters(self):
        return []


class FlattenConsecutive(BaseLayer):
    def __init__(self, n):
        super().__init__()
        self.n = n  # sum n consecutive elements

    def __call__(self, x):
        B, T, C = x.shape
        self.out = x.view(B, T // self.n, C * self.n)
        if self.out.shape[1] == 1:
            self.out = self.out.squeeze(1)
        return self.out

    def parameters(self):
        return []


class Sequential(BaseLayer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

        for i, l in enumerate(self.layers):
            self.__dict__[f"layer_{i}"] = l

    def __call__(self, x):
        self.out = x
        for layer in self.layers:
            self.out = layer(self.out)
            # print(f"{layer.__class__(BaseLayer).__name__:20s}:", self.out.shape)

        return self.out

    def parameters(self):
        params = {}
        layernames = [l.__class__.__name__.lower() for l in self.layers]
        counts = {}
        for name in layernames:
            counts[name] = counts.get(name, 0) + 1

        for name, l in zip(layernames, self.layers):
            ix = counts[name] - 1
            if ix == 0:
                ix = ""

            layername = l.__class__.__name__.lower()
            params[f"{layername}{ix}"] = l.parameters()

        return params


class HierarchicalMLP(BaseLayer):
    def __init__(
        self, vocab_size, n_consecutive, n_embed, n_hidden, block_size, n_layers=4
    ):
        assert (
            2**n_layers == block_size
        ), "`2^n_layers` must be equal to `block_size` because of `FlattenConsecutive`!"
        super().__init__()
        self.vocab_size = vocab_size
        self.n_consecutive = n_consecutive
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.block_size = block_size
        self.n_layers = n_layers

        self.special_tokens = {}
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

    def __call__(self, x, y=None):
        self.x = x
        self.out = self.model(self.x)

        if y is None:
            loss = None
        else:
            loss = F.cross_entropy(self.out, y)

        return self.out, loss

    def add_special_token(self, key, val):
        self.special_tokens[key] = val

    def eval(self):
        for layer in self.layers:
            if isinstance(layer, BatchNorm1d):
                layer.training = False

    def train(self):
        for layer in self.layers:
            if isinstance(layer, BatchNorm1d):
                layer.training = True

    def parameters(self):
        return self.model.parameters()

    def generate(self, idx, max_new_tokens):
        if idx.shape[0] != 1:
            raise NotImplementedError(
                "batched generation is not supported at the moment."
            )
        self.eval()
        for _ in range(max_new_tokens):
            input_tensor = idx[:, -self.block_size :]
            out, loss = self(input_tensor)
            probs = F.softmax(out, dim=-1)
            # sample the next character
            next_ix = torch.multinomial(probs, 1)  # (1, 1)

            # if next_ix[0] == self.special_tokens.get("EOS_TOKEN", None):
            #     break

            idx = torch.concat([idx, next_ix], -1)  # (B, block_size+1)
        self.train()
        return idx


class MLP(BaseLayer):
    def __init__(self, vocab_size, block_size, n_embed, n_hidden, n_layers=4):
        super().__init__()
        self.layers = []
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.special_tokens = {}

        self.embedding = Embedding(vocab_size, n_embed)

        self.layers.append(Linear(n_embed * block_size, n_hidden))

        self.layers += [LinearBlock(n_hidden, n_hidden) for _ in range(n_layers - 2)]

        self.layers.append(Linear(n_hidden, vocab_size))

        with torch.no_grad():
            self.layers[-1].weight *= 0.1

        self.net = Sequential(self.layers)

    def __call__(self, x, y=None):
        self.x = x  # (B, T)

        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.net(x)

        if y is None:
            loss = None
        else:
            loss = F.cross_entropy(x, y)

        self.out = x
        return self.out, loss

    def add_special_token(self, key, val):
        self.special_tokens[key] = val

    def eval(self):
        for layer in self.layers:
            if isinstance(layer, BatchNorm1d):
                layer.training = False
            elif isinstance(layer, LinearBlock):
                layer.bn.training = False

    def train(self):
        for layer in self.layers:
            if isinstance(layer, BatchNorm1d):
                layer.training = True
            elif isinstance(layer, LinearBlock):
                layer.bn.training = True

    def parameters(self):
        return {
            "embedding": self.embedding.parameters(),
            "sequential": self.net.parameters(),
        }

    def generate(self, idx, max_new_tokens):
        if idx.shape[0] != 1:
            raise NotImplementedError(
                "batched generation is not supported at the moment."
            )
        self.eval()
        for _ in range(max_new_tokens):
            input_tensor = idx[:, -self.block_size :]
            out, loss = self(input_tensor)
            probs = F.softmax(out, dim=-1)
            # sample the next character
            next_ix = torch.multinomial(probs, 1)  # (1, 1)

            # if next_ix[0] == self.special_tokens.get("EOS_TOKEN", None):
            #     break

            idx = torch.concat([idx, next_ix], -1)  # (B, block_size+1)
        self.train()
        return idx


class GPT(BaseLayer):
    def __init__(self, n_embd, vocab_size, num_heads, num_blocks, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.token_embeddings_table = Embedding(vocab_size, n_embd)
        self.pos_embeddings_table = Embedding(block_size, n_embd)
        self.blocks = Sequential(
            [
                DecoderTransformerBlock(num_heads, n_embd, context_length=block_size)
                for _ in range(num_blocks)
            ]
        )
        self.ln_f = LayerNorm(n_embd)
        self.ln_head = Linear(n_embd, vocab_size)

    def __call__(self, idx, targets=None):
        # inputs and targets are (B, T) shaped
        B, T = idx.shape

        tok_emb = self.token_embeddings_table(idx)  # (B, T, n_embd)
        pos_emb = self.pos_embeddings_table(torch.arange(T))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        x = self.blocks(x)  # (B, T, n_embd)
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.ln_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def parameters(self):
        return {
            "token_embeddings_table": self.token_embeddings_table.parameters(),
            "pos_embeddings_table": self.pos_embeddings_table.parameters(),
            "blocks": self.blocks.parameters(),
            "linear_final": self.ln_f.parameters(),
            "language_head": self.ln_head.parameters(),
        }

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array where T is the context length
        for _ in range(max_new_tokens):
            # crop the idx so we'll stay in the dimensions of positional embedding table
            cropped_idx = idx[:, -self.block_size :]
            logits, loss = self(cropped_idx)  # (B, T, C=vocab_size)
            # pick the last context window to sample the next token
            logits = logits[:, -1, :]  # (B, C)
            # apply softmax to map the logits to probs
            probs = F.softmax(logits, -1)
            # sample the next index
            next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.concat([idx, next_idx], dim=-1)  # (B, T+1)

        return idx
