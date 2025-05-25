import os
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from torch.nn import functional as F

from utils import debug, flatten_dict


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
        attrs = vars(self)
        for attr_name in vars(self):
            attr = attrs[attr_name]
            if isinstance(attr, BaseLayer) or isinstance(attr, torch.Tensor):
                self.__setattr__(attr_name, attr.to(device))
            elif isinstance(attr, list):
                if all(isinstance(a, BaseLayer) for a in attr):
                    self.__setattr__(attr_name, [a.to(device) for a in attr])
        return self

    def train(self, log=False):
        for layer in self._layers:
            layer.training = True

    def eval(self):
        for layer in self._layers:
            layer.training = False

    def start_debug(self):
        self.log_outputs = True
        for layer in self._layers:
            layer.start_debug()
            if isinstance(layer, Sequential):
                for l in layer.layers:
                    l.start_debug()

    def stop_debug(self):
        self.log_outputs = False
        for layer in self._layers:
            layer.stop_debug()
            if isinstance(layer, Sequential):
                for l in layer.layers:
                    l.stop_debug()


class Head(BaseLayer):
    """one head of self-attention"""

    def __init__(self, n_in, n_head, context_length):
        # the head size that'll be used to map the
        # tokens to n_head-dimensional space, without additional bias.
        self.head_size = n_head
        # define the key, query and value transformations
        # these will be used to set up the affinity calculation between tokens.
        self.key = Linear(n_in, n_head, bias=False)  # (n_in, n_head)
        self.query = Linear(n_in, n_head, bias=False)  # (n_in, n_head)
        # this will be used to represent the tokens in a higher dimensional space
        # which will let self to learn more concise token representations (my take)
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
        out = wei @ v  # (B, T, hs)
        return out

    def parameters(self):
        return flatten_dict(
            {
                "key": self.key.parameters(),
                "query": self.query.parameters(),
                "value": self.value.parameters(),
            }
        )


class MultiHeadAttention(BaseLayer):
    """implements multi-headed masked self-attention using tensor operations"""

    def __init__(self, num_heads, n_embd, head_size, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size

        self.key = torch.randn(num_heads, n_embd, head_size)
        self.query = torch.randn(num_heads, n_embd, head_size)
        self.value = torch.randn(num_heads, n_embd, head_size)

        self.proj = Linear(n_embd, n_embd)

        # (B, T, n_embd) x (num_heads, n_embd, head_size) --> (B, num_heads, T, head_size)
        self.tril = torch.tril(torch.ones(num_heads, block_size, block_size))

    def __call__(self, x):
        """
        x: (B, T, n_embd) tensor

        returns: (B, T, n_embd) tensor


        """

        # Naming convention for the comments in the code:
        # bs: batch size
        # nh: number of heads
        # cl: context length
        # hs: head size
        # ne: n_embd

        B, T, C = x.shape
        x = x.unsqueeze(1)  # (batch_size, 1, context_length, n_embd)
        k = x @ self.key  # (batch_size, num_heads, context_length, head_size)
        q = x @ self.query  # (batch_size, num_heads, context_length, head_size)

        wei = q @ k.transpose(
            -2, -1
        )  # (bs, nh, cl, hs) x (bs, nh, hs, cl) -> (bs, nh, cl, cl)
        wei = wei.masked_fill(self.tril[:, :T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (bs, nh, cl, cl)

        if hasattr(self, 'log_outputs') and self.log_outputs:
            self.attn_weights_viz = wei.detach() # Save for visualization

        v = x @ self.value  # (bs, 1, cl, ne) x (nh, ne, hs) -> (bs, nh, cl, hs)
        out = wei @ v  # (bs, nh, cl, cl) x (bs, nh, cl, hs) -> (bs, nh, cl, hs)
        out = out.transpose(1, 2)  # (bs, cl, nh, hs)
        out = out.reshape(
            out.size(0), out.size(1), self.n_embd
        )  # (bs, cl, n_embd) = (B, T, C)

        out = self.proj(out)

        return out

    def parameters(self):
        params = {
            "key": self.key,
            "query": self.query,
            "value": self.value,
        }
        params["proj"] = self.proj.parameters()
        return flatten_dict(params)


class MultiHeadAttentionConcat(BaseLayer):
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
        out = torch.concat(out, -1)  # (B, T, head_size * num_heads)
        # apply the projection transformation
        out = self.proj(out)
        return out

    def parameters(self):
        params = {}
        for ix, h in enumerate(self.heads):
            params[f"head{ix}"] = h.parameters()
        params["proj"] = self.proj.parameters()
        return flatten_dict(params)


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
        out = self.net(x)
        return out

    def parameters(self):
        return self.net.parameters()


class ReLU(BaseLayer):
    def __call__(self, x):
        return (x > 0) * x

    def parameters(self):
        return {}


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
        out = x

        return out

    def parameters(self):
        return flatten_dict(
            {
                "layernorm1": self.ln1.parameters(),
                "self_attn": self.self_attn.parameters(),
                "layernorm2": self.ln2.parameters(),
                "ffwd_net": self.ffwd_net.parameters(),
            }
        )


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
            out = x @ self.weight + self.bias
        else:
            out = x @ self.weight
        return out

    def parameters(self):
        params = {"weight": self.weight}
        if self.has_bias:
            params["bias"] = self.bias
        return flatten_dict(params)


class Tanh(BaseLayer):
    def __call__(self, x):
        self.x = x
        out = F.tanh(x)
        return out

    def parameters(self):
        return {}


class Embedding(BaseLayer):
    def __init__(self, n_vocab, n_embed):
        super().__init__()
        self.weight = torch.randn(n_vocab, n_embed)

    def __call__(self, x):
        self.x = x
        out = self.weight[x]
        return out

    def parameters(self):
        return flatten_dict({"weight": self.weight})


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
        out = self.gamma * xhat + self.beta
        return out

    def parameters(self):
        return flatten_dict({"gamma": self.gamma, "beta": self.beta})


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
        out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    self.running_mean * (1 - self.momentum) + xmean * self.momentum
                )
                self.running_var = (
                    self.running_var * (1 - self.momentum) + xvar * self.momentum
                )
        return out

    def parameters(self):
        return flatten_dict({"gamma": self.gamma, "beta": self.beta})


class LinearBlock(BaseLayer):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.linear = Linear(n_in, n_out)
        self.bn = BatchNorm1d(n_out)
        self.tanh = Tanh()

    def __call__(self, x):
        self.x = x
        out = self.tanh(self.bn(self.linear(x)))

        return out

    def parameters(self):
        return flatten_dict(
            {"linear": self.linear.parameters(), "batchnorm": self.bn.parameters()}
        )


class Flatten(BaseLayer):
    def __call__(self, x: torch.tensor) -> torch.tensor:
        out = x.view(x.size(0), -1)
        return out

    def parameters(self):
        return {}


class FlattenConsecutive(BaseLayer):
    def __init__(self, n):
        super().__init__()
        self.n = n  # sum n consecutive elements

    def __call__(self, x):
        B, T, C = x.shape
        out = x.view(B, T // self.n, C * self.n)
        if out.shape[1] == 1:
            out = out.squeeze(1)
        return out

    def parameters(self):
        return {}


class Sequential(BaseLayer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
            # print(f"{layer.__class__(BaseLayer).__name__:20s}:", out.shape)

        return out

    def parameters(self):
        params = {}
        layernames = [l.__class__.__name__.lower() for l in self.layers]
        counts = {}
        for name in layernames:
            counts[name] = counts.get(name, 0) + 1

        for name, l in zip(layernames, self.layers):
            ix = counts[name] - 1
            counts[name] -= 1

            layername = l.__class__.__name__.lower()
            params[f"{layername}{ix}"] = l.parameters()

        return flatten_dict(params)


class HierarchicalMLP(BaseLayer):
    def __init__(
        self, vocab_size, n_consecutive, n_embed, n_hidden, block_size, n_layers=4
    ):
        assert (
            n_consecutive**n_layers == block_size
        ), "`n_consecutive^n_layers` must be equal to `block_size` because of `FlattenConsecutive`!"
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
        out = self.model(self.x)

        if y is None:
            loss = None
        else:
            loss = F.cross_entropy(out, y)

        return out, loss

    def add_special_token(self, key, val):
        self.special_tokens[key] = val

    def eval(self):
        for layer in self.layers:
            layer.training = False

    def train(self):
        for layer in self.layers:
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

    def get_layer_output_histograms(
        self,
        sample_input: torch.Tensor = None,
        save_affix: str = "pretraining",
        save_path: str = "artifacts",
    ):
        if sample_input is None:
            sample_input = torch.randint(
                0, self.vocab_size - 1, (1, self.block_size), dtype=torch.long
            )
        self.eval()
        self.start_debug()
        self(sample_input)
        self.stop_debug()
        outs = []
        titles = []

        for layer in self.model.layers:
            name = layer.__class__.__name__
            if name != "FlattenConsecutive":
                out = layer.out
                outs.append(out)
                titles.append(name)

        # generate Nx3 subplots
        fig, axs = plt.subplots(len(outs) // 3 + 1, 3, figsize=(15, 5))

        for title, out, ax in zip(titles, outs, axs.flatten()):
            ax.set_title(title)
            ax.hist(out.view(-1).tolist(), 50)
        # plt.savefig(f"{save_path}/layer_output_histograms_{save_affix}.png") # Commented out as this is now handled by utils.plot_layer_output_histograms

        self.train()
        return

    def plot_emb_weights(
        self,
        plot_text: bool = False,
        save_affix: str = "pretraining",
        save_path: str = "artifacts",
        ndims=1,
        tokenizer=None,
    ):
        assert ndims == 1 or ndims == 2, "ndims must be 1 or 2 for plotting the embeddings"
        
        embedding_tensor = self.model.layers[0].weight.detach().cpu().numpy()
        if embedding_tensor.shape[1] > ndims:
            pca = PCA(n_components=ndims)
            weights = pca.fit_transform(embedding_tensor)
        else:
            # If original embedding dim is <= ndims, use original weights (or up to original dim)
            weights = embedding_tensor[:, :ndims]

        # Prepare token labels
        token_labels = {}
        if tokenizer is not None and hasattr(tokenizer, 'itos'):
            # Ensure tokenizer.itos provides a list/dict of characters/strings
            # If itos is a dictionary {index: token_char}
            token_labels = {i: tokenizer.itos.get(i, str(i)) for i in range(self.vocab_size)}
        else:
            token_labels = {i: str(i) for i in range(self.vocab_size)}
        
        # Ensure weights match vocab_size for consistent indexing with token_labels
        if weights.shape[0] != self.vocab_size:
            print(f"Warning: PCA transformed weights count ({weights.shape[0]}) does not match vocab_size ({self.vocab_size}). Truncating/padding may occur in labels.")
            # Adjust token_labels to match weights length if necessary, though ideally they should match.
            # This case should be rare if vocab_size is correct for the embedding layer.

        path = Path(save_path)
        os.makedirs(path, exist_ok=True)
        filename = path / f"HierarchicalMLP_embedding_weights_{save_affix}_pca{ndims}d.png"

        if ndims == 1:
            plt.figure(figsize=(max(10, int(self.vocab_size * 0.5)), 6))
            plt.bar(range(weights.shape[0]), weights[:, 0] if weights.ndim > 1 else weights)
            
            # Prepare x-tick labels carefully
            tick_positions = range(weights.shape[0])
            tick_labels_list = [token_labels.get(i, str(i)) for i in tick_positions]
            
            plt.xticks(tick_positions, tick_labels_list, rotation=90, fontsize=8)
            plt.ylabel(f"1st Principal Component Value")
            plt.title(f"HierarchicalMLP Embedding Weights (1D PCA)\n{save_affix}")
            
            if plot_text:
                for i in range(weights.shape[0]):
                    val = weights[i, 0] if weights.ndim > 1 else weights[i]
                    plt.text(i, val, tick_labels_list[i], ha='center', va='bottom' if val >= 0 else 'top', fontsize=7)
        
        else: # ndims == 2
            plt.figure(figsize=(10, 8))
            plt.scatter(weights[:, 0], weights[:, 1], alpha=0.7)
            plt.xlabel("1st Principal Component")
            plt.ylabel("2nd Principal Component")
            plt.title(f"HierarchicalMLP Embedding Weights (2D PCA)\n{save_affix}")
            
            if plot_text:
                for i in range(weights.shape[0]):
                    # Ensure we use the correct label for the i-th point
                    label = token_labels.get(i, str(i))
                    plt.text(weights[i, 0], weights[i, 1], label, fontsize=8)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved HierarchicalMLP embedding weights plot to {filename}")
        return


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

        out = x
        return out, loss

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
        return flatten_dict(
            {
                "embedding": self.embedding.parameters(),
                "sequential": self.net.parameters(),
            }
        )

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

    # Removed get_layer_output_histograms, as it's now consolidated in utils.py

    def plot_emb_weights(
        self,
        plot_text: bool = False,
        save_affix: str = "pretraining",
        save_path: str = "artifacts",
        ndims=1,
        tokenizer=None,
    ):
        assert ndims == 1 or ndims == 2, "ndims must be 1 or 2 for plotting the embeddings"

        embedding_tensor = self.embedding.weight.detach().cpu().numpy()
        if embedding_tensor.shape[1] > ndims:
            pca = PCA(n_components=ndims)
            weights = pca.fit_transform(embedding_tensor)
        else:
            weights = embedding_tensor[:, :ndims]

        token_labels = {}
        if tokenizer is not None and hasattr(tokenizer, 'itos'):
            token_labels = {i: tokenizer.itos.get(i, str(i)) for i in range(self.vocab_size)}
        else:
            token_labels = {i: str(i) for i in range(self.vocab_size)}

        if weights.shape[0] != self.vocab_size:
            print(f"Warning: PCA transformed weights count ({weights.shape[0]}) does not match vocab_size ({self.vocab_size}).")

        path = Path(save_path)
        os.makedirs(path, exist_ok=True)
        filename = path / f"MLP_embedding_weights_{save_affix}_pca{ndims}d.png"

        if ndims == 1:
            plt.figure(figsize=(max(10, int(self.vocab_size * 0.5)), 6))
            plt.bar(range(weights.shape[0]), weights[:, 0] if weights.ndim > 1 else weights)
            
            tick_positions = range(weights.shape[0])
            tick_labels_list = [token_labels.get(i, str(i)) for i in tick_positions]
            
            plt.xticks(tick_positions, tick_labels_list, rotation=90, fontsize=8)
            plt.ylabel("1st Principal Component Value")
            plt.title(f"MLP Embedding Weights (1D PCA)\n{save_affix}")

            if plot_text:
                for i in range(weights.shape[0]):
                    val = weights[i, 0] if weights.ndim > 1 else weights[i]
                    plt.text(i, val, tick_labels_list[i], ha='center', va='bottom' if val >= 0 else 'top', fontsize=7)
        
        else: # ndims == 2
            plt.figure(figsize=(10, 8))
            plt.scatter(weights[:, 0], weights[:, 1], alpha=0.7)
            plt.xlabel("1st Principal Component")
            plt.ylabel("2nd Principal Component")
            plt.title(f"MLP Embedding Weights (2D PCA)\n{save_affix}")

            if plot_text:
                for i in range(weights.shape[0]):
                    label = token_labels.get(i, str(i))
                    plt.text(weights[i, 0], weights[i, 1], label, fontsize=8)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved MLP embedding weights plot to {filename}")
        return


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
        return flatten_dict(
            {
                "token_embeddings_table": self.token_embeddings_table.parameters(),
                "pos_embeddings_table": self.pos_embeddings_table.parameters(),
                "blocks": self.blocks.parameters(),
                "linear_final": self.ln_f.parameters(),
                "language_head": self.ln_head.parameters(),
            }
        )

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

    def get_layer_output_histograms(
        self,
        sample_input: torch.Tensor = None,
        save_affix: str = "pretraining",
        save_path: str = "artifacts",
    ):
        """
        Generate histograms of layer outputs for a given model.

        This function creates histograms for various layer outputs of the model,
        including embeddings, block outputs, and logits.

        Args:
        sample_input (torch.Tensor, optional): Input tensor to use. If None, random input is generated.

        Returns:
        None. Displays the histograms using matplotlib.
        """
        assert os.path.exists(save_path), "The save path does not exist."
        self.eval()
        if sample_input is None:
            sample_input = torch.randint(
                0, self.vocab_size - 1, (1, self.block_size), dtype=torch.long
            )
        layer_outputs = []
        # get the embeddings
        emb1 = self.token_embeddings_table(sample_input)
        # positional embeddings
        emb2 = self.pos_embeddings_table(torch.arange(self.block_size))
        # combine the embeddings
        emb = emb1 + emb2
        # pass the embeddings through the model
        block_outputs = self.blocks(emb)
        lnf_out = self.ln_f(block_outputs)
        logits = self.ln_head(lnf_out)

        # store the layer outputs and their titles
        layer_outputs = [emb1, emb2, emb, block_outputs, lnf_out, logits]
        plot_titles = [
            "Token embd",
            "Pos embd",
            "Combined emb",
            "Block outputs",
            "Layer norm",
            "Language head",
        ]

        # generate 2x3 subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        path = Path(save_path)

        # set the plot titles and plot the layer outputs according to subplots
        for title, output, ax in zip(plot_titles, layer_outputs, axs.flatten()):
            ax.set_title(title)
            ax.hist(output.view(-1).tolist(), 50)
        fig.savefig(path / f"layer_output_histograms_{save_affix}.png")

        # now get into the decoder transformer blocks (dtb)
        out = emb
        for i, dtb in enumerate(self.blocks.layers):
            # get the outputs of the decoder transformer block
            first_ln = dtb.ln1(out)  # layer norm
            self_attn_outs = dtb.self_attn(first_ln)  # self attention
            first_res = out + self_attn_outs  # residual connection
            second_ln = dtb.ln2(first_res)  # layer norm
            mlp_outs = dtb.ffwd_net(second_ln)  # feedforward network
            out = first_res + mlp_outs  # second residual connection
            outputs = [first_ln, self_attn_outs, first_res, second_ln, mlp_outs, out]

            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            for title, output, ax in zip(plot_titles, outputs, axs.flatten()):
                ax.set_title(title)
                ax.hist(output.view(-1).tolist(), 50)

            fig.suptitle(f"DecoderTransformerBlock {i+1}")
            fig.savefig(path / f"layer_output_histograms_{save_affix}_dtb_{i+1}.png")

        self.train()
        return

    def plot_emb_weights(
        self,
        plot_text: bool = False,
        save_affix: str = "pretraining",
        save_path: str = "artifacts",
        tokenizer=None,
    ):
        """
        Plot heatmaps of the embedding weights.

        This function creates heatmaps for positional and token embedding weights,
        applying PCA to reduce dimensionality if necessary.

        Args:
        plot_text (bool, optional): If True, overlay weight values on the heatmap. Defaults to False.

        Returns:
        None. Displays the heatmaps using matplotlib.
        """
        path = Path(save_path)
        os.makedirs(path, exist_ok=True)
        
        embedding_tables = {
            "token_embeddings": self.token_embeddings_table.weight,
            "positional_embeddings": self.pos_embeddings_table.weight,
        }
        
        num_tables = len(embedding_tables)
        fig, axs = plt.subplots(num_tables, 1, figsize=(max(12, self.vocab_size // 3), 6 * num_tables))
        if num_tables == 1: # if only one embedding table for some reason
            axs = [axs]

        plot_idx = 0
        for name, emb_weight_tensor in embedding_tables.items():
            ax = axs[plot_idx]
            original_weights = emb_weight_tensor.detach().cpu().numpy()
            
            # Always use 1D PCA for GPT embeddings as per original logic, but make it robust
            pca_dim = 1
            if original_weights.shape[1] > pca_dim:
                pca = PCA(n_components=pca_dim)
                transformed_weights = pca.fit_transform(original_weights)
            else: # If original embedding dim is already 1 or less
                transformed_weights = original_weights[:, :pca_dim]

            data_to_plot = transformed_weights[:, 0] if transformed_weights.ndim > 1 else transformed_weights.flatten()
            num_elements = len(data_to_plot)
            
            ax.bar(range(num_elements), data_to_plot)
            ax.set_ylabel("1st Principal Component Value")
            ax.set_title(f"GPT {name.replace('_', ' ').title()} (1D PCA)\n{save_affix}")

            tick_labels_list = []
            if name == "token_embeddings":
                if tokenizer is not None and hasattr(tokenizer, 'itos'):
                    tick_labels_list = [tokenizer.itos.get(i, str(i)) for i in range(num_elements)]
                else:
                    tick_labels_list = [str(i) for i in range(num_elements)]
                # Adjust font size for token labels if many tokens
                xtick_fontsize = 8 if num_elements < 50 else (6 if num_elements < 100 else 4)
                ax.set_xticks(range(num_elements))
                ax.set_xticklabels(tick_labels_list, rotation=90, fontsize=xtick_fontsize)

            elif name == "positional_embeddings":
                tick_labels_list = [str(i) for i in range(num_elements)] # Positional indices
                ax.set_xticks(range(num_elements))
                ax.set_xticklabels(tick_labels_list, rotation=0, fontsize=8)
            
            if plot_text:
                text_fontsize = 7 if num_elements < 50 else (5 if num_elements < 100 else 3)
                for i in range(num_elements):
                    label = tick_labels_list[i] if tick_labels_list else str(i)
                    ax.text(i, data_to_plot[i], label, ha='center', 
                            va='bottom' if data_to_plot[i] >= 0 else 'top', fontsize=text_fontsize)
            
            ax.grid(True, axis='y', alpha=0.3)
            plot_idx += 1

        plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust for suptitle
        fig.suptitle(f"GPT Embedding Weights Visualizations ({save_affix})", fontsize=16)
        filename = path / f"GPT_embedding_weights_{save_affix}_pca1d.png"
        plt.savefig(filename)
        plt.close(fig)
        print(f"Saved GPT embedding weights plot to {filename}")
        return

    def plot_attn_heatmaps(
        self,
        sample_input: torch.Tensor = None,
        plot_text: bool = False,
        save_affix: str = "pretraining",
        save_path: str = "artifacts",
    ):
        """
        Plot heatmaps of softmaxed attention weights for each layer and head.
        This function visualizes the actual attention distributions (post-softmax)
        used by each attention head in each DecoderTransformerBlock.

        It assumes that `model.start_debug()` has been called and a forward pass
        (e.g., via `model(sample_input)`) has occurred, so that MultiHeadAttention
        layers have populated their `attn_weights_viz` attribute.

        Args:
        sample_input (torch.Tensor, optional): A sample input tensor. While the function
            primarily relies on pre-populated `attn_weights_viz`, this input can be
            used by other plotting functions or if a fresh forward pass is needed
            (though current design expects `attn_weights_viz` to be ready).
            If None, and `attn_weights_viz` is not found, the function may not plot.
        plot_text (bool, optional): If True, overlay attention probability values on the heatmap.
        save_affix (str): Affix for the saved plot filename.
        save_path (str): Directory to save the plot.

        Returns:
        None. Displays and saves the heatmaps.
        """
        path = Path(save_path)
        os.makedirs(path, exist_ok=True)

        if not (hasattr(self, 'log_outputs') and self.log_outputs):
            print("Warning: `plot_attn_heatmaps` called but model.log_outputs is False. "
                  "Attention weights might not be available in `attn_weights_viz`."
                  "Ensure model.start_debug() was called and a forward pass occurred.")

        # Ensure sample_input is on the correct device if provided and used for a new fwd pass
        # However, the primary goal is to use already populated .attn_weights_viz
        if sample_input is not None:
            device = next(self.parameters()).device # Get model's device
            sample_input = sample_input.to(device)
            # Trigger a forward pass to ensure attn_weights_viz are populated if not already
            # This is a bit redundant if main.py's debug block already did a forward pass
            # but makes the function more robust if called standalone after start_debug().
            with torch.no_grad():
                 self(sample_input) # This ensures MultiHeadAttention layers are run

        num_blocks = len(self.blocks.layers)
        if num_blocks == 0 or not isinstance(self.blocks.layers[0].self_attn, MultiHeadAttention):
            print("No suitable DecoderTransformerBlocks with MultiHeadAttention found.")
            return
            
        num_heads = self.blocks.layers[0].self_attn.num_heads
        
        # Determine figure size dynamically
        # Max width of 20 inches, height based on number of blocks
        fig_width = min(20, 5 * num_heads) 
        fig_height = min(20, 4 * num_blocks if num_blocks > 1 else 5) # Slightly more height for single block

        fig, axs = plt.subplots(num_blocks, num_heads, figsize=(fig_width, fig_height), squeeze=False)
        
        plotted_anything = False
        for i, dtb in enumerate(self.blocks.layers):
            if not isinstance(dtb.self_attn, MultiHeadAttention) or \
               not hasattr(dtb.self_attn, 'attn_weights_viz') or \
               dtb.self_attn.attn_weights_viz is None:
                print(f"Skipping DecoderTransformerBlock {i+1}, no 'attn_weights_viz' found or layer is not MultiHeadAttention.")
                for j in range(num_heads): # Still need to handle axs if skipping
                    axs[i,j].set_title(f"Block {i+1} Head {j+1}\n(No data)")
                    axs[i,j].text(0.5, 0.5, "No data", ha="center", va="center", transform=axs[i,j].transAxes)
                continue

            # attn_weights_viz has shape (B, num_heads, T, T)
            # sample_input is (1, T), so B=1. Squeeze the batch dim.
            head_weights_all_heads = dtb.self_attn.attn_weights_viz.squeeze(0).cpu().numpy() # Shape: (num_heads, T, T)
            
            context_len = head_weights_all_heads.shape[-1] # Should be T

            for j in range(num_heads):
                ax = axs[i, j]
                if j < head_weights_all_heads.shape[0]:
                    plot_data = head_weights_all_heads[j, :, :] # Shape: (T, T)
                    
                    im = ax.imshow(plot_data, cmap="viridis", interpolation="nearest", vmin=0, vmax=1) # Softmax output is [0,1]
                    ax.set_title(f"Block {i+1} Head {j+1}", fontsize=10)
                    
                    # Add colorbar for context
                    # fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)


                    if plot_text and context_len <= 20: # Only plot text for smaller contexts for legibility
                        text_fontsize = max(2, 10 - context_len // 2) # Adaptive font size
                        for row in range(context_len):
                            for col in range(context_len):
                                ax.text(col, row, f"{plot_data[row, col]*100:.0f}",
                                        color="white" if plot_data[row, col] < 0.5 else "black", # Contrast for text
                                        ha="center", va="center", fontsize=text_fontsize)
                    plotted_anything = True
                else: # Should not happen if num_heads is consistent
                    ax.set_title(f"Block {i+1} Head {j+1}\n(Error)")
                    ax.text(0.5, 0.5, "Head index error", ha="center", va="center", transform=ax.transAxes)

                # Set axis ticks for clarity (optional, can be noisy)
                # ax.set_xticks(torch.arange(context_len))
                # ax.set_yticks(torch.arange(context_len))
                # ax.set_xlabel("Key Positions")
                # ax.set_ylabel("Query Positions")


        if not plotted_anything:
            print("No attention weights were plotted. Check debug state and forward pass.")
            plt.close(fig)
            return

        fig.suptitle(f"GPT Attention Heatmaps (Softmaxed Weights) - {save_affix}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle
        
        filename = path / f"GPT_attention_heatmaps_{save_affix}.png"
        plt.savefig(filename)
        plt.close(fig)
        print(f"Saved GPT attention heatmaps to {filename}")
        return
