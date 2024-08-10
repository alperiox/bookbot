import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F

from tokenizers import Tokenizer


class Head(nn.Module):
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
        self.register_buffer(
            "tril", torch.tril(torch.ones(context_length, context_length))
        )

    def forward(self, x):
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


class MultiHeadAttention(nn.Module):
    """implements multi-headed masked self-attention using tensor operations"""

    def __init__(self, num_heads, n_embd, head_size, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size

        self.key = (
            nn.Parameter(torch.randn(num_heads, n_embd, head_size))
            * (num_heads * n_embd) ** -0.5
        )
        self.query = (
            nn.Parameter(torch.randn(num_heads, n_embd, head_size))
            * (num_heads * n_embd) ** -0.5
        )
        self.value = (
            nn.Parameter(torch.randn(num_heads, n_embd, head_size))
            * (num_heads * n_embd) ** -0.5
        )

        self.proj = Linear(n_embd, n_embd)

        # (B, T, n_embd) x (num_heads, n_embd, head_size) --> (B, num_heads, T, head_size)
        self.register_buffer(
            "tril", torch.tril(torch.ones(num_heads, block_size, block_size))
        )

    def forward(self, x):
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

        v = x @ self.value  # (bs, 1, cl, ne) x (nh, ne, hs) -> (bs, nh, cl, hs)
        out = wei @ v  # (bs, nh, cl, cl) x (bs, nh, cl, hs) -> (bs, nh, cl, hs)
        out = out.transpose(1, 2)  # (bs, cl, nh, hs)
        out = out.reshape(
            out.size(0), out.size(1), self.n_embd
        )  # (bs, cl, n_embd) = (B, T, C)

        out = self.proj(out)

        return out


class MultiHeadAttentionConcat(nn.Module):
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

    def forward(self, x):
        # calculate the outputs of the heads
        out = [h(x) for h in self.heads]  # list of (B, T, head_size)
        # just concatenate them all
        out = torch.concat(out, -1)  # (B, T, head_size * num_heads)
        # apply the projection transformation
        out = self.proj(out)
        return out


class FeedForwardBlock(nn.Module):
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

    def forward(self, x):
        out = self.net(x)
        return out


class ReLU(nn.Module):
    def forward(self, x):
        return (x > 0) * x


class DecoderTransformerBlock(nn.Module):
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

    def forward(self, x):
        # add the residual connections and normalize given inputs
        # just as in the paper.
        x = x + self.self_attn(self.ln1(x))
        x = x + self.ffwd_net(self.ln2(x))
        out = x

        return out


class Linear(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_in, n_out) / n_in**0.5)

        self.has_bias = bias
        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        self.x = x
        if self.has_bias:
            out = x @ self.weight + self.bias
        else:
            out = x @ self.weight
        return out


class Tanh(nn.Module):
    def forward(self, x):
        self.x = x
        out = F.tanh(x)
        return out


class Embedding(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_vocab, n_embed))

    def forward(self, x):
        self.x = x
        out = self.weight[x]
        return out


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # parameters to be trained with backprop
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        xmean = x.mean(1, keepdim=True)  # layers mean
        xvar = x.var(1, keepdim=True)  # layers var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        out = self.gamma * xhat + self.beta
        return out


class BatchNorm1d(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters to be trained with backprop
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        # buffers, trained with a running `momentum update`
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            else:
                raise NotImplementedError("Number of input dimensions must be 2 or 3.")

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


class LinearBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.linear = Linear(n_in, n_out)
        self.bn = BatchNorm1d(n_out)
        self.tanh = Tanh()

    def forward(self, x):
        self.x = x
        out = self.tanh(self.bn(self.linear(x)))

        return out


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.view(x.size(0), -1)
        return out


class FlattenConsecutive(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n  # sum n consecutive elements

    def forward(self, x):
        B, T, C = x.shape
        out = x.view(B, T // self.n, C * self.n)
        if out.shape[1] == 1:
            out = out.squeeze(1)
        return out


class Sequential(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        return out


class HierarchicalMLP(nn.Module):
    def __init__(
        self, vocab_size, n_consecutive, n_embed, n_hidden, block_size, n_layers=4
    ):
        assert (
            n_consecutive**n_layers == block_size
        ), "`n_consecutive^n_layers` must be equal to `block_size` because of `FlattenConsecutive`!"
        super().__init__()
        self.vocab_size = vocab_size
        self.n_consecutiv = n_consecutive
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

    def forward(self, x, y=None):
        self.x = x
        out = self.model(self.x)

        if y is None:
            loss = None
        else:
            loss = F.cross_entropy(out, y)

        return out, loss

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
        sample_input: torch.Tensor | None = None,
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
        plt.savefig(f"{save_path}/layer_output_histograms_{save_affix}.png")

        self.train()
        return

    def plot_emb_weights(
        self,
        plot_text: bool = False,
        save_affix: str = "pretraining",
        save_path: str = "artifacts",
        ndims=1,
        tokenizer: Tokenizer | None = None,
    ):
        assert ndims < 2, "ndims must be 1 or 2 for plotting the embeddings"
        pca = PCA(n_components=ndims)
        weights = pca.fit_transform(self.model.layers[0].weight.detach().numpy())

        plt.figure(figsize=(self.n_embed // 5, 10))
        plt.title("Embedding weights")
        if isinstance(tokenizer, Tokenizer):
            tokens = tokenizer.itos
        else:
            tokens = {i: i for i in range(self.vocab_size)}

        if ndims == 1:
            plt.imshow(weights.T, cmap="gray", interpolation="nearest")
            # set the x-axis labels
            plt.xticks(range(self.vocab_size), tokens.values())

            if plot_text:
                for i in range(self.vocab_size):
                    plt.text(i, 0, weights[i, 0])
        else:
            plt.scatter(weights[:, 0], weights[:, 1])
            if plot_text:
                for i in range(self.vocab_size):
                    plt.text(weights[i, 0], weights[i, 1], tokens[i])
        plt.savefig(f"{save_path}/embedding_weights_{save_affix}.png")

        return


class MLP(nn.Module):
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

    def forward(self, x, y=None):
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
        sample_input: torch.Tensor | None = None,
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

        layer_counts = {}
        for i in range(len(self.layers)):
            name = self.layers[i].__class__.__name__
            layer_counts[name] = layer_counts.get(name, 0) + 1

        plot_titles = ["Embedding"]

        # add the counts of each layer
        for i in range(len(self.layers)):
            name = self.layers[i].__class__.__name__
            plot_titles.append(f"{name} {layer_counts[name]}")
            layer_counts[name] -= 1

        # generate a (L//3 + 2)x3 grid of subplots,
        # so every row will have 3 subplots, plus one row for the embeddings
        fig, axs = plt.subplots(len(self.layers) // 3 + 2, 3, figsize=(15, 5))

        # plot the embedding outputs
        ax = axs[0, 0]
        ax.set_title("Embedding")
        ax.hist(self.embedding.out.view(-1).tolist(), 50)

        # plot the layer outputs
        for title, layer, ax in zip(plot_titles[1:], self.layers, axs.flatten()[1:]):
            ax.set_title(title)
            ax.hist(layer.out.view(-1).tolist(), 50)
        plt.savefig(f"{save_path}/layer_output_histograms_{save_affix}.png")
        self.stop_debug()
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
        assert ndims < 2, "ndims must be 1 or 2 for plotting the embeddings"
        pca = PCA(n_components=ndims)
        weights = pca.fit_transform(self.embedding.weight.detach().numpy())
        plt.figure(figsize=(self.n_embed // 5, 10))
        plt.title("Embedding weights")
        if tokenizer is not None:
            tokens = tokenizer.itos
        else:
            tokens = {i: i for i in range(self.vocab_size)}

        if ndims == 1:
            plt.imshow(weights.T, cmap="gray", interpolation="nearest")
            # set the x-axis labels
            plt.xticks(range(self.vocab_size), tokens.values())
            if plot_text:
                for i in range(len(weights)):
                    plt.text(i, 0, str(i))
        else:
            plt.scatter(weights[:, 0], weights[:, 1])
            if plot_text:
                for i in range(len(weights)):
                    plt.text(weights[i, 0], weights[i, 1], str(i))
        plt.savefig(f"{save_path}/embedding_weights_{save_affix}.png")

        return


class GPT(nn.Module):
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

    def forward(self, idx, targets=None) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        sample_input: torch.Tensor | None = None,
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
        assert os.path.exists(save_path), "The save path does not exist."
        if plot_text:
            print(
                Warning(
                    "Plotting text on the heatmaps may not be feasible for large embeddings. (WIP feature)"
                )
            )

        fig, axs = plt.subplots(2, 1, figsize=(25, 15))
        path = Path(save_path)
        weights = [self.pos_embeddings_table.weight, self.token_embeddings_table.weight]
        font_sizes = [3, 4]
        # apply PCA to the weights
        pca = PCA(n_components=1)  # reduce to 1D
        # apply PCA to the weights
        weights = [pca.fit_transform(weight.detach().numpy()) for weight in weights]
        titles = ["Positional embeddings (weight)", "Token embeddings (weight)"]

        for ix, (title, weight, ax, font_size) in enumerate(
            zip(titles, weights, axs.flatten(), font_sizes)
        ):
            ax.set_title(title)
            ax.imshow(weight.T, cmap="gray", interpolation="nearest")
            #
            if tokenizer is not None and ix == 1:
                # set the token names to x-axis
                ax.set_xticks(range(tokenizer.vocab_size))
                ax.set_xticklabels(tokenizer.itos.values())

            if plot_text:
                for x in range(weight.shape[0]):
                    for y in range(weight.shape[1]):
                        ax.text(
                            x,
                            y,
                            f"{weight[x, y]:.2f}",
                            color="red",
                            ha="center",
                            va="center",
                            fontsize=font_size,
                        )

        plt.suptitle("Embedding heatmaps")
        fig.savefig(path / f"embedding_heatmaps_{save_affix}.png")

        return

    def plot_attn_heatmaps(
        self,
        sample_input: torch.Tensor | None = None,
        plot_text: bool = False,
        save_affix: str = "pretraining",
        save_path: str = "artifacts",
    ):
        """
        Plot heatmaps of attention weights for each layer and head.

        This function visualizes the attention weights for each attention head
        in each layer of the model.

        Args:
        model (GPT): The GPT model instance
        sample_input (torch.Tensor, optional): Input tensor to use. If None, random input is generated.
        plot_text (bool, optional): If True, overlay attention values on the heatmap. Defaults to False.

        Returns:
        None. Displays the heatmaps using matplotlib.
        """
        assert os.path.exists(save_path), "The save path does not exist."
        if plot_text:
            print(
                Warning(
                    "Plotting text on the heatmaps may not be feasible for large embeddings. (WIP feature)"
                )
            )

        if sample_input is None:
            sample_input = torch.randint(
                0, self.vocab_size - 1, (1, self.block_size), dtype=torch.long
            )

        attention_outputs = []
        out = self.token_embeddings_table(sample_input) + self.pos_embeddings_table(
            torch.arange(self.block_size)
        )

        for i, dtb in enumerate(self.blocks.layers):
            k = out @ dtb.self_attn.key
            q = out @ dtb.self_attn.query
            w = (q @ k.transpose(-2, -1)) * (
                dtb.self_attn.head_size**-0.5
            )  # (num_heads, T, T)
            w = w.detach()  # (num_heads, T, T)

            attention_outputs.append(w)

            out = dtb(out)

        attention_outputs = [
            [w[head_ix, :, :] for head_ix in range(self.num_heads)]
            for w in attention_outputs
        ]

        # plot the heatmaps
        font_size = self.num_heads
        figsize = (int(7.5 * self.num_heads), int(5 * self.num_heads))
        path = Path(save_path)

        fig, axs = plt.subplots(
            len(self.blocks.layers), self.num_heads, figsize=figsize
        )
        for i, dtb_attn_out in enumerate(attention_outputs):
            for j, attn_head_out in enumerate(dtb_attn_out):
                # normalize the attention weights
                plot_data = attn_head_out.detach().abs()
                plot_data /= plot_data.max()
                # plot the heatmap
                axs[i, j].imshow(
                    plot_data.numpy(), cmap="gray", interpolation="nearest"
                )
                axs[i, j].set_title(f"DecoderTransformerBlock {i+1} Head {j+1}")
                # text annotations
                if plot_text:
                    for x in range(plot_data.size(0)):
                        for y in range(plot_data.size(1)):
                            axs[i, j].text(
                                y,
                                x,
                                f"{plot_data[x, y] * 100:.0f}",
                                color="red",
                                ha="center",
                                va="center",
                                fontsize=font_size,
                            )

        plt.suptitle("Attention heatmaps")
        fig.savefig(path / f"attention_heatmaps_{save_affix}.png")

        return
