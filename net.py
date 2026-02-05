import torch
from torch import nn
from torch.nn import functional as F


class BaseModel(nn.Module):
    """
    Simple model class that has layer-specific backprop calculation.

    """


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


class MultiHeadDifferentialAttention(nn.Module):
    """implements multi-head differential attention as shown in https://arxiv.org/pdf/2410.05258"""

    def __init__(self, num_heads, n_embd, head_size, block_size, l_ix):
        super().__init__()
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size
        self.l_ix = l_ix

        self.key1 = nn.Parameter(
            torch.randn(num_heads, n_embd, head_size) * (num_heads * n_embd) ** -0.5
        )
        self.key2 = nn.Parameter(
            torch.randn(num_heads, n_embd, head_size) * (num_heads * n_embd) ** -0.5
        )
        self.query1 = nn.Parameter(
            torch.randn(num_heads, n_embd, head_size) * (num_heads * n_embd) ** -0.5
        )
        self.query2 = nn.Parameter(
            torch.randn(num_heads, n_embd, head_size) * (num_heads * n_embd) ** -0.5
        )
        self.value = nn.Parameter(
            torch.randn(num_heads, n_embd, head_size) * (num_heads * n_embd) ** -0.5
        )
        # Detach to prevent computation graph from persisting across forward passes
        self.initial_lambda = (
            0.8 - (0.6 * torch.exp(torch.tensor(-0.3 * (l_ix - 1))))
        ).detach()
        self.lambdas = nn.Parameter(torch.randn(4))  # [q1, k1, q2, k2]
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

        # Use local variable to avoid storing computation graph state
        reparameterized_lambda = (
            torch.exp(self.lambdas[0] * self.lambdas[1])
            - torch.exp(self.lambdas[2] * self.lambdas[3])
            + self.initial_lambda
        )

        k1 = x @ self.key1  # (batch_size, num_heads, context_length, head_size)
        q1 = x @ self.query1  # (batch_size, num_heads, context_length, head_size)

        k2 = x @ self.key2  # (batch_size, num_heads, context_length, head_size)
        q2 = x @ self.query2  # (batch_size, num_heads, context_length, head_size)

        s = 1 / torch.sqrt(torch.tensor(self.head_size))

        wei1 = (q1 @ k1.transpose(-2, -1)).masked_fill(
            self.tril[:, :T, :T] == 0, float("-inf")
        ) * s
        wei2 = (q2 @ k2.transpose(-2, -1)).masked_fill(
            self.tril[:, :T, :T] == 0, float("-inf")
        ) * s

        wei1 = F.softmax(wei1, dim=-1)
        wei2 = F.softmax(wei2, dim=-1)

        wei = wei1 - reparameterized_lambda * wei2

        v = x @ self.value  # (bs, 1, cl, ne) x (nh, ne, hs) -> (bs, nh, cl, hs)
        out = wei @ v  # (bs, nh, cl, cl) x (bs, nh, cl, hs) -> (bs, nh, cl, hs)
        out = out.transpose(1, 2)  # (bs, cl, nh, hs)
        out = out.reshape(
            out.size(0), out.size(1), self.n_embd
        )  # (bs, cl, n_embd) = (B, T, C)

        # scale the output with (1- self.initial_lambda) as its stated in the paper
        out = out * (1 - self.initial_lambda)

        out = self.proj(out)

        return out


class MultiHeadAttention(nn.Module):
    """implements multi-headed masked self-attention using tensor operations"""

    def __init__(self, num_heads, n_embd, head_size, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size

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


class DecoderDifferentialTransformerBlock(nn.Module):
    def __init__(self, num_heads, n_hidden, context_length, l_ix=None):
        super().__init__()
        # using multiple heads will let us to provide more
        # communication channels between the tokens
        # but there's a caveat, the implementation downscales
        # the attention layers' dimensions, resulting in
        # more condensed communication channels.
        self.head_size = n_hidden // num_heads
        self.self_attn = MultiHeadDifferentialAttention(
            num_heads, n_hidden, self.head_size, context_length, l_ix
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
        pos_emb = self.pos_embeddings_table(
            torch.arange(T, device=idx.device)
        )  # (T, n_embd)
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


class GPDT(GPT):
    """GPT implementation extended to include differential attention in https://arxiv.org/pdf/2410.05258"""

    def __init__(self, n_embd, vocab_size, num_heads, num_blocks, block_size):
        super().__init__(n_embd, vocab_size, num_heads, num_blocks, block_size)

        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.token_embeddings_table = Embedding(vocab_size, n_embd)
        self.pos_embeddings_table = Embedding(block_size, n_embd)
        self.blocks = Sequential(
            [
                DecoderDifferentialTransformerBlock(
                    num_heads, n_embd, context_length=block_size, l_ix=i
                )
                for i in range(1, num_blocks + 1)
            ]
        )
        self.ln_f = LayerNorm(n_embd)
        self.ln_head = Linear(n_embd, vocab_size)
