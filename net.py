import torch
from torch.nn import functional as F

# TODO: saving the model weights
# TODO: loading the model weights
# TODO: generation method for the MLP models


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
        # we might want to work with shorted sequences rather than defined context length, hence `self.tril[:T, :T]`.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (B,T,T)

        self.out = wei @ v  # (B, T, hs)
        return self.out

    def parameters(self):
        return [*self.key.parameters(),
                *self.query.parameters(),
                *self.value.parameters()]

class MultiHeadAttention:
    """multi-head self-attention that'll be used in GPT implementation"""

    def __init__(self, num_head, n_in, head_size, context_length):
        self.head_size = head_size
        self.num_head = num_head

        self.heads = [Head(n_in, head_size, context_length) for _ in range(num_head)]
        self.proj = Linear(n_in, n_in)

    def __call__(self, x):
        out = [h(x) for h in self.heads]  # list of (B, T, head_size)
        self.out = torch.concat(out, -1)  # (B, T, head_size * num_heads)
        self.out = self.proj(self.out)
        return self.out

    def parameters(self):
        params = []
        for h in self.heads:
            params.extend(h.parameters())
        return params + self.proj.parameters()


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

    def parameters(self):
        return self.net.parameters()

class ReLU:
    def __call__(self, x):
        return (x > 0) * x

    def parameters(self): return []


class DecoderTransformerBlock:
    def __init__(self, num_heads, n_hidden, context_length):
        self.head_size = n_hidden // num_heads
        self.self_attn = MultiHeadAttention(
            num_heads, n_hidden, self.head_size, context_length
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

    def parameters(self):
        return [
            *self.self_attn.parameters(),
            *self.ffwd_net.parameters(),
            *self.ln1.parameters(),
            *self.ln2.parameters()
        ]


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


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        self.out = x
        for layer in self.layers:
            self.out = layer(self.out)
            # print(f"{layer.__class__.__name__:20s}:", self.out.shape)

        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class HierarchicalMLP:
    def __init__(
        self, vocab_size, n_consecutive, n_embed, n_hidden, block_size, n_layers=4
    ):
        assert (
            2**n_layers == block_size
        ), "`2^n_layers` must be equal to `block_size` because of `FlattenConsecutive`!"
        
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
            raise NotImplementedError("batched generation is not supported at the moment.")
        self.eval()
        for _ in range(max_new_tokens):
            input_tensor = idx[:, -self.block_size:]
            out, loss = self(input_tensor)
            probs = F.softmax(out, dim=-1)
            # sample the next character
            next_ix = torch.multinomial(probs, 1) # (1, 1)

            # if next_ix[0] == self.special_tokens.get("EOS_TOKEN", None):
            #     break
            
            idx = torch.concat([idx, next_ix], -1) # (B, block_size+1)
        self.train()
        return idx


class MLP:
    def __init__(self, vocab_size, block_size, n_embed, n_hidden, n_layers=4):
        self.layers = []
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.special_tokens = {}

        self.embedding = Embedding(vocab_size, n_embed)

        self.layers.append(Linear(n_embed * block_size, n_hidden))

        self.layers += [LinearBlock(n_hidden, n_hidden) for _ in range(n_layers - 2)]

        self.layers.append(Linear(n_hidden, vocab_size))

    def __call__(self, x, y=None):
        self.x = x  # (B, T)

        x = self.embedding(x)
        x = x.view(x.size(0), -1)

        for layer in self.layers:
            x = layer(x)

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

    def parameters(self):
        return [self.embedding.weight] + [
            param for layer in self.layers for param in layer.parameters()
        ]

    def generate(self, idx, max_new_tokens):
        if idx.shape[0] != 1:
            raise NotImplementedError("batched generation is not supported at the moment.")

        for _ in range(max_new_tokens):
            input_tensor = idx[:, -self.block_size:]
            out, loss = self(input_tensor)
            probs = F.softmax(out, dim=1)
            # sample the next character
            next_ix = torch.multinomial(probs, 1).item() # (1, 1)

            if next_ix[0] == self.special_tokens.get("EOS_TOKEN", None):
                break
            
            idx = torch.concat([idx, next_ix], -1) # (B, block_size+1)

        return idx


class GPT:
    def __init__(self, n_embd, vocab_size, num_heads, num_blocks, block_size):
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
        return [
            *self.token_embeddings_table.parameters(),
            *self.pos_embeddings_table.parameters(),
            *self.blocks.parameters(),
            *self.ln_f.parameters(),
            *self.ln_head.parameters()
        ]

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
