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


class HierarchicalMLP:
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

    def generate(self, idx, max_new_tokens):
        # TODO
        raise NotImplementedError("Not implemented yet.")


class MLP:
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

    def generate(self, idx, max_new_tokens):
        # TODO
        raise NotImplementedError("Not implemented yet.")
class GPT:
    def __init__(self, n_embd, vocab_size, num_heads, num_blocks, block_size):
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.token_embeddings_table = Embedding(vocab_size, n_embd)
        self.pos_embeddings_table = Embedding(block_size, n_embd)
        self.blocks = Sequential([
            DecoderTransformerBlock(num_heads, n_embd, context_length=block_size)
            for _ in range(num_blocks)
        ])
        self.ln_f = LayerNorm()
        self.ln_head = Linear(n_embd, vocab_size)

    def __call__(self, idx, targets=None):
        # inputs and targets are (B, T) shaped
        B, T = idx.shape

        tok_emb = self.token_embeddings_table(idx) # (B, T, n_embd)
        pos_emb = self.pos_embeddings_table(torch.arange(T)) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        x = self.blocks(x) # (B, T, n_embd)
        x = self.ln_f(x) # (B, T, n_embd)
        logits = self.ln_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            logits = logits.view(B*T, -1)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array where T is the context length
        for _ in range(max_new_tokens):
            # crop the idx so we'll stay in the dimensions of positional embedding table
            cropped_idx = idx[:, -self.block_size:]
            logits, loss = self(cropped_idx) # (B, T, C=vocab_size)
            # pick the last context window to sample the next token
            logits = logits[:, -1, :] # (B, C)
            # apply softmax to map the logits to probs
            probs = F.softmax(logits, -1)
            # sample the next index
            next_idx = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.concat([idx, next_idx], dim=-1) # (B, T+1)
        
        return idx