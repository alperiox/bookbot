# bookbot

a project that reads the given file and uses a neural network to generate text that looks like from the book.

the built-in neural network is MLP, Wavenet-inspired Hierarchical MLP, and a GPT network that's built with pure Pytorch (from scratch) along with batch normalization layer and Kaiming initialization.

Thanks to Andrej Karpathy for his great course on deep learning.

Available file types as of the moment:

- PDF
- TXT

## Usage

### Installation

You can try the project out by cloning the git repository

```bash
git clone https://github.com/alperiox/bookbot.git
```

Then just install the `poetry` environment and move on to the next steps.

### How to train the network?

Simply run the `main.py` by setting up the arguments below.

You can start the training using the script like in the following:

```bash
python main.py --file=romeo-and-juliet.pdf --n_embed=20 --n_hidden=200 
            \ --block_size=15 --batch_size=64 --epochs=20 --lr=0.01
```

Or you can just start the training using the default arguments:

```bash
python main.py --file=romeo-and-juliet.pdf
```

Or if you want to have more control over the whole training, consider using a more detailed configuration:

| Argument | Default Value | Description |
|----------|---------------|-------------|
| train_ratio | 0.8 | Ratio of the input data that will be used for training |
| file | - | Path to the PDF/TXT file |
| n_embed | 15 | Embedding vector's dimension |
| n_hidden | 400 | Hidden layer's dimensions (the hidden layers will be defined as n_hidden x n_hidden) |
| block_size | 10 | Block size to set up the dataset, it's our context window in this project |
| batch_size | 32 | The amount of samples that'll be processed in one go |
| epochs | 10 | Number of epochs to train the model |
| lr | 0.001 | Learning rate to update the weights |
| generate | False | To run the generation mode, it's required to generate text using the pre-trained model. So you should train a model first |
| max_new_tokens | 100 | The amount of tokens that will be generated if `generate` flag is active |
| model | gpt | Hierarchical mlp (hmlp), mlp model (mlp) or gpt (gpt) model to train |
| n_consecutive | 2 | The amount of consecutive tokens to concatenate in the hierarchical model |
| n_layers | 4 | Number of processor blocks in the model, check out the models in `layers.py` for more information about its usage |
| num_heads | 3 | Number of self-attention heads in the multi-head self-attention layer in GPT implementation |
| num_blocks | 2 | Number of layer blocks given the model. Sequential linear blocks for MLP and Hierarchical MLP, DecoderTransformerBlocks for GPT |
| context | None | The context for the text generation, please try to use a longer context than the `block_size` (required if `generate` is True) |
| device | cpu | The device to train the models on, available values are `mps`, `cpu` and `cuda`. |

The training will generate several artifacts and will save them in the `artifacts` directory. The saved artifacts include the model, the data loaders, calculated losses along the training, and finally the tokenizer to use the constructed character-level vocabulary.

### How to generate new text?

You can generate text __after training a model first.__ That's because the generation pipeline makes use of the saved artifacts. In order to start the generation, you need to pass the `generate` flag:

```bash
python main.py --generate --context="Juliet," --max_new_tokens=100
>>> juliet, and have know lie thee why!
```

The generation will run until the wanted character length is matched.

## Further plans

- [x] Implement debugging tools to analyze the neural network's training performance. (more like useful graphs and statistics.)
  - [x] graphs to check out the layer outputs' distributions.
        layer output distributions (with extra information about mean, std and the distribution plot)
  - [x] graphs to check the gradient flow
    - [x] layer gradient means
    - [x] layer gradient stds
    - [x] ratio of amount of change in the parameters given the weights
          we multiply the learning rate with the layer's gradient's std and divide it by
          parameters' std. this ratio will be higher if gradient std is larger (grads vary too much from the mean)
          and the params are smaller in comparison.
    - [x] layer grad distributions (with extra information about mean, std and the distribution plot)
    - [x] ratio of the gradient of a specific layer to its input
          so if the ratio is too high, it means that the gradients are too high wtr to the input
          and we actually want constant but smaller updates throughout the network to not miss any local minimas etc
    - [x] ratio of the amount of change vs the weights, the stats should be saved in L7 here
  - [x] summary for the training
- [ ] More modeling options such as LSTMs, RNNs, and Transformer-based architectures.
  - [x] Wavenet? (implemented the hierarchical architecture)
  - [x] GPT
  - [ ] GPT-2
- [ ] GPT tokenizer implementation to further improve the generation quality.

## Contributing

While I'm open to new feature ideas and stuff, please let me do the coding part since I'm trying to improve my overall understanding. Thus, I'd love to accept any feature requests as new PRs. You can reach me from Discord (@alperiox) or my e-mail address (<alper_balbay@hacettepe.edu.tr>)
