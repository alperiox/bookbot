# bookbot

a project that reads the given file and uses a neural network to generate text that looks like from the book.

the built-in neural network is a MLP, Wavenet-inspired Hierarchical MLP, and a GPT network that's built with pure Pytorch (from scratch) along with batch normalization layer and Kaiming initialization.

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

- __train_ratio__ (default: 0.8): ratio of the input data that will be used for the training
- __file__: Path to the PDF/TXT file
- __n_embed__ (default: 15): embedding vector's dimension
- __n_hidden__ (default: 400): hidden layer's dimensions (the hidden layers will be defined as n_hidden x n_hidden)
- __block_size__ (default: 10): block size to set up the dataset, it's our context window in this project.
- __batch_size__ (default: 32): the amount of samples that'll be processed in one go
- __epochs__ (default: 10): number of epochs to train the model.
- __lr__ (default: 0.001): learning rate to update the weights
- __generate__ (default: False): to run the generation mode, it's required to generate text using the pre-trained model. So you should train a model first.
- __max_new_tokens__ (default: 100): the amount of tokens that will be generated if `generate` flag is active
- __model__ (default: gpt): hierarchical mlp (hmlp), mlp model (mlp) or gpt (gpt) model to train.
- __n_consecutive__ (default: 2): the amount of consecutive tokens to concatenate in the hierarchical model.
- __n_layers__ (default: 4): number of processor blocks in the model, check out the models in `layers.py` for more information about its usage.
- __num_heads__ (default: 3): number of self-attention heads in the multi-head self-attention layer in GPT implementation.
- __num_blocks__ (default: 2): number of layer blocks given the model. Sequential linear blocks for MLP and Hierarchical MLP, DecoderTransformerBlocks for GPT.
- __context__ (default: None, required if `generate`): the context for the text generation, please try to use a longer context than the `block_size`

The training will generate several artifacts and will save them in the `artifacts` directory. The saved artifacts include the model, the data loaders, calculated losses along the training, and finally the tokenizer to use the constructed character-level vocabulary.

### How to generate new text?

You can generate text __after training a model first.__ That's because the generation pipeline makes use of the saved artifacts. In order to start the generation, you need to pass the `generate` flag:

```bash
python main.py --generate --context="Juliet," --max_new_tokens=100
>>> juliet, and have know lie thee why!
```

The generation will run until the wanted character length is matched.

## Further plans

- [ ] Implement debugging tools to analyze the neural network's training performance. (more like useful graphs and statistics.)
  - [ ] graphs to check out the layer outputs' distributions.
  - [ ] graphs to check the gradient flow
  - [ ] summary for the training
- [ ] More modeling options such as LSTMs, RNNs, and Transformer-based architectures.
  - [x] Wavenet? (implemented the hierarchical architecture)
  - [x] GPT
  - [ ] GPT-2
- [ ] GPT tokenizer implementation to further improve the generation quality.

## Contributing

While I'm open to new feature ideas and stuff, please let me do the coding part since I'm trying to improve my overall understanding. Thus, I'd love to accept any feature requests as new PRs. You can reach me from Discord (@alperiox) or my e-mail address (<alper_balbay@hacettepe.edu.tr>)
