# to turn this into a cli, we need an argument parser
import argparse

import torch

from net import MLP, HierarchicalMLP
from nnutils import generate_text, train_loop
from processors import CharLevelMLPProcessor
from tokenizers import CharTokenizer

parser = argparse.ArgumentParser(
    description="Train a neural net with given file to generate text"
)
parser.add_argument(
    "--train_ratio",
    type=float,
    default=0.8,
    help="ratio of the input data that will be used for the training",
)
parser.add_argument("--file", type=str, help="Path to the file")
parser.add_argument("--n_embed", type=int, default=15, help="Embedding size")
parser.add_argument("--n_hidden", type=int, default=400, help="Hidden size")
parser.add_argument("--block_size", type=int, default=10, help="Block size")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument(
    "--lrsche", action="store_true", help="Learning rate scheduler", default=False
)
parser.add_argument(
    "--generate", action="store_true", help="Generate text", default=False
)
parser.add_argument(
    "--n_chars", type=int, default=100, help="Number of characters to generate"
)
parser.add_argument(
    "--model",
    type=str,
    default="hmlp",
    help="(hmlp) hierarchical, (mlp) mlp, or (gpt) GPT model to train",
)
parser.add_argument(
    "--n_consecutive",
    type=int,
    default=2,
    help="number of token to concatenate hierarchically (only used when the model is hierarchical)",
)
parser.add_argument(
    "--n_layers",
    type=int,
    default=4,
    help="number of consecutive hidden layer blocks, blocks are different for each model and can be seen in `layers.py`.",
)
parser.add_argument("--seedtext", type=str, help="Starting text for generation")
args = parser.parse_args()
args = vars(args)

if __name__ == "__main__":
    if args["generate"]:
        # we need to load the model and start generating text
        import os

        assert os.path.exists("artifacts/model.pt"), "Model not found"
        assert os.path.exists("artifacts/char_to_ix.pt"), "char_to_ix not found"
        assert os.path.exists("artifacts/ix_to_char.pt"), "ix_to_char not found"

        model = torch.load("artifacts/model.pt")
        text = generate_text(
            args["seedtext"].lower(), model=model, n_chars=args["n_chars"]
        )
        print(text)
    else:
        if args["model"] in ["hmlp", "mlp"]:
            tokenizer = CharTokenizer()
            processor = CharLevelMLPProcessor(
                paths=args["file"],
                tokenizer=tokenizer,
                context_length=args["block_size"],
            )
        elif args["model"] == "gpt":
            pass

        train_loader, test_loader = processor.get_dataloaders(
            args["batch_size"], args["train_ratio"]
        )

        vocab_size = tokenizer.vocab_size + 1
        print("vocab_size:", vocab_size)

        if args["model"] == "hmlp":
            model = HierarchicalMLP(
                vocab_size,
                n_consecutive=args["n_consecutive"],
                n_embed=args["n_embed"],
                n_hidden=args["n_hidden"],
                n_layers=args["n_layers"],
                block_size=args["block_size"],
            )
        elif args["model"] == "mlp":
            model = MLP(
                vocab_size,
                n_embed=args["n_embed"],
                block_size=args["block_size"],
                n_hidden=args["n_hidden"],
                n_layers=args["n_layers"],
            )

        train_loop(
            model,
            train_loader,
            test_loader,
            epochs=args["epochs"],
            learning_rate=args["lr"],
            lrsche=args["lrsche"],
        )
        print("-" * 50)
        print("TRAINED THE MODEL!")
        print("-" * 50)
