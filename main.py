# to turn this into a cli, we need an argument parser
import argparse

import torch
from pypdf import PdfReader

from layers import Model
from nnutils import generate_text, train_loop
from preprocessing import process_data

parser = argparse.ArgumentParser(
    description="Train a neural net with given PDF generate text"
)
parser.add_argument("--pdf", type=str, help="Path to the PDF file")
parser.add_argument("--n_embed", type=int, default=15, help="Embedding size")
parser.add_argument("--n_hidden", type=int, default=400, help="Hidden size")
parser.add_argument("--block_size", type=int, default=10, help="Block size")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument(
    "--generate", action="store_true", help="Generate text", default=False
)
parser.add_argument(
    "--n_chars", type=int, default=100, help="Number of characters to generate"
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
        # read the pdf file
        reader = PdfReader(args["pdf"])
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            text += t

        with open(f"books/{args['pdf'].split('/')[-1]}.txt", "w") as f:
            f.write(text)

        # construct the dataset and get the vocabulary size
        text = text.lower()
        vocab_size = len(set(text)) + 1

        train_loader, test_loader = process_data(
            text, block_size=args["block_size"], batch_size=args["batch_size"]
        )
        model = Model(
            vocab_size,
            n_embed=args["n_embed"],
            block_size=args["block_size"],
            n_hidden=args["n_hidden"],
        )
        train_loop(
            model,
            train_loader,
            test_loader,
            epochs=args["epochs"],
            learning_rate=args["lr"],
        )
        print("-" * 50)
        print("TRAINED THE MODEL!")
        print("-" * 50)
