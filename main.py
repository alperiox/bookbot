from preprocessing import process_data
from layers import Model
from nnutils import train_loop, generate_text

from torch import nn


with open("books/frankenstein.txt", "r") as f:
    text = f.read()
    text = text.lower()
    vocab_size = len(set(text))+1

if __name__ == "__main__":
    train_loader, test_loader = process_data(text, block_size=10, batch_size=32)
    model = Model(vocab_size, n_embed=15, block_size=10, n_hidden=400)
    train_loop(model, train_loader, test_loader, epochs=30, learning_rate=0.001)
    start_text = "the monster"
    text = generate_text(start_text,model=model, n_chars=100)

    print(text)