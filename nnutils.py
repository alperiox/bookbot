import torch
from torch.nn import functional as F
from tqdm import tqdm

from utils import load_artifacts, save_artifacts


def optimize_step(parameters, learning_rate=0.001):
    for param in parameters:
        param.data += -learning_rate * param.grad

    for param in parameters:
        param.grad = None


def evaluate(model, x, y):
    with torch.no_grad():
        outs = model(x)
        loss = F.cross_entropy(outs, y)
    return loss.item()


def train_loop(model, train_loader, test_loader, epochs, learning_rate):
    # a single training loop
    parameters = model.parameters()
    for p in parameters:
        p.requires_grad = True

    train_losses = torch.zeros(epochs)
    valid_losses = torch.zeros(epochs)
    for epoch in range(epochs):
        bar = tqdm(enumerate(train_loader), total=len(train_loader))
        print("========")
        print("TRAINING (epoch:%d/%d)" % (epoch + 1, epochs))
        for i, (x, y) in bar:
            outs = model(x)
            loss = F.cross_entropy(outs, y)
            loss.backward()
            optimize_step(parameters, learning_rate)
            loss = loss.item()

            train_losses[epoch] += loss / len(train_loader)
            desc_text = f"({epoch*train_loader.batch_size + i*train_loader.batch_size}/{len(train_loader.dataset)}): loss {train_losses[epoch]:.4f}"
            bar.set_description(desc_text)

        print("TESTING")
        bar = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, (x, y) in bar:
            loss = evaluate(model, x, y)

            valid_losses[epoch] += loss / len(test_loader)
            desc_text = f"({epoch*test_loader.batch_size + i*test_loader.batch_size}/{len(test_loader.dataset)}): loss {valid_losses[epoch]:.4f}"
            bar.set_description(desc_text)

    save_artifacts(
        model=model,
        train_losses=train_losses,
        valid_losses=valid_losses,
        train_loader=train_loader,
        test_loader=test_loader,
    )


def generate_text(seed_text, model=None, char_to_ix=None, ix_to_char=None, n_chars=100):
    if not model:
        model = load_artifacts("model")["model"]
    if not char_to_ix:
        char_to_ix = load_artifacts("char_to_ix")["char_to_ix"]
    if not ix_to_char:
        ix_to_char = load_artifacts("ix_to_char")["ix_to_char"]

    model.eval()
    with torch.no_grad():
        num_generated_chars = 0
        generated_text = seed_text
        # pad or truncate the seed text to the block size
        if len(generated_text) > model.block_size:
            input_text = generated_text[: model.block_size]
        else:
            input_text = " " * (model.block_size - len(generated_text)) + generated_text

        while num_generated_chars < n_chars and generated_text[-1] != "|":
            input_block = [char_to_ix[char] for char in input_text]
            input_tensor = torch.tensor(input_block, dtype=torch.long).unsqueeze(0)
            out = model(input_tensor)
            # sample the next character
            next_char_ix = torch.multinomial(F.softmax(out, dim=1), 1).item()
            next_char = ix_to_char[next_char_ix]
            generated_text += next_char
            input_text = input_text[1:] + next_char

            num_generated_chars += 1

    return generated_text.strip("|")
