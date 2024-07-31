import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from tqdm import tqdm


def save_loss_figures(
    train_losses: torch.Tensor, valid_losses: torch.Tensor, save_path: str = "artifacts"
):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses.tolist(), label="train loss")
    plt.plot(valid_losses.tolist(), label="valid loss")
    plt.legend()
    path = Path(save_path)
    plt.savefig(path / "losses.png")
    return


def get_baseline_score(vocabulary_size):
    """
    Calculate the baseline loss for the model.

    This function computes a baseline loss assuming uniform probability
    distribution over the vocabulary.

    Args:
    model: The GPT model instance

    Returns:
    float: The baseline loss
    """
    baseline_loss = (-torch.log(torch.tensor(1 / vocabulary_size))).item()
    print("BASELINE LOSS:", baseline_loss)
    return baseline_loss


def debug(func):
    """a decorator to save the object's __call__ outputs to the `out` attribute."""

    def wrapper(obj, *args, **kwargs):
        out = func(obj, *args, **kwargs)
        if hasattr(obj, "log_outputs"):
            if obj.log_outputs:
                obj.out = out
        else:
            obj.out = None
        return out

    return wrapper


def flatten_dict(d: dict) -> dict:
    flattened_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flattened_dict.update(
                {f"{k}.{key}": val for key, val in flatten_dict(v).items()}
            )
        else:
            flattened_dict[k] = v

    return flattened_dict


def load_artifact(save_path, name):
    artifact = torch.load(f"{save_path}/{name}.pt")
    return artifact


def save_artifacts(save_path, **kwargs):
    os.makedirs(save_path, exist_ok=True)
    for key, value in kwargs.items():
        torch.save(value, f"{save_path}/{key}.pt")


def optimize_step(parameters, learning_rate=0.001):
    """optimize the model parameters just once using SGD."""
    for k, param in parameters.items():
        param.data += -learning_rate * param.grad

    for k, param in parameters.items():
        param.grad = None


def train_loop(model, train_loader, test_loader, epochs, learning_rate, lrsche, device):
    """
    Trains the given model using the provided data loaders.

    model: the model to be trained, current available ones are MLP, hMLP, and GPT.
    train_loader (torch DataLoader): the training loader that loads the training batches for the training
    test_loader (torch DataLoader) : the testing loader that loads the data batches for the testing phase
    epochs (int): number of epochs (iterations) to train the model
    learning_rate (float): the learning rate that'll be used to scale the gradients in the optimization phase
    lrsche (bool): whether apply a learning rate decay or not
    """
    # allow the model parameters to calculate gradients

    # loss vectors
    train_losses = torch.zeros(epochs)
    valid_losses = torch.zeros(epochs)

    # the initial learning_rate
    lr = learning_rate

    model.to(device)

    parameters = model.parameters()
    for k, p in parameters.items():
        p.requires_grad = True
    # training loop
    for epoch in range(epochs):
        # set up the tqdm bar
        bar = tqdm(enumerate(train_loader), total=len(train_loader))
        # decay the learning rate if it's provided
        if lrsche:
            n_epochs = int(epochs * 0.3)
            if n_epochs == 0:
                n_epochs = 1
            if (epoch + 1) % n_epochs == 0:
                lr = lr / 10
        print("========")
        print("TRAINING (epoch:%d/%d)" % (epoch + 1, epochs))
        model.train()
        for i, (x, y) in bar:
            x, y = x.to(device), y.to(device)
            # get the logits and the loss
            logits, loss = model(x, y)
            # backward pass
            loss.backward()
            optimize_step(parameters, lr)
            loss = loss.item()
            # statistics and logging
            train_losses[epoch] += loss
            desc_text = f"({epoch*train_loader.batch_size + i*train_loader.batch_size}/{len(train_loader.dataset)}) (lr={lr:.4f}): loss {train_losses[epoch]/(i+1):.4f}"
            bar.set_description(desc_text)

        print("TESTING")
        model.eval()
        bar = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, (x, y) in bar:
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                # same as above, don't calculate the gradients this time
                # and just calculate the loss
                logits, loss = model(x, y)
            # statistics and logging, again.
            valid_losses[epoch] += loss.item()
            desc_text = f"({epoch*test_loader.batch_size + i*test_loader.batch_size}/{len(test_loader.dataset)}): loss {valid_losses[epoch]/(i+1):.4f}"
            bar.set_description(desc_text)

    model.to("cpu")  # move the model to the cpu
    # return the losses.
    return train_losses, valid_losses
