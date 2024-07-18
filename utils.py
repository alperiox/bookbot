import torch
from torch.nn import functional as F
from tqdm import tqdm
import os

def load_artifacts(*args):
    artifacts = {}
    for key in args:
        artifacts[key] = torch.load(f"artifacts/{key}.pt")
    return artifacts


def save_artifacts(**kwargs):
    os.makedirs("artifacts", exist_ok=True)
    for key, value in kwargs.items():
        torch.save(value, f"artifacts/{key}.pt")


def optimize_step(parameters, learning_rate=0.001):
    """optimize the model parameters just once using SGD."""
    for param in parameters:
        param.data += -learning_rate * param.grad

    for param in parameters:
        param.grad = None

def train_loop(model, train_loader, test_loader, epochs, learning_rate, lrsche):
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
    parameters = model.parameters()
    for p in parameters:
        p.requires_grad = True

    # loss vectors
    train_losses = torch.zeros(epochs)
    valid_losses = torch.zeros(epochs)

    # the initial learning_rate
    lr = learning_rate

    # training loop
    for epoch in range(epochs):
        # set up the tqdm bar
        bar = tqdm(enumerate(train_loader), total=len(train_loader))
        # decay the learning rate if it's provided
        if lrsche:
            if epoch == int(epochs * 0.5):
                lr = lr / 10
        print("========")
        print("TRAINING (epoch:%d/%d)" % (epoch + 1, epochs))
        for i, (x, y) in bar:
            # get the logits and the loss
            logits, loss = model(x, y)
            # backward pass
            loss.backward()
            optimize_step(parameters, lr)
            loss = loss.item()
            # statistics and loggign
            train_losses[epoch] += loss
            desc_text = f"({epoch*train_loader.batch_size + i*train_loader.batch_size}/{len(train_loader.dataset)}) (lr={lr:.4f}): loss {train_losses.sum()/(i+1):.4f}"
            bar.set_description(desc_text)

        print("TESTING")
        bar = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, (x, y) in bar:
            with torch.no_grad():
                # same as above, don't calculate the gradients this time
                # and just calculate the loss
                logits, loss = model(x, y)
            # statistics and logging, again.
            valid_losses[epoch] += loss
            desc_text = f"({epoch*test_loader.batch_size + i*test_loader.batch_size}/{len(test_loader.dataset)}): loss {valid_losses.sum()/(i+1):.4f}"
            bar.set_description(desc_text)
    # return the losses.
    return train_losses, valid_losses
