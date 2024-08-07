import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from optimizers import SGD


def plot_aoc_ratio(ud, model, save_path: str = "artifacts"):
    """
    this plot let us to see the ratio of the amount of change in the parameters given the weights.
    this ratio is calculated by multiplying the learning rate with the layer's gradient's standard deviation and dividing it by the layer's parameters' standard deviation
    so this ratio will be higher if gradients are vary too much from the mean (so they're large) and the parameters are too small in comparison
    a usual value for this ratio is 1e-3, so if the ratio is too high, it means that the updates are too large with respect to the weights
    meaning a faster learning, if it's rather lower than 1e-3 that typically means that the updates are too less to make an impact on the weights
    so, consider that we lower the learning rate, so this would result in a slower learning process, meaning ratios of the most layers will probably be less than 1e-3
    so the learning will slow down and the model will learn the patterns in the data more slowly.
    """
    # visualize the ratio of the amount of change vs the weights
    # and this is the ratio of the amount of change
    path = Path(save_path)
    parameters = model.parameters()
    plt.figure(figsize=(20, 4))
    legends = []
    for i, (param, p) in enumerate(parameters.items()):
        if p.ndim == 2:
            plt.plot([ud[j][i] for j in range(len(ud))])
            legends.append(param)

    plt.plot(
        [0, len(ud)], [-3, -3], "k"
    )  # these ratios should be ~1e-3, indicate on the plot (the initial data fed to log10 in the training phase, so -3 means 10^-3)
    # put the legend outside of the plot
    plt.legend(legends, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("update to data ratio")
    plt.savefig(path / "amount of change.png", bbox_inches="tight")


def plot_grad2data_ratio(model, save_path: str = "artifacts"):
    """
    this plot let us to see the ratio of the magnitude of the gradients to the magnitude of the data
    so if the ratio is too big, it means that the gradients are too large with respect to the data so the updates will be too large
    if the ratio is too small, we then expect the step update to be too small to make an impact. While this could still work, we might need to increase the learning rate since
    it'll effect the network's overall learning speed.
    try to train the layers for just 1 epoch and then 1000 full epochs
    in the initial epochs, the ratio of the last layer will be too large compared to the others but it'll decrease (or other's will increase)
    as the training goes on. Of course, this plot is not that informative since what we need is the actual amount of update compared to the layer input.
    """

    # this is the ratio of the gradient of a specific layer to its input.
    # so if the ratio is too high, it means that the gradients are too high with regard to the input so the update will be larger
    # and we actually want constant but smaller updates throughout the network so we don't miss any local minimas etc.
    path = Path(save_path)
    parameters = model.parameters()
    plt.figure(figsize=(20, 4))
    legends = []
    print("-" * 20)
    print("Grad to data ratio")
    for i, (param, p) in enumerate(parameters.items()):
        t = p.grad
        if p.ndim == 2:
            print(
                "weight %10s | mean %+f | std %e | grad:data ratio %e"
                % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std())
            )
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f"{param} ({tuple(p.shape)})")
    plt.legend(legends)
    plt.title("weights gradient to data ratio")
    plt.savefig(path / "grad2data.png", bbox_inches="tight")
    print("-" * 20)


def plot_layer_grads(model, save_path: str = "artifacts"):
    """
    this plot let us to see the distribution of the gradients of the layers
    """
    path = Path(save_path)
    # visualize histograms
    plt.figure(figsize=(20, 4))
    legends = []
    layers = model._layers

    print("-" * 20)
    print("Layers: grads distribution")

    for i, layer in enumerate(layers):
        layer_name = layer.__class__.__name__
        t = layer.out.grad
        assert t is not None, f"Grads for {layer_name} is None!"

        print(
            "layer %d (%10s): mean %+.8f, std %.8f"
            % (i, layer_name, t.mean().item(), t.std().item())
        )
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} ({layer_name})")

    plt.legend(legends)
    plt.title("Layers: grads distribution")
    plt.savefig(path / "grads.png")

    print("-" * 20)

    return


def plot_layer_outputs(model, save_path: str = "artifacts"):
    path = Path(save_path)
    # visualize histograms
    layer_name = None
    plt.figure(figsize=(20, 4))
    legends = []
    layers = model._layers
    print("-" * 20)
    print("Layers: output distribution")
    for i, layer in enumerate(layers):
        t = layer.out
        layer_name = layer.__class__.__name__
        print(
            "layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%"
            % (
                i,
                layer_name,
                t.mean().item(),
                t.std().item(),
                (torch.abs(t) > 0.97).float().mean().item() * 100,
            )
        )
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} ({layer_name})")
    plt.legend(legends)
    plt.title("Layers: output distribution")
    plt.savefig(path / "outputs.png")
    print("-" * 20)
    return


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
            pass
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


def calc_debug_stats(
    learning_rate, parameters
) -> tuple[list[float], list[float], list[float]]:
    ratio = [
        (learning_rate * p.grad.std() / p.data.std()).log10().item()
        for p in parameters.values()
    ]
    means = [p.mean().item() for p in parameters.values()]
    stds = [p.std().item() for p in parameters.values()]

    return ratio, means, stds


def retain_grads(layer):
    layer.out.retain_grad()
    for l in layer._layers:
        retain_grads(l)


def train_step(
    model, parameters, optimizer, lr, x, y, device, debug_stats
) -> tuple[float, list[float] | None, list[float] | None, list[float] | None]:
    """optimizes the model weights using the given optimizer object for a batch"""
    x, y = x.to(device), y.to(device)

    # get the logits and the loss
    logits, loss = model(x, y)
    # optimize the model parameters and log the gradients if needed
    if debug_stats:
        for layer in model._layers:
            retain_grads(layer)

    optimizer.zero_grad()  # cast the grads to None

    # backward pass
    loss.backward()

    with torch.no_grad():
        optimizer.step(lr)

        ratio, means, stds = (
            calc_debug_stats(lr, parameters) if debug_stats else (None, None, None)
        )

    return loss.item(), ratio, means, stds


def evaluate(model, loader, device, progress_bar=True):
    valid_losses = torch.zeros(len(loader))

    bar = (
        tqdm(enumerate(loader), total=len(loader))
        if progress_bar
        else enumerate(loader)
    )

    for i, (x, y) in bar:
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            # same as above, don't calculate the gradients this time
            # and just calculate the loss
            logits, loss = model(x, y)
        # statistics and logging, again.
        valid_losses[i] = loss
        if progress_bar:
            desc_text = (
                f"({(i+1)}/{len(loader)}): loss {valid_losses.sum()/((i+1)):.4f}"
            )
            bar.set_description(desc_text)

    return valid_losses.mean().item()


def train_loop(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    learning_rate: float,
    lrsche: bool,
    device: str,
    debug_stats: bool = True,
    max_steps: int | None = None,
    epochs: int | None = None,
):
    """
    Trains the given model using the provided data loaders.

    model: the model to be trained, current available ones are MLP, hMLP, and GPT.
    train_loader (torch DataLoader): the training loader that loads the training batches for the training
    test_loader (torch DataLoader) : the testing loader that loads the data batches for the testing phase
    epochs (int): number of epochs (iterations) to train the model
    learning_rate (float): the learning rate that'll be used to scale the gradients in the optimization phase
    lrsche (bool): whether apply a learning rate decay or not
    """
    assert not (
        max_steps and epochs
    ), "You cannot pass both `max_steps` and `epochs` at the same time!"
    # allow the model parameters to calculate gradients

    # the initial learning_rate
    lr = learning_rate

    model.to(device)

    parameters = model.parameters()
    for k, p in parameters.items():
        p.requires_grad = True

    optimizer = SGD()
    optimizer.register_params(parameters)

    ratios, means_list, stds_list = [], [], []

    # training loop
    if epochs:
        # loss vectors
        train_losses = torch.zeros(epochs)
        valid_losses = torch.zeros(epochs)

        for epoch in range(epochs):
            # set up the tqdm bar
            bar = tqdm(enumerate(train_loader), total=len(train_loader))
            # decay the learning rate if it's provided
            if lrsche and epochs > 1:
                n_epochs = round(epochs * 0.33)
                if (epoch + 1) % n_epochs == 0:
                    lr = lr / 10
            print("========")
            print("TRAINING (epoch:%d/%d)" % (epoch + 1, epochs))

            model.train()
            model.start_debug()
            for i, (x, y) in bar:
                loss, ratio, means, stds = train_step(
                    model, parameters, optimizer, lr, x, y, device, debug_stats
                )
                ratios.append(ratio)
                means_list.append(means)
                stds_list.append(stds)
                # statistics and logging
                train_losses[epoch] += loss
                desc_text = f"({epoch*train_loader.batch_size + i*train_loader.batch_size}/{len(train_loader.dataset)}) (lr={lr:.4f}): loss {train_losses[epoch]/(i+1):.4f}"
                bar.set_description(desc_text)

            train_losses[epoch] /= len(train_loader)
            model.eval()
            model.stop_debug()
            # valid_loss = evaluate(model, test_loader, device)
            # valid_losses[epoch] = valid_loss

    elif max_steps is not None:
        # if `max_steps` is given instead of `epochs`,
        # just train the model for the given max_steps,
        # we will assume the dataloader has InfiniteRandomSampler as it's sampler
        # but it can be used with default dataloader as well
        bar = tqdm(total=max_steps)

        # loss vectors
        train_losses = torch.zeros(max_steps)
        valid_losses = torch.zeros(max_steps)
        batch_size = train_loader.batch_size
        total = 0
        frac = round(max_steps * 0.33)
        valid_loss = 0

        model.train()
        if debug_stats:
            model.start_debug()

        for ix, (x, y) in enumerate(train_loader):
            if ix == max_steps:
                break

            loss, ratio, means, stds = train_step(
                model, parameters, optimizer, lr, x, y, device, debug_stats
            )

            ratios.append(ratio)
            means_list.append(means)
            stds_list.append(stds)

            bar.update(1)
            total += batch_size

            train_losses[ix] = loss

            desc_text = f"(lr={lr:.4f}): loss {train_losses.sum()/(ix+1):.4f} val_loss {valid_loss:.4f}"
            bar.set_description(desc_text)

            if frac > 1 and ((ix + 1) % frac == 0):
                model.eval()
                model.stop_debug()
                valid_loss = evaluate(model, test_loader, device, progress_bar=False)
                model.train()
                if debug_stats:
                    model.start_debug()

        model.to("cpu")

        bar.close()

    # return the losses.
    return train_losses, valid_losses, ratios, means_list, stds_list
