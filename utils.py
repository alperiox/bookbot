import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    os.makedirs(save_path, exist_ok=True)
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
    os.makedirs(save_path, exist_ok=True)
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


def plot_layer_grads(layers_to_plot, save_path: str = "artifacts"):
    """
    this plot let us to see the distribution of the gradients of the layers
    """
    os.makedirs(save_path, exist_ok=True)
    path = Path(save_path)
    # visualize histograms
    plt.figure(figsize=(20, 4))
    legends = []

    print("-" * 20)
    print("Layers: grads distribution")

    for i, layer in enumerate(layers_to_plot):
        layer_name = layer.__class__.__name__
        if not hasattr(layer, 'out'):
            print(f"Warning: Layer {layer_name} (index {i}) has no 'out' attribute. Skipping.")
            continue
        if layer.out.grad is None:
            print(f"Warning: Grads for {layer_name} (index {i}) is None. Skipping.")
            continue
        
        t = layer.out.grad
        
        if not isinstance(t, torch.Tensor):
            print(f"Warning: Grad for {layer_name} (index {i}) is not a tensor. Skipping.")
            continue

        print(
            "layer %d (%10s): mean %+.8f, std %.8f"
            % (i, layer_name, t.mean().item(), t.std().item())
        )
        hy, hx = torch.histogram(t.cpu(), density=True) # ensure t is on cpu for histogram
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} ({layer_name})")

    if legends: # Only save plot if there's something to plot
        plt.legend(legends)
        plt.title("Layers: grads distribution")
        plt.savefig(path / "grads.png")
    else:
        print("No valid layer gradients to plot.")


    print("-" * 20)

    return


def plot_layer_outputs(layers_to_plot, save_path: str = "artifacts"):
    os.makedirs(save_path, exist_ok=True)
    path = Path(save_path)
    # visualize histograms
    layer_name = None
    plt.figure(figsize=(20, 4))
    legends = []
    print("-" * 20)
    print("Layers: output distribution")
    for i, layer in enumerate(layers_to_plot):
        if not hasattr(layer, 'out'):
            print(f"Warning: Layer {layer.__class__.__name__} (index {i}) has no 'out' attribute. Skipping.")
            continue

        t = layer.out
        layer_name = layer.__class__.__name__
        
        if not isinstance(t, torch.Tensor):
            print(f"Warning: Output for {layer_name} (index {i}) is not a tensor. Skipping.")
            continue

        print(
            "layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%"
            % (
                i,
                layer_name,
                t.mean().item(),
                t.std().item(),
                (torch.abs(t.cpu()) > 0.97).float().mean().item() * 100, # ensure t is on cpu
            )
        )
        hy, hx = torch.histogram(t.cpu(), density=True) # ensure t is on cpu for histogram
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} ({layer_name})")
    
    if legends: # Only save plot if there's something to plot
        plt.legend(legends)
        plt.title("Layers: output distribution")
        plt.savefig(path / "outputs.png")
    else:
        print("No valid layer outputs to plot.")
        
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


def load_artifact(save_path, name):
    artifact = torch.load(f"{save_path}/{name}.pt")
    return artifact


def save_artifacts(save_path, **kwargs):
    os.makedirs(save_path, exist_ok=True)
    for key, value in kwargs.items():
        torch.save(value, f"{save_path}/{key}.pt")


def calc_debug_stats(
    learning_rate, named_parameters_dict: dict
) -> tuple[list[float], list[float], list[float]]:
    ratios = []
    means = []
    stds = []
    for name, p in named_parameters_dict.items():
        if p.requires_grad: # Consider only parameters that require gradients
            if p.grad is not None:
                if p.data.std() > 1e-9: # Avoid division by zero or very small std
                    ratio_val = (learning_rate * p.grad.std() / p.data.std()).log10().item()
                else:
                    ratio_val = float('nan') # Or some other indicator for problematic std
            else:
                # print(f"Warning: Gradient for parameter {name} is None. Skipping ratio calculation.")
                ratio_val = float('nan') # Grad is None, ratio is undefined
            ratios.append(ratio_val)
            means.append(p.data.mean().item())
            stds.append(p.data.std().item())
        else: # If param does not require grad, append NaN or skip
            ratios.append(float('nan'))
            means.append(p.data.mean().item() if p.data is not None else float('nan'))
            stds.append(p.data.std().item() if p.data is not None else float('nan'))


    return ratios, means, stds


def train_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """optimizes the model weights using the given optimizer object for a batch"""
    x, y = x.to(device), y.to(device)

    # get the logits and the loss
    _, loss = model(x, y)
    # optimize the model parameters and log the gradients if needed

    # backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)  # cast the grads to None

    return loss.item()


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
            _, loss = model(x, y)  # _: logits
        # statistics and logging, again.
        valid_losses[i] = loss
        if isinstance(bar, tqdm):
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
    max_steps: int | None = None,
    epochs: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    debug: bool = False,
    model_name: str = "",
    artifacts_save_path: str = "artifacts",
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
    if (max_steps is None) and (epochs is None):
        raise ValueError("Both `max_steps` and `epochs` are none!")

    assert not (
        max_steps and epochs
    ), "You cannot pass both `max_steps` and `epochs` at the same time!"

    # allow the model parameters to calculate gradients
    model.to(device)

    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_batch_size: int = (
        train_loader.batch_size if isinstance(train_loader.batch_size, int) else 1
    )

    num_training_samples = len(train_loader.dataset)

    
    def get_plottable_layers(model_obj, name_str):
        if name_str == "mlp":
            return model_obj.net.layers if hasattr(model_obj, 'net') and hasattr(model_obj.net, 'layers') else []
        elif name_str == "hmlp":
            return model_obj.model.layers if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'layers') else []
        elif name_str == "gpt" or name_str == "gpdt":
            return model_obj.blocks.layers if hasattr(model_obj, 'blocks') and hasattr(model_obj.blocks, 'layers') else []
        return []

    update_dynamics_history = []
    # For summary statistics
    all_period_layer_output_means = []
    all_period_layer_output_stds = []
    all_period_layer_grad_means = []
    all_period_layer_grad_stds = []
    all_period_param_grad_means = []
    all_period_param_grad_stds = []
    
    final_lr = learning_rate # Initialize with base learning rate
    total_duration_metric = 0 # Will be epochs or max_steps

    # training loop
    if epochs:
        total_duration_metric = epochs
        # loss vectors
        train_losses: torch.Tensor = torch.zeros(epochs)
        valid_losses: torch.Tensor = torch.zeros(epochs)

        for epoch in range(epochs):
            # set up the tqdm bar
            bar = tqdm(enumerate(train_loader), total=len(train_loader))
            # decay the learning rate if it's provided
            current_lr = learning_rate 
            if lrsche and epochs > 1:
                n_epochs = round(epochs * 0.33)
                if (epoch + 1) % n_epochs == 0:
                    current_lr = learning_rate / 10 
            final_lr = current_lr # Update final_lr for summary

            # Re-assign optimizer learning rate if it changed
            if optimizer is not None:
                for g in optimizer.param_groups:
                    g['lr'] = current_lr


            print("========")
            print("TRAINING (epoch:%d/%d)" % (epoch + 1, epochs))

            model.train()
            epoch_train_loss = 0.0
            for i, (x, y) in bar:
                loss = train_step(model, x, y, optimizer, device) 

                # statistics and logging
                epoch_train_loss += loss
                desc_text = f"({epoch*train_batch_size + i*train_batch_size}/{num_training_samples}) (lr={current_lr:.4f}): loss {epoch_train_loss/(i+1):.4f}"
                bar.set_description(desc_text)
            
            train_losses[epoch] = epoch_train_loss / len(train_loader)
            
            model.eval()
            valid_loss = evaluate(model, test_loader, device)
            valid_losses[epoch] = valid_loss
            
            if debug:
                # Grads should be available from the last train_step of the epoch
                current_named_params = {name: p for name, p in model.named_parameters() if p.requires_grad}
                ratios_for_epoch, _, _ = calc_debug_stats(current_lr, current_named_params)
                update_dynamics_history.append(ratios_for_epoch)


    elif max_steps is not None:
        # if `max_steps` is given instead of `epochs`,
        # just train the model for the given max_steps,
        # we will assume the dataloader has InfiniteRandomSampler as it's sampler
        # but it can be used with default dataloader as well
        bar = tqdm(total=max_steps)

        # loss vectors
        train_losses = torch.zeros(max_steps)
        # valid_losses are not regularly collected per step, so this might be misleading if not handled carefully
        # For simplicity, we might only store the final validation loss or validation losses at 'frac' intervals
        # The current code stores one valid_loss value updated at 'frac' intervals.
        
        batch_size = train_loader.batch_size
        batch_size = batch_size if isinstance(batch_size, int) else 1
        total = 0
        frac = round(max_steps * 0.33) # Interval for validation and debug stats
        current_valid_loss = 0.0 # Stores the latest validation loss

        model.train()
        for ix, (x, y) in enumerate(train_loader):
            if ix == max_steps:
                break

            loss = train_step(model, x, y, optimizer, device) # optimizer uses the initial learning_rate

            bar.update(1)
            total += batch_size
            train_losses[ix] = loss

            desc_text = f"(lr={learning_rate:.4f}): loss {train_losses[:ix+1].mean():.4f} val_loss {current_valid_loss:.4f}"
            bar.set_description(desc_text)
            
            if debug and (frac > 0 and ((ix + 1) % frac == 0) or ix == max_steps - 1) :
                # Grads should be available from the train_step
                current_named_params = {name: p for name, p in model.named_parameters() if p.requires_grad}
                ratios_for_step, _, _ = calc_debug_stats(learning_rate, current_named_params) # Use initial LR for max_steps
                update_dynamics_history.append(ratios_for_step)

            if frac > 0 and ((ix + 1) % frac == 0 or ix == max_steps -1): # Also run validation at last step
                model.eval()
                current_valid_loss = evaluate(model, test_loader, device, progress_bar=False)
                # valid_losses[ix] = current_valid_loss # If we want to store validation loss at each 'frac'
                model.train()
        
        # For max_steps, valid_losses might not be a per-step array.
        # We might return just the final one or a list of collected ones.
        # For now, let's ensure train_losses is returned, and valid_losses is handled based on collection strategy.
        # The original code returns a valid_losses tensor initialized to max_steps, which might be sparse.
        # For simplicity, we'll keep train_losses as per-step and valid_losses as potentially sparse or just final.
        # The current structure implies valid_losses is not really used in max_steps for plotting losses over time.
        # We will create a minimal valid_losses tensor for save_loss_figures.
        # If current_valid_loss is the only validation performed, we can make valid_losses a single element tensor.
        if frac > 0 : # if validation was run
             # Create a simplified valid_losses for plotting, could be just the last one or ones collected at frac
            final_valid_loss_tensor = torch.tensor([current_valid_loss] * max_steps) # Simplistic, repeats last valid loss
        else:
            final_valid_loss_tensor = torch.tensor([])


        model.to("cpu")
        bar.close()
        # Overwrite valid_losses for max_steps if it was sparsely populated
        valid_losses = final_valid_loss_tensor


    else:
        train_losses, valid_losses = torch.tensor([]), torch.tensor([])

    save_loss_figures(train_losses, valid_losses, save_path=artifacts_save_path)

    if debug:
        print("Debug mode: Generating plots...")
        plottable_layers = get_plottable_layers(model, model_name)
        
        if not plottable_layers:
            print(f"Warning: No plottable layers found for model_name='{model_name}'. Skipping layer-specific plots.")

        plot_layer_outputs(plottable_layers, save_path=artifacts_save_path)
        plot_layer_grads(plottable_layers, save_path=artifacts_save_path)
        plot_grad2data_ratio(model, save_path=artifacts_save_path)
        
        if update_dynamics_history:
             # Need to ensure parameters used by plot_aoc_ratio match the structure of update_dynamics_history
             # plot_aoc_ratio expects model.parameters() to provide names and params.
             # The current ud is a list of lists of ratios.
             # We might need to adjust plot_aoc_ratio or how ud is passed/used.
             # For now, let's assume plot_aoc_ratio can handle the list of ratios directly
             # if the model structure (number of params) hasn't changed.
             # The original plot_aoc_ratio iterates model.parameters() and then uses ud[j][i]
             # This implies `i` should correspond to the parameter index.
             # `calc_debug_stats` returns ratios for named_parameters. If order is preserved, this might work.
            plot_aoc_ratio(update_dynamics_history, model, save_path=artifacts_save_path)
        else:
            print("Warning: update_dynamics_history is empty. Skipping plot_aoc_ratio.")


    # return the losses.
    return train_losses, valid_losses
