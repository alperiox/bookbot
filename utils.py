import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
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
    path = Path(save_path)
    os.makedirs(path, exist_ok=True)
    
    fig = plt.figure(figsize=(15, 7)) # Adjusted figure size
    parameters = model.parameters()
    legends = []
    
    num_batches = len(ud)
    if num_batches == 0:
        print("Warning: No data provided for plot_aoc_ratio. Skipping plot.")
        plt.close(fig)
        return

    for i, (param_name, p) in enumerate(parameters.items()):
        if p.ndim == 2: # Plot only for 2D parameters (typically weight matrices)
            # Ensure that 'i' is a valid index for ud[j]
            # This assumes 'ud' rows correspond to all parameters iterated by model.parameters()
            # and that each ud[j] has ratios for all parameters.
            try:
                ratios_for_param_i = [ud_batch[i] for ud_batch in ud if i < len(ud_batch)]
                if ratios_for_param_i: # Only plot if there's data for this parameter
                    plt.plot(range(len(ratios_for_param_i)), ratios_for_param_i, label=param_name)
                    legends.append(param_name) # This list is actually not used for legend if label is set in plot
            except IndexError:
                print(f"Warning: Index {i} out of bounds for a batch in 'ud' for parameter {param_name} in plot_aoc_ratio.")
                continue


    plt.plot(
        [0, num_batches], [-3, -3], "k--", label="1e-3 threshold (log10 scale)"
    ) 
    # Corrected legend call
    if legends: # Only show legend if there are items to show
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.xlabel("Batch Index")
    plt.ylabel("log10(Update-to-Data Ratio)")
    plt.title("Log10 Update-to-Data Ratio per Parameter Over Batches")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for external legend
    
    filename = path / "amount_of_change_log_ratio.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Amount of Change (log ratio) plot to {filename}")


def plot_parameter_gradient_distributions(model, save_path: str = "artifacts"):
    """
    Plots the distribution of gradients for each model parameter (typically weights with ndim==2).
    Also prints statistics including the ratio of the standard deviation of gradients
    to the standard deviation of parameter values for each parameter.
    """
    path = Path(save_path)
    os.makedirs(path, exist_ok=True)

    fig = plt.figure(figsize=(15, 7)) # Adjusted figure size
    parameters = model.parameters()
    legends = []
    
    print("-" * 20)
    print("Parameter Gradient Statistics & Distributions:")
    plotted_anything = False
    for i, (param_name, p) in enumerate(parameters.items()):
        if p.grad is None:
            print(f"Parameter '{param_name}': No gradient available (p.grad is None). Skipping.")
            continue
        
        t = p.grad.detach().cpu() # Gradient tensor
        param_data = p.data.detach().cpu() # Parameter tensor

        # Plot only for 2D parameters (typically weight matrices) for the histogram
        if p.ndim == 2:
            print(
                "Parameter %s (%s) | grad_mean %+f | grad_std %e | data_std %e | grad_std/data_std ratio %e"
                % (
                    param_name,
                    tuple(p.shape),
                    t.mean(),
                    t.std(),
                    param_data.std(),
                    t.std() / (param_data.std() + 1e-9), # Added epsilon to prevent div by zero if param_data.std() is 0
                )
            )
            if t.numel() > 0 : # Ensure tensor is not empty
                hy, hx = torch.histogram(t, density=True)
                plt.plot(hx[:-1].numpy(), hy.numpy(), label=f"{param_name} {tuple(p.shape)}")
                legends.append(f"{param_name} ({tuple(p.shape)})") # This list is actually not used for legend if label is set in plot
                plotted_anything = True
            else:
                print(f"Parameter '{param_name}': Gradient tensor is empty. Skipping histogram.")
        else:
             print(
                "Parameter %s (%s) | grad_mean %+f | grad_std %e | data_std %e | grad_std/data_std ratio %e (Not plotted)"
                % (
                    param_name,
                    tuple(p.shape),
                    t.mean(),
                    t.std(),
                    param_data.std(),
                    t.std() / (param_data.std() + 1e-9),
                )
            )


    if not plotted_anything:
        print("Warning: No parameter gradient distributions were plotted (e.g. no 2D params with grads).")
        plt.close(fig)
        return

    if legends: # Only show legend if there are items to show
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.xlabel("Gradient Value")
    plt.ylabel("Density")
    plt.title("Parameter Gradient Distributions")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for external legend

    filename = path / "parameter_gradient_distributions.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved parameter gradient distributions plot to {filename}")
    print("-" * 20)


def plot_layer_grads(model, save_path: str = "artifacts", save_affix: str = "results"):
    """
    Plots the distribution of gradients of the '.out' attribute for each layer in model._layers.
    These are typically gradients of activations (outputs of layers), not parameter gradients.
    """
    path = Path(save_path)
    os.makedirs(path, exist_ok=True)

    fig = plt.figure(figsize=(15, 7)) # Adjusted figure size
    legends = []
    plotted_anything = False

    # Ensure model._layers exists and is iterable
    if not hasattr(model, '_layers') or not model._layers:
        print("Warning: model._layers not found or empty in plot_layer_grads. Skipping plot.")
        plt.close(fig)
        return

    print("-" * 20)
    print(f"Activation Gradient Distributions for Layers ({save_affix}):")
    for i, layer in enumerate(model._layers):
        layer_name = layer.__class__.__name__
        
        if not hasattr(layer, 'out') or layer.out is None:
            print(f"Layer {i} ({layer_name}): '.out' attribute missing or None. Skipping.")
            continue
        if not hasattr(layer.out, 'grad') or layer.out.grad is None:
            print(f"Layer {i} ({layer_name}): '.out.grad' attribute missing or None. Skipping.")
            continue

        t = layer.out.grad.detach().cpu()
        if t.numel() == 0:
            print(f"Layer {i} ({layer_name}): '.out.grad' tensor is empty. Skipping.")
            continue
            
        print(
            "Layer %d (%10s): grad_mean %+.8f, grad_std %.8f"
            % (i, layer_name, t.mean().item(), t.std().item())
        )
        hy, hx = torch.histogram(t.numpy(), density=True, bins=50) # Use .numpy() for histogram input
        plt.plot(hx[:-1], hy, label=f"L{i} {layer_name} ({tuple(layer.out.shape)})")
        legends.append(f"L{i} {layer_name} ({tuple(layer.out.shape)})") # This list is actually not used for legend if label is set in plot
        plotted_anything = True

    if not plotted_anything:
        print("Warning: No layer activation gradient distributions were plotted.")
        plt.close(fig)
        return

    if legends: # Only show legend if there are items to show
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.xlabel("Gradient Value of Layer Output")
    plt.ylabel("Density")
    plt.title(f"Layer Activation Gradient Distributions ({save_affix})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for external legend
    
    filename = path / f"layer_activation_gradient_distributions_{save_affix}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved layer activation gradient distributions plot to {filename}")
    print("-" * 20)
    return


# The old plot_layer_outputs function is removed.
# New consolidated function and its helper:

def _plot_histograms_to_file(layers_data: list, filename: Path, suptitle: str):
    """
    Helper function to plot a list of (name, tensor_data) tuples to a specified file.
    Each tuple represents a layer's name and its output tensor.
    """
    num_items = len(layers_data)
    if num_items == 0:
        print(f"No data provided for plotting. File '{filename}' will not be created.")
        return

    # Determine subplot layout (e.g., 3 columns)
    n_cols = 3
    n_rows = (num_items + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, max(5, 4 * n_rows))) # Min height of 5
    if num_items == 1: # If only one plot, axs is not an array
        axs = [axs]
    else:
        axs = axs.flatten()

    print(f"Generating plot with {num_items} histograms for '{filename}'...")
    for i, (item_name, output_tensor) in enumerate(layers_data):
        ax = axs[i]
        if isinstance(output_tensor, torch.Tensor):
            data_for_hist = output_tensor.detach().cpu().view(-1).tolist()
            
            if not data_for_hist: # Check if tensor became an empty list (e.g. tensor had 0 elements)
                print(f"Warning: Output tensor for '{item_name}' is empty. Skipping histogram.")
                ax.set_title(f"{item_name}\n(Empty Output Tensor)")
                ax.text(0.5, 0.5, "Empty Tensor", ha='center', va='center', transform=ax.transAxes)
                continue

            ax.hist(data_for_hist, bins=50, density=True)
            mean_val = output_tensor.mean().item()
            std_val = output_tensor.std().item()
            saturation = (torch.abs(output_tensor) > 0.97).float().mean().item() * 100
            ax.set_title(f"{item_name}\nMean: {mean_val:.2f}, Std: {std_val:.2f}\nSat: {saturation:.2f}%")
        else:
            print(f"Warning: Output for '{item_name}' is not a tensor (type: {type(output_tensor)}). Skipping histogram.")
            ax.set_title(f"{item_name}\n(Output not a Tensor)")
            ax.text(0.5, 0.5, "Data not a Tensor", ha='center', va='center', transform=ax.transAxes)

    # Hide any unused subplots
    for j in range(num_items, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle and avoid overlap
    
    try:
        plt.savefig(filename)
        print(f"Successfully saved plot to '{filename}'")
    except Exception as e:
        print(f"Error saving plot to '{filename}': {e}")
    plt.close(fig) # Close the figure to free memory


def plot_layer_output_histograms(model, save_path: str = "artifacts", save_affix: str = "training"):
    """
    Consolidated function to plot layer output histograms for various model types.
    Assumes that the model has been run in debug mode (model.start_debug() followed by a forward pass)
    so that layer.out attributes are populated.
    """
    path = Path(save_path)
    os.makedirs(path, exist_ok=True)
    
    model_name = model.__class__.__name__
    # Need to import these here to avoid circular dependency if they are at the top level
    # And also to use isinstance checks without importing them globally in utils.py
    from net import GPT, DecoderTransformerBlock, MLP, HierarchicalMLP, Sequential, BaseLayer


    if isinstance(model, GPT):
        # GPT specific logic: Replicate its previous multi-plot behavior.
        gpt_main_outputs = []
        if hasattr(model, 'token_embeddings_table') and hasattr(model.token_embeddings_table, 'out') and model.token_embeddings_table.out is not None:
            gpt_main_outputs.append(("Token Embeddings", model.token_embeddings_table.out))
        if hasattr(model, 'pos_embeddings_table') and hasattr(model.pos_embeddings_table, 'out') and model.pos_embeddings_table.out is not None:
            gpt_main_outputs.append(("Positional Embeddings", model.pos_embeddings_table.out))
        
        # Example for combined embeddings (if GPT model is modified to save it):
        # if hasattr(model, 'combined_embeddings') and hasattr(model.combined_embeddings, 'out') and model.combined_embeddings.out is not None:
        #     gpt_main_outputs.append(("Combined Embeddings", model.combined_embeddings.out))

        if hasattr(model, 'blocks') and hasattr(model.blocks, 'out') and model.blocks.out is not None: # Output of Sequential container
            gpt_main_outputs.append(("All Blocks Output (Sequential)", model.blocks.out))
        if hasattr(model, 'ln_f') and hasattr(model.ln_f, 'out') and model.ln_f.out is not None:
            gpt_main_outputs.append(("Final LayerNorm (ln_f)", model.ln_f.out))
        if hasattr(model, 'ln_head') and hasattr(model.ln_head, 'out') and model.ln_head.out is not None:
            gpt_main_outputs.append(("Language Head (logits)", model.ln_head.out))
        
        if gpt_main_outputs:
            _plot_histograms_to_file(
                gpt_main_outputs,
                path / f"{model_name}_main_outputs_{save_affix}.png",
                f"{model_name} - Main Component Outputs ({save_affix})"
            )

        # Detailed plots for each DecoderTransformerBlock
        if hasattr(model, 'blocks') and hasattr(model.blocks, 'layers'): # model.blocks is Sequential
            for i, dtb_block in enumerate(model.blocks.layers):
                if not isinstance(dtb_block, DecoderTransformerBlock): 
                    continue # Skip if not a DecoderTransformerBlock
                
                dtb_internal_outputs = []
                if hasattr(dtb_block, 'ln1') and hasattr(dtb_block.ln1, 'out') and dtb_block.ln1.out is not None:
                    dtb_internal_outputs.append(("LayerNorm1 (ln1)", dtb_block.ln1.out))
                if hasattr(dtb_block, 'self_attn') and hasattr(dtb_block.self_attn, 'out') and dtb_block.self_attn.out is not None:
                    dtb_internal_outputs.append(("SelfAttention (self_attn)", dtb_block.self_attn.out))
                if hasattr(dtb_block, 'ln2') and hasattr(dtb_block.ln2, 'out') and dtb_block.ln2.out is not None:
                    dtb_internal_outputs.append(("LayerNorm2 (ln2)", dtb_block.ln2.out))
                if hasattr(dtb_block, 'ffwd_net') and hasattr(dtb_block.ffwd_net, 'out') and dtb_block.ffwd_net.out is not None:
                    dtb_internal_outputs.append(("FeedForwardNet (ffwd_net)", dtb_block.ffwd_net.out))
                if hasattr(dtb_block, 'out') and dtb_block.out is not None: # Final output of the block itself
                    dtb_internal_outputs.append(("Block Output (final)", dtb_block.out))

                if dtb_internal_outputs:
                    _plot_histograms_to_file(
                        dtb_internal_outputs,
                        path / f"{model_name}_DecoderBlock{i+1}_internals_{save_affix}.png",
                        f"{model_name} - DecoderTransformerBlock {i+1} Internals ({save_affix})"
                    )
        
    elif isinstance(model, (MLP, HierarchicalMLP)) or (hasattr(model, '_layers') and isinstance(model, BaseLayer)):
        # General approach for models with _layers (populated by MetaClass)
        # or specifically MLP and HierarchicalMLP
        layers_to_plot = []
        q = [(model, model_name)]  # queue of (object_to_inspect, name_prefix_for_its_children)
        
        # Keep track of object IDs whose .out has been added for plotting to avoid duplicates
        # and container objects already traversed for their _layers or .layers.
        plotted_instance_ids = set()
        # traversed_container_ids = set() # May not be needed if plotted_instance_ids handles layers uniquely

        while q:
            current_obj, current_name_prefix = q.pop(0)

            # If current_obj is a layer itself (not the root model object usually) and has an output
            if hasattr(current_obj, 'out') and current_obj.out is not None and current_obj is not model:
                if id(current_obj) not in plotted_instance_ids:
                    # Use current_name_prefix as the layer's name if it's descriptive enough
                    # or fallback to class name if prefix seems like a container name
                    layer_display_name = current_name_prefix 
                    layers_to_plot.append((layer_display_name, current_obj.out))
                    plotted_instance_ids.add(id(current_obj))

            # Determine which attribute holds the sub-layers
            sub_layers_attr = None
            if hasattr(current_obj, '_layers') and current_obj._layers: # Custom _layers from metaclass
                sub_layers_attr = current_obj._layers
            elif isinstance(current_obj, Sequential) and hasattr(current_obj, 'layers') and current_obj.layers: # Sequential's .layers
                sub_layers_attr = current_obj.layers
            # Special handling for MLP's top-level structure if not fully covered by _layers recursion
            elif isinstance(current_obj, MLP) and current_obj is model:
                 # MLP has embedding then net (Sequential)
                 if hasattr(model.embedding, 'out') and model.embedding.out is not None and id(model.embedding) not in plotted_instance_ids:
                     layers_to_plot.append((f"{model_name}.Embedding", model.embedding.out))
                     plotted_instance_ids.add(id(model.embedding))
                 if hasattr(model.net, '_layers') or hasattr(model.net, 'layers'): # model.net is Sequential
                     q.append((model.net, f"{model_name}.SequentialNet"))
                 sub_layers_attr = None # Already handled specific MLP structure

            if sub_layers_attr:
                for i, layer in enumerate(sub_layers_attr):
                    layer_class_name = layer.__class__.__name__
                    # Prefix for sub-layer: if current_obj is the model, use model_name. Else, use current_name_prefix.
                    parent_display_name = model_name if current_obj is model else current_name_prefix
                    # For items in _layers of a Sequential, current_name_prefix might be like "Model.Sequential.Linear_0"
                    # So, the layer itself might be "Model.Sequential.Linear_0.some_sub_layer" if it's a container
                    # If it's a direct layer, its name is what we want.
                    
                    # Try to create a meaningful name
                    # If layer is from _layers of the main model: "model_name.LayerClass_i"
                    # If layer is from a Sequential's .layers: "prefix_to_sequential.LayerClass_i"
                    # If layer is from _layers of a nested BaseLayer: "prefix_to_baselayer.LayerClass_i"
                    descriptive_name = f"{parent_display_name}.{layer_class_name}_{i}"


                    if hasattr(layer, 'out') and layer.out is not None:
                        if id(layer) not in plotted_instance_ids:
                            layers_to_plot.append((descriptive_name, layer.out))
                            plotted_instance_ids.add(id(layer))
                    
                    # If this sub-layer is itself a container (e.g. Sequential, or a BaseLayer with _layers)
                    # add it to the queue to inspect its children.
                    if (hasattr(layer, '_layers') and layer._layers) or \
                       (isinstance(layer, Sequential) and hasattr(layer, 'layers') and layer.layers):
                        if id(layer) not in plotted_instance_ids: # Check if this container itself has been processed as a plottable layer
                             # If it's a container, we want to explore it.
                             # The name prefix for its children will be `descriptive_name`
                             q.append((layer, descriptive_name))
        
        if layers_to_plot:
             _plot_histograms_to_file(
                layers_to_plot,
                path / f"{model_name}_all_layer_outputs_{save_affix}.png",
                f"{model_name} - All Layer Output Histograms ({save_affix})"
            )
        else:
            print(f"No plottable outputs found for {model_name}. Ensure model ran in debug mode and layers populate '.out'.")

    else:
        print(f"Model type {model_name} is not explicitly handled for histogram plotting. Please check model structure or extend this function.")


def save_loss_figures(
    train_losses: torch.Tensor, valid_losses: torch.Tensor, save_path: str = "artifacts"
):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses.tolist(), label="train loss")
    plt.plot(valid_losses.tolist(), label="valid loss")
    plt.legend()
    path = Path(save_path)
    os.makedirs(path, exist_ok=True) # Ensure directory exists
    
    fig_filename = path / "losses.png"
    plt.title("Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_filename)
    plt.close() # Close the figure to free memory
    print(f"Saved loss figures to {fig_filename}")
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


def train_loop(
    model,
    train_loader,
    test_loader,
    epochs,
    learning_rate,
    lrsche,
    device,
    debug_stats=True,
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
        ratios, means_list, stds_list = [], [], []

        model.train()
        model.start_debug()
        for i, (x, y) in bar:
            x, y = x.to(device), y.to(device)
            # get the logits and the loss
            logits, loss = model(x, y)
            # optimize the model parameters and log the gradients if needed
            if debug_stats:
                for l in model._layers:
                    l.out.retain_grad()

            for k, param in parameters.items():
                param.grad = None
            # backward pass
            loss.backward()

            with torch.no_grad():
                for k, param in parameters.items():
                    param.data += -learning_rate * param.grad

                if debug_stats:
                    ratio = [
                        (learning_rate * p.grad.std() / p.data.std()).log10().item()
                        for p in parameters.values()
                    ]
                    means = [p.mean().item() for p in parameters.values()]
                    stds = [p.std().item() for p in parameters.values()]
                else:
                    ratio = None
                    means = None
                    stds = None

            ratios.append(ratio)
            means_list.append(means)
            stds_list.append(stds)
            # log the loss
            loss = loss.item()
            # statistics and logging
            train_losses[epoch] += loss
            desc_text = f"({epoch*train_loader.batch_size + i*train_loader.batch_size}/{len(train_loader.dataset)}) (lr={lr:.4f}): loss {train_losses[epoch]/(i+1):.4f}"
            bar.set_description(desc_text)

        print("TESTING")
        model.eval()
        model.stop_debug()
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
    return train_losses, valid_losses, ratios, means_list, stds_list
