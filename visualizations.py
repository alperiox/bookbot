# TODO: a simple baseline where the probability of picking any token as the next token is uniform
# TODO: layer output distributions (histogram)
# TODO: layer output heatmap? to take a look at the absolute values of activations and stuff?
# TODO: loss graphs
# TODO: layer gradient means
# TODO: layer gradient stds
# TODO: ratio of amount of change in the parameters given the weights
#       we multiply the learning rate with the layer's gradient's std and divide it by
#       parameters' std. this ratio will be higher if gradient std is larger (grads vary too much from the mean)
#       and the params are smaller in comparison.
# TODO: layer output distributions (there's a ready-to-use function in lesson3 dev)
# TODO: layer grad distributions (there's a ready-to-use function in lesson3 dev)
# TODO: ratio of the gradient of a specific layer to its input
#       so if the ratio is too high, it means that the gradients are too high wtr to the input
#       and we actually want constant but smaller updates throughout the network to not miss any local minimas et.c
# TODO: ratio of the amount of change vs the weights, the stats should be saved in L7 here
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

from net import GPT


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


# 1: Baseline
def get_baseline_score(model):
    """
    Calculate the baseline loss for the model.

    This function computes a baseline loss assuming uniform probability
    distribution over the vocabulary.

    Args:
    model: The GPT model instance

    Returns:
    float: The baseline loss
    """
    baseline_loss = (-torch.log(torch.tensor(1 / model.vocab_size))).item()
    print("BASELINE LOSS:", baseline_loss)
    return baseline_loss


# 2: Layer output distributions as histograms (specific to the model)


def get_layer_output_histograms(
    model: GPT,
    sample_input: torch.Tensor = None,
    save_affix: str = "pretraining",
    save_path: str = "artifacts",
):
    """
    Generate histograms of layer outputs for a given model.

    This function creates histograms for various layer outputs of the model,
    including embeddings, block outputs, and logits.

    Args:
    model (GPT): The GPT model instance
    sample_input (torch.Tensor, optional): Input tensor to use. If None, random input is generated.

    Returns:
    None. Displays the histograms using matplotlib.
    """
    if model.__class__.__name__ != "GPT":
        print(
            NotImplementedError(
                "Plotting layer outputs are only available for GPT models for now."
            )
        )
        return

    if sample_input is None:
        sample_input = torch.randint(
            0, model.vocab_size - 1, (1, model.block_size), dtype=torch.long
        )
    layer_outputs = []
    # get the embeddings
    emb1 = model.token_embeddings_table(sample_input)
    # positional embeddings
    emb2 = model.pos_embeddings_table(torch.arange(model.block_size))
    # combine the embeddings
    emb = emb1 + emb2
    # pass the embeddings through the model
    block_outputs = model.blocks(emb)
    lnf_out = model.ln_f(block_outputs)
    logits = model.ln_head(lnf_out)

    # store the layer outputs and their titles
    layer_outputs = [emb1, emb2, emb, block_outputs, lnf_out, logits]
    plot_titles = [
        "Token embd",
        "Pos embd",
        "Combined emb",
        "Block outputs",
        "Layer norm",
        "Language head",
    ]

    # generate 2x3 subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    path = Path(save_path)

    # set the plot titles and plot the layer outputs according to subplots
    for title, output, ax in zip(plot_titles, layer_outputs, axs.flatten()):
        ax.set_title(title)
        ax.hist(output.view(-1).tolist(), 50)
    fig.savefig(path / f"layer_output_histograms_{save_affix}.png")

    # now get into the decoder transformer blocks (dtb)
    out = emb
    for i, dtb in enumerate(model.blocks.layers):
        # get the outputs of the decoder transformer block
        first_ln = dtb.ln1(out)  # layer norm
        self_attn_outs = dtb.self_attn(first_ln)  # self attention
        first_res = out + self_attn_outs  # residual connection
        second_ln = dtb.ln2(first_res)  # layer norm
        mlp_outs = dtb.ffwd_net(second_ln)  # feedforward network
        out = first_res + mlp_outs  # second residual connection
        outputs = [first_ln, self_attn_outs, first_res, second_ln, mlp_outs, out]

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        for title, output, ax in zip(plot_titles, outputs, axs.flatten()):
            ax.set_title(title)
            ax.hist(output.view(-1).tolist(), 50)

        fig.suptitle(f"DecoderTransformerBlock {i+1}")
        fig.savefig(path / f"layer_output_histograms_{save_affix}_dtb_{i+1}.png")


# 3: layer output heatmaps

# plot the heatmaps
# we'll use emb1, emb2 and emb


def plot_emb_weights(
    model: GPT,
    plot_text: bool = False,
    save_affix: str = "pretraining",
    save_path: str = "artifacts",
):
    """
    Plot heatmaps of the embedding weights.

    This function creates heatmaps for positional and token embedding weights,
    applying PCA to reduce dimensionality if necessary.

    Args:
    model (GPT): The GPT model instance
    plot_text (bool, optional): If True, overlay weight values on the heatmap. Defaults to False.

    Returns:
    None. Displays the heatmaps using matplotlib.
    """
    if model.__class__.__name__ != "GPT":
        print(
            NotImplementedError(
                "Plotting embedding weights are only available for GPT models for now."
            )
        )
        return

    fig, axs = plt.subplots(2, 1, figsize=(15, 5))
    path = Path(save_path)
    weights = [model.pos_embeddings_table.weight, model.token_embeddings_table.weight]
    # apply PCA to the weights
    pca = PCA(n_components=1)  # reduce to 1D
    # apply PCA to the weights
    weights = [pca.fit_transform(weight.detach().numpy()) for weight in weights]

    titles = ["Positional embeddings (weight)", "Token embeddings (weight)"]

    for title, weight, ax in zip(titles, weights, axs.flatten()):
        ax.set_title(title)
        ax.imshow(weight.T, cmap="gray", interpolation="nearest")
        if plot_text:
            for x in range(weight.shape[0]):
                for y in range(weight.shape[1]):
                    ax.text(
                        x,
                        y,
                        f"{weight[x, y]:.2f}",
                        color="red",
                        ha="center",
                        va="center",
                    )

    plt.suptitle("Embedding heatmaps")
    fig.savefig(path / f"embedding_heatmaps_{save_affix}.png")

    # attention heatmaps
    # we'll use the attention heads' outputs from the decoder transformer blocks


def plot_attn_heatmaps(
    model: GPT,
    sample_input: torch.Tensor = None,
    plot_text: bool = False,
    save_affix: str = "pretraining",
    save_path: str = "artifacts",
):
    """
    Plot heatmaps of attention weights for each layer and head.

    This function visualizes the attention weights for each attention head
    in each layer of the model.

    Args:
    model (GPT): The GPT model instance
    sample_input (torch.Tensor, optional): Input tensor to use. If None, random input is generated.
    plot_text (bool, optional): If True, overlay attention values on the heatmap. Defaults to False.

    Returns:
    None. Displays the heatmaps using matplotlib.
    """
    if sample_input is None:
        sample_input = torch.randint(
            0, model.vocab_size - 1, (1, model.block_size), dtype=torch.long
        )

    attention_outputs = []
    out = model.token_embeddings_table(sample_input) + model.pos_embeddings_table(
        torch.arange(model.block_size)
    )
    for i, dtb in enumerate(model.blocks.layers):
        k = out @ dtb.self_attn.key
        q = out @ dtb.self_attn.query
        w = (q @ k.transpose(-2, -1)) * (
            dtb.self_attn.head_size**-0.5
        )  # (num_heads, T, T)
        w = w.detach()  # (num_heads, T, T)

        attention_outputs.append(w)

        out = dtb(out)

    attention_outputs = [
        [w[head_ix, :, :] for head_ix in range(model.num_heads)]
        for w in attention_outputs
    ]

    # plot the heatmaps
    font_size = 2 * model.num_heads
    figsize = (int(7.5 * model.num_heads), int(5 * model.num_heads))
    path = Path(save_path)

    fig, axs = plt.subplots(len(model.blocks.layers), model.num_heads, figsize=figsize)
    for i, dtb_attn_out in enumerate(attention_outputs):
        for j, attn_head_out in enumerate(dtb_attn_out):
            # normalize the attention weights
            plot_data = attn_head_out.detach().abs()
            plot_data /= plot_data.max()
            # plot the heatmap
            axs[i, j].imshow(plot_data.numpy(), cmap="gray", interpolation="nearest")
            axs[i, j].set_title(f"DecoderTransformerBlock {i+1} Head {j+1}")
            # text annotations
            if plot_text:
                for x in range(plot_data.size(0)):
                    for y in range(plot_data.size(1)):
                        axs[i, j].text(
                            y,
                            x,
                            f"{plot_data[x, y] * 100:.0f}",
                            color="red",
                            ha="center",
                            va="center",
                            fontsize=font_size,
                        )

    plt.suptitle("Attention heatmaps")
    fig.savefig(path / f"attention_heatmaps_{save_affix}.png")
