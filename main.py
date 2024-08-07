# to turn this into a cli, we need an argument parser
import argparse
import os

import torch

from net import GPT, MLP, HierarchicalMLP
from processors import CharLevelMLPProcessor, GPTProcessor
from tokenizers import CharTokenizer
from utils import (
    get_baseline_score,
    load_artifact,
    plot_aoc_ratio,
    plot_grad2data_ratio,
    plot_layer_grads,
    plot_layer_outputs,
    save_artifacts,
    save_loss_figures,
    train_loop,
)

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
parser.add_argument("--block_size", type=int, default=16, help="Block size")
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
    "--max_new_tokens", type=int, default=100, help="Number of characters to generate"
)
parser.add_argument(
    "--model",
    type=str,
    default="gpt",
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
parser.add_argument(
    "--num_heads",
    type=int,
    default=3,
    help="Number of self-attention heads for the MultiHead self-attention implementation",
)
parser.add_argument(
    "--num_blocks",
    type=int,
    default=2,
    help="Number of decoder transformer blocks in GPT implementation",
)
parser.add_argument("--context", type=str, help="Starting text for generation")
parser.add_argument(
    "--save_path",
    type=str,
    help="The path to save the artifacts. Default is 'artifacts'",
    default="artifacts",
)
parser.add_argument(
    "--device",
    type=str,
    help="The device to train the models on. default: cpu",
    default="cpu",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Print debug statistics and save statistical plots on artifacts folder",
    default=False,
)
parser.add_argument(
    "--max_steps",
    help="Train the model for given number of steps instead of full epochs.",
    default=None,
    type=int,
)
args = parser.parse_args()
args = vars(args)

if __name__ == "__main__":
    if args["generate"]:
        # we need to load the model and start generating text

        # check if there's a pretrained model and tokenizer
        assert os.path.exists(f"{args['save_path']}/model.pt"), "Model not found"
        assert os.path.exists(
            f"{args['save_path']}/tokenizer.pt"
        ), "Tokenizer not found"
        # load the model and the tokenizer
        model = load_artifact(args["save_path"], "model")
        tokenizer = load_artifact(args["save_path"], "tokenizer")
        # tokenize the context
        context = tokenizer.encode(args["context"])
        # if the model is hmlp, we should pad the input tokens in case they will be less than the context length
        # it's because the hierarchical architecture makes use of the context length, so it requires inputs to be at least the `context length`-sized vectors
        if args["model"] == "hmlp":
            context = tokenizer.pad(context, length=model.block_size)
        context = torch.tensor(context, dtype=torch.long)
        # if the model has special tokens and tokenizer has special_token_mappings attrs
        # then map the special token names to the special token's integer value.
        if hasattr(model, "special_tokens") and hasattr(
            tokenizer, "special_token_mappings"
        ):
            # rather complex mapping for special_token_name -> token_char -> token_ix for each special token
            model.special_tokens = {
                k: tokenizer.special_tokens[v]
                for k, v in tokenizer.special_token_mappings.items()
            }
        # generate the output tokens
        output_tokens = model.generate(
            idx=context, max_new_tokens=args["max_new_tokens"]
        )
        # decode the output and join to the text
        text = "".join(tokenizer.decode(output_tokens.tolist())[0])
        # strip the special tokens
        for k in tokenizer.special_tokens:
            text = text.strip(k)
        print(text)  # print the generated text
    else:
        # load the corresponding tokenizer and the processor for different models
        # their behavior is rather model-agnostic, one might want to check out
        # the source code to understand how they work internally.
        if args["model"] in ["hmlp", "mlp"]:
            tokenizer = CharTokenizer()
            processor = CharLevelMLPProcessor(
                paths=args["file"],
                tokenizer=tokenizer,
                context_length=args["block_size"],
            )
        elif args["model"] == "gpt":
            tokenizer = CharTokenizer()
            # I didn't use these in my first implementation, maybe later.
            tokenizer.BOS_TOKEN = ""
            tokenizer.EOS_TOKEN = ""
            tokenizer.special_tokens = {}

            processor = GPTProcessor(
                paths=args["file"],
                tokenizer=tokenizer,
                context_length=args["block_size"],
            )

        # get the data loaders
        train_loader, test_loader = processor.get_dataloaders(
            args["batch_size"], args["train_ratio"]
        )
        # the vocabulary size is calculated in tokenizer already.
        vocab_size = tokenizer.vocab_size
        # load the model
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
        elif args["model"] == "gpt":
            model = GPT(
                n_embd=args["n_embed"],
                vocab_size=vocab_size,
                num_heads=args["num_heads"],
                num_blocks=args["num_blocks"],
                block_size=args["block_size"],
            )

        else:
            raise NotImplementedError("Available models: MLP, HierarchicalMLP, GPT.")

        # get the baseline score
        # print("sample input: ", sample_input[: model.block_size])
        # sample_input = torch.tensor(
        #     tokenizer.encode(sample_input[: model.block_size]), dtype=torch.long
        # )

        if args["debug"]:
            get_baseline_score(tokenizer.vocab_size)
            model.get_layer_output_histograms(
                save_affix="pretraining",
                save_path=args["save_path"],
            )
            model.plot_emb_weights(
                save_affix="pretraining",
                plot_text=False,
                save_path=args["save_path"],
                tokenizer=tokenizer,
            )
            if args["model"] == "gpt":
                model.plot_attn_heatmaps(
                    save_affix="pretraining",
                    plot_text=False,
                    save_path=args["save_path"],
                )

            print("VOCABULARY:")
            print(tokenizer.vocabulary)

        # train the model
        train_losses, valid_losses, ratios, means, stds = train_loop(
            model,
            train_loader,
            test_loader,
            epochs=args["epochs"],
            learning_rate=args["lr"],
            lrsche=args["lrsche"],
            device=args["device"],
            debug_stats=args["debug"],
        )

        if args["debug"]:
            plot_layer_outputs(model)
            plot_layer_grads(model)
            plot_grad2data_ratio(model)
            plot_aoc_ratio(ratios, model)

            save_loss_figures(train_losses, valid_losses, save_path=args["save_path"])
            model.get_layer_output_histograms(
                save_affix="results",
                save_path=args["save_path"],
            )
            model.plot_emb_weights(
                save_affix="results", plot_text=False, save_path=args["save_path"]
            )
            if args["model"] == "gpt":
                model.plot_attn_heatmaps(
                    save_affix="results",
                    plot_text=False,
                    save_path=args["save_path"],
                )
        # save the results
        save_artifacts(
            model=model,
            tokenizer=tokenizer,
            train_losses=train_losses,
            valid_losses=valid_losses,
            train_loader=train_loader,
            test_loader=test_loader,
            save_path=args["save_path"],
        )

        print("-" * 50)
        print("TRAINED THE MODEL!")
        print("-" * 50)
