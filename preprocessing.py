import torch
from torch.utils.data import DataLoader, TensorDataset

from fileloaders import AVAILABLE_LOADERS
from utils import save_artifacts


def load_data(filepath: str) -> list:
    """takes in the filepath and loads the data using corresponding loader

    currently supports the following file types:
    - pdf
    - txt

    """
    file_ext = filepath.split(".")[-1].lower()
    loader = AVAILABLE_LOADERS[
        file_ext
    ]  # huggingface's transformers module handles this kind of behavior quiet nicely, might need to take a look
    text = loader(filepath)

    vocab_size = (
        len(set("".join(text).lower())) + 2
    )  # plus one for the beginning and ending token.

    return text, vocab_size


def process_data(input_text: str, block_size: int = 3, batch_size: int = 32):
    """
    given the input text, context length and the batch size,
    returns train and test loaders, saving the vocabulary along the way
    """
    char_to_ix, ix_to_char = prepare_vocabulary(input_text)

    paragraphs = get_paragraphs(input_text)

    blocks, targets = setup_blocks(paragraphs, char_to_ix, block_size)

    train_loader, test_loader = prepare_dataloaders(blocks, targets, batch_size)

    save_artifacts(char_to_ix=char_to_ix, ix_to_char=ix_to_char)

    return train_loader, test_loader


def prepare_vocabulary(input_text: str):
    """
    builds a vocabulary by mapping unique characters in the given text
    """
    character_set = set(input_text)
    char_to_ix = {char: ix for ix, char in enumerate(character_set, 1)}
    ix_to_char = {ix: char for char, ix in char_to_ix.items()}
    # add the stop char too
    char_to_ix["|"] = 0
    ix_to_char[0] = "|"

    return char_to_ix, ix_to_char


def get_paragraphs(input_text: str):
    """returns the individual paragraphs in the given markdown/pdf text."""
    paragraphs = input_text.split("\n")
    paragraphs = list(filter(lambda x: len(x) > 0, paragraphs))
    return paragraphs


def setup_blocks(paragraphs, char_to_ix, block_size):
    """
    returns the input blocks along with targets using given paragraphs.
    """
    blocks = []
    targets = []

    for paragraph in paragraphs:
        p = "|" + paragraph + "|"
        length = len(p)
        for i in range(length - block_size):
            block = p[i : i + block_size]
            target = p[i + block_size]
            block = [char_to_ix[char] for char in block]
            target = char_to_ix[target]
            blocks.append(block)
            targets.append(target)

    return blocks, targets


def prepare_dataloaders(
    blocks,
    targets,
    batch_size,
    split_ratio=0.2,
    generator=torch.Generator().manual_seed(42),
):
    """returns the data loaders for training and testing phase given blocks, targets and batch_size."""
    # now we'll need to convert the singular samples to tensors, and then to batches
    inputs = torch.tensor(blocks, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    random_indices = torch.randperm(inputs.size(0), generator=generator)
    inputs = inputs[random_indices]
    targets = targets[random_indices]

    n_samples = inputs.size(0)
    n_train = int(n_samples * (1 - split_ratio))

    train_inputs = inputs[:n_train]
    train_targets = targets[:n_train]

    test_inputs = inputs[n_train:]
    test_targets = targets[n_train:]

    train_dataset = TensorDataset(train_inputs, train_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
