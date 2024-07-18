from abc import ABC, abstractmethod

import torch
from fileloaders import AVAILABLE_LOADERS
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, TensorDataset

class DataProcessor(ABC):
    """
    a simple data processor that can be customized from model to model,
    this base processor provides the training, validation and test data loaders by:
    - loading the raw data from supported data sources
    - using the provided tokenizer to tokenize the loaded raw data.
    - setting up the tensors for the training phase (splitting them into blocks etc.)
    - preparing the torch data loaders with given data splits
    """

    def __init__(self, paths: list[str], tokenizer: Tokenizer):
        """
        paths (list): list of filepaths for the input files
        """
        self.sources = paths if isinstance(paths, list) else [paths]
        self.tokenizer = tokenizer

        for source in self.sources:
            ext = source.split(".")[-1]
            if ext not in AVAILABLE_LOADERS:
                raise NotImplementedError(f"`{ext}` extension is not covered for the moment! please remove it from the provided sources.")

        self.load_data()

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def process_raw_data(self):
        pass

    @abstractmethod
    def get_dataloaders(self):
        pass

class CharLevelMLPProcessor(DataProcessor):
    """ Data processor for the character-level Hierarchical MLP and MLP models. """
    def __init__(self, paths: list[str], tokenizer: Tokenizer, context_length: int):
        """
        paths (list): list of filepaths for the input files
        tokenizer (Tokenizer): tokenizer to tokenize the raw data
        context_length (int): context length (or the block size) to prepare the input data
        """
        super().__init__(paths, tokenizer)

        self.context_length = context_length


    def load_data(self):
        raw_data = ""
        for filepath in self.sources:
            ext = filepath.split(".")[-1]
            loader = AVAILABLE_LOADERS[ext]
            data = loader(filepath)
            raw_data += data
        
        self.raw_data = raw_data

        self.tokenizer.fit(raw_data)
    
        return raw_data

    def process_raw_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        1- Splits the raw data into paragraphs
        2- Removes the empty lines from the `paragraphs` list
        3- Sets up the blocks and the targets using the defined context length
          and tokenizer
        4- The blocks and targets are returned as `torch.long` tensors.
        
        For each paragraph, the beginning and the ending token from the tokenizer is 
        added to the text. 
        
        To put it simply, we use a sliding window of length `context_length` to slide
        through the paragraph. 
        Each context window is a training sample whereas the `context_length+1`th character
        is the target.

        I think the following example provides an intuitive explanation:

        context_length = 3

        input paragraph = "Hello there!"
        
        length of the paragraph = 12
        
        there will be 12-3=9 iterations:
        
        1: block = "Hel", target = "l"
        
        so "Hello there!"
            ---+
        
        2: block = "ell", target = "o""
        
        so "Hello there!"
             ---+
        ...


        """
        # get the paragraphs
        paragraphs = self.raw_data.split("\n")
        # filter out the empty lines
        paragraphs = list(filter(lambda x: len(x) > 0, paragraphs))

        blocks = []
        targets = []

        for paragraph in paragraphs:
            p = self.tokenizer.BOS_TOKEN + paragraph + self.tokenizer.EOS_TOKEN
            length = len(p)
            for i in range(length - self.context_length):
                block = p[i : i + self.context_length]
                target = p[i + self.context_length]
                # block is just a piece of string, so the tokenizer will do the following
                # transformation: string -> [encoded_string] 
                # therefore we will take the first index since it returns a list
                block = self.tokenizer.encode(block)[0]
                target = self.tokenizer.encode(target)[0]
                blocks.append(block)
                targets.append(target)

        blocks = torch.tensor(blocks, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long).view(-1)

        return blocks, targets

    def get_dataloaders(self, batch_size, train_ratio=0.8, generator=torch.Generator().manual_seed(42)):
        """ 
        Given the batch size, training ratio and an optional generator object, returns the training and testing data loaders
        """

        # get the blocks and the targets
        inputs, targets = self.process_raw_data() # (n_samples, context_length), (n_samples, ) shaped tensors

        # shuffle the data
        random_indices = torch.randperm(inputs.size(0), generator=generator)
        inputs = inputs[random_indices]
        targets = targets[random_indices]

        n_samples = inputs.size(0)
        n_train = int(n_samples * train_ratio)

        train_inputs = inputs[:n_train]
        train_targets = targets[:n_train]

        test_inputs = inputs[n_train:]
        test_targets = targets[n_train:]

        train_dataset = TensorDataset(train_inputs, train_targets)
        test_dataset = TensorDataset(test_inputs, test_targets)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader