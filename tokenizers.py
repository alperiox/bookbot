import json
from abc import ABC, abstractmethod
from pathlib import Path


### Template for the future tokenizers, all of them should follow this structure
class Tokenizer(ABC):
    @abstractmethod
    def encode(self, texts: list[str | list[int]] | str) -> list[list[int]]:
        pass

    @abstractmethod
    def decode(self, tokens: list[list[int]]) -> list[str]:
        pass

    @abstractmethod
    def fit(self, source: list[str] | str):
        """
        requires a list of strings.
        - defines vocabulary
        - defines vocab_size
        - defines two dictionaries for mapping text -> token (stoi) or token -> text (itos)
        """

    # we won't need to implement the methods below since they are not abstract methods
    def save(self, path, name="tokenizer"):
        # check if the tokenizer fit to a text
        for k in ["vocabulary", "vocab_size", "stoi", "itos"]:
            if k not in self.__dict__:
                raise NotImplementedError(
                    f"`{k}` is not implemented in the `fit` method or the tokenizer is not fitted to any text yet!"
                )
        # save the tokenizer
        p = Path(path)
        with open(p / f"{name}.json", "w") as f:
            json.dump(self.__dict__, f, indent=2)

        return

    def load(self, path):
        with open(path) as f:
            config = json.loads(f.read())
        self.__dict__ = config
        self.itos = {int(k): v for k, v in self.itos.items()}

        return


class CharTokenizer(Tokenizer):
    """
    A simple character-level tokenizer, simply maps unique character to an integer
    the first two tokens are reserved for the following EOS and BOS tokens:
    0: |
    """

    def __init__(self):
        self.special_tokens = {
            "|": 0,
        }
        self.special_token_mappings = {
            "EOS_TOKEN": "|",
            "BOS_TOKEN": "|",
        }
        self.__dict__.update(self.special_token_mappings)

    def fit(self, source: list[str] | str) -> None:
        if isinstance(source, str):
            source = [source]

        tokens = set()
        for string in source:
            unique_chars = set(string)
            tokens.update(unique_chars)

        # define the vocabulary
        self.vocabulary: list = sorted(list(tokens))
        # define the vocab_size
        self.vocab_size: int = len(tokens) + len(self.special_tokens)
        # define the stoi and itos dictionaries
        # stoi: string -> ix
        # itos: ix -> string
        self.stoi: dict = self.special_tokens.copy()
        self.stoi.update({t: i for i, t in enumerate(tokens, len(self.special_tokens))})
        self.itos: dict = {i: t for t, i in self.stoi.items()}

        return

    def pad(self, sequences: list[list[int]], length: int) -> list[list[int]]:
        """pads the given list of sequences (a list of tokens) to the given length"""
        if isinstance(sequences, list):
            if all([isinstance(s, int) for s in sequences]):
                sequences = list(sequences)

        padded_tokens = []
        for s in sequences:
            if not isinstance(s, list):
                raise ValueError(f"Sequence {s} is not a list!")

            L = len(s)
            if L < length:
                # pad the sequence
                padded_seq = [self.special_tokens[self.BOS_TOKEN]] * (length - L) + s
            else:
                padded_seq = s

            padded_tokens.append(padded_seq)

        return padded_tokens

    def encode(self, texts: list[str | list[int]] | str) -> list[list[int]]:
        if isinstance(texts, str):
            texts = [texts]

        tokens = []
        for text in texts:
            # map every character to the corresponding token
            text_tokens = map(lambda c: self.stoi[c], text)
            # convert the generator to list, it will make it yield every result
            # so we will have all the tokens in the end
            text_tokens = list(text_tokens)
            # append the tokens to the `tokens` list
            tokens.append(text_tokens)

        return tokens

    def decode(self, tokens: list[list[int]]) -> list[str]:
        if isinstance(tokens, list):
            if all([isinstance(s, int) for s in tokens]):
                tokens = list(tokens)

        texts = []
        for token_list in tokens:
            text = "".join([self.itos[ix] for ix in token_list])
            texts.append(text)

        return texts
