import json
from abc import ABC, abstractmethod
from pathlib import Path


### Template for the future tokenizers, all of them should follow this structure
class Tokenizer(ABC):
    @abstractmethod
    def __init__(self):
        self.special_tokens = {
            "<BEGIN>": 0,
            "<END>": 1,
        }
        self.EOS_TOKEN = "<BEGIN>"
        self.BOS_TOKEN = "<END>"

    @abstractmethod
    def encode(self, text: list):
        pass

    @abstractmethod
    def decode(self, tokens: list):
        pass

    @abstractmethod
    def fit(self, source: list[str]):
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
    1: |
    """

    def __init__(self):
        self.special_tokens = {
            "|": 0,
        }
        self.EOS_TOKEN = "|"
        self.BOS_TOKEN = "|"

    def fit(self, source: list[str]) -> None:
        if isinstance(source, str):
            source = [source]

        tokens = set()
        for string in source:
            unique_chars = set(string)
            tokens.update(unique_chars)

        # define the vocabulary
        self.vocabulary = list(tokens)
        # define the vocab_size
        self.vocab_size = len(tokens) + len(self.special_tokens)
        # define the stoi and itos dictionaries
        # stoi: string -> ix
        # itos: ix -> string
        self.stoi = self.special_tokens.copy()
        self.stoi.update(
            {t: i for i, t in enumerate(tokens, 2)}
        )  # +2 for the special tokens
        self.itos = {i: t for t, i in self.stoi.items()}

        return

    def encode(self, texts: list[str]) -> list[list[int]]:
        if isinstance(texts, str):
            texts = [texts]

        tokens = []
        for text in texts:
            # map every character to the corresponding token
            text_tokens = map(
                lambda c: self.stoi[c], text
            )
            # convert the generator to list, it will make it yield every result
            # so we will have all the tokens in the end
            text_tokens = list(text_tokens)
            # append the tokens to the `tokens` list
            tokens.append(text_tokens)

        return tokens

    def decode(self, tokens: list[list[int]]) -> list[str]:
        if not isinstance(tokens[0], list):  # if it's not a list of lists
            tokens = [tokens]

        texts = []
        for token_list in tokens:
            text = "".join([self.itos[ix] for ix in token_list])
            texts.append(text)

        return texts
