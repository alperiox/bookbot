import torch
from torch.utils import data


# src: https://discuss.pytorch.org/t/implementing-an-infinite-loop-dataset-dataloader-combo/35567/5
class InfiniteRandomSampler(data.Sampler):
    """Return random indices from [0-n) infinitely.

    Arguments:
        dset_size (int): Size of the dataset to sample.
    """

    def __init__(self, dset_size):
        self.dset_size = dset_size

    def __iter__(self):
        # Create a random number generator (optional, makes the sampling independent of the base RNG)
        rng = torch.Generator()
        seed = torch.empty((), dtype=torch.int64).random_().item()
        rng.manual_seed(seed)

        return _infinite_generator(self.dset_size, rng)

    def __len__(self):
        return float("inf")


def _infinite_generator(n, rng):
    """Inifinitely returns a number in [0, n)."""
    while True:
        yield from torch.randperm(n, generator=rng).tolist()
