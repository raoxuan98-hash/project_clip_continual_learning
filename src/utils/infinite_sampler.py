"""Infinite sampler that reshuffles at each epoch boundary without restarting DataLoader workers."""
import torch
from torch.utils.data import Sampler


class InfiniteSampler(Sampler):
    """Yield indices from a dataset forever, reshuffling at every epoch.

    Using this sampler with ``persistent_workers=True`` avoids the pause
    caused by recreating a ``DataLoader`` iterator at each epoch boundary.
    The length of the sampler equals one epoch, matching the semantics used
    by training-budget calculations.
    """

    def __init__(self, data_source, shuffle=True, seed=0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._g = torch.Generator()

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        n = len(self.data_source)
        while True:
            self._g.manual_seed(self.seed + self.epoch)
            if self.shuffle:
                indices = torch.randperm(n, generator=self._g).tolist()
            else:
                indices = list(range(n))
            for idx in indices:
                yield idx
            self.epoch += 1
