import torch
from torch import Tensor

from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
import math
import random


T_co = TypeVar('T_co', covariant=True)


class ConstantSampler(Sampler[int]):
    r"""Samples elements for each worker from a given constant list of example indices.

    """
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,
                 examples_allocations: dict = {},
                 ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should in the interval [0, {num_replicas-1}]"
            )


        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.examples_allocations = examples_allocations

        # testing subsampling
        self.indices_0 = [] # indices on epoch 0
        self.indices_1 = [] # indices on epoch 1


        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # split to nearest available length
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

        if len(self.examples_allocations) == 0:
            self.examples_allocations = {}
            portion_of_each_worker = int(self.total_size / self.num_replicas)
            all_indices = [x for x in range(0, self.total_size)]
            for i in range(0, self.num_replicas):
                self.examples_allocations[i] = all_indices[i * portion_of_each_worker:(i + 1) * portion_of_each_worker]

    def __iter__(self) -> Iterator[T_co]:
        """Return an iterator to be used by the DataLoader.
        The difference between this implementation and DistributedSampler is that each worker only takes care of its own indices.
        For unevenly divisable datasets, if workers have to add examples, they add from their own examples
        Also it uses Python's `random` rather than PyTorch's `Generator`."""
        if self.shuffle:
            # deterministically shuffle based on epoch and seed

            random.seed(self.seed + self.epoch)
            indices = self.examples_allocations[self.rank].copy()
            random.shuffle(indices)
        else:
            # do not shuffle - use the same order everytime
            indices = self.examples_allocations[self.rank]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.num_samples - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # drop the tail of dataset
            indices = indices[:self.num_samples]

        assert len(indices) == self.num_samples

        ###################### test block for allocations
        # can be replaced with an assertion
        # if self.rank == 0:
        #     print("in iterator if, epoch is {}".format(self.epoch))
        #     if self.epoch == 0:
        #         self.indices_0 = indices
        #     elif self.epoch == 1:
        #         print("iterator in epoch 1")
        #         self.indices_1 = indices
        #         indices_0_as_set = set(self.indices_0)
        #         indices_1_as_set = set(self.indices_1)
        #
        #         common_indices = indices_0_as_set.intersection(indices_1_as_set)
        #         common_indices = list(common_indices)
        #         print(f"length of common indices is {len(common_indices)}, num of worker's samples is {self.num_samples}")
        #         print(common_indices)

        # print(f"Indices of worker {self.rank} in epoch {self.epoch}: " + str(indices))
        ########################

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

