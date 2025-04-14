from abc import ABC, abstractmethod
from typing import Literal, get_args, Protocol

import logging
import nearness
import numpy as np
import torch
from einops import rearrange
from nearness import NearestNeighbors

from .anomaly_detector import query_safe_distances

__all__ = [
    "DistanceDistribution",
    "NormalDistanceDistribution",
    "EmpiricalDistanceDistribution",
    "HistogramDistanceDistribution"
]


class CDF(Protocol):
    # Empty method body (explicit '...')
    def cdf(self, x: torch.Tensor) -> torch.Tensor: ...


class DistanceDistribution(ABC):
    def __init__(
        self,
        n_neighbors: int,
        min_samples: int = 8
    ) -> None:
        self.n_neighbors = n_neighbors
        self.min_samples = min_samples
        self.n_samples = 0  # to be updated in ``update``
        self.computed_distribution = None  # to be updated in ``compute``

    @property
    def is_available(self):
        return self.n_samples >= self.min_samples

    def get(self, recompute: bool = False) -> CDF:
        assert self.is_available, "Cannot compute distribution, n_samples <= min_samples."

        if recompute or self.computed_distribution is None:
            self.compute()

        return self.computed_distribution

    @abstractmethod
    def update(self, x: np.ndarray, index: nearness.NearestNeighbors) -> None:
        ...

    @abstractmethod
    def compute(self) -> None:
        ...


class HistogramDistanceDistribution(DistanceDistribution):
    def __init__(
        self,
        n_neighbors: int,
        min_samples: int = 8,
        n_buckets: int = 1024,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        """
        Determine the distribution of distances based on a histogram of values (an approximation of the ECDF).

        :param n_neighbors: Number of neighbors to determine the distance distribution.
        :param min_samples: Minimum number of samples required for distribution computation.
        :param max_value: Max value for the largest bucket, if none use the max value of the first batch
        """
        super().__init__(n_neighbors=n_neighbors, min_samples=min_samples)
        self.n_buckets = n_buckets
        self.min_value = min_value
        self.max_value = max_value

        self.buckets = None
        self.counts = None

    def update(self, x: np.ndarray, index: nearness.NearestNeighbors) -> None:
        assert index.is_fitted
        n, c, h, w = x.shape

        # calculate the patch-wise neighborhood scores
        x_flat = reshape_embedding(x)
        dist = query_safe_distances(x_flat, index, n_neighbors=self.n_neighbors)  # n_flat, k
        dist = torch.from_numpy(dist.reshape(n, h, w, self.n_neighbors))
        dist = rearrange(dist, "n h w k -> (h w) (n k)")

        if self.buckets is None:
            # determine the buckets based on the initial
            min_value = dist.min() if self.min_value is None else self.min_value
            max_value = dist.max() if self.max_value is None else self.max_value
            self.buckets = torch.linspace(min_value, max_value, steps=self.n_buckets)

        bins = torch.bucketize(dist.contiguous(), boundaries=self.buckets, right=False)

        if self.counts is None:
            # set the initial counts to zero (n_buckets + 1 to open both boundaries)
            self.counts = bins.new_zeros(bins.size(0), self.n_buckets + 1)

        self.counts = bincount(bins, counts=self.counts)
        self.n_samples += n

    def compute(self) -> None:
        self.computed_distribution = HistCDF(self.buckets, self.counts)


class EmpiricalDistanceDistribution(DistanceDistribution):
    def __init__(
        self,
        n_neighbors: int,
        min_samples: int = 8,
    ):
        super().__init__(n_neighbors=n_neighbors, min_samples=min_samples)

        # to be set in update
        self.values = None

    def update(self, x: np.ndarray, index: nearness.NearestNeighbors) -> None:
        assert index.is_fitted
        n, c, h, w = x.shape

        # calculate the patch-wise neighborhood scores
        x_flat = reshape_embedding(x)
        dist = query_safe_distances(x_flat, index, n_neighbors=self.n_neighbors)  # n_flat, k
        dist = torch.from_numpy(dist.reshape(n, h, w, self.n_neighbors))
        dist = rearrange(dist, "n h w k -> (h w) (n k)")

        if self.values is None:
            # initialize an empty tensor if values is none (now we now h and w)
            self.values = torch.empty((h * w, 0), dtype=dist.dtype)

        self.values = torch.cat([self.values, dist], dim=-1)
        self.n_samples += n

    def compute(self) -> None:
        self.computed_distribution = ECDF(values=self.values)


class ECDF:
    def __init__(self, values: torch.Tensor) -> None:
        sorted_values = torch.sort(values, dim=-1).values
        self.sorted_values = sorted_values.contiguous()

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        result = torch.searchsorted(
            self.sorted_values,
            rearrange(x, "n 1 h w -> (h w) n").contiguous()
        ) / self.sorted_values.shape[-1]
        return rearrange(result, "(h w) n -> n 1 h w", h=h, w=w)


class HistCDF:
    def __init__(
        self,
        buckets: torch.Tensor,
        counts: torch.Tensor
    ) -> None:
        cum_sum = torch.cumsum(counts, dim=-1)
        # cum_sum.max() is actually the number of total elements, which should be same
        # for all rows in the counts matrix (the total cumulative count)
        self.proba = cum_sum / cum_sum.max()
        self.buckets = buckets

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x = rearrange(x, "n 1 h w -> (h w) n")
        idx = torch.bucketize(x.contiguous(), boundaries=self.buckets, right=False)
        result = torch.gather(self.proba, dim=-1, index=idx)
        return rearrange(result, "(h w) n -> n 1 h w", h=h, w=w)


class NormalDistanceDistribution(DistanceDistribution):
    valid_distribution = Literal["normal", "log", "half"]

    def __init__(
        self,
        n_neighbors: int,
        distribution: valid_distribution = "normal",
        min_samples: int = 8,
    ):
        super().__init__(n_neighbors=n_neighbors, min_samples=min_samples)
        self.distribution = distribution

        # to be updated in ``update``
        self.mean = None
        self.var = None
        self.computed_distribution = None

    def update(self, x: np.ndarray, index: nearness.NearestNeighbors) -> None:  # (n, c, h, w)
        """Use Chan's algorithm [1] to calculate the running mean and variance.

        [1]  Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (November 1979). "Updating Formulae and a Pairwise
        Algorithm for Computing Sample Variances"

        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        https://stackoverflow.com/questions/75545944/efficient-algorithm-for-online-variance-over-image-batches
        """
        assert index.is_fitted
        n, c, h, w = x.shape

        # initialize the parameters if not already initialized
        if self.mean is None:
            self.mean = torch.zeros(h, w)
        if self.var is None:
            self.var = torch.zeros(h, w)

        # calculate the patch-wise neighborhood scores
        x_flat = reshape_embedding(x)
        dist = query_safe_distances(x_flat, index, n_neighbors=self.n_neighbors)  # n, k
        dist_patch = torch.from_numpy(dist.reshape(n, h, w, self.n_neighbors))

        # update the distribution parameters
        var, mean = torch.var_mean(dist_patch, dim=(0, -1))
        self.mean = self.combine_means(self.mean, mean, self.n_samples, n)
        self.var = self.combine_vars(self.var, var, self.mean, mean, self.n_samples, n)
        self.n_samples += n

    def compute(self) -> None:
        assert self.is_available
        std, mean = torch.clip(self.std, min=1e-8), self.mean

        logger.info("Distribution mean: %s %s %s %s", mean.min(), mean.mean(), mean.max(), mean.std())
        logger.info("Distribution std:  %s %s %s %s", std.min(), std.mean(), std.max(), std.std())

        if self.distribution == "normal":
            self.computed_distribution = torch.distributions.normal.Normal(loc=mean, scale=std)
        elif self.distribution == "lognormal":
            self.computed_distribution = torch.distributions.log_normal.LogNormal(loc=mean, scale=std)
        elif self.distribution == "halfnormal":
            self.computed_distribution = torch.distributions.half_normal.HalfNormal(scale=std)
        else:
            raise AssertionError(f"Distribution must be one of {self.valid_distribution}")

    @staticmethod
    def combine_means(
        agg_mean: torch.Tensor,
        batch_mean: torch.Tensor,
        agg_n: int,
        batch_n: int
    ) -> torch.Tensor:
        """
        Updates old mean mu1 from m samples with mean mu2 of n samples.
        Returns the mean of the m+n samples.
        """
        return (agg_n / (agg_n + batch_n)) * agg_mean + (batch_n / (agg_n + batch_n)) * batch_mean

    @staticmethod
    def combine_vars(
        agg_var: torch.Tensor,
        batch_var: torch.Tensor,
        agg_mean: torch.Tensor,
        batch_mean: torch.Tensor,
        agg_n: int,
        batch_n: int
    ) -> torch.Tensor:
        """
        Updates old variance v1 from m samples with variance v2 of n samples.
        Returns the variance of the m+n samples.
        """
        return (agg_n / (agg_n + batch_n)) * agg_var + batch_n / (agg_n + batch_n) * batch_var + agg_n * batch_n / (
            agg_n + batch_n) ** 2 * (agg_mean - batch_mean) ** 2

    @property
    def std(self):
        return torch.sqrt(self.var) if self.is_available else None


def reshape_embedding(embedding: np.array) -> np.array:
    """Reshape Embedding.

    Reshapes Embedding to the following format:
        - [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

    Args:
        embedding (torch.Tensor): Embedding tensor extracted from CNN features.

    Returns:
        Tensor: Reshaped embedding tensor.
    """
    return rearrange(embedding, "n c h w -> (n h w) c")


def bincount(x: torch.Tensor, counts: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Bincounting along a given dimension, adapted from https://github.com/pytorch/pytorch/issues/32306"""
    assert x.dtype is torch.int64, "only integral (int64) tensor is supported"
    assert dim != 0, "dim cannot be 0, zero is the counting dimension"
    # no scalar or broadcasting `src` support yet
    # c.f. https://github.com/pytorch/pytorch/issues/5740
    return counts.scatter_add_(dim=dim, index=x, src=x.new_ones(()).expand_as(x))
