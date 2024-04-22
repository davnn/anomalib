from abc import ABC
from typing import Literal, get_args

import logging
import nearness
import numpy as np
import torch
from nearness import NearestNeighbors

__all__ = [
    "KNNDetector",
    "LOFDetector",
    "DistanceDistribution"
]

logger = logging.getLogger(__name__)

KNNReductionT = Literal["max", "mean", "median"]


class Detector(ABC):

    def fit(self, x, index: NearestNeighbors):
        ...

    def predict(self, x, index: NearestNeighbors):
        ...


class KNNDetector(Detector):
    def __init__(self, n_neighbors: int, reduction: KNNReductionT = "max"):
        self.n_neighbors = n_neighbors
        self.reduction = reduction

    def fit(self, x: np.ndarray, index: NearestNeighbors) -> None:
        assert index.is_fitted

    def predict(self, x: np.ndarray, index: NearestNeighbors) -> np.ndarray:
        assert index.is_fitted
        dist = query_safe_distances(x, index, n_neighbors=self.n_neighbors)
        logger.info("KNN prediction: %s %s %s %s", dist.min(), dist.mean(), dist.max(), dist.std())

        if self.reduction == "max":
            return dist[:, -1]
        if self.reduction == "mean":
            return np.mean(dist, axis=1)
        if self.reduction == "median":
            return np.median(dist, axis=1)
        raise_reduction(self.reduction)  # noqa: RET503


def query_safe_distances(x: np.ndarray, index: NearestNeighbors, n_neighbors: int) -> np.ndarray:
    idx, dist = index.query_batch(x, n_neighbors=n_neighbors)

    if np.any(missing_idx := (idx == -1)):
        # some indexing structures might return -1 indices if no element was found
        dist[missing_idx] = np.nan

        # TODO(David): Evaluate if np.max over all axis is a better choice
        # assign each missing element the max distance (for later mean and median)
        max_value = np.nanmax(dist, axis=1, keepdims=True)
        dist = np.where(missing_idx, max_value, dist)

    return dist


class LOFDetector(Detector):
    def __init__(self, n_neighbors: int):
        self.n_neighbors = n_neighbors

        self.lrd_train = None
        self.idx_train = None
        self.dist_train = None

    def fit(self, x: np.ndarray, index: NearestNeighbors) -> None:
        # we expect that the provided index is already fit
        assert index.is_fitted
        idx, dist = index.query_batch(x, n_neighbors=self.n_neighbors + 1)
        self.idx_train = idx[:, 1:]  # ignore first neighbor of training points
        self.dist_train = dist[:, 1:]  # ignore first neighbor of training points
        self.lrd_train = local_reachability_density(self.dist_train, self.dist_train, self.idx_train, self.n_neighbors)

    def predict(self, x: np.ndarray, index: NearestNeighbors) -> np.ndarray:
        assert self.lrd_train is not None, "LOF must be 'fit' before 'predict' can be used."
        idx, dist = index.query_batch(x, n_neighbors=self.n_neighbors)
        lrd_test = local_reachability_density(self.dist_train, dist, idx, self.n_neighbors)
        return local_outlier_factor(self.lrd_train, lrd_test, idx)


class DistanceDistribution:
    valid_distribution = Literal["normal", "lognormal", "halfnormal"]

    def __init__(
            self,
            n_neighbors: int,
            distribution: valid_distribution = "normal",
            min_samples: int = 8,
    ):
        super().__init__()
        self.distribution = distribution
        self.n_neighbors = n_neighbors
        self.min_samples = min_samples

        # to be updated in ``update``
        self.n_samples = 0
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
        x_flat = self.reshape_embedding(x)
        dist = query_safe_distances(x_flat, index, n_neighbors=self.n_neighbors)  # n, k
        dist_patch = torch.from_numpy(dist.reshape(n, h, w, self.n_neighbors))

        # update the distribution parameters
        var, mean = torch.var_mean(dist_patch, dim=(0, -1))
        self.mean = self.combine_means(self.mean, mean, self.n_samples, n)
        self.var = self.combine_vars(self.var, var, self.mean, mean, self.n_samples, n)
        self.n_samples += n

    def compute(self) -> torch.distributions.Distribution:
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

        return self.computed_distribution

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

    @property
    def is_available(self):
        return self.n_samples >= self.min_samples

    @staticmethod
    def reshape_embedding(embedding: np.array) -> np.array:
        """Reshape Embedding.

        Reshapes Embedding to the following format:
            - [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (torch.Tensor): Embedding tensor extracted from CNN features.

        Returns:
            Tensor: Reshaped embedding tensor.
        """
        embedding_size = embedding.shape[1]
        return np.transpose(embedding, axes=(0, 2, 3, 1)).reshape(-1, embedding_size)


def raise_reduction(reduction: KNNReductionT) -> None:
    msg = f"Invalid value for argument `method`, got {reduction}, but expected one of {get_args(KNNReductionT)}"
    raise AssertionError(msg)


def numpy_ecdf(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    # reference input must be a flat (1d) ascending sorted array
    result = (np.searchsorted(reference, query, side="right")) / len(reference)
    return result.astype(query.dtype)


def local_reachability_density(
        dist_train: np.ndarray,
        dist_test: np.ndarray,
        idx_test: np.ndarray,
        k: int
) -> np.ndarray:
    dist_k = dist_train[idx_test, k - 1]
    reachability_distance = np.maximum(dist_test, dist_k)
    return 1.0 / (np.mean(reachability_distance, axis=1) + 1e-8)


def local_outlier_factor(
        lrd_train: np.ndarray,
        lrd_test: np.ndarray,
        idx_test: np.ndarray
) -> np.ndarray:
    lrd_ratio = lrd_train[idx_test] / (lrd_test[:, np.newaxis] + 1e-8)
    return np.mean(lrd_ratio, axis=1)
