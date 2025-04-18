import abc
from abc import ABC
from typing import Literal, get_args

import logging
import numpy as np
from nearness import NearestNeighbors

__all__ = [
    "KNNDetector",
    "LOFDetector",
]

logger = logging.getLogger(__name__)

KNNReductionT = Literal["max", "mean", "median"]


class Detector(abc.ABC):
    def __init__(self):
        self._is_fitted = False

    @property
    def is_fitted(self):
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self, value: bool) -> None:
        self._is_fitted = value

    @abc.abstractmethod
    def fit(self, x, index: NearestNeighbors):
        ...

    @abc.abstractmethod
    def predict(self, x, index: NearestNeighbors):
        ...


class KNNDetector(Detector):
    def __init__(self, n_neighbors: int, reduction: KNNReductionT = "max"):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.reduction = reduction

    def fit(self, x: np.ndarray, index: NearestNeighbors) -> None:
        assert index.is_fitted
        self.is_fitted = True

    def predict(self, x: np.ndarray, index: NearestNeighbors) -> np.ndarray:
        assert index.is_fitted
        assert self.is_fitted
        dist = query_safe_distances(x, index, n_neighbors=self.n_neighbors)
        logger.debug(
            "KNN (prediction): %s (min) %s (mean) %s (max) %s (std)",
            dist.min(),
            dist.mean(),
            dist.max(),
            dist.std()
        )

        if self.reduction == "max":
            return dist[:, -1]
        if self.reduction == "mean":
            return np.mean(dist, axis=1)
        if self.reduction == "median":
            return np.median(dist, axis=1)
        raise_reduction(self.reduction)  # noqa: RET503


class LOFDetector(Detector):
    def __init__(self, n_neighbors: int):
        super().__init__()
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
        self.is_fitted = True

    def predict(self, x: np.ndarray, index: NearestNeighbors) -> np.ndarray:
        assert self.is_fitted
        assert index.is_fitted
        idx, dist = index.query_batch(x, n_neighbors=self.n_neighbors)
        lrd_test = local_reachability_density(self.dist_train, dist, idx, self.n_neighbors)
        return local_outlier_factor(self.lrd_train, lrd_test, idx)


def query_safe_distances(x: np.ndarray, index: NearestNeighbors, n_neighbors: int) -> np.ndarray:
    idx, dist = index.query_batch(x, n_neighbors=n_neighbors)

    # some indexing structures might return -1 indices if no element was found
    if np.any(missing_idx := (idx == -1)):
        # assign each missing element the max distance (for later mean and median)
        # note that the max value is chosen over all samples, because it might be that there are only NaN
        # values for a single sample (all-NaN-slice)
        # the risk here is that the max value over all samples disturbes the anomaly result of the sample,
        # but assuming that a NaN value is even worse it might be a reasonable assumption
        dist[missing_idx] = dist[~missing_idx].max()

    # prevent possible negative distances (e.g. rounding errors)
    # this is ~2x faster than np.maximum(dist, 0.0) because it does not copy
    return np.clip(dist, a_min=0.0, a_max=None, out=dist)


def raise_reduction(reduction: KNNReductionT) -> None:
    msg = f"Invalid value for argument `method`, got {reduction}, but expected one of {get_args(KNNReductionT)}"
    raise AssertionError(msg)


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
