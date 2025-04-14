import numpy as np
import torch
from copy import deepcopy
from pathlib import Path
from typing import Literal

from nearness import NearestNeighbors
from anomalib.models.components import KCenterGreedy, Random

from .anomaly_detector import Detector, KNNReductionT


class EnsembleIndex(NearestNeighbors):
    """
    An ensemble of nearest neighbor indices, each trained on a different subsample of the input data.

    Parameters
    ----------
    base_index : NearestNeighbors
        A fitted or uninitialized nearest neighbor index to clone for each ensemble member.
    sampler : {"kcenter", "random"}, default="kcenter"
        The method used to subsample the training data for each index.
    reduction : {"mean", "max", "median"}, default="mean"
        Aggregation strategy for combining results (not used directly here but included for consistency).
    random_seed : int or None, optional
        Seed for random sampling, if applicable.
    sampling_ratio : float or None, optional
        Fraction of data to sample for each index. If None, defaults to 1 / n_indices.
    n_indices : int, default=1
        Number of ensemble members to create.
    """

    def __init__(
        self,
        *,
        base_index: NearestNeighbors,
        sampler: Literal["kcenter", "random"] = "kcenter",
        reduction: Literal["mean", "max", "median"] = "mean",
        random_seed: int | None = None,  # currently not used..
        sampling_ratio: float | None = None,
        n_indices: int = 1,
        device: str | torch.device = "auto"
    ) -> None:
        super().__init__()
        assert n_indices > 0, "Parameter 'n_indices' must be greater than zero."
        self.sampling_ratio = sampling_ratio if sampling_ratio is not None else 1 / n_indices
        self.indices = [deepcopy(base_index) for _ in range(n_indices)]
        self.sampler_class = {
            "kcenter": KCenterGreedy,
            "random": Random,
        }[sampler]
        self.device = self._get_device(device)

    def _get_device(self, device: str | torch.device) -> torch.device:
        match device:
            case "auto":
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")
            case str() as device_name:
                return torch.device(device_name)
            case torch.device() as device:
                return device
            case _ as value:
                msg = f"Invalid torch device specified, found '{value}'."
                raise ValueError(msg)

    def fit(self, data: np.ndarray) -> "EnsembleIndex":
        """
        Fit each index in the ensemble to a subsampled version of the data.

        Parameters
        ----------
        data : np.ndarray
            Input data to build the nearest neighbor indices on.

        Returns
        -------
        EnsembleIndex
            The fitted ensemble object.
        """
        tensor = torch.from_numpy(data).to(self.device)
        for index in self.indices:
            coreset = self.subsample_embedding(tensor, sampling_ratio=self.sampling_ratio)
            index.fit(coreset)
        return self

    def add(self, data: np.ndarray) -> "EnsembleIndex":
        """
        Add data to the existing index.

        Parameters
        ----------
        data : np.ndarray
            Input data to build the nearest neighbor indices on.

        Returns
        -------
        EnsembleIndex
            The fitted ensemble object.
        """
        for index in self.indices:
            index.add(data)
        return self

    def query(self, point: np.ndarray, n_neighbors: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Query all ensemble indices for nearest neighbors of a single point.

        Parameters
        ----------
        point : np.ndarray
            A single data point to query.
        n_neighbors : int
            Number of neighbors to retrieve.

        Returns
        -------
        list of tuple[np.ndarray, np.ndarray]
            List of (distances, indices) tuples from each ensemble member.
        """
        return [index.query(point, n_neighbors) for index in self.indices]

    def query_batch(self, points: np.ndarray, n_neighbors: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Query all ensemble indices for nearest neighbors of a batch of points.

        Parameters
        ----------
        points : np.ndarray
            A batch of data points to query.
        n_neighbors : int
            Number of neighbors to retrieve.

        Returns
        -------
        list of tuple[np.ndarray, np.ndarray]
            List of (distances, indices) tuples from each ensemble member.
        """
        return [index.query_batch(points, n_neighbors) for index in self.indices]

    def subsample_embedding(self, embedding: torch.Tensor, sampling_ratio: float) -> np.ndarray:
        """
        Subsample the embedding using the configured sampling method.

        Parameters
        ----------
        embedding : np.ndarray
            The input feature embedding.
        sampling_ratio : float
            Ratio of points to sample.

        Returns
        -------
        np.ndarray
            Subsampled coreset embedding.
        """
        sampler = self.sampler_class(embedding=embedding, sampling_ratio=sampling_ratio)
        return sampler.sample_coreset().cpu().numpy()


class EnsembleDetector(Detector):
    """
    Ensemble of anomaly detectors, each trained on a different index subset.

    Parameters
    ----------
    base_detector : Detector
        A base detector to clone across the ensemble.
    reduction : {"mean", "max", "median"}, default="max"
        Aggregation function used to reduce predictions from each detector.
    """

    def __init__(
        self,
        base_detector: Detector,
        reduction: KNNReductionT = "max",
    ) -> None:
        self.base_detector = base_detector
        self.reduction = {
            "mean": np.mean,
            "max": np.max,
            "median": np.median
        }[reduction]

        self.detectors: list[Detector] | None = None

    def fit(self, x: np.ndarray, index: NearestNeighbors):
        """
        Fit each detector in the ensemble using its corresponding index.

        Parameters
        ----------
        x : np.ndarray
            Input data to train on.
        index : EnsembleIndex
            Ensemble of nearest neighbor indices.

        Raises
        ------
        AssertionError
            If the given index is not an instance of EnsembleIndex.
        """
        assert isinstance(index, EnsembleIndex), "'EnsembleDetector' requires 'EnsembleIndex'."
        self.detectors = [deepcopy(self.base_detector) for _ in index.indices]
        for detector, sub_index in zip(self.detectors, index.indices, strict=True):
            detector.fit(x, sub_index)

    def predict(self, x: np.ndarray, index: NearestNeighbors) -> np.ndarray:
        """
        Predict anomaly scores using all detectors and aggregate them.

        Parameters
        ----------
        x : np.ndarray
            Input data for prediction.
        index : EnsembleIndex
            Ensemble of nearest neighbor indices.

        Returns
        -------
        np.ndarray
            Reduced anomaly scores after applying the specified aggregation function.

        Raises
        ------
        AssertionError
            If the given index is not an instance of EnsembleIndex.
        """
        assert isinstance(index, EnsembleIndex), "'EnsembleDetector' requires 'EnsembleIndex'."
        predictions = [det.predict(x, idx) for det, idx in zip(self.detectors, index.indices, strict=True)]
        return self.reduction(predictions, axis=0)
