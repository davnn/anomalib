import contextlib
import logging
from functools import partial

import numpy as np
import torch
from copy import deepcopy
from typing import Literal

from joblib import cpu_count
from nearness import NearestNeighbors
from anomalib.models.components import KCenterGreedy, Random
from concurrent.futures import ThreadPoolExecutor, as_completed

from .detector import Detector, KNNReductionT

logger = logging.getLogger(__name__)


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
    IndexMethodT = Literal["fit", "add"]
    CoresetMethodT = Literal["kcenter", "random"]

    def __init__(
        self,
        *,
        base_index: NearestNeighbors,
        coreset_method: CoresetMethodT = "kcenter",
        coreset_ratio_fit: int | float | None = None,
        coreset_ratio_add: int | float | None = None,
        n_indices: int = 1,
        n_jobs: int = 0,
        device: str | torch.device = "auto",
    ) -> None:
        super().__init__()
        assert n_indices > 0, "Parameter 'n_indices' must be greater than zero."
        self.coreset_ratio_fit = coreset_ratio_fit if coreset_ratio_fit is not None else 1 / n_indices
        self.coreset_ratio_add = coreset_ratio_add if coreset_ratio_add is not None else 1 / n_indices
        self.indices = [deepcopy(base_index) for _ in range(n_indices)]
        self.coreset_class = {
            "kcenter": partial(KCenterGreedy, progress=False),
            "random": Random,
        }[coreset_method]
        self.device = self._get_device(device)
        self.n_jobs = min(cpu_count(), n_indices) if n_jobs == -1 else n_jobs
        self.thread_pool = self._create_pool()
        self.index_streams = self._create_streams()
        self.apply_coreset = self._apply_coreset_loop if n_jobs == 0 else self._apply_coreset_parallel

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
        return self.apply_coreset(data, method="fit")

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
        return self.apply_coreset(data, method="add")

    @staticmethod
    def _get_device(device: str | torch.device) -> torch.device:
        match device:
            case "auto":
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")
            case str():
                return torch.device(device)
            case torch.device():
                return device
            case _ as value:
                msg = f"Invalid torch device specified, found '{value}'."
                raise ValueError(msg)

    def _apply_coreset_loop(self, data: np.ndarray, *, method: IndexMethodT) -> "EnsembleIndex":
        tensor = torch.from_numpy(data).to(self.device, non_blocking=True)
        sampling_ratio = self._get_sampling_ratio(len(data), method=method)
        for idx, index in enumerate(self.indices):
            self._process_index(
                index,
                idx=idx,
                method=method,
                tensor=tensor,
                sampling_ratio=sampling_ratio
            )
        return self

    def _apply_coreset_parallel(self, data: np.ndarray, *, method: IndexMethodT) -> "EnsembleIndex":
        """
        Apply coreset sampling to the given data to spread the data over multiples indices.

        Parameters
        ----------
        data : np.ndarray
            Input data to build the nearest neighbor indices on.

        Returns
        -------
        EnsembleIndex
            The ensemble index.
        """
        tensor = torch.from_numpy(data).to(self.device, non_blocking=True)
        sampling_ratio = self._get_sampling_ratio(len(data), method=method)

        # Wait for all tasks to finish, but no need to collect results
        futures = [self.thread_pool.submit(
            self._process_index,
            index,
            idx=idx,
            method=method,
            tensor=tensor,
            sampling_ratio=sampling_ratio
        ) for idx, index in enumerate(self.indices)]
        for future in as_completed(futures):
            try:  # Ensure all futures complete
                future.result()  # Will raise exception if one occurred
            except Exception as e:
                logger.error("Exception in subsampling thread: %s", e)
            except BaseException as e:
                logger.critical("Critical error in subsampling thread: %s", e)
        return self

    def _get_sampling_ratio(self, n_samples: int, *, method: IndexMethodT) -> float:
        sampling_ratio = self.coreset_ratio_fit if method == "fit" else self.coreset_ratio_add
        sampling_ratio = sampling_ratio / n_samples if isinstance(sampling_ratio, int) else sampling_ratio
        return sampling_ratio

    def _process_index(
            self,
            index: NearestNeighbors,
            *,
            idx: int,
            method: IndexMethodT,
            sampling_ratio: float,
            tensor: torch.Tensor,
    ) -> None:
        with self.index_streams[idx] as stream:  # make sure the device copy on default stream is done
            if isinstance(stream, torch.cuda.Stream):
                stream.wait_stream(torch.cuda.default_stream(device=self.device))
            coreset = self.subsample_embedding(tensor, sampling_ratio=sampling_ratio)
            getattr(index, method)(coreset.cpu().numpy())

    def __getstate__(self):
        state = super().__getstate__()
        # Remove thread pool since it's not pickleable
        state["thread_pool"] = None
        state["index_streams"] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.update(state)
        # Recreate the thread pool after unpickling
        self.thread_pool = self._create_pool()
        self.index_streams = self._create_streams()

    def _create_streams(self) -> list[torch.cuda.StreamContext]:
        device_use_cuda = self.device.type == "cuda"

        def cuda_stream_context(use_cuda: bool = device_use_cuda):
            if use_cuda:
                stream = torch.cuda.Stream(device=self.device)
                return torch.cuda.stream(stream)
            else:
                # No-op context manager when CUDA is not available
                return contextlib.nullcontext()

        return [cuda_stream_context() for _ in range(len(self.indices))]

    def _create_pool(self) -> ThreadPoolExecutor | None:
        return ThreadPoolExecutor(max_workers=self.n_jobs) if self.n_jobs > 0 else None

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
        result = [index.query(point, n_neighbors) for index in self.indices]
        idx, dist = zip(*result)
        return np.stack(idx), np.stack(dist)

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
        result = [index.query_batch(points, n_neighbors) for index in self.indices]
        idx, dist = zip(*result)
        return np.stack(idx), np.stack(dist)

    @torch.inference_mode()
    def subsample_embedding(self, embedding: torch.Tensor, sampling_ratio: float) -> torch.Tensor:
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
        sampler = self.coreset_class(embedding=embedding, sampling_ratio=sampling_ratio)
        return sampler.sample_coreset()


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
        super().__init__()
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
        self.is_fitted = True

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
        assert self.is_fitted
        predictions = [det.predict(x, idx) for det, idx in zip(self.detectors, index.indices, strict=True)]
        return self.reduction(predictions, axis=0)
