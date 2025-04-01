"""k-Medoids Clustering.

Erich Schubert, Peter J. Rousseeuw:
Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms
In: 12th International Conference on Similarity Search and Applications (SISAP 2019), 171-187.
"""
import logging

logger = logging.getLogger(__name__)

try:
    from kmedoids import KMedoids as FasterKMediods
except ImportError:
    msg = ("K-Medoids sampling requires the 'kmedoids' package to be installed "
           ", you can install the package using 'pip install kmedoids', or using"
           "'conda install conda-forge::kmedoids'.")
    logger.debug(msg)

import torch
from anomalib.models.components.dimensionality_reduction import SparseRandomProjection


class KMedoids:
    """Implements k-center-greedy method.

    Args:
        embedding (torch.Tensor): Embedding vector extracted from a CNN
        sampling_ratio (float): Ratio to choose coreset size from the embedding size.

    Example:
        >>> embedding.shape
        torch.Size([219520, 1536])
        >>> sampler = KMedoids(embedding=embedding)
        >>> sampled_idxs = sampler.select_coreset_idxs()
        >>> coreset = embedding[sampled_idxs]
        >>> coreset.shape
        torch.Size([219, 1536])
    """

    def __init__(self, embedding: torch.Tensor, sampling_ratio: float) -> None:
        self._embedding = embedding
        self._model = SparseRandomProjection(eps=0.9)
        coreset_size = int(embedding.shape[0] * sampling_ratio)
        self._cluster = FasterKMediods(
            init="random",
            random_state=0,
            max_iter=10,
            n_clusters=coreset_size,
            method="fasterpam"
        )

    def sample_coreset(self, selected_idxs: list[int] | None = None) -> torch.Tensor:
        """Select coreset from the embedding.

        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
            Tensor: Output coreset

        Example:
            >>> embedding.shape
            torch.Size([219520, 1536])
            >>> sampler = KCenterGreedy(...)
            >>> coreset = sampler.sample_coreset()
            >>> coreset.shape
            torch.Size([219, 1536])
        """

        features = self._model.fit(self._embedding).transform(self._embedding)
        distances = torch.cdist(features, features, compute_mode="use_mm_for_euclid_dist")
        model = self._cluster.fit(distances.cpu().numpy())
        coreset = model.medoid_indices_.astype(int)
        return self._embedding[coreset]
