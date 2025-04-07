"""k-Medoids Clustering.

Erich Schubert, Peter J. Rousseeuw:
Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms
In: 12th International Conference on Similarity Search and Applications (SISAP 2019), 171-187.
"""
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


class Uncertainty:
    """Implements uncertainty sampling based on the anomaly score.

    Args:
        embedding (torch.Tensor): Embedding vector extracted from a CNN
        sampling_ratio (float): Ratio to choose coreset size from the embedding size.

    Example:
        >>> embedding.shape
        torch.Size([219520, 1536])
        >>> sampler = Uncertainty(embedding=embedding)
        >>> sampled_idxs = sampler.select_coreset_idxs()
        >>> coreset = embedding[sampled_idxs]
        >>> coreset.shape
        torch.Size([219, 1536])
    """

    def __init__(
        self,
        embedding: torch.Tensor,
        sampling_ratio: float,
        detector: "KNNDetector",
        index: "NearestNeighbors"
    ) -> None:
        self._embedding = embedding
        self._coreset_size = int(embedding.shape[0] * sampling_ratio)
        self._detector = detector
        self._index = index

    def sample_coreset(self, selected_idxs: list[int] | None = None) -> torch.Tensor:
        """Select coreset from the embedding.

        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
            Tensor: Output coreset

        Example:
            >>> embedding.shape
            torch.Size([219520, 1536])
            >>> sampler = Uncertainty(...)
            >>> coreset = sampler.sample_coreset()
            >>> coreset.shape
            torch.Size([219, 1536])
        """
        patch_scores = self._detector.predict(self._embedding.cpu().numpy(), self._index)
        # sort scores ascending and choose highest scored embeddings for coreset
        coreset = np.argsort(patch_scores)[len(patch_scores) - self._coreset_size:]
        return self._embedding[coreset]
