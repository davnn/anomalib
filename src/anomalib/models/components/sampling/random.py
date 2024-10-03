from typing import Literal

import torch


class Random:
    """Implements random sampling.

    Args:
        embedding (torch.Tensor): Embedding vector extracted from a CNN
        sampling_ratio (float): Ratio to choose coreset size from the embedding size.
        sampling_method ("permutation" | "randint"): Corresponds to sampling with and without replacement.

    Example:
        >>> embedding.shape
        torch.Size([219520, 1536])
        >>> sampler = Random(embedding=embedding)
        >>> sampled_idxs = sampler.select_coreset_idxs()
        >>> coreset = embedding[sampled_idxs]
        >>> coreset.shape
        torch.Size([219, 1536])
    """
    SamplingMethod = Literal["permutation", "randint"]

    def __init__(
            self,
            embedding: torch.Tensor,
            sampling_ratio: float,
            sampling_method: SamplingMethod = "permutation"
    ) -> None:
        self._embedding = embedding
        self._coreset_size = int(embedding.shape[0] * sampling_ratio)
        self._sampling_method = sampling_method

    def sample_coreset(self) -> torch.Tensor:
        """Randomly select points from the embedding.

        Returns:
            Tensor: Random sample.

        Example:
            >>> embedding.shape
            torch.Size([219520, 1536])
            >>> sampler = Random(...)
            >>> coreset = sampler.sample_coreset()
            >>> coreset.shape
            torch.Size([219, 1536])
        """
        idx = self.generate_sample_idx_torch(
            embedding=self._embedding,
            sample_size=self._coreset_size,
            method=self._sampling_method
        )
        return self._embedding[idx]

    @staticmethod
    def generate_sample_idx_torch(
            embedding: torch.Tensor,
            sample_size: int,
            rng: torch.Generator | None = None,
            method: SamplingMethod = "permutation"
    ) -> torch.Tensor:
        total_size = len(embedding)
        device = embedding.device

        if rng is None:
            rng = torch.Generator(device=device)

        if method == "randint":
            return torch.randint(low=0, high=total_size, size=sample_size, generator=rng, device=device)
        if method == "permutation":
            return (torch.ones(total_size, device=device).multinomial(num_samples=sample_size, generator=rng))

        msg = f"Sampling method '{method}' not found, use one 'integers' or 'permutation'."
        raise AssertionError(msg)
