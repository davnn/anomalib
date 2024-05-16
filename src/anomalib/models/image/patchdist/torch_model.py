"""PyTorch model for the PatchDist model implementation."""

import torch
from nearness import TorchNeighbors, NearestNeighbors
from safecheck import typecheck
from torch import nn

from anomalib.models.components import TimmFeatureExtractor, AnomalyMapGenerator
from .anomaly_detector import KNNDetector, DistanceDistribution

__all__ = [
    "PatchDistModel",
]

PatchDistDefaultIndex = TorchNeighbors()
PatchDistDefaultDetector = KNNDetector(n_neighbors=3)


class PatchDistModel(nn.Module):
    """PatchDist Module.

    Args:
        input_size (tuple[int, int]): Input size for the model.
        layer (str): Layer used for feature extraction
        backbone (str, optional): Pre-trained model backbone.
            Defaults to ``resnet18``.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.
    """

    @typecheck
    def __init__(
        self,
        input_size: tuple[int, int],
        layer: str = "layer2",
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        index: NearestNeighbors = PatchDistDefaultIndex,
        detector: KNNDetector = PatchDistDefaultDetector,
        score_quantile: float = 0.99,
        score_distribution: DistanceDistribution | None = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.layer = layer
        self.input_size = input_size
        self.index = index
        self.detector = detector
        self.score_quantile = score_quantile
        self.score_distribution = score_distribution
        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            pre_trained=pre_trained,
            layers=[layer]
        )
        self.anomaly_map_generator = AnomalyMapGenerator(sigma=3).eval()

    @torch.inference_mode
    def forward(
        self,
        input_tensor: torch.Tensor,
        *,
        use_for_normalization: bool = False,
        disable_normalization: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.

        Args:
            input_tensor (torch.Tensor): Input tensor
            use_for_normalization (bool): Use the input tensor to learn a normalization distribution
            disable_normalization (bool): Do not normalize even if a distribution is available (useful for debugging)

        Returns:
            Tensor | dict[str, torch.Tensor]: Embedding for training, anomaly map and anomaly score for testing.
        """
        # generate the embedding
        embedding = self.feature_extractor(input_tensor)[self.layer]
        batch_size, _, height, width = embedding.shape
        embedding_flat = self.reshape_embedding(embedding)
        device = input_tensor.device

        if self.training:
            # directly return the flattened embedding if in training mode
            return embedding_flat

        # distance to the nearest patches
        patch_anomaly_score = self.detector.predict(embedding_flat.cpu().numpy(), self.index)
        # convert scores to tensor (as expected in the anomaly_map module)
        patch_anomaly_score = torch.as_tensor(patch_anomaly_score)
        # reshape to height and width
        patch_anomaly_score = patch_anomaly_score.reshape((batch_size, 1, height, width))

        # learn a normalization distribution
        if use_for_normalization and self.score_distribution is not None:
            self.score_distribution.update(embedding.cpu().numpy(), self.index)

        # normalize the patch-wise scores if a normalization distribution is available
        if self.score_distribution is not None and self.score_distribution.is_available:
            # if an update has been performed, we (re-) compute the distribution, otherwise use the computed value
            distribution = self.score_distribution.get(recompute=use_for_normalization)
            patch_anomaly_score = distribution.cdf(
                patch_anomaly_score
            ) if not disable_normalization else patch_anomaly_score

        # determine image score based on the patch score quantile
        image_anomaly_score = torch.quantile(patch_anomaly_score.reshape(batch_size, -1), self.score_quantile, dim=-1)
        # upscale the patch score to determine the image anomaly map
        image_anomaly_map = self.anomaly_map_generator(patch_anomaly_score.to(device), self.input_size)

        return {
            "anomaly_map": image_anomaly_map,
            "pred_score": image_anomaly_score.to(device),
            "patch_score": patch_anomaly_score
        }

    @staticmethod
    def reshape_embedding(embedding: torch.Tensor) -> torch.Tensor:
        """Reshape Embedding.

        Reshapes Embedding to the following format:
            - [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (torch.Tensor): Embedding tensor extracted from CNN features.

        Returns:
            Tensor: Reshaped embedding tensor.
        """
        embedding_size = embedding.size(1)
        return embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
