"""PyTorch model for the PatchDist model implementation."""
import enum
from importlib.metadata import PathDistribution
from pathlib import Path

import torch
import timm
import warnings
from anomalib.models.components import AnomalyMapGenerator
from fontTools.misc.cython import returns
from nearness import TorchNeighbors, NearestNeighbors
from safecheck import typecheck
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import get_model

from .anomaly_detector import KNNDetector, DistanceDistribution
from .model_registry import get_local_model

__all__ = [
    "PatchDistModel",
]

PatchDistDefaultIndex = TorchNeighbors()
PatchDistDefaultDetector = KNNDetector(n_neighbors=3)


class PatchDistBackbone(enum.Enum):
    sam2 = "SAM2"
    timm = "TIMM"
    torchvision = "torchvision"
    local = "local"


def get_backbone_kind(backbone: str) -> PatchDistBackbone:
    if backbone.startswith("facebook/sam2"):
        return PatchDistBackbone.sam2

    if backbone.startswith("torchvision/"):
        return PatchDistBackbone.torchvision

    if backbone.startswith("local/"):
        return PatchDistBackbone.local

    return PatchDistBackbone.timm


class PatchDistModel(nn.Module):
    """PatchDist Module.

    Args:
        input_size (tuple[int, int]): Input size for the model.
        layer (str): Layer used for feature extraction
        backbone (str, optional): Pre-trained model backbone.
            Defaults to ``resnet18``.
    """

    @typecheck
    def __init__(
        self,
        input_size: tuple[int, int],
        layer: str | None = "layer2",
        backbone: str = "wide_resnet50_2",
        backbone_path: str | Path | None = None,
        index: NearestNeighbors = PatchDistDefaultIndex,
        detector: KNNDetector = PatchDistDefaultDetector,
        score_quantile: float = 0.99,
        score_distribution: DistanceDistribution | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone_path = backbone_path
        self.backbone_kind = get_backbone_kind(backbone)
        self.layer = layer
        self.input_size = input_size
        self.index = index
        self.detector = detector
        self.score_quantile = score_quantile
        self.score_distribution = score_distribution
        self.feature_extractor = self._get_feature_extractor(self.backbone, layer)
        self.anomaly_map_generator = AnomalyMapGenerator(sigma=3).eval()

    def _get_feature_extractor(self, backbone: str, layer: str | None):
        kind = self.backbone_kind
        if kind == PatchDistBackbone.sam2:
            if self.backbone_path is not None:
                warnings.warn("Cannot load SAM2 model from path, ignoring configured backbone path.")

            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                predictor = SAM2ImagePredictor.from_pretrained(backbone, device="cpu")
                model = predictor.model.image_encoder

            except ModuleNotFoundError:
                msg = (f"Cannot use backbone '{backbone}', if the SAM2 package is not installed, "
                       f"make sure to install SAM2 from https://github.com/facebookresearch/sam2.")
                raise ModuleNotFoundError(msg)

        if kind == PatchDistBackbone.timm or kind == PatchDistBackbone.torchvision or kind == PatchDistBackbone.local:
            # cannot use features only, not available for vision transformers (for example)
            model = load_model(model_name=self.backbone, model_path=self.backbone_path, model_kind=kind)

        return model if layer is None else create_feature_extractor(model, return_nodes=[layer])

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        res = self.feature_extractor(x)
        return res if self.layer is None else res[self.layer]

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
        embedding = self.extract_features(input_tensor)
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
        # note that, for a given first batch used for normalization of size >= ``score_distribution.min_samples``,
        # the entire batch is normalized, but for given batches with ``use_for_normalization=True`` of
        # total size < ``score_distribution.min_samples``, the returned values are un-normalized
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


def load_model(model_name: str, model_path: str | Path | None, model_kind: PatchDistBackbone) -> nn.Module:
    # cannot use features only for timm, not available for vision transformers (for example)
    load_fn = {
        PatchDistBackbone.timm: timm.create_model,
        PatchDistBackbone.torchvision: get_model,
        PatchDistBackbone.local: get_local_model
    }[model_kind]
    model_name = model_name.replace(f"{model_kind.value}/", "")

    if model_path is not None:
        model = load_fn(self.backbone)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    # catch deprecation warning loading model with pretrained=True for torchvision
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return load_fn(model_name, pretrained=True)
