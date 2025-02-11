"""TODO: Add publication.
"""

import logging
from typing import Any, Literal, Dict

import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from nearness import NearestNeighbors
from torchvision.transforms.v2 import Compose, Normalize, Transform, Resize, InterpolationMode

from anomalib import LearningType
from anomalib.models.components import AnomalyModule, MemoryBankMixin, KCenterGreedy, KMedoids, Random
from .anomaly_detector import KNNDetector, DistanceDistribution
from .torch_model import PatchDistModel, PatchDistDefaultDetector, PatchDistDefaultIndex

logger = logging.getLogger(__name__)


class PatchDist(MemoryBankMixin, AnomalyModule):
    """PatchcoreLightning Module to train PatchCore algorithm.

    Args:
        input_size (tuple[int, int]): Size of the model input.
            Defaults to ``(224, 224)``.
        backbone (str): Backbone CNN network
            Defaults to ``wide_resnet50_2``.
        layer (str): Layer to extract features from the backbone CNN
            Defaults to ``["layer2", "layer3"]``.
    """

    def __init__(
        self,
        input_size: tuple[int, int] = (224, 224),
        backbone: str = "wide_resnet50_2",
        layer: str | None = "layer2",
        index: NearestNeighbors = PatchDistDefaultIndex,
        detector: KNNDetector = PatchDistDefaultDetector,
        score_distribution: DistanceDistribution | None = None,
        score_quantile: float = 0.99,
        incremental_indexing: bool = False,
        coreset_sampling_ratio_start: float | None = None,
        coreset_sampling_ratio_step: float | None = 0.1,
        coreset_sampling_ratio_end: float | None = None,
        coreset_sampling_type: Literal["random", "kcenter", "kmedoids"] = "kcenter",
        coreset_sampling_device: Literal["initial", "cpu", "auto"] = "initial"
    ) -> None:
        super().__init__()
        is_valid_coreset_value = lambda value: (isinstance(value, float) and (0 < value < 1)) or value is None
        if not (
            is_valid_coreset_value(coreset_sampling_ratio_start)
            and is_valid_coreset_value(coreset_sampling_ratio_step)
            and is_valid_coreset_value(coreset_sampling_ratio_end)
        ):
            msg = (
                f"The coreset sampling ratio must be a floating point number x with 0 < x < 1 or None, but found "
                f"start={coreset_sampling_ratio_start}, step={coreset_sampling_ratio_step}, "
                f"end={coreset_sampling_ratio_end:}."
            )
            raise ValueError(msg)
        self.coreset_sampling_ratio_start = coreset_sampling_ratio_start
        self.coreset_sampling_ratio_step = coreset_sampling_ratio_step
        self.coreset_sampling_ratio_end = coreset_sampling_ratio_end
        self.coreset_sampling_type = coreset_sampling_type
        # auto = use the lightning device for start and step-sampling and cpu for end sampling
        # initial = use the lightning device for all sampling
        # cpu = use cpu for all sampling
        self.coreset_sampling_device = coreset_sampling_device
        self.incremental_indexing = incremental_indexing
        self.model: PatchDistModel = PatchDistModel(
            input_size=input_size,
            layer=layer,
            backbone=backbone,
            index=index,
            detector=detector,
            score_quantile=score_quantile,
            score_distribution=score_distribution
        )
        self.embeddings: list[torch.Tensor] = []

    def get_coreset_sampling_device(
        self,
        step: Literal["start", "step", "end"]
    ) -> torch.device | str:
        if self.coreset_sampling_device == "auto":
            if step == "end":
                return "cpu"
            return self.device
        if self.coreset_sampling_device == "cpu":
            return "cpu"
        if self.coreset_sampling_device == "initial":
            return self.device

        raise AssertionError(f"Invalid coreset sampling device {self.coreset_sampling_device}")

    @staticmethod
    def get_coreset_sampling_ratio(
        input: float | None | tuple[float | None, float | None, float | None]
    ) -> tuple[float | None, float | None, float | None]:

        msg = f"Input for coreset sampling ratio is invalid, expected float, None or 3-tuple, but found {input}."
        if isinstance(input, tuple) and len(input) == 3:  # a valid tuple
            for ratio in input:
                assert is_valid_value(ratio), msg
            return input
        elif is_valid_value(input):  # a valid value
            return (input, input, input)
        else:
            raise AssertionError(msg)

    @staticmethod
    def determine_coreset_ratio(coreset_ratio: int | float, embedding: torch.Tensor):
        if isinstance(coreset_ratio, int):
            coreset_ratio = coreset_ratio / len(embedding)

        return coreset_ratio

    def configure_optimizers(self) -> None:
        """Configure optimizers.

        Returns:
            None: Do not set optimizers by returning None.
        """
        return

    @torch.inference_mode()
    def on_train_start(self) -> None:
        """Train the index with a random subset of the data.

        This is not necesary for most indexing structures, but some require a training stage, see:
        <https://github.com/facebookresearch/faiss/wiki/Faster-search>
        """
        if self.incremental_indexing:
            embeddings = []
            for item in self.trainer.datamodule.train_data:
                image = torch.unsqueeze(item["image"], 0)
                embedding = self.model(image.to(self.device))

                if self.coreset_sampling_ratio_start is not None:
                    ratio = self.determine_coreset_ratio(self.coreset_sampling_ratio_start, embedding)
                    embedding = torch.as_tensor(embedding, device=self.get_coreset_sampling_device("start"))
                    embedding = self.subsample_embedding(embedding, ratio)

                embeddings.append(embedding.cpu())

            embeddings = torch.vstack(embeddings)
            logger.info(f"Training index with data of size {embeddings.shape}.")
            self.model.index.fit(embeddings.numpy())

    @torch.inference_mode()
    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> None:
        """Generate feature embedding of the batch.

        Args:
            batch (dict[str, str | torch.Tensor]): Batch containing image filename, image, label and mask
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            dict[str, np.ndarray]: Embedding Vector
        """
        del args, kwargs  # These variables are not used.
        embedding = self.model(batch["image"].to(self.device))

        # Log the embedding size for the first batch
        is_first_batch = self.trainer.global_step == 0
        if is_first_batch:
            logger.info(f"First training batch with embeddings {embedding.shape}")

        # Coreset sample the embedding
        if self.coreset_sampling_ratio_step is not None:
            ratio = self.determine_coreset_ratio(self.coreset_sampling_ratio_step, embedding)
            embedding = torch.as_tensor(embedding, device=self.get_coreset_sampling_device("step"))
            embedding = self.subsample_embedding(embedding, ratio)

        # Add elements incrementally to the index
        if self.incremental_indexing:
            self.model.index.add(embedding.cpu().numpy())

        # Store the embedding for later indexing and training
        self.embeddings.append(embedding.cpu())

    @torch.inference_mode()
    def fit(self) -> None:
        """Learn an indexing structure """
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        if self.coreset_sampling_ratio_end is not None:
            logger.info(f"Coreset sampling the embedding of size {embeddings.shape}.")
            ratio = self.determine_coreset_ratio(self.coreset_sampling_ratio_end, embeddings)
            embeddings = torch.as_tensor(embeddings, device=self.get_coreset_sampling_device("end"))
            embeddings = self.subsample_embedding(embeddings, ratio)

        # convert coreset sample to numpy
        embeddings = embeddings.cpu().numpy()

        if not self.incremental_indexing:
            logger.info(f"Learning an index structure with embeddings {embeddings.shape}.")
            self.model.index.fit(embeddings)

        logger.info(f"Learning an outlier detector with embeddings {embeddings.shape}.")
        self.model.detector.fit(embeddings, self.model.index)

    @torch.inference_mode()
    def validation_step(self, batch: dict[str, str | torch.Tensor], use_for_normalization: bool = True) -> STEP_OUTPUT:
        """Get batch of anomaly maps from input image batch.

        Args:
            batch (dict[str, str | torch.Tensor]): Batch containing image filename, image, label and mask
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: Image filenames, test images, GT and predicted label/masks
        """
        # Get anomaly maps and predicted scores from the model.
        output = self.model(batch["image"].to(self.device), use_for_normalization=use_for_normalization)

        # Add anomaly maps and predicted scores to the batch.
        batch["anomaly_maps"] = output["anomaly_map"]
        batch["pred_scores"] = output["pred_score"]

        return batch

    def predict_step(
        self,
        batch: dict[str, str | torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> STEP_OUTPUT:
        """Step function called during :meth:`~lightning.pytorch.trainer.Trainer.predict`.

        By default, it calls :meth:`~lightning.pytorch.core.lightning.LightningModule.forward`.
        Override to add any processing logic.

        Args:
            batch (Any): Current batch
            batch_idx (int): Index of current batch
            dataloader_idx (int): Index of the current dataloader

        Return:
            Predicted output
        """
        del batch_idx, dataloader_idx  # These variables are not used.
        return self.validation_step(batch, use_for_normalization=False)

    def configure_transforms(self, image_size: tuple[int, int] | None = None) -> Transform:
        # use default imagenet normalization with resizing to the given input size
        # antialiasing is not relevant for LANCZOS mode, therefore false
        # choosing the right interpolation algorithm appears to be black art, but LANCZOS
        # should be a good option:
        # https://stackoverflow.com/questions/384991/what-is-the-best-image-downscaling-algorithm-quality-wise
        return Compose(
            [
                Resize(self.model.input_size, antialias=True),
                # Resize(self.model.input_size, interpolation=InterpolationMode.LANCZOS, antialias=False),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return PatchDist trainer arguments."""
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

    def subsample_embedding(self, embedding: torch.Tensor, sampling_ratio: float) -> torch.Tensor:
        """Subsample embedding based on coreset sampling and store to memory.

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio
        """
        if self.coreset_sampling_type == "kcenter":
            sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
        elif self.coreset_sampling_type == "kmedoids":
            sampler = KMedoids(embedding=embedding, sampling_ratio=sampling_ratio)
        elif self.coreset_sampling_type == "random":
            sampler = Random(embedding=embedding, sampling_ratio=sampling_ratio)
        else:
            raise AssertionError(f"Coreset sampling type {self.coreset_sampling_type} does not exist.")

        coreset = sampler.sample_coreset()
        return coreset

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["index"] = self.model.index
        checkpoint["detector"] = self.model.detector
        checkpoint["score_distribution"] = self.model.score_distribution

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.model.index = checkpoint["index"]
        self.model.detector = checkpoint["detector"]
        self.model.score_distribution = checkpoint["score_distribution"]
