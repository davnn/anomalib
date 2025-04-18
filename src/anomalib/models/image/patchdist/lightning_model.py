"""TODO: Add publication.
"""
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from typing import Literal

import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from nearness import NearestNeighbors
from torchvision.transforms.v2 import Compose, Normalize, Resize, CenterCrop
from tqdm import tqdm

from anomalib import LearningType
from anomalib.data import Batch, AnomalibDataModule
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule, MemoryBankMixin, KCenterGreedy, KMedoids, Random, Uncertainty
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer
from .detector import Detector
from .distribution import (
    DistanceDistribution
)
from .torch_model import PatchDistModel, PatchDistDefaultDetector, PatchDistDefaultIndex

__all__ = [
    "PatchDist",
    "CoresetRatioT",
    "CoresetMethodT",
    "CoresetDeviceT"
]

logger = logging.getLogger(__name__)

CoresetRatioT = float | int | None
CoresetMethodT = Literal["random", "kcenter", "kmedoids", "uncertainty"]
CoresetDeviceT = torch.device | str | None


@dataclass(frozen=True)
class CoresetMethod:
    start: CoresetMethodT
    step: CoresetMethodT
    end: CoresetMethodT


@dataclass(frozen=True)
class CoresetRatio:
    start: CoresetRatioT
    step: CoresetRatioT
    end: CoresetRatioT


@dataclass(frozen=True)
class CoresetDevice:
    start: CoresetDeviceT
    step: CoresetDeviceT
    end: CoresetDeviceT


class PatchDist(MemoryBankMixin, AnomalibModule):
    """PatchDist.
    """

    def __init__(
            self,
            backbone: str = "wide_resnet50_2",
            backbone_path: str | Path | None = None,
            layer: str | None = "layer2",
            index: NearestNeighbors = PatchDistDefaultIndex,
            detector: Detector = PatchDistDefaultDetector,
            score_distribution: DistanceDistribution | None = None,
            score_quantile: float = 0.99,
            incremental_indexing: bool = False,
            coreset_ratio_start: CoresetRatioT = None,
            coreset_ratio_step: CoresetRatioT = 0.1,
            coreset_ratio_end: CoresetRatioT = None,
            coreset_method_start: CoresetMethodT = "kcenter",
            coreset_method_step: CoresetMethodT = "kcenter",
            coreset_method_end: CoresetMethodT = "kcenter",
            coreset_device_start: CoresetDeviceT = None,
            coreset_device_step: CoresetDeviceT = None,
            coreset_device_end: CoresetDeviceT = None,
            coreset_aggregate_start: bool = False,
            coreset_batch_start: bool = False,
            pre_processor: torch.nn.Module | bool = True,
            post_processor: torch.nn.Module | bool = True,
            evaluator: Evaluator | bool = True,
            visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        self.coreset_ratio = CoresetRatio(
            start=coreset_ratio_start,
            step=coreset_ratio_step,
            end=coreset_ratio_end
        )
        self.coreset_method = CoresetMethod(
            start=coreset_method_start,
            step=coreset_method_step,
            end=coreset_method_end
        )
        self.coreset_device = CoresetDevice(
            start=coreset_device_start,
            step=coreset_device_step,
            end=coreset_device_end
        )
        self.coreset_aggregate_start = coreset_aggregate_start
        self.coreset_batch_start = coreset_batch_start
        self.incremental_indexing = incremental_indexing
        self.model: PatchDistModel = PatchDistModel(
            layer=layer,
            backbone=backbone,
            backbone_path=backbone_path,
            index=index,
            detector=detector,
            score_quantile=score_quantile,
            score_distribution=score_distribution,
        )
        self.embeddings: list[torch.Tensor] = []

    @staticmethod
    def get_coreset_ratio(coreset_ratio: int | float, embedding: torch.Tensor):
        coreset_ratio = coreset_ratio / len(embedding) if isinstance(coreset_ratio, int) else coreset_ratio
        is_valid_ratio = lambda value: (isinstance(value, float) and (0 < value < 1)) or value is None
        if not is_valid_ratio(coreset_ratio):
            msg = (
                f"The coreset sampling ratio must be a floating point number x with 0 < x < 1 or None, "
                f"but found {coreset_ratio}."
            )
            raise ValueError(msg)

        return coreset_ratio

    def get_coreset_device(self, device: str | torch.device | None) -> torch.device:
        """
        Determine the torch device used for coreset sampling

        Note: The pytorch lightning device (self.device) is not usable in __init__ as it changes
        when moved to the real device, therefore this method has to be called when the model has
        already been moved to the correct device.
        """
        match device:
            case None:
                return self.device
            case torch.device():
                return device
            case str():
                return torch.device(device)
        raise ValueError(f"Invalid coreset sampling device: {device}")

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
        fully_fitted = self.model.detector.is_fitted and self.model.index.is_fitted
        uses_uncertainty_on_start = self.coreset_method.start == "uncertainty" and self.incremental_indexing
        if uses_uncertainty_on_start and not fully_fitted:
            msg = "Cannot use uncertainty sampling on train start without a fitted index and detector."
            raise ValueError(msg)

        uses_uncertainty = "uncertainty" in [self.coreset_method.step, self.coreset_method.end]
        if uses_uncertainty and not self.incremental_indexing:
            msg = "Using uncertainty sampling without incremental indexing."
            warnings.warn(msg)

        if self.incremental_indexing:
            embeddings = []
            use_coreset = self.coreset_ratio.start is not None
            logger.info(f"Using incremental indexing: coreset={use_coreset} aggregate={self.coreset_aggregate_start}")
            datamodule: AnomalibDataModule = self.trainer.datamodule  # type: ignore
            dataloader = datamodule.train_dataloader() if self.coreset_batch_start else datamodule.train_data
            total_size = self.trainer.num_training_batches if self.coreset_batch_start else len(dataloader)
            for item in tqdm(dataloader, desc="Generating training data embedding", total=total_size):
                image = item.image if self.coreset_batch_start else item.image.unsqueeze(dim=0)
                embedding = self.model(image.to(self.device))
                if use_coreset and not self.coreset_aggregate_start:
                    embedding = self.subsample_embedding(embedding, "start")
                embeddings.append(embedding)

            # stack the (possibly coreset-sampled) embeddings to a single tensor
            embeddings = torch.vstack(embeddings)

            # coreset-sample the aggregated embeddings if configured to do so
            if use_coreset and self.coreset_aggregate_start:
                logger.info(f"Coreset sampling aggregate embedding of size {embeddings.shape}.")
                embeddings = self.subsample_embedding(embeddings, step="start")

            # convert the embeddings to numpy to fit the index and detector
            embeddings = embeddings.numpy(force=True)
            logger.info(f"Training index with data of size {embeddings.shape}.")
            self.model.index.fit(embeddings)
            self.model.detector.fit(embeddings, self.model.index)

    @torch.inference_mode()
    def training_step(self, batch: Batch, *args, **kwargs) -> None:
        """Generate feature embedding of the batch."""
        del args, kwargs  # These variables are not used.
        embedding = self.model(batch.image.to(self.device))

        # Log the embedding size for the first batch
        is_first_batch = self.trainer.global_step == 0
        is_first_batch and logger.info(f"First training batch embedding: {embedding.shape}")

        # Coreset sample the embedding
        if self.coreset_ratio.step is not None:
            embedding = self.subsample_embedding(embedding, step="step")
            is_first_batch and logger.info(f"First training batch coreset: {embedding.shape} ")

        # Move embedding to cpu (possibly on other device)
        embedding = embedding.cpu()

        # Add elements incrementally to the index
        if self.incremental_indexing:
            self.model.index.add(embedding.numpy())

        # Store the embedding for later indexing and training
        self.embeddings.append(embedding)

    @torch.inference_mode()
    def fit(self) -> None:
        """Learn an indexing structure """
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        if self.coreset_ratio.end is not None:
            logger.info(f"Coreset sampling embedding of size: {embeddings.shape}")
            embeddings = self.subsample_embedding(embeddings, step="end")

        # convert coreset sample to numpy
        embeddings = embeddings.cpu().numpy()

        if not self.incremental_indexing:
            logger.info(f"Learning an index structure with embeddings: {embeddings.shape}.")
            self.model.index.fit(embeddings)

        logger.info(f"Learning an outlier detector with embeddings: {embeddings.shape}.")
        self.model.detector.fit(embeddings, self.model.index)

    @torch.inference_mode()
    def validation_step(self, batch: Batch, *args: Any, use_for_normalization: bool = True, **kwargs) -> STEP_OUTPUT:
        """Get batch of anomaly maps from input image batch.

        Args are any additional (unused) arguments provided by anomalib.

        Args:
            batch (dict[str, str | torch.Tensor]): Batch containing image filename, image, label and mask
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: Image filenames, test images, GT and predicted label/masks
        """
        # Get anomaly maps and predicted scores from the model.
        predictions = self.model(batch.image.to(self.device), use_for_normalization=use_for_normalization)
        return batch.update(**predictions)

    def predict_step(
            self,
            batch: Batch,
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

    @classmethod
    def configure_pre_processor(
            cls,
            image_size: tuple[int, int] | None = None,
            center_crop_size: tuple[int, int] | None = None,
    ) -> PreProcessor:
        image_size = image_size or (256, 256)
        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]
        if center_crop_size is not None:
            # scale center crop size proportional to image size
            return PreProcessor(
                transform=Compose([
                    Resize(image_size, antialias=True),
                    CenterCrop(center_crop_size),
                    Normalize(mean=normalize_mean, std=normalize_std),
                ])
            )

        return PreProcessor(
            transform=Compose([
                Resize(image_size, antialias=True),
                Normalize(mean=normalize_mean, std=normalize_std),
            ])
        )

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        """Configure the default post-processor.

        Returns:
            PostProcessor: Post-processor for one-class models that
                converts raw scores to anomaly predictions
        """
        return PostProcessor()

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return PatchDist trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

    def subsample_embedding(
            self,
            embedding: torch.Tensor,
            step: Literal["start", "step", "end"]
    ) -> torch.Tensor:
        """Subsample embedding based on coreset sampling and store to memory.
        """
        coreset_method = getattr(self.coreset_method, step)
        coreset_ratio = getattr(self.coreset_ratio, step)
        coreset_ratio = self.get_coreset_ratio(coreset_ratio, embedding=embedding)
        coreset_device = getattr(self.coreset_device, step)
        coreset_device = self.get_coreset_device(coreset_device)
        embedding = torch.as_tensor(embedding, device=coreset_device)

        if coreset_method == "kcenter":
            hide_progress = step == "step" or (step == "start" and not self.coreset_aggregate_start)
            sampler = KCenterGreedy(
                embedding=embedding,
                sampling_ratio=coreset_ratio,
                progress=not hide_progress
            )
        elif coreset_method == "kmedoids":
            sampler = KMedoids(embedding=embedding, sampling_ratio=coreset_ratio)
        elif coreset_method == "random":
            sampler = Random(embedding=embedding, sampling_ratio=coreset_ratio)
        elif coreset_method == "uncertainty":
            sampler = Uncertainty(
                embedding=embedding,
                sampling_ratio=coreset_ratio,
                detector=self.model.detector,
                index=self.model.index
            )
        else:
            raise AssertionError(f"Coreset sampling method {coreset_method} does not exist.")

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
