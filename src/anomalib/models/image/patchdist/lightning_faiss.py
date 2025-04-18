from importlib.util import find_spec
from pathlib import Path
from typing import Literal, Union
from typing import get_args

import torch

from anomalib.metrics import Evaluator
from anomalib.visualization import Visualizer
from .detector import KNNDetector
from .distribution import (
    DistanceDistribution,
    EmpiricalDistanceDistribution,
    NormalDistanceDistribution,
    HistogramDistanceDistribution
)
from .ensemble import EnsembleDetector, EnsembleIndex
from .lightning_model import PatchDist, CoresetMethodT, CoresetDeviceT, CoresetRatioT

__all__ = [
    "PatchDistFaiss"
]

ValidDistributionT = Union[Literal["empirical", "histogram"], NormalDistanceDistribution.valid_distribution]
PatchDistFaiss = None

if find_spec("faiss") is not None:
    from nearness import FaissNeighbors


    class PatchDistFaiss(PatchDist):
        def __init__(
                self,
                backbone: str = "wide_resnet50_2",
                backbone_path: str | Path | None = None,
                layer: str | None = "layer2",
                index: str = "Flat",
                score_distribution: ValidDistributionT | None = None,
                n_neighbors: int = 1,
                n_neighborhood: int = 128,
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
                ensemble_size: int = 1,  # (ensemble_size = 1 means no ensemble)
                ensemble_coreset_method: Literal["random", "kcenter", "kmedoids"] = "kcenter",
                ensemble_coreset_ratio_fit: float | int | None = None,
                ensemble_coreset_ratio_add: float | int | None = None,
                pre_processor: torch.nn.Module | bool = True,
                post_processor: torch.nn.Module | bool = True,
                evaluator: Evaluator | bool = True,
                visualizer: Visualizer | bool = True,
        ) -> None:
            base_detector = KNNDetector(n_neighbors=n_neighbors, reduction="max")
            detector = base_detector if ensemble_size == 1 else EnsembleDetector(base_detector=base_detector)
            base_index = FaissNeighbors(index=index, add_data_on_fit=True)
            index = base_index if ensemble_size == 1 else EnsembleIndex(
                base_index=base_index,
                n_indices=ensemble_size,
                coreset_method=ensemble_coreset_method,
                coreset_ratio_fit=ensemble_coreset_ratio_fit,
                coreset_ratio_add=ensemble_coreset_ratio_add,
            )
            distribution = get_distribution(
                score_distribution, n_neighborhood
            ) if score_distribution is not None else None
            super().__init__(
                backbone=backbone,
                backbone_path=backbone_path,
                layer=layer,
                index=index,
                detector=detector,
                score_distribution=distribution,
                score_quantile=score_quantile,
                incremental_indexing=incremental_indexing,
                coreset_ratio_start=coreset_ratio_start,
                coreset_ratio_step=coreset_ratio_step,
                coreset_ratio_end=coreset_ratio_end,
                coreset_method_start=coreset_method_start,
                coreset_method_step=coreset_method_step,
                coreset_method_end=coreset_method_end,
                coreset_device_start=coreset_device_start,
                coreset_device_step=coreset_device_step,
                coreset_device_end=coreset_device_end,
                coreset_aggregate_start=coreset_aggregate_start,
                coreset_batch_start=coreset_batch_start,
                pre_processor=pre_processor,
                post_processor=post_processor,
                evaluator=evaluator,
                visualizer=visualizer
            )


def get_distribution(
        name: ValidDistributionT,
        n_neighborhood: int
) -> DistanceDistribution:
    match name:
        case "empirical":
            return EmpiricalDistanceDistribution(n_neighbors=n_neighborhood, )
        case "histogram":
            return HistogramDistanceDistribution(
                n_neighbors=n_neighborhood,
                min_value=0.0,  # the minimum distance
            )
        case value if value in get_args(NormalDistanceDistribution.valid_distribution):
            return NormalDistanceDistribution(
                n_neighbors=n_neighborhood,
                distribution=value
            )
        case other:
            msg = f"Found invalid value for distribution: {other}."
            raise ValueError(msg)
