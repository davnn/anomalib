from typing import Any

import torch
from anomalib.metrics.threshold import WrapperThreshold
from anomalib.metrics.threshold.wrapper import ThresholdMethodT
from anomalib.post_processing import PostProcessor

__all__ = ["WrapperPostProcessor"]


class WrapperPostProcessor(PostProcessor):
    """A wrapper for the PostProcessor class with additional thresholding capabilities.

    This class extends the base PostProcessor with configurable thresholding methods
    for both image-level and pixel-level anomaly detection. It allows for different
    thresholding strategies and scales to be applied to the anomaly scores.

    Parameters
    ----------
    enable_normalization : bool, optional
        Whether to normalize the anomaly scores, by default True
    enable_thresholding : bool, optional
        Whether to apply thresholding to the anomaly scores, by default True
    enable_threshold_matching : bool, optional
        Whether to match thresholds between image and pixel levels, by default True
    image_sensitivity : float, optional
        Sensitivity parameter for image-level anomaly detection, by default 0.5
    pixel_sensitivity : float, optional
        Sensitivity parameter for pixel-level anomaly detection, by default 0.5
    image_threshold_method : ThresholdMethodT | None, optional
        Method to use for image-level thresholding, by default None
    image_threshold_scale : float, optional
        Scale factor for image-level thresholding, by default 3.0
    pixel_threshold_method : ThresholdMethodT | None, optional
        Method to use for pixel-level thresholding, by default None
    pixel_threshold_scale : float, optional
        Scale factor for pixel-level thresholding, by default 3.0
    pixel_threshold_shape : tuple[int, int] | None, optional
        Shape of the pixel-level threshold tensor, by default None, resulting in a scalar threshold
    **kwargs : Any
        Additional keyword arguments passed to the parent class

    """

    def __init__(
            self,
            enable_normalization: bool = True,
            enable_thresholding: bool = True,
            enable_threshold_matching: bool = True,
            image_sensitivity: float = 0.5,
            pixel_sensitivity: float = 0.5,
            image_threshold_method: ThresholdMethodT | None = None,
            image_threshold_scale: float = 3.0,
            pixel_threshold_method: ThresholdMethodT | None = None,
            pixel_threshold_scale: float = 3.0,
            pixel_threshold_shape: tuple[int, int] | None = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            enable_normalization=enable_normalization,
            enable_thresholding=enable_thresholding,
            enable_threshold_matching=enable_threshold_matching,
            image_sensitivity=image_sensitivity,
            pixel_sensitivity=pixel_sensitivity,
            **kwargs,
        )
        if image_threshold_method is not None:
            self._image_threshold_metric = WrapperThreshold(
                fields=["pred_score", "gt_label"],
                strict=False,
                method=image_threshold_method,
                scale=image_threshold_scale,
            )
        if pixel_threshold_method is not None:
            self._pixel_threshold_metric = WrapperThreshold(
                fields=["anomaly_map", "gt_mask"],
                strict=False,
                method=pixel_threshold_method,
                scale=pixel_threshold_scale,
                flatten_scores=pixel_threshold_shape is None,
            )
            if pixel_threshold_shape is not None:
                self.register_buffer("_pixel_threshold", torch.full(size=pixel_threshold_shape, fill_value=torch.nan))
