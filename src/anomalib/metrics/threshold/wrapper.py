"""Implementation of different thresholds in a shared wrapper."""

import logging
from typing import Literal

import torch
from typing_extensions import Any

from .base import BaseThreshold

logger = logging.getLogger(__name__)


class WrapperThreshold(BaseThreshold):
    """Anomaly Score Threshold.

    A wrapper to bundle alternative unsupervised thresholding approaches.

    Args:
        default_value: Default value of the threshold.
            Defaults to ``0.5``.
    """
    full_state_update: bool = False
    threshold_type = Literal["fixed", "survival", "iqr", "mad", "sigma"]

    def __init__(
            self,
            default_value: float = 0.5,
            method: threshold_type = "fixed",
            # useful for patchdist, to ensure only well-normalized values are used for thresholding
            ignore_first_n_values: int = 0,
            **method_kwargs: Any
    ) -> None:
        super().__init__()
        self.add_state("scores", default=[], persistent=True)
        self.value = torch.tensor(default_value)
        self.ignore_first_n_values = ignore_first_n_values
        self.method = method
        self.method_kwargs = method_kwargs

    def update(self, preds: torch.Tensor, target: torch.Tensor | None = None) -> None:
        """ Append the batch of image-wise or pixel-wise scores to the current scores."""
        # we don't need to aggregate scores if the method is fixed
        if self.method != "fixed":
            self.scores.append(preds)

    def compute(self) -> torch.Tensor:
        """ Determine the threshold for all gathered ``scores``"""
        scores = self.scores[self.ignore_first_n_values:]  # ignore first values inclusive
        if len(scores) > 0 and self.method != "fixed":
            cat_scores = torch.cat(scores, dim=0)
            value = threshold(cat_scores, method=self.method, **self.method_kwargs)
            self.value = value

        return self.value

    def compute_components(self) -> tuple[torch.Tensor, torch.Tensor]:
        """ Determine the threshold for all gathered ``scores``"""
        if self.method == "fixed" or self.method == "survival":
            val = self.compute()
            return val, torch.zeros_like(val)

        scores = self.scores[self.ignore_first_n_values:]  # ignore first values inclusive
        if len(scores) > 0:
            cat_scores = torch.cat(scores, dim=0)
            return threshold(cat_scores, method=self.method, return_components=True, **self.method_kwargs)
        else:
            raise ValueError("Can not compute components without scores.")

    def update_threshold_scale(self, scale: float) -> None:
        """Update the scale value of this object, used in thresholding."""
        self.method_kwargs["scale"] = scale


def threshold(scores: torch.Tensor,
              method: WrapperThreshold.threshold_type,
              **kwargs: Any) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    threshold_map = {
        "survival": threshold_survival,
        "iqr": threshold_iqr,
        "mad": threshold_mad,
        "sigma": threshold_sigma,
    }
    return threshold_map[method](scores, **kwargs)


def cutoff(scores: torch.Tensor, threshold: float | torch.Tensor) -> torch.Tensor:
    labels = torch.zeros_like(scores, dtype=torch.int)
    labels[scores >= threshold] = 1
    return labels


def threshold_survival(scores: torch.Tensor,
                       scale: float = 0.01,
                       return_components=False,
                       ) -> torch.Tensor:
    assert 0 <= scale <= 1
    x_sorted_val, _ = torch.sort(scores, dim=0, descending=True)
    survival_idx = int(len(scores) * scale)
    if return_components:
        logger.warning("Components are not supported for threshold survival - returning single threshold.")
    return x_sorted_val[survival_idx].squeeze()


def threshold_iqr(scores: torch.Tensor,
                  scale: float = 1.5,
                  return_components=False,
                  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    x_sorted_val, _ = torch.sort(scores, dim=0, descending=False)
    q1_idx, q3_idx = int(len(scores) * 0.25), int(len(scores) * 0.75)
    q1, q3 = x_sorted_val[q1_idx].squeeze(), x_sorted_val[q3_idx].squeeze()
    iqr = q3 - q1
    return (q3, iqr) if return_components else q3 + scale * iqr


def threshold_mad(scores: torch.Tensor,
                  scale: float = 3.0,
                  return_components=False,
                  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    median = torch.median(scores, dim=0).values
    abs_deviation = torch.abs(scores - median)
    median_abs_deviation = torch.median(abs_deviation, dim=0).values

    mean = torch.mean(scores, dim=0).squeeze()
    median_abs_deviation = median_abs_deviation.squeeze()
    return (mean, median_abs_deviation) if return_components else mean + scale * median_abs_deviation


def threshold_sigma(scores: torch.Tensor,
                    scale: float = 3.0,
                    return_components=False,
                    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    std, mean = torch.std_mean(scores, dim=0)
    mean = mean.squeeze()
    std = std.squeeze()
    return (mean.squeeze(), std.squeeze()) if return_components else mean.squeeze() + scale * std.squeeze()
