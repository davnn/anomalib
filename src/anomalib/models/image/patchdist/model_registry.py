import torch
import warnings
from typing import TypedDict
from pathlib import Path

from ..efficient_ad.torch_model import SmallPatchDescriptionNetwork, MediumPatchDescriptionNetwork

__all__ = ["get_local_model"]


class ModelRegistry(dict[str, torch.nn.Module]):
    ...


registry = ModelRegistry()
registry["efficientad_pdn_s"] = SmallPatchDescriptionNetwork(out_channels=384)
registry["efficientad_pdn_m"] = MediumPatchDescriptionNetwork(out_channels=384)


def get_local_model(model_name: str, pretrained: bool = False):
    if pretrained:
        msg = "Local model is not available with pretrained=True, please specify the path to the weights manually."

    model = registry.get(model_name)
    return model
