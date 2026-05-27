"""Temporal curvature regularizer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from typing import Optional


class CurveLoss(nn.Module):
    """Uniform cosine curvature: mean(1 - cos(Δz_t, Δz_{t+1}))."""

    log_key = "curve_loss"

    def __init__(self, step_thresh: float = 1e-6):
        super().__init__()
        self.step_thresh = step_thresh

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"Expected features of shape (B, T, D), got {tuple(features.shape)}"
            )
        if features.size(1) < 3:
            return features.new_tensor(0.0)

        v1 = features[:, 1:-1, :] - features[:, :-2, :]
        v2 = features[:, 2:, :] - features[:, 1:-1, :]
        loss = 1.0 - F.cosine_similarity(v1, v2, dim=-1, eps=1e-6)

        if self.step_thresh > 0:
            step1 = v1.norm(dim=-1)
            step2 = v2.norm(dim=-1)
            mask = (step1 > self.step_thresh) & (step2 > self.step_thresh)
            if mask.any():
                loss = loss[mask]

        return loss.mean()


CURVE_LOSS_TYPES = {
    "curve": CurveLoss,
}


def build_curve_loss(curve_cfg: Optional[DictConfig]) -> Optional[nn.Module]:
    """Instantiate the curve loss module, or None if curve_cfg is missing."""
    if curve_cfg is None:
        return None

    kind = str(OmegaConf.select(curve_cfg, "type", default="curve")).lower()
    cls = CURVE_LOSS_TYPES.get(kind)
    if cls is None:
        raise ValueError(
            f"Unknown loss.curve.type={kind!r}; choose from {list(CURVE_LOSS_TYPES)}."
        )

    step_thresh = float(OmegaConf.select(curve_cfg, "step_thresh", default=1e-6))
    return CurveLoss(step_thresh=step_thresh)
