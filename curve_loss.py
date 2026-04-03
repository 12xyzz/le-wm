import torch
import torch.nn.functional as F


def cosine_curvature_loss(
    features: torch.Tensor,
    step_thresh: float = 1e-6,
) -> torch.Tensor:
    """
    Cosine-based curvature regularizer over temporal trajectories.

    Args:
        features: Tensor of shape (B, T, D) representing a sequence of embeddings.
        step_thresh: Minimum step size for both adjacent differences to be
            included in the curvature computation.

    Returns:
        Scalar tensor with the average curvature loss.
    """
    if features.ndim != 3:
        raise ValueError(
            f"Expected features of shape (B, T, D), got {tuple(features.shape)}"
        )

    if features.size(1) < 3:
        # Not enough timesteps to define curvature; return zero.
        return features.new_tensor(0.0)

    # First and second temporal differences
    v1 = features[:, 1:-1, :] - features[:, :-2, :]
    v2 = features[:, 2:, :] - features[:, 1:-1, :]

    cos = F.cosine_similarity(v1, v2, dim=-1, eps=1e-6)
    loss = 1.0 - cos

    if step_thresh > 0:
        step1 = v1.norm(dim=-1)
        step2 = v2.norm(dim=-1)
        mask = (step1 > step_thresh) & (step2 > step_thresh)
        if mask.any():
            loss = loss[mask]

    return loss.mean()

