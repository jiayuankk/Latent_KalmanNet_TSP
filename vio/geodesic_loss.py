"""
SE(3) Geodesic Loss Function.

Replaces the standard Euclidean nn.MSELoss with a geodesic distance on
the SE(3) manifold, properly weighting rotation and translation errors.

The geodesic loss decomposes into:

    L = α · ‖Δt‖² + β · ‖Log(ΔR)‖²

where:
    ΔT = T_pred⁻¹ ∘ T_gt  ∈ SE(3)
    Δt = translational component of ΔT
    ΔR = rotational component of ΔT ∈ SO(3)
    Log(ΔR) ∈ so(3) is the rotation angle-axis vector

This ensures the loss respects the manifold geometry of rotations
and avoids the discontinuities of Euler-angle or quaternion L2 losses.
"""

import torch
import torch.nn as nn
import pypose as pp


class GeodesicLossSE3(nn.Module):
    """SE(3) geodesic loss combining translation L2 and rotation geodesic.

    Args:
        alpha: Weight for the translational component.
        beta: Weight for the rotational component.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute geodesic loss between predicted and target SE(3) poses.

        Args:
            pred: Predicted poses as SE3 LieTensor or raw (B, 7) tensor
                  with format [tx,ty,tz, qx,qy,qz,qw].
            target: Ground-truth poses, same format as pred.

        Returns:
            Scalar geodesic loss.
        """
        if not isinstance(pred, pp.LieTensor):
            pred = pp.SE3(pred)
        if not isinstance(target, pp.LieTensor):
            target = pp.SE3(target)

        # Relative transform: ΔT = T_pred⁻¹ ∘ T_gt
        delta = pred.Inv() @ target  # SE3 LieTensor (B, 7)

        # Logarithmic map to se(3) tangent space: (B, 6)
        # [v_x, v_y, v_z, ω_x, ω_y, ω_z]
        log_delta = pp.Log(delta).tensor()  # (B, 6)

        # Decompose into translation (first 3) and rotation (last 3)
        trans_err = log_delta[..., :3]  # (B, 3)
        rot_err = log_delta[..., 3:]  # (B, 3)

        loss_trans = torch.mean(torch.sum(trans_err ** 2, dim=-1))
        loss_rot = torch.mean(torch.sum(rot_err ** 2, dim=-1))

        return self.alpha * loss_trans + self.beta * loss_rot
