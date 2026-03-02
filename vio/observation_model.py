"""
Latent Observation Model h_φ(·).

Implements a learnable MLP that maps the physical prior state
(an SE(3) pose, represented as a 7-D vector [tx,ty,tz, qx,qy,qz,qw])
to the visual latent space ℝ^d, so that the Kalman innovation can
be computed as:

    Δz_t = z_t − h_φ( x̂_{t|t-1} )

where z_t is the encoder output and x̂_{t|t-1} is the prior state.
"""

import torch
import torch.nn as nn
import pypose as pp


class LatentObservationModel(nn.Module):
    """MLP mapping an SE(3) physical state to the visual latent space.

    The input is a PyPose SE3 LieTensor (or its raw 7-D tensor
    representation [tx,ty,tz, qx,qy,qz,qw]).  The model first
    converts the SE(3) pose into a local tangent-space representation
    via the logarithmic map (6-D se(3) vector), then maps it through
    an MLP to produce the predicted latent observation ẑ_t.

    Args:
        latent_dim: Dimension d of the latent observation space.
        state_dim: Dimension of the tangent-space input (default 6
                   for se(3) = ℝ^6).
        hidden_dim: Hidden layer width.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        state_dim: int = 6,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x_prior: torch.Tensor) -> torch.Tensor:
        """Map prior state to predicted latent observation.

        Args:
            x_prior: Prior state as either:
                - A PyPose SE3 LieTensor of shape (B, 7), or
                - A raw se(3) tangent vector of shape (B, 6).

        Returns:
            Predicted latent observation ẑ_t of shape (B, latent_dim).
        """
        if isinstance(x_prior, pp.LieTensor):
            # Convert SE(3) → se(3) via logarithmic map
            tangent = pp.Log(x_prior)  # (B, 6)
            tangent = tangent.tensor()
        elif x_prior.shape[-1] == 7:
            # Raw SE(3) tensor representation → wrap and log-map
            tangent = pp.Log(pp.SE3(x_prior))  # (B, 6)
            tangent = tangent.tensor()
        else:
            # Already in tangent space (B, 6)
            tangent = x_prior

        return self.mlp(tangent)  # (B, latent_dim)
