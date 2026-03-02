"""
Manifold-Aware KalmanNet for Visual-Inertial Odometry.

Extends the original KalmanNet architecture with Lie-group-constrained
operations.  The four difference features fed into the GRU networks
(GRU_Q, GRU_Sigma, GRU_S) are computed using the logarithmic map
(boxminus ⊟) on the SE(3) manifold, rather than Euclidean subtraction.

The posterior update uses the exponential map (boxplus ⊞):
    x̂_{t|t} = x̂_{t|t-1} ⊞ (K_t · Δz_t)

Key manifold-difference features:
    1. fw_evol_diff  = Log( x̂_{t|t-1}⁻¹ ∘ x̂_{t-1|t-1} )  — prior evolution
    2. fw_update_diff = Log( x̂_{t|t}⁻¹ ∘ x̂_{t|t-1} )      — posterior update
    3. obs_diff       = z_t − z_{t-1}                         — observation diff (Euclidean, latent space)
    4. obs_innov_diff  = z_t − ẑ_t                            — innovation (Euclidean, latent space)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pypose as pp

from vio.observation_model import LatentObservationModel


class ManifoldKalmanNet(nn.Module):
    """KalmanNet with SE(3) manifold-aware state estimation.

    Args:
        state_dim: Dimension of the Lie algebra tangent space (6 for se(3)).
        latent_dim: Dimension d of the latent observation space.
        hidden_mult: Multiplier for GRU hidden dimensions.
    """

    def __init__(
        self,
        state_dim: int = 6,
        latent_dim: int = 128,
        hidden_mult: int = 1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim  # se(3) tangent dimension
        self.latent_dim = latent_dim
        self.m = state_dim
        self.d = latent_dim

        # ---- GRU networks for implicit covariance tracking ----

        # GRU_Q: tracks process noise covariance via forward evolution diff
        self.d_input_Q = state_dim * hidden_mult
        self.d_hidden_Q = state_dim ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q, batch_first=True)

        # GRU_Sigma: tracks state covariance
        self.d_input_Sigma = self.d_hidden_Q + state_dim * hidden_mult
        self.d_hidden_Sigma = state_dim ** 2
        self.GRU_Sigma = nn.GRU(
            self.d_input_Sigma, self.d_hidden_Sigma, batch_first=True
        )

        # GRU_S: tracks innovation covariance
        self.d_input_S = latent_dim ** 2 + 2 * latent_dim * hidden_mult
        self.d_hidden_S = latent_dim ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S, batch_first=True)

        # ---- Fully Connected layers ----

        # FC5: process forward evolution diff
        self.FC5 = nn.Sequential(
            nn.Linear(state_dim, state_dim * hidden_mult),
            nn.ReLU(),
        )

        # FC6: process forward update diff
        self.FC6 = nn.Sequential(
            nn.Linear(state_dim, state_dim * hidden_mult),
            nn.ReLU(),
        )

        # FC7: process observation diffs [obs_diff, obs_innov_diff]
        self.FC7 = nn.Sequential(
            nn.Linear(latent_dim * 2, 2 * latent_dim * hidden_mult),
            nn.ReLU(),
        )

        # FC1: Sigma → innovation space
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_hidden_Sigma, latent_dim ** 2),
            nn.ReLU(),
        )

        # FC2: compute Kalman gain (Sigma + S → KG)
        self.d_output_FC2 = state_dim * latent_dim
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_hidden_S + self.d_hidden_Sigma, self.d_hidden_S),
            nn.ReLU(),
            nn.Linear(self.d_hidden_S, self.d_output_FC2),
        )

        # FC3: backward flow
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_hidden_S + self.d_output_FC2, state_dim ** 2),
            nn.ReLU(),
        )

        # FC4: update hidden state of Sigma-GRU
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_hidden_Sigma + state_dim ** 2, self.d_hidden_Sigma),
            nn.ReLU(),
        )

        # ---- Priors for hidden states ----
        self.register_buffer(
            "prior_Q", torch.eye(state_dim).flatten().unsqueeze(0).unsqueeze(0)
        )
        self.register_buffer(
            "prior_Sigma",
            torch.eye(state_dim).flatten().unsqueeze(0).unsqueeze(0),
        )
        self.register_buffer(
            "prior_S",
            torch.eye(latent_dim).flatten().unsqueeze(0).unsqueeze(0),
        )

    def init_hidden(self, batch_size: int = 1) -> None:
        """Initialize GRU hidden states from covariance priors."""
        device = self.prior_Q.device
        self.h_Q = self.prior_Q.expand(1, batch_size, -1).contiguous().to(device)
        self.h_Sigma = (
            self.prior_Sigma.expand(1, batch_size, -1).contiguous().to(device)
        )
        self.h_S = self.prior_S.expand(1, batch_size, -1).contiguous().to(device)

    @staticmethod
    def _manifold_diff(pose_a: pp.LieTensor, pose_b: pp.LieTensor) -> torch.Tensor:
        """Compute manifold difference using log map: Log(a⁻¹ ∘ b).

        This is the Lie-algebraic ⊟ (boxminus) operator.

        Args:
            pose_a: SE(3) pose, shape (B, 7).
            pose_b: SE(3) pose, shape (B, 7).

        Returns:
            Tangent vector in se(3), shape (B, 6).
        """
        relative = pose_a.Inv() @ pose_b
        return pp.Log(relative).tensor()

    def compute_kalman_gain(
        self,
        obs_diff: torch.Tensor,
        obs_innov_diff: torch.Tensor,
        fw_evol_diff: torch.Tensor,
        fw_update_diff: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the data-driven Kalman gain from difference features.

        All inputs are assumed to be (B, 1, dim) for GRU batch_first.

        Returns:
            Kalman gain of shape (B, state_dim, latent_dim).
        """
        # Normalize features
        obs_diff = F.normalize(obs_diff, p=2, dim=-1, eps=1e-12)
        obs_innov_diff = F.normalize(obs_innov_diff, p=2, dim=-1, eps=1e-12)
        fw_evol_diff = F.normalize(fw_evol_diff, p=2, dim=-1, eps=1e-12)
        fw_update_diff = F.normalize(fw_update_diff, p=2, dim=-1, eps=1e-12)

        # ---- Forward Flow ----

        # FC5 → GRU_Q
        out_FC5 = self.FC5(fw_evol_diff)  # (B, 1, m*mult)
        out_Q, self.h_Q = self.GRU_Q(out_FC5, self.h_Q)

        # FC6 → GRU_Sigma
        out_FC6 = self.FC6(fw_update_diff)
        in_Sigma = torch.cat([out_Q, out_FC6], dim=-1)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC1
        out_FC1 = self.FC1(out_Sigma)

        # FC7 → GRU_S
        in_FC7 = torch.cat([obs_diff, obs_innov_diff], dim=-1)
        out_FC7 = self.FC7(in_FC7)
        in_S = torch.cat([out_FC1, out_FC7], dim=-1)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        # FC2: Kalman gain
        in_FC2 = torch.cat([out_Sigma, out_S], dim=-1)
        KG_flat = self.FC2(in_FC2)  # (B, 1, m*d)

        # ---- Backward Flow ----

        in_FC3 = torch.cat([out_S, KG_flat], dim=-1)
        out_FC3 = self.FC3(in_FC3)

        in_FC4 = torch.cat([out_Sigma, out_FC3], dim=-1)
        out_FC4 = self.FC4(in_FC4)

        # Update Sigma hidden state
        self.h_Sigma = out_FC4.transpose(0, 1).contiguous()

        # Reshape gain to matrix
        B = KG_flat.shape[0]
        KG = KG_flat.view(B, self.state_dim, self.latent_dim)

        return KG

    def step(
        self,
        z_t: torch.Tensor,
        x_prior: pp.LieTensor,
        z_hat: torch.Tensor,
        x_posterior_prev: pp.LieTensor,
        x_prior_prev: pp.LieTensor,
        z_prev: torch.Tensor,
    ) -> pp.LieTensor:
        """Perform one Kalman filter step on the SE(3) manifold.

        Args:
            z_t: Current latent observation, shape (B, d).
            x_prior: Prior SE(3) state x̂_{t|t-1}, shape (B, 7).
            z_hat: Predicted latent observation ẑ_t = h_φ(x̂_{t|t-1}), shape (B, d).
            x_posterior_prev: Previous posterior x̂_{t-1|t-1}, shape (B, 7).
            x_prior_prev: Previous prior x̂_{t-1|t-2}, shape (B, 7).
            z_prev: Previous latent observation z_{t-1}, shape (B, d).

        Returns:
            Posterior SE(3) state x̂_{t|t} as a PyPose SE3 LieTensor.
        """
        B = z_t.shape[0]

        # ---- Compute difference features ----

        # Observation difference (Euclidean in latent space)
        obs_diff = (z_t - z_prev).unsqueeze(1)  # (B, 1, d)

        # Innovation (Euclidean in latent space)
        obs_innov_diff = (z_t - z_hat).unsqueeze(1)  # (B, 1, d)

        # Forward evolution diff on manifold: Log(x̂_{t-1|t-1}⁻¹ ∘ x̂_{t|t-1})
        fw_evol_diff = self._manifold_diff(
            x_posterior_prev, x_prior
        ).unsqueeze(1)  # (B, 1, 6)

        # Forward update diff on manifold: Log(x̂_{t|t-1}⁻¹ ∘ x̂_{t-1|t-1})
        fw_update_diff = self._manifold_diff(
            x_prior_prev, x_posterior_prev
        ).unsqueeze(1)  # (B, 1, 6)

        # ---- Compute Kalman Gain ----
        KG = self.compute_kalman_gain(
            obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff
        )  # (B, 6, d)

        # ---- Innovation in latent space ----
        delta_z = (z_t - z_hat).unsqueeze(-1)  # (B, d, 1)

        # Kalman gain times innovation → tangent vector
        correction = torch.bmm(KG, delta_z).squeeze(-1)  # (B, 6)

        # ---- Posterior update via exponential map (boxplus ⊞) ----
        # x̂_{t|t} = x̂_{t|t-1} ⊞ correction
        # i.e., x̂_{t|t} = x̂_{t|t-1} ∘ Exp(correction)
        correction_lie = pp.se3(correction).Exp()  # SE3 LieTensor (B, 7)
        x_posterior = x_prior @ correction_lie

        return x_posterior
