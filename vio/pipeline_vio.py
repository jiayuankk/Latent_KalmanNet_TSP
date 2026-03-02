"""
VIO Pipeline with 3-Stage Curriculum Learning.

Implements the end-to-end training and inference pipeline for the
Latent VIO system, with a three-stage curriculum learning strategy:

    Stage 1: Freeze KalmanNet; train visual encoder E_θ and
             observation model h_φ with supervision.
    Stage 2: Freeze encoder; open-loop train KalmanNet GRUs using
             ground-truth trajectories.
    Stage 3: Joint fine-tuning of all components via BPTT.

The loss function uses the SE(3) geodesic loss (GeodesicLossSE3)
instead of Euclidean MSE.
"""

import time

import torch
import torch.nn as nn
import pypose as pp

from vio.spatiotemporal_encoder import SpatiotemporalLatentEncoder
from vio.observation_model import LatentObservationModel
from vio.manifold_kalmannet import ManifoldKalmanNet
from vio.geodesic_loss import GeodesicLossSE3


class VIOPipeline(nn.Module):
    """End-to-End Latent VIO Pipeline.

    Combines the visual encoder, observation model, and manifold KalmanNet
    into a unified pipeline with curriculum-learning-based training.

    Args:
        latent_dim: Dimension of the latent observation space.
        state_dim: Dimension of the se(3) tangent space (6).
        backbone: Backbone name for the visual encoder.
        pretrained: Whether to use ImageNet-pretrained backbone.
        alpha: Geodesic loss weight for translation.
        beta: Geodesic loss weight for rotation.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        state_dim: int = 6,
        backbone: str = "resnet18",
        pretrained: bool = True,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim

        # Visual encoder: image pairs → latent observations
        self.encoder = SpatiotemporalLatentEncoder(
            latent_dim=latent_dim,
            backbone_name=backbone,
            pretrained=pretrained,
        )

        # Observation model: SE(3) state → latent space
        self.obs_model = LatentObservationModel(
            latent_dim=latent_dim,
            state_dim=state_dim,
        )

        # Manifold KalmanNet
        self.kalman_net = ManifoldKalmanNet(
            state_dim=state_dim,
            latent_dim=latent_dim,
        )

        # Geodesic loss
        self.geodesic_loss = GeodesicLossSE3(alpha=alpha, beta=beta)

    def _freeze_params(self, module: nn.Module) -> None:
        """Freeze all parameters of a module."""
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_params(self, module: nn.Module) -> None:
        """Unfreeze all parameters of a module."""
        for param in module.parameters():
            param.requires_grad = True

    def set_stage(self, stage: int) -> None:
        """Configure parameter freezing for curriculum learning stage.

        Args:
            stage: Training stage (1, 2, or 3).
                Stage 1: Train encoder + obs_model only.
                Stage 2: Train KalmanNet only.
                Stage 3: Joint fine-tuning of all components.
        """
        if stage == 1:
            self._unfreeze_params(self.encoder)
            self._unfreeze_params(self.obs_model)
            self._freeze_params(self.kalman_net)
        elif stage == 2:
            self._freeze_params(self.encoder)
            self._freeze_params(self.obs_model)
            self._unfreeze_params(self.kalman_net)
        elif stage == 3:
            self._unfreeze_params(self.encoder)
            self._unfreeze_params(self.obs_model)
            self._unfreeze_params(self.kalman_net)
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, or 3.")

        n_trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        print(f"Stage {stage}: {n_trainable} trainable parameters")

    def forward_sequence(
        self,
        images: torch.Tensor,
        x_init: torch.Tensor,
        x_priors: torch.Tensor = None,
    ) -> tuple:
        """Process a sequence of image pairs through the full pipeline.

        Args:
            images: Image sequence of shape (B, T, C, H, W).
            x_init: Initial SE(3) pose, shape (B, 7).
            x_priors: Optional precomputed SE(3) prior states from IMU,
                      shape (B, T-1, 7). If None, uses identity propagation.

        Returns:
            Tuple of:
                - x_posteriors: Posterior SE(3) states, shape (B, T-1, 7).
                - z_observations: Latent observations, shape (B, T-1, d).
                - z_predictions: Predicted observations, shape (B, T-1, d).
        """
        B, T, C, H, W = images.shape
        device = images.device

        # Initialize
        if not isinstance(x_init, pp.LieTensor):
            x_init = pp.SE3(x_init)

        x_posterior = x_init
        x_prior_prev = x_init
        x_posterior_prev = x_init

        # First latent observation from initial frame pair (use frame 0 twice)
        z_prev = self.encoder(images[:, 0], images[:, 0])

        # Storage
        x_posteriors = []
        z_observations = []
        z_predictions = []

        self.kalman_net.init_hidden(B)

        for t in range(1, T):
            # --- Visual Encoding ---
            z_t = self.encoder(images[:, t - 1], images[:, t])  # (B, d)

            # --- Prior State ---
            if x_priors is not None:
                x_prior = pp.SE3(x_priors[:, t - 1])
            else:
                # Identity propagation (no IMU): prior = posterior
                x_prior = x_posterior

            # --- Predicted Observation ---
            z_hat = self.obs_model(x_prior)  # (B, d)

            # --- KalmanNet Step ---
            x_posterior_new = self.kalman_net.step(
                z_t=z_t,
                x_prior=x_prior,
                z_hat=z_hat,
                x_posterior_prev=x_posterior_prev,
                x_prior_prev=x_prior_prev,
                z_prev=z_prev,
            )

            # --- Update history ---
            x_prior_prev = x_prior
            x_posterior_prev = x_posterior
            x_posterior = x_posterior_new
            z_prev = z_t

            # --- Store ---
            x_posteriors.append(x_posterior.tensor())
            z_observations.append(z_t)
            z_predictions.append(z_hat)

        x_posteriors = torch.stack(x_posteriors, dim=1)  # (B, T-1, 7)
        z_observations = torch.stack(z_observations, dim=1)  # (B, T-1, d)
        z_predictions = torch.stack(z_predictions, dim=1)  # (B, T-1, d)

        return x_posteriors, z_observations, z_predictions

    def compute_loss(
        self,
        x_posteriors: torch.Tensor,
        x_gt: torch.Tensor,
        z_observations: torch.Tensor = None,
        z_predictions: torch.Tensor = None,
        stage: int = 3,
    ) -> dict:
        """Compute training loss based on the current curriculum stage.

        Args:
            x_posteriors: Predicted SE(3) posteriors, shape (B, T-1, 7).
            x_gt: Ground-truth SE(3) poses, shape (B, T-1, 7).
            z_observations: Latent observations (for Stage 1 aux loss).
            z_predictions: Predicted observations (for Stage 1 aux loss).
            stage: Current training stage.

        Returns:
            Dictionary with loss components.
        """
        B, T_seq, _ = x_posteriors.shape

        # Flatten for geodesic loss
        pred_flat = x_posteriors.reshape(B * T_seq, 7)
        gt_flat = x_gt.reshape(B * T_seq, 7)

        geo_loss = self.geodesic_loss(pred_flat, gt_flat)

        losses = {"geodesic": geo_loss, "total": geo_loss}

        # Stage 1: add latent-space consistency loss
        if stage == 1 and z_observations is not None and z_predictions is not None:
            latent_loss = nn.functional.mse_loss(z_predictions, z_observations)
            losses["latent"] = latent_loss
            losses["total"] = geo_loss + 0.1 * latent_loss

        return losses

    def train_epoch(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        stage: int,
        device: torch.device,
    ) -> dict:
        """Run one training epoch.

        Args:
            dataloader: DataLoader yielding (images, x_gt, x_init) tuples.
                - images: (B, T, C, H, W)
                - x_gt: (B, T-1, 7) ground-truth SE(3) poses
                - x_init: (B, 7) initial pose
            optimizer: Optimizer instance.
            stage: Current curriculum learning stage.
            device: Torch device.

        Returns:
            Dictionary of averaged loss components.
        """
        self.train()
        self.set_stage(stage)

        epoch_losses = {"geodesic": 0.0, "total": 0.0, "latent": 0.0}
        n_batches = 0

        for batch in dataloader:
            images, x_gt, x_init = batch
            images = images.to(device)
            x_gt = x_gt.to(device)
            x_init = x_init.to(device)

            optimizer.zero_grad()

            x_posteriors, z_obs, z_pred = self.forward_sequence(
                images, x_init
            )

            losses = self.compute_loss(
                x_posteriors, x_gt, z_obs, z_pred, stage=stage
            )

            losses["total"].backward()
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] += v.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        device: torch.device,
    ) -> dict:
        """Evaluate on a dataset.

        Args:
            dataloader: DataLoader yielding (images, x_gt, x_init).
            device: Torch device.

        Returns:
            Dictionary of averaged loss components.
        """
        self.eval()

        eval_losses = {"geodesic": 0.0, "total": 0.0}
        n_batches = 0

        for batch in dataloader:
            images, x_gt, x_init = batch
            images = images.to(device)
            x_gt = x_gt.to(device)
            x_init = x_init.to(device)

            x_posteriors, z_obs, z_pred = self.forward_sequence(
                images, x_init
            )

            losses = self.compute_loss(x_posteriors, x_gt)

            for k, v in losses.items():
                eval_losses[k] += v.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in eval_losses.items()}

    def curriculum_train(
        self,
        train_loader,
        val_loader,
        device: torch.device,
        stage1_epochs: int = 10,
        stage2_epochs: int = 20,
        stage3_epochs: int = 30,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        save_path: str = "vio_best.pt",
    ) -> dict:
        """Full 3-stage curriculum training.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            device: Torch device.
            stage1_epochs: Epochs for Stage 1.
            stage2_epochs: Epochs for Stage 2.
            stage3_epochs: Epochs for Stage 3.
            lr: Learning rate.
            weight_decay: L2 regularization weight.
            save_path: Path to save the best model.

        Returns:
            Training history dictionary.
        """
        self.to(device)
        history = {"stage": [], "epoch": [], "train_loss": [], "val_loss": []}
        best_val = float("inf")

        stages = [
            (1, stage1_epochs),
            (2, stage2_epochs),
            (3, stage3_epochs),
        ]

        for stage, n_epochs in stages:
            self.set_stage(stage)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )

            print(f"\n{'='*50}")
            print(f"Stage {stage}: {n_epochs} epochs")
            print(f"{'='*50}")

            for epoch in range(n_epochs):
                t0 = time.time()
                train_losses = self.train_epoch(
                    train_loader, optimizer, stage, device
                )
                val_losses = self.evaluate(val_loader, device)
                dt = time.time() - t0

                history["stage"].append(stage)
                history["epoch"].append(epoch)
                history["train_loss"].append(train_losses["total"])
                history["val_loss"].append(val_losses["total"])

                print(
                    f"  [{epoch+1:3d}/{n_epochs}] "
                    f"train={train_losses['total']:.6f} "
                    f"val={val_losses['total']:.6f} "
                    f"({dt:.1f}s)"
                )

                if val_losses["total"] < best_val:
                    best_val = val_losses["total"]
                    torch.save(self.state_dict(), save_path)
                    print(f"    → Saved best model (val={best_val:.6f})")

        return history
