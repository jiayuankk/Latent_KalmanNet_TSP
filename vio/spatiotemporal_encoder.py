"""
Spatiotemporal Latent Encoder for Visual-Inertial Odometry.

Implements a deep backbone (ResNet18) visual encoder that processes
consecutive image pairs (I_{t-1}, I_t) to extract spatiotemporal
features, producing a fixed-dimensional latent observation vector
z_t ∈ ℝ^d.

Architecture:
    - Siamese ResNet18 backbone (shared weights) extracts per-frame features
    - Temporal fusion layer merges the two feature vectors
    - Projection head maps fused features to the latent observation space
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SpatiotemporalLatentEncoder(nn.Module):
    """Siamese ResNet18 encoder for consecutive image pairs.

    Given two consecutive frames I_{t-1} and I_t, extracts per-frame
    features via a shared ResNet18 backbone, fuses them through a
    temporal fusion layer, and projects to the latent observation space.

    Args:
        latent_dim: Dimension d of the output latent observation z_t.
        backbone_name: Name of the torchvision backbone.
                       Currently supports ``'resnet18'`` and ``'resnet50'``.
        pretrained: Whether to load ImageNet-pretrained weights.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Build backbone (shared for siamese streams)
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            feat_dim = 512
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Remove the final FC layer; keep up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_dim = feat_dim

        # Temporal fusion: concatenate features from two frames then fuse
        self.temporal_fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

        # Projection head to latent observation space
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, latent_dim),
        )

    def _extract_features(self, img: torch.Tensor) -> torch.Tensor:
        """Extract features from a single image through the backbone.

        Args:
            img: Image tensor of shape (B, C, H, W).

        Returns:
            Feature vector of shape (B, feat_dim).
        """
        feat = self.backbone(img)  # (B, feat_dim, 1, 1)
        return feat.flatten(1)  # (B, feat_dim)

    def forward(
        self, img_prev: torch.Tensor, img_curr: torch.Tensor
    ) -> torch.Tensor:
        """Encode a pair of consecutive images into a latent observation.

        Args:
            img_prev: Previous frame I_{t-1}, shape (B, C, H, W).
            img_curr: Current frame I_t, shape (B, C, H, W).

        Returns:
            Latent observation z_t of shape (B, latent_dim).
        """
        feat_prev = self._extract_features(img_prev)  # (B, feat_dim)
        feat_curr = self._extract_features(img_curr)  # (B, feat_dim)

        # Temporal fusion via concatenation
        fused = torch.cat([feat_prev, feat_curr], dim=-1)  # (B, 2*feat_dim)
        fused = self.temporal_fusion(fused)  # (B, feat_dim)

        # Project to latent observation space
        z_t = self.projection(fused)  # (B, latent_dim)
        return z_t
