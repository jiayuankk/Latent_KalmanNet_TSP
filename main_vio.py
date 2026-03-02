"""
Main entry point for the End-to-End Latent Visual-Inertial Odometry system.

This script demonstrates the instantiation and forward pass of the
complete VIO pipeline, including:
    - SpatiotemporalLatentEncoder (ResNet18 backbone)
    - LatentObservationModel (MLP h_φ)
    - ManifoldKalmanNet (SE(3) manifold-aware Kalman gain)
    - GeodesicLossSE3 (SE(3) geodesic loss)
    - VIOPipeline (3-stage curriculum learning)

Usage:
    python main_vio.py [--latent_dim 128] [--backbone resnet18]
                       [--stage1_epochs 10] [--stage2_epochs 20]
                       [--stage3_epochs 30] [--lr 1e-4]
"""

import argparse

import torch
import pypose as pp

from vio import VIOPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-End Latent VIO Pipeline"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=128, help="Latent observation dim"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
    )
    parser.add_argument("--stage1_epochs", type=int, default=10)
    parser.add_argument("--stage2_epochs", type=int, default=20)
    parser.add_argument("--stage3_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Build Pipeline ----
    pipeline = VIOPipeline(
        latent_dim=args.latent_dim,
        state_dim=6,
        backbone=args.backbone,
        pretrained=True,
        alpha=1.0,
        beta=1.0,
    ).to(device)

    n_params = sum(p.numel() for p in pipeline.parameters())
    print(f"Total parameters: {n_params:,}")

    # ---- Synthetic Demo Forward Pass ----
    B = args.batch_size
    T = args.seq_len
    C, H, W = 3, 224, 224

    # Random images
    images = torch.randn(B, T, C, H, W, device=device)

    # Identity initial pose
    x_init = pp.identity_SE3(B).to(device).tensor()

    # Ground-truth poses (identity + small perturbations)
    x_gt_list = []
    for t in range(T - 1):
        noise = torch.randn(B, 6, device=device) * 0.01
        x_gt_t = (pp.identity_SE3(B).to(device) @ pp.se3(noise).Exp()).tensor()
        x_gt_list.append(x_gt_t)
    x_gt = torch.stack(x_gt_list, dim=1)  # (B, T-1, 7)

    print(f"\nImages shape: {images.shape}")
    print(f"Initial pose shape: {x_init.shape}")
    print(f"GT poses shape: {x_gt.shape}")

    # ---- Forward Pass ----
    pipeline.eval()
    with torch.no_grad():
        x_posteriors, z_obs, z_pred = pipeline.forward_sequence(images, x_init)

    print(f"\nPosterior poses shape: {x_posteriors.shape}")
    print(f"Latent observations shape: {z_obs.shape}")
    print(f"Latent predictions shape: {z_pred.shape}")

    # ---- Compute Loss ----
    losses = pipeline.compute_loss(x_posteriors, x_gt, z_obs, z_pred, stage=3)
    print(f"\nGeodesic loss: {losses['geodesic'].item():.6f}")
    print(f"Total loss: {losses['total'].item():.6f}")

    # ---- Verify Manifold Constraint ----
    # Check that posteriors are valid SE(3) elements
    posteriors_se3 = pp.SE3(x_posteriors.reshape(-1, 7))
    identity_check = posteriors_se3 @ posteriors_se3.Inv()
    log_identity = pp.Log(identity_check).tensor()
    max_error = log_identity.abs().max().item()
    print(f"\nManifold consistency check (max Log(T @ T^-1)): {max_error:.2e}")

    # ---- Stage Configuration Demo ----
    for stage in [1, 2, 3]:
        pipeline.set_stage(stage)

    print("\nVIO Pipeline instantiation and forward pass successful.")


if __name__ == "__main__":
    main()
