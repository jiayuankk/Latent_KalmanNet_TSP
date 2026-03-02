"""
Unit tests for the VIO (Visual-Inertial Odometry) module.

Tests cover instantiation and forward-pass correctness of all
new VIO components:
    - SpatiotemporalLatentEncoder
    - LatentObservationModel
    - GeodesicLossSE3
    - ManifoldKalmanNet
    - VIOPipeline
"""

import torch
import pypose as pp
import pytest

from vio.spatiotemporal_encoder import SpatiotemporalLatentEncoder
from vio.observation_model import LatentObservationModel
from vio.geodesic_loss import GeodesicLossSE3
from vio.manifold_kalmannet import ManifoldKalmanNet
from vio.pipeline_vio import VIOPipeline


# ---------------------------------------------------------------------------
# SpatiotemporalLatentEncoder
# ---------------------------------------------------------------------------
class TestSpatiotemporalLatentEncoder:
    def test_output_shape(self):
        encoder = SpatiotemporalLatentEncoder(
            latent_dim=64, backbone_name="resnet18", pretrained=False
        )
        B, C, H, W = 2, 3, 224, 224
        img_prev = torch.randn(B, C, H, W)
        img_curr = torch.randn(B, C, H, W)
        z = encoder(img_prev, img_curr)
        assert z.shape == (B, 64)

    def test_gradient_flow(self):
        encoder = SpatiotemporalLatentEncoder(
            latent_dim=32, backbone_name="resnet18", pretrained=False
        )
        img = torch.randn(1, 3, 224, 224, requires_grad=True)
        z = encoder(img, img)
        z.sum().backward()
        assert img.grad is not None
        assert not torch.all(img.grad == 0)


# ---------------------------------------------------------------------------
# LatentObservationModel
# ---------------------------------------------------------------------------
class TestLatentObservationModel:
    def test_output_shape_from_lie_tensor(self):
        model = LatentObservationModel(latent_dim=64, state_dim=6)
        x_prior = pp.identity_SE3(3)
        z_hat = model(x_prior)
        assert z_hat.shape == (3, 64)

    def test_output_shape_from_raw_se3(self):
        model = LatentObservationModel(latent_dim=64, state_dim=6)
        x_prior = pp.identity_SE3(2).tensor()  # raw (2, 7)
        z_hat = model(x_prior)
        assert z_hat.shape == (2, 64)

    def test_output_shape_from_tangent(self):
        model = LatentObservationModel(latent_dim=64, state_dim=6)
        tangent = torch.zeros(4, 6)
        z_hat = model(tangent)
        assert z_hat.shape == (4, 64)

    def test_gradient_flow(self):
        model = LatentObservationModel(latent_dim=32, state_dim=6)
        tangent = torch.randn(1, 6, requires_grad=True)
        z_hat = model(tangent)
        z_hat.sum().backward()
        assert tangent.grad is not None


# ---------------------------------------------------------------------------
# GeodesicLossSE3
# ---------------------------------------------------------------------------
class TestGeodesicLossSE3:
    def test_zero_loss_for_identical_poses(self):
        loss_fn = GeodesicLossSE3()
        pose = pp.identity_SE3(5).tensor()
        loss = loss_fn(pose, pose)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss_for_different_poses(self):
        loss_fn = GeodesicLossSE3()
        pred = pp.identity_SE3(3).tensor()
        noise = torch.randn(3, 6) * 0.1
        target = (pp.identity_SE3(3) @ pp.se3(noise).Exp()).tensor()
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_symmetry(self):
        loss_fn = GeodesicLossSE3()
        a = pp.identity_SE3(2).tensor()
        noise = torch.randn(2, 6) * 0.05
        b = (pp.identity_SE3(2) @ pp.se3(noise).Exp()).tensor()
        loss_ab = loss_fn(a, b)
        loss_ba = loss_fn(b, a)
        assert loss_ab.item() == pytest.approx(loss_ba.item(), abs=1e-5)

    def test_gradient_flow(self):
        loss_fn = GeodesicLossSE3()
        pred = pp.identity_SE3(2).tensor().detach().requires_grad_(True)
        target = pp.identity_SE3(2).tensor()
        loss = loss_fn(pred, target)
        loss.backward()
        # At identity, gradient may be zero; just check no errors occurred
        assert pred.grad is not None


# ---------------------------------------------------------------------------
# ManifoldKalmanNet
# ---------------------------------------------------------------------------
class TestManifoldKalmanNet:
    def test_manifold_diff(self):
        a = pp.identity_SE3(2)
        b = pp.identity_SE3(2)
        diff = ManifoldKalmanNet._manifold_diff(a, b)
        assert diff.shape == (2, 6)
        assert torch.allclose(diff, torch.zeros(2, 6), atol=1e-6)

    def test_step_output_is_se3(self):
        net = ManifoldKalmanNet(state_dim=6, latent_dim=16)
        B = 2
        net.init_hidden(B)

        z_t = torch.randn(B, 16)
        x_prior = pp.identity_SE3(B)
        z_hat = torch.randn(B, 16)
        x_post_prev = pp.identity_SE3(B)
        x_prior_prev = pp.identity_SE3(B)
        z_prev = torch.randn(B, 16)

        x_posterior = net.step(z_t, x_prior, z_hat, x_post_prev, x_prior_prev, z_prev)

        assert isinstance(x_posterior, pp.LieTensor)
        assert x_posterior.tensor().shape == (B, 7)

    def test_gradient_flow(self):
        net = ManifoldKalmanNet(state_dim=6, latent_dim=8)
        B = 1
        net.init_hidden(B)

        z_t = torch.randn(B, 8, requires_grad=True)
        x_prior = pp.identity_SE3(B)
        z_hat = torch.randn(B, 8)
        x_post_prev = pp.identity_SE3(B)
        x_prior_prev = pp.identity_SE3(B)
        z_prev = torch.randn(B, 8)

        x_posterior = net.step(z_t, x_prior, z_hat, x_post_prev, x_prior_prev, z_prev)
        x_posterior.tensor().sum().backward()
        assert z_t.grad is not None


# ---------------------------------------------------------------------------
# VIOPipeline
# ---------------------------------------------------------------------------
class TestVIOPipeline:
    def test_forward_sequence(self):
        pipeline = VIOPipeline(
            latent_dim=16, state_dim=6, backbone="resnet18", pretrained=False
        )
        B, T = 2, 4
        images = torch.randn(B, T, 3, 224, 224)
        x_init = pp.identity_SE3(B).tensor()

        x_post, z_obs, z_pred = pipeline.forward_sequence(images, x_init)

        assert x_post.shape == (B, T - 1, 7)
        assert z_obs.shape == (B, T - 1, 16)
        assert z_pred.shape == (B, T - 1, 16)

    def test_compute_loss(self):
        pipeline = VIOPipeline(
            latent_dim=16, state_dim=6, backbone="resnet18", pretrained=False
        )
        B, T_seq = 2, 3
        x_post = pp.identity_SE3(B * T_seq).tensor().reshape(B, T_seq, 7)
        x_gt = pp.identity_SE3(B * T_seq).tensor().reshape(B, T_seq, 7)

        losses = pipeline.compute_loss(x_post, x_gt)
        assert "geodesic" in losses
        assert "total" in losses
        assert losses["geodesic"].item() == pytest.approx(0.0, abs=1e-5)

    def test_stage_configuration(self):
        pipeline = VIOPipeline(
            latent_dim=8, state_dim=6, backbone="resnet18", pretrained=False
        )

        # Stage 1: encoder + obs_model trainable, kalman_net frozen
        pipeline.set_stage(1)
        assert any(p.requires_grad for p in pipeline.encoder.parameters())
        assert any(p.requires_grad for p in pipeline.obs_model.parameters())
        assert not any(p.requires_grad for p in pipeline.kalman_net.parameters())

        # Stage 2: kalman_net trainable, rest frozen
        pipeline.set_stage(2)
        assert not any(p.requires_grad for p in pipeline.encoder.parameters())
        assert not any(p.requires_grad for p in pipeline.obs_model.parameters())
        assert any(p.requires_grad for p in pipeline.kalman_net.parameters())

        # Stage 3: everything trainable
        pipeline.set_stage(3)
        assert any(p.requires_grad for p in pipeline.encoder.parameters())
        assert any(p.requires_grad for p in pipeline.obs_model.parameters())
        assert any(p.requires_grad for p in pipeline.kalman_net.parameters())

    def test_manifold_consistency(self):
        """Verify that posteriors remain valid SE(3) elements."""
        pipeline = VIOPipeline(
            latent_dim=8, state_dim=6, backbone="resnet18", pretrained=False
        )
        B, T = 1, 3
        images = torch.randn(B, T, 3, 224, 224)
        x_init = pp.identity_SE3(B).tensor()

        with torch.no_grad():
            x_post, _, _ = pipeline.forward_sequence(images, x_init)

        # Every output should be a valid SE(3) element
        for t in range(T - 1):
            pose = pp.SE3(x_post[:, t, :])
            identity = pose @ pose.Inv()
            log_err = pp.Log(identity).tensor().abs().max().item()
            assert log_err < 1e-4, f"Manifold error at t={t}: {log_err}"
