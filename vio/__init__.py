"""
End-to-End Latent Visual-Inertial Odometry (VIO) Module.

This package implements the manifold-aware VIO pipeline that replaces
the traditional Euclidean-space KalmanNet with a Lie-group-constrained
architecture using PyPose for SE(3)/SO(3) operations.

Core components:
    - SpatiotemporalLatentEncoder: ResNet-based visual encoder for image pairs
    - LatentObservationModel: Learnable MLP mapping physical state to latent space
    - ManifoldKalmanNet: KalmanNet with Lie algebra manifold differences
    - IMUPreintegrator: PyPose-based IMU preintegration for state prediction
    - GeodesicLossSE3: SE(3) geodesic loss replacing Euclidean MSE
    - VIOPipeline: 3-stage curriculum learning pipeline
"""

from vio.spatiotemporal_encoder import SpatiotemporalLatentEncoder
from vio.observation_model import LatentObservationModel
from vio.geodesic_loss import GeodesicLossSE3
from vio.manifold_kalmannet import ManifoldKalmanNet
from vio.imu_preintegration import IMUPreintegrator
from vio.pipeline_vio import VIOPipeline

__all__ = [
    "SpatiotemporalLatentEncoder",
    "LatentObservationModel",
    "GeodesicLossSE3",
    "ManifoldKalmanNet",
    "IMUPreintegrator",
    "VIOPipeline",
]
