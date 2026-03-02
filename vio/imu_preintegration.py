"""
IMU Preintegration Module using PyPose.

Wraps PyPose's IMUPreintegrator to provide SE(3) state prediction
(prior state x̂_{t|t-1}) from high-frequency IMU measurements
(accelerometer + gyroscope).

The preintegrated IMU measurements are applied to the previous
posterior state to produce the prior state for the next time step,
strictly on the SE(3) manifold.
"""

import torch
import torch.nn as nn
import pypose as pp


class IMUPreintegrator(nn.Module):
    """PyPose-based IMU preintegration for SE(3) state prediction.

    Given the posterior state at time t-1 and IMU measurements between
    t-1 and t, computes the prior state at time t via preintegration
    on the SE(3) manifold.

    The state vector is defined as a PyPose SE3 LieTensor:
        x = [tx, ty, tz, qx, qy, qz, qw]  ∈ SE(3)

    Additionally, velocity (3-D) is tracked separately.

    Args:
        gravity: Gravity vector in the world frame, shape (3,).
        prop_cov: Whether to propagate covariance (for uncertainty).
        reset: Whether to reset the integrator after each use.
    """

    def __init__(
        self,
        gravity: float = 9.81,
        prop_cov: bool = True,
        reset: bool = True,
    ) -> None:
        super().__init__()
        self.prop_cov = prop_cov
        self.reset_flag = reset

        self.register_buffer(
            "gravity",
            torch.tensor([0.0, 0.0, -gravity], dtype=torch.float64),
        )

        # PyPose IMU preintegrator
        self.integrator = pp.module.IMUPreintegrator(
            gravity=self.gravity,
            prop_cov=self.prop_cov,
            reset=self.reset_flag,
        )

    def forward(
        self,
        x_posterior: torch.Tensor,
        velocity: torch.Tensor,
        imu_data: dict,
        dt: torch.Tensor,
    ) -> tuple:
        """Predict prior state from posterior state via IMU preintegration.

        Args:
            x_posterior: Previous posterior SE(3) pose, shape (B, 7).
            velocity: Previous velocity, shape (B, 3).
            imu_data: Dictionary with keys:
                - ``'gyro'``: Gyroscope readings, shape (B, N, 3).
                - ``'acc'``: Accelerometer readings, shape (B, N, 3).
                where N is the number of IMU samples between frames.
            dt: Time intervals for each IMU sample, shape (B, N).

        Returns:
            Tuple of:
                - x_prior: Predicted SE(3) pose, shape (B, 7).
                - vel_prior: Predicted velocity, shape (B, 3).
                - cov: Preintegration covariance (if prop_cov=True).
        """
        if not isinstance(x_posterior, pp.LieTensor):
            x_posterior = pp.SE3(x_posterior)

        gyro = imu_data["gyro"].double()  # (B, N, 3)
        acc = imu_data["acc"].double()  # (B, N, 3)
        dt_ = dt.double()  # (B, N)

        # Run preintegration
        state = self.integrator(
            dt=dt_,
            gyro=gyro,
            acc=acc,
            rot=x_posterior.rotation(),
            vel=velocity.double(),
            pos=x_posterior.translation(),
        )

        # Extract predicted pose components
        pos_prior = state["pos"][:, -1, :]  # (B, 3)
        rot_prior = state["rot"][:, -1, :]  # (B, 4) quaternion
        vel_prior = state["vel"][:, -1, :]  # (B, 3)

        # Construct SE(3) prior: [tx,ty,tz, qx,qy,qz,qw]
        x_prior = pp.SE3(torch.cat([pos_prior, rot_prior], dim=-1))

        cov = state.get("cov", None)
        if cov is not None:
            cov = cov[:, -1, :, :]  # (B, 9, 9)

        return x_prior, vel_prior.float(), cov
