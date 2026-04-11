"""INDI (Incremental Nonlinear Dynamic Inversion) attitude controller.

Ported from reference_code/osprey_rl/osprey_rl/mdp/controller/indi.py.
Adapted for Newton's XYZW quaternion convention and config-driven parameters.

The INDI controller takes collective thrust + desired body rates from the RL policy
and outputs individual rotor speed commands via the G1 allocation matrix.
"""

from __future__ import annotations

import math

import torch

from controllers.config import OspreyConfig, RotorConfig
from controllers.low_pass_filter import LowPassFilter


class IndiController:
    """Batched INDI attitude controller for the Osprey platform.

    INDI computes incremental control torques based on measured angular acceleration
    error, then allocates to individual rotor thrusts via the G1 inverse matrix.

    Roll/pitch use full INDI (incremental), yaw uses NDI (non-incremental).

    Args:
        num_envs: Number of parallel environments.
        config: Full Osprey platform configuration.
        device: Torch device.
    """

    def __init__(
        self,
        num_envs: int,
        config: OspreyConfig,
        device: torch.device,
    ) -> None:
        self.num_envs = num_envs
        self.device = device

        rcfg = config.rotor
        icfg = config.inertia
        ccfg = config.control
        fcfg = config.filter

        # G1 allocation matrix: maps [total_thrust, tau_roll, tau_pitch, tau_yaw]
        # to individual rotor thrusts [FR, FL, BL, BR]
        self.G1 = self._build_g1_matrix(rcfg).to(device)
        self.G1_inv = torch.linalg.inv(self.G1)

        # Thrust limits
        self.thrust_min = 0.0
        self.thrust_max = rcfg.thrust_max
        self.thrust_coeff = torch.tensor(
            rcfg.thrust_coeff, device=device, dtype=torch.float32
        )
        self.omega_min = rcfg.omega_min
        self.omega_max = rcfg.omega_max

        # Inertia matrix (static, no arm coupling for Phase 0)
        self.inertia_mat = torch.diag(
            torch.tensor(icfg.base_inertia_diag, device=device, dtype=torch.float32)
        )

        # Control gains
        self.k_alpha_cmd = torch.tensor(
            ccfg.k_alpha_cmd, device=device, dtype=torch.float32
        )

        # Filters — sampling freq must match actual sim freq
        sim_freq = config.sim.sim_freq
        fc = torch.full((num_envs, 1), fcfg.cutoff_freq, device=device)
        fs = torch.full((num_envs, 1), sim_freq, device=device)

        init_gyr = torch.zeros(num_envs, 3, device=device)
        init_mot = (
            torch.tensor(ccfg.init_omega, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(num_envs, -1)
            .clone()
        )

        self.filter_gyr = LowPassFilter(fc, fs, init_gyr)
        self.filter_mot = LowPassFilter(fc, fs, init_mot)

    @staticmethod
    def _build_g1_matrix(rcfg: RotorConfig) -> torch.Tensor:
        """Build the 4x4 control allocation matrix.

        Maps [total_thrust, tau_roll, tau_pitch, tau_yaw] to per-rotor thrusts.
        Rotor order: [front_right, front_left, back_left, back_right].

        The G1 matrix encodes the rotor geometry:
        - Row 0: thrust contribution (all ones)
        - Row 1: roll torque from lateral moment arms
        - Row 2: pitch torque from longitudinal moment arms
        - Row 3: yaw torque from reaction moments
        """
        beta1 = math.radians(rcfg.tilt_angles_deg[0])  # front tilt
        beta2 = math.radians(rcfg.tilt_angles_deg[1])  # back tilt
        l1 = rcfg.arm_lengths[0]  # front arm
        l2 = rcfg.arm_lengths[1]  # back arm
        kappa = rcfg.moment_constant

        G1 = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [
                    l1 * math.sin(beta1),
                    -l1 * math.sin(beta1),
                    -l2 * math.sin(beta2),
                    l2 * math.sin(beta2),
                ],
                [
                    -l1 * math.cos(beta1),
                    -l1 * math.cos(beta1),
                    l2 * math.cos(beta2),
                    l2 * math.cos(beta2),
                ],
                [kappa, -kappa, kappa, -kappa],
            ],
            dtype=torch.float32,
        )
        return G1

    def get_command(
        self,
        omega_body: torch.Tensor,
        rotor_speeds: torch.Tensor,
        collective_thrust: torch.Tensor,
        desired_rates: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target rotor speeds from INDI control law.

        The control law:
        1. Filters angular velocity and computes angular acceleration (omega_dot)
        2. Filters rotor speeds and computes current torques via G1
        3. Computes desired torques using INDI (roll/pitch) and NDI (yaw)
        4. Solves for per-rotor thrusts via G1_inv
        5. Converts thrusts to rotor speeds

        Args:
            omega_body: Body-frame angular velocity (num_envs, 3) [rad/s].
            rotor_speeds: Current rotor speeds (num_envs, 4) [rad/s].
            collective_thrust: Desired total thrust (num_envs,) [N].
            desired_rates: Desired body rates (num_envs, 3) [p, q, r] [rad/s].

        Returns:
            Target rotor speeds (num_envs, 4) [rad/s].
        """
        # Filter angular velocity and get derivative (angular acceleration)
        self.filter_gyr.add(omega_body)
        omega_f_dot = self.filter_gyr.derivative()

        # Compute desired angular acceleration from rate error
        alpha_cmd = self.k_alpha_cmd * (desired_rates - omega_body)

        # Filter rotor speeds and compute current thrusts
        self.filter_mot.add(rotor_speeds)
        rotor_speeds_filtered = self.filter_mot()

        thrusts_state = self.thrust_coeff * rotor_speeds_filtered**2
        thrusts_state = torch.clamp(thrusts_state, self.thrust_min, self.thrust_max)

        # Current torques from filtered thrusts: tau = G1[1:4, :] @ thrusts
        tau_f = torch.matmul(self.G1, thrusts_state.transpose(0, 1)).transpose(0, 1)[
            :, 1:
        ]

        # Build desired control wrench mu = [thrust, tau_x, tau_y, tau_z]
        mu = torch.zeros(self.num_envs, 4, device=self.device)

        # Collective thrust (clamped)
        mu[:, 0] = torch.clamp(collective_thrust, 0.0, self.thrust_max * 4.0)

        # INDI for roll/pitch: mu[1:3] = tau_f + J @ (alpha_cmd - omega_dot)
        indi_increment = self.inertia_mat.matmul(
            (alpha_cmd - omega_f_dot).transpose(0, 1)
        ).transpose(0, 1)
        mu[:, 1:3] = tau_f[:, 0:2] + indi_increment[:, 0:2]

        # NDI for yaw: mu[3] = J @ alpha_cmd + omega x (J @ omega)
        moments_ndi = self.inertia_mat.matmul(alpha_cmd.transpose(0, 1)).transpose(
            0, 1
        ) + torch.linalg.cross(
            omega_body,
            self.inertia_mat.matmul(omega_body.transpose(0, 1)).transpose(0, 1),
        )
        mu[:, 3] = moments_ndi[:, 2]

        # Solve for per-rotor thrusts: thrusts = G1_inv @ mu
        thrusts = self.G1_inv.matmul(mu.transpose(0, 1))
        thrusts = torch.clamp(thrusts, self.thrust_min, self.thrust_max)

        # Convert thrusts to rotor speeds: omega = sqrt(T / C_T)
        target_speeds = torch.sqrt(thrusts / self.thrust_coeff[:, None]).transpose(0, 1)
        target_speeds = torch.clamp(target_speeds, self.omega_min, self.omega_max)

        return target_speeds

    def reset(self, env_ids: torch.Tensor, config: OspreyConfig) -> None:
        """Reset filter state for specific environments.

        Args:
            env_ids: Environment indices to reset.
            config: Configuration for initial values.
        """
        self.filter_gyr.reset(
            env_ids,
            torch.tensor([0.0, 0.0, 0.0], device=self.device).unsqueeze(-1),
        )
        self.filter_mot.reset(
            env_ids,
            torch.tensor(config.control.init_omega, device=self.device).unsqueeze(-1),
        )
