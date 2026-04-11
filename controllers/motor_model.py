"""First-order motor dynamics model for quadcopter rotors.

Ported from reference_code/osprey_rl/osprey_rl/mdp/controller/motor_model.py.
Simulates motor spin-up/spin-down lag and computes thrust/moment from rotor speed.
"""

from __future__ import annotations

import torch

from controllers.config import MotorConfig, RotorConfig


class RotorMotor:
    """Batched first-order motor dynamics for 4 rotors.

    Computes thrust and yaw reaction moment from rotor angular velocity,
    with asymmetric spin-up/spin-down time constants.

    Args:
        num_envs: Number of parallel environments.
        rotor_cfg: Rotor configuration (thrust coefficients, limits, etc.).
        motor_cfg: Motor dynamics configuration (time constants).
        init_omega: Initial rotor speeds (num_envs, 4) [rad/s].
        device: Torch device.
    """

    def __init__(
        self,
        num_envs: int,
        rotor_cfg: RotorConfig,
        motor_cfg: MotorConfig,
        init_omega: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.num_envs = num_envs
        self.device = device

        self.tau_up = motor_cfg.tau_up
        self.tau_down = motor_cfg.tau_down
        self.omega_min = rotor_cfg.omega_min
        self.omega_max = rotor_cfg.omega_max

        self.thrust_coeff = torch.tensor(
            rotor_cfg.thrust_coeff, device=device, dtype=torch.float32
        )
        self.moment_constant = rotor_cfg.moment_constant
        self.directions = torch.tensor(
            rotor_cfg.directions, device=device, dtype=torch.float32
        )

        self.current_omega = init_omega.clone()

    def step(
        self, target_speeds: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance motor dynamics by one timestep.

        The motor response uses asymmetric first-order dynamics:
        alpha = exp(-dt / tau), where tau = tau_up if accelerating, tau_down if decelerating.

        Thrust and moment are computed from the CURRENT omega (before update),
        then omega is updated toward the target.

        Args:
            target_speeds: Desired rotor speeds (num_envs, 4) [rad/s].
            dt: Timestep [s].

        Returns:
            Tuple of:
                thrusts: Per-rotor thrust (num_envs, 4) [N].
                moments: Per-rotor yaw reaction moment (num_envs, 4) [N*m].
                omega: Updated rotor speeds (num_envs, 4) [rad/s].
        """
        # Asymmetric time constant
        tau = torch.where(
            target_speeds > self.current_omega, self.tau_up, self.tau_down
        )
        alpha = torch.exp(-dt / tau)

        # Compute thrust and moment from current omega
        thrusts = self.thrust_coeff * self.current_omega**2
        moments = self.moment_constant * thrusts * self.directions

        # First-order dynamics update
        self.current_omega = alpha * self.current_omega + (1.0 - alpha) * target_speeds
        self.current_omega = torch.clamp(
            self.current_omega, self.omega_min, self.omega_max
        )

        return thrusts, moments, self.current_omega

    def reset(self, env_ids: torch.Tensor, init_omega: torch.Tensor) -> None:
        """Reset motor state for specific environments.

        Args:
            env_ids: Environment indices to reset.
            init_omega: Initial rotor speeds (4,) [rad/s].
        """
        self.current_omega[env_ids] = init_omega
