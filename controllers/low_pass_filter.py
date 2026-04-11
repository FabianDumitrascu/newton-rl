"""Second-order Butterworth low-pass IIR filter.

Ported from reference_code/osprey_rl/osprey_rl/mdp/controller/low_pass_filter.py.
Operates on batched PyTorch tensors for GPU-parallel environments.
"""

from __future__ import annotations

import math

import torch


class LowPassFilter:
    """Batched 2nd-order Butterworth low-pass filter.

    Supports (num_envs, dim) signals. Each environment gets independent filter state.

    Args:
        fc: Cutoff frequency per env (num_envs, 1) [Hz].
        fs: Sampling frequency per env (num_envs, 1) [Hz].
        initial_value: Initial signal value (num_envs, dim).
    """

    def __init__(
        self,
        fc: torch.Tensor,
        fs: torch.Tensor,
        initial_value: torch.Tensor,
    ) -> None:
        self.sampling_freq = fs  # (num_envs, 1)

        # Compute Butterworth coefficients
        self.num = self._init_num(fc, fs)  # (num_envs, 1, 2)
        self.den = self._init_den(fc, fs)  # (num_envs, 1, 2)

        # Filter state: last 2 inputs and outputs per dimension
        # Shape: (num_envs, dim, 2) where [:, :, 0] = current, [:, :, 1] = previous
        self.input = initial_value.unsqueeze(2).repeat(1, 1, 2)
        self.output = initial_value.unsqueeze(2).repeat(1, 1, 2)

    @staticmethod
    def _init_num(fc: torch.Tensor, fs: torch.Tensor) -> torch.Tensor:
        """Compute numerator coefficients via bilinear transform."""
        K = torch.tan(math.pi * fc / fs)
        poly = K * K + math.sqrt(2.0) * K + 1.0
        num = torch.zeros_like(fc).repeat(1, 2)
        num[:, 0] = (K * K / poly).squeeze(1)
        num[:, 1] = 2.0 * num[:, 0]
        return num.unsqueeze(1)

    @staticmethod
    def _init_den(fc: torch.Tensor, fs: torch.Tensor) -> torch.Tensor:
        """Compute denominator coefficients via bilinear transform."""
        K = torch.tan(math.pi * fc / fs)
        poly = K * K + math.sqrt(2.0) * K + 1.0
        den = torch.zeros_like(fc).repeat(1, 2)
        den[:, 0] = (2.0 * (K * K - 1.0) / poly).squeeze(1)
        den[:, 1] = ((K * K - math.sqrt(2.0) * K + 1.0) / poly).squeeze(1)
        return den.unsqueeze(1)

    def add(self, sample: torch.Tensor) -> torch.Tensor:
        """Feed a new sample and return the filtered output.

        Args:
            sample: New input signal (num_envs, dim).

        Returns:
            Filtered output (num_envs, dim).
        """
        # Shift input history
        x2 = self.input[:, :, 1]
        self.input[:, :, 1] = self.input[:, :, 0]
        self.input[:, :, 0] = sample

        # IIR filter: y[n] = num[0]*x[n-2] + num[1]*x[n-1] + num[0]*x[n]
        #                    - den[0]*y[n-1] - den[1]*y[n-2]
        out = self.num[:, :, 0] * x2 + (
            self.num * self.input - self.den * self.output
        ).sum(dim=2)

        # Shift output history
        self.output[:, :, 1] = self.output[:, :, 0]
        self.output[:, :, 0] = out

        return out

    def derivative(self) -> torch.Tensor:
        """Compute the time derivative of the filtered signal.

        Returns:
            Derivative (num_envs, dim) = fs * (y[n] - y[n-1]).
        """
        return self.sampling_freq * (self.output[:, :, 0] - self.output[:, :, 1])

    def __call__(self) -> torch.Tensor:
        """Return the current filtered output.

        Returns:
            Current output (num_envs, dim).
        """
        return self.output[:, :, 0]

    def reset(self, env_ids: torch.Tensor, val: torch.Tensor) -> None:
        """Reset filter state for specific environments.

        Args:
            env_ids: Environment indices to reset (K,).
            val: Reset value (dim,) broadcast, or (K, dim, 1) for per-env values.
        """
        self.input[env_ids, :, :] = val
        self.output[env_ids, :, :] = val
