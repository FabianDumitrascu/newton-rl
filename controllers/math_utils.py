"""Quaternion and rotation utilities for the INDI controller.

All functions use the XYZW quaternion convention (Newton/Warp format).
All operations are pure PyTorch for GPU compatibility.
"""

from __future__ import annotations

import torch


def quat_rotate(q_xyzw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) by quaternion(s) (body-to-world).

    Args:
        q_xyzw: Quaternions (..., 4) in XYZW format.
        v: Vectors (..., 3) to rotate.

    Returns:
        Rotated vectors (..., 3).
    """
    x, y, z, w = q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2], q_xyzw[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

    # t = 2 * cross(q_xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    # result = v + w * t + cross(q_xyz, t)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)

    return torch.stack([rx, ry, rz], dim=-1)


def quat_rotate_inverse(q_xyzw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) by the inverse of quaternion(s) (world-to-body).

    Args:
        q_xyzw: Quaternions (..., 4) in XYZW format.
        v: Vectors (..., 3) to rotate.

    Returns:
        Inversely rotated vectors (..., 3).
    """
    x, y, z, w = q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2], q_xyzw[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

    # t = 2 * cross(q_xyz, v) with negated q_xyz (inverse = conjugate for unit quat)
    tx = 2.0 * (-y * vz + z * vy)
    ty = 2.0 * (-z * vx + x * vz)
    tz = 2.0 * (-x * vy + y * vx)

    rx = vx + w * tx + (-y * tz + z * ty)
    ry = vy + w * ty + (-z * tx + x * tz)
    rz = vz + w * tz + (-x * ty + y * tx)

    return torch.stack([rx, ry, rz], dim=-1)


def euler_from_quat_xyzw(
    q_xyzw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract Euler angles (roll, pitch, yaw) from XYZW quaternion.

    Uses ZYX intrinsic rotation convention (aerospace standard).

    Args:
        q_xyzw: Quaternions (..., 4) in XYZW format.

    Returns:
        Tuple of (roll, pitch, yaw) each with shape (...).
    """
    x, y, z, w = q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2], q_xyzw[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """Compute the 3x3 skew-symmetric matrix from a vector.

    For batched input (N, 3), returns (N, 3, 3).
    For single input (3,), returns (3, 3).

    Args:
        v: Vector(s) (..., 3).

    Returns:
        Skew-symmetric matrices (..., 3, 3).
    """
    batch = v.shape[:-1]
    zero = (
        torch.zeros(batch, dtype=v.dtype, device=v.device)
        if batch
        else torch.tensor(0.0, dtype=v.dtype, device=v.device)
    )

    # fmt: off
    mat = torch.stack([
        zero,     -v[..., 2],  v[..., 1],
        v[..., 2], zero,      -v[..., 0],
       -v[..., 1], v[..., 0],  zero,
    ], dim=-1).reshape(*batch, 3, 3)
    # fmt: on

    return mat
