# Phase 0: INDI Controller & Hover Validation — Implementation Summary

## What Was Built

Seven files implementing the INDI flight controller stack and an interactive hover validation script, all ported from the old Isaac Lab reference code (`reference_code/osprey_rl/`) and adapted for Newton's API.

### Files Created

```
controllers/
├── __init__.py           # Package init
├── config.py             # All platform parameters as dataclasses
├── math_utils.py         # Quaternion rotation (XYZW convention)
├── low_pass_filter.py    # 2nd-order Butterworth IIR filter
├── motor_model.py        # First-order motor dynamics
└── indi.py               # INDI attitude controller

testing/
└── validate_hover.py     # Interactive Newton viewer with GUI
```

## How to Run

### Prerequisites

- Python 3.12, NVIDIA GPU, CUDA 12+
- Environment synced: `uv sync --extra torch-cu12`

### Launch the Interactive Viewer

```bash
uv run --extra torch-cu12 python testing/validate_hover.py
```

This opens a Newton viewer window with the Osprey drone hovering at 2m altitude. The side panel has:

- **Flight Commands**: Thrust (default ~8.4N = hover), roll/pitch/yaw rate sliders
- **Arm Commands**: Arm pitch, arm roll, gripper position sliders
- **Telemetry**: Live altitude, total thrust, rotor speeds, sim time
- **Reset button** (or press `R`)

### What to Verify

| Check | Expected |
|-------|----------|
| Drone stays near 2m altitude | Slow drift downward (~0.06m/s) — see Known Issues |
| Total thrust in telemetry | ~8.4N |
| Rotor speeds | Front ~834, back ~1173 rad/s (stable) |
| Move thrust slider up | Drone rises |
| Move thrust slider down | Drone falls |
| Move roll/pitch rate sliders | Drone tilts, INDI tries to compensate |
| Move arm pitch slider | Visible coupling, INDI compensates |
| Press R or click Reset | Returns to initial hover |

### Run Component Tests

```bash
# Config validation
uv run --extra torch-cu12 python controllers/config.py

# Quick smoke test of all components
uv run --extra torch-cu12 python -c "
from controllers.config import default_osprey_config
from controllers.indi import IndiController
from controllers.motor_model import RotorMotor
import torch

cfg = default_osprey_config()
init_omega = torch.tensor(cfg.control.init_omega).unsqueeze(0)
indi = IndiController(1, cfg, torch.device('cpu'))
motor = RotorMotor(1, cfg.rotor, cfg.motor, init_omega, torch.device('cpu'))

# Hover test: 50 steps with zero rates
for _ in range(50):
    target = indi.get_command(torch.zeros(1,3), motor.current_omega, torch.tensor([cfg.hover_thrust]), torch.zeros(1,3))
    thrusts, _, _ = motor.step(target, cfg.sim.sim_dt)

print(f'Hover thrust: {thrusts.sum():.2f}N (expected {cfg.hover_thrust:.2f}N)')
print(f'Rotor speeds: {motor.current_omega.squeeze().tolist()}')
"

# Lint
uvx ruff check controllers/ testing/validate_hover.py
```

## Architecture Decisions

### Force Application (Key Design Decision)

Newton's XPBD solver does not propagate external forces through joint constraints from child bodies to the root body. Forces applied to individual rotor bodies do NOT create torques on the base body.

**Solution:** All rotor forces are applied to the base body. The G1 allocation matrix (which encodes rotor geometry) computes the actual control wrench `[total_thrust, tau_roll, tau_pitch, tau_yaw]` from per-rotor thrusts. This wrench is transformed from body frame to world frame and applied as force + torque on the base body.

### Quaternion Convention

Newton uses **XYZW** quaternion format `(qx, qy, qz, qw)`. The old reference code used WXYZ (Isaac Lab convention). All `math_utils` functions operate in XYZW.

### Body-Frame Angular Velocity

Newton stores `body_qd` as `(vx, vy, vz, wx, wy, wz)` in **world frame**. The INDI controller needs body-frame angular velocity. The `quat_rotate_inverse()` function converts world → body frame.

### Center of Mass

The base body COM is set to `[0.0506, 0.0, 0.0963]` (in body-local frame) so that the thrust moment arms from the asymmetric rotor layout produce zero net pitch torque at hover. Without this, the front rotors (which are stronger and farther out) create a nose-down pitch torque.

## Known Issues & Next Steps

### Slow Altitude Drift

The drone drifts downward at ~0.06m/s during hover. Root cause: gravity acts on all articulated bodies (arm, differential, fingers, rotors) individually, but thrust force is only applied to the base body. The XPBD solver doesn't fully compensate this through joint constraints.

**Possible fixes to explore:**
- Apply compensating gravity forces on child bodies
- Adjust hover thrust to account for total system weight distribution
- Try SolverMuJoCo (may handle articulated body forces differently)
- Set all child body masses to near-zero and concentrate mass on base body

### Rotor Speed Convergence

At hover, rotor speeds converge to [834, 834, 1173, 1173] instead of the expected [935, 935, 1002, 1002]. The total thrust is correct (8.44N), but the distribution differs from the reference because the G1-based torque computation creates a slightly different equilibrium than per-rotor force application.

### Not Yet Implemented

- Arm coupling compensation via `get_total_J(joint_pos)` (dynamic inertia)
- CUDA graph capture for simulation performance
- Headless validation script (auto-check hover stability without viewer)

## Key Parameters (from `controllers/config.py`)

| Parameter | Value | Source |
|-----------|-------|--------|
| Total mass | 0.86 kg | Real platform |
| Hover thrust | 8.44 N | mass * g |
| Front rotor C_T | 3.1e-6 | Reference code |
| Back rotor C_T | 1.5e-6 | Reference code |
| Init hover omega | [935, 935, 1002, 1002] rad/s | Reference code |
| Sim frequency | 300 Hz (60fps * 5 substeps) | Configurable |
| Filter cutoff | 12 Hz | Reference code |
| k_alpha_cmd | [100, 100, 10] | Reference code |
| Motor tau_up/down | 0.033 s | Reference code |
| G1 arm lengths | [0.143, 0.1133] m | Reference code |
| G1 tilt angles | [53.53, 41.42] deg | Reference code |
