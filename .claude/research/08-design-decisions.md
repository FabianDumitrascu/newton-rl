# Design Decisions — Resolved

## Drone Platform
| Question | Decision |
|----------|----------|
| Model source | **`flattened-osprey.usd`** — primary USD model, joint names use `dof_*` prefix |
| Platform specs | **Full specs from real platform** — asymmetric rotors (front stronger for payload) |
| Propeller model | **T = C_T * omega^2** — simplified coefficients, front: 3.1e-6, back: 1.5e-6 |
| Force application | **External body forces** — not spinning joints (solver stability at high RPM) |
| INDI controller | **Collective thrust + body rates (T, p, q, r) → G1 allocation → 4 motor commands** |
| Control mode | **INDI only** — no Geometric controller needed |

## Arm & Gripper
| Question | Decision |
|----------|----------|
| Arm DOF | **2 DOF (pitch + roll)** — joints: `dof_differential`, `dof_arm` |
| Gripper type | **Parallel gripper — 1 DOF (open/close), rack-and-pinion** — both fingers move together |
| Actuators | **PD position control** with configurable gains |
| Arm mass ratio | **~25% of total** (0.2kg arm, 0.6kg base) — significant coupling, INDI compensates via `get_total_J()` |

## Grasping
| Question | Decision |
|----------|----------|
| Grasping approach | **Hybrid** — contact-force primary, fixed joint fallback if too hard |
| Objects | **Start simple (spheres, boxes), add variety later** |
| Success criteria | **Task-specific** — e.g., valve turning: open 360 deg with custom torque curve, progressive difficulty via curriculum. Success = task complete + drone didn't crash |

## RL Training
| Question | Decision |
|----------|----------|
| RL library | **skrl** — clean GPU env integration, multi-algorithm |
| GPU | **RTX 4070 laptop** (now), **RTX 5090** (future) |
| Training strategy | **Curriculum learning**, upgrade to hierarchical if needed |
| Observation space | **Rich, configurable**: quaternion orientation, body-frame velocities, arm joints, target object, contact forces, IMU, angular accel. Must be easy to add/remove terms. Normalized. |
| Action space | **Hybrid**: collective thrust + body rates (T, p, q, r) via INDI for drone + joint position targets (PD) for arm/gripper |
| Reward design | **Framework for iteration** — easily add/remove/weight reward terms. Not optimized yet. |

## Simulation
| Question | Decision |
|----------|----------|
| Solver | **Start with MuJoCo or XPBD** (mature), potentially switch to Kamino later |
| Sim frequency | **Configurable** — all frequencies (sim, control, policy) must be easy to change |
| Domain randomization | **None initially**, add later for sim-to-real |
| Termination | **Per scenario**, but universally: excessive tilt/crash + timeout + task success |
| Force model | **External forces** on rotor bodies (not spinning joints) for solver stability |

## Integration
| Question | Decision |
|----------|----------|
| Code organization | **Subdirectory structure**: `envs/`, `training/`, `controllers/` |
| Visualization | **Periodic video recording** (every N episodes) + TensorBoard |
| Experiment tracking | **TensorBoard only** |
| Stretch tasks | **General manipulation scenarios**: gate valve, sliding door, microwave door, etc. |
