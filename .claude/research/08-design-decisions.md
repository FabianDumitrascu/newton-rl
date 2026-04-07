# Design Decisions — Resolved

## Drone Platform
| Question | Decision |
|----------|----------|
| Model source | **USD file available** |
| Platform specs | **Full specs from real platform** |
| Propeller model | **Similar but not identical to Newton's** — need to compare side by side |
| INDI controller | **Rate commands (p, q, r) + thrust → 4 motor RPMs** |

## Arm & Gripper
| Question | Decision |
|----------|----------|
| Arm DOF | **2 DOF (pitch + roll)** |
| Gripper type | **Parallel gripper — 1 DOF (open/close)** |
| Actuators | **Servos with position control (PD targets)** |
| Arm mass ratio | **Moderate (10-30%)** — noticeable coupling, INDI compensates |

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
| Observation space | **Rich**: drone pose + velocity + arm joints + target object + contact forces + IMU + angular accel |
| Action space | **Hybrid**: attitude commands (via INDI) for drone + joint position targets for arm/gripper |

## Simulation
| Question | Decision |
|----------|----------|
| Solver | **Start with MuJoCo or XPBD** (mature), potentially switch to Kamino later |
| Sim frequency | **TBD** — benchmark what's stable |
| Domain randomization | **None initially**, add later for sim-to-real |
| Termination | **Per scenario**, but universally: excessive tilt/crash + timeout + task success |

## Integration
| Question | Decision |
|----------|----------|
| Code organization | **Subdirectory structure**: `envs/`, `training/`, `controllers/` |
| Visualization | **Periodic video recording** (every N episodes) + TensorBoard |
| Experiment tracking | **TensorBoard only** |
| Stretch tasks | **General manipulation scenarios**: gate valve, sliding door, microwave door, etc. |
