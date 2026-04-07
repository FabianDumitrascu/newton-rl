# Product Requirements Document: newton-rl

## 1. Executive Summary

**newton-rl** is a reinforcement learning framework for aerial manipulation built on top of the [Newton physics engine](https://github.com/newton-physics/newton) — a GPU-accelerated simulator powered by NVIDIA Warp. The project targets a real aerial manipulator platform: a quadcopter equipped with a 2-DOF robotic arm and a parallel gripper, designed to perform contact-rich manipulation tasks such as valve turning, door opening, and object grasping while in flight.

The framework bridges Newton's high-performance batched simulation (1000+ parallel GPU environments) with modern RL training via the skrl library. A key architectural decision is the **hybrid action space**: an INDI inner-loop controller handles low-level attitude stabilization at high frequency, while the RL policy operates at a lower frequency outputting attitude commands for the drone and joint position targets for the arm/gripper. This mirrors the real hardware control stack, improving sim-to-real transfer potential.

**MVP goal:** A fully functional RL training pipeline that can train a policy to hover, navigate to an object, and grasp it in simulation — with the drone physics validated against real platform parameters.

---

## 2. Mission

**Mission statement:** Enable learning-based aerial manipulation by providing a fast, modular, and physically accurate simulation-to-training pipeline on top of the Newton physics engine.

**Core principles:**

1. **Physics fidelity first** — Validate all dynamics against the real platform before training. A policy trained on wrong physics is worthless.
2. **Hierarchical control** — Separate concerns: INDI handles flight stability, RL handles task intelligence. This matches the real hardware stack.
3. **Progressive complexity** — Start with hover, end with valve turning. Curriculum learning guides the policy from simple to hard.
4. **GPU-native throughput** — All simulation and observation computation stays on GPU. No CPU bottlenecks in the training loop.
5. **Modularity** — Tasks, rewards, observations, and controllers are swappable components, not monolithic code.

---

## 3. Target Users

**Primary user:** Fabian (thesis researcher) — building and training RL policies for aerial manipulation, with deep knowledge of drone control (INDI) and real platform hardware specs. Needs to iterate quickly on reward design, observation tuning, and curriculum progression.

**Secondary users:**
- Thesis advisors reviewing simulation results and trained policy behavior
- Future researchers extending the framework to new tasks or platforms

**Key needs:**
- Reproducible experiments with TensorBoard logging
- Visual debugging (periodic video renders of policy behavior)
- Easy task definition (new manipulation scenarios without rewriting the sim)
- Confidence that simulated dynamics match the real platform

---

## 4. MVP Scope

### In Scope (Core Functionality)

- ✅ Quadcopter model loaded from USD with real platform parameters
- ✅ Propeller force model (thrust + reaction torque from angular velocity)
- ✅ INDI inner-loop controller ported from existing project
- ✅ 2-DOF arm (pitch + roll) with PD position control
- ✅ 1-DOF parallel gripper with PD position control
- ✅ Contact-force grasping with tuned friction
- ✅ Gymnasium-compatible vectorized environment wrapper
- ✅ Multi-world batched simulation (512+ parallel envs on RTX 4070)
- ✅ skrl integration for PPO training
- ✅ Curriculum learning framework (staged difficulty progression)
- ✅ Reward function framework (modular, per-task)
- ✅ TensorBoard logging + periodic video recording
- ✅ Basic tasks: hover stabilization, waypoint navigation, approach & grasp

### In Scope (Technical)

- ✅ Substepping: high-frequency sim (INDI rate) with low-frequency policy
- ✅ Observation pipeline: drone state, arm joints, contact forces, IMU, angular acceleration
- ✅ Action pipeline: attitude commands → INDI → motors; joint targets → PD → arm/gripper
- ✅ Episode termination: crash detection (tilt/altitude), timeout, task success
- ✅ Per-world reset with initial state randomization

### Out of Scope (Deferred)

- ❌ Domain randomization (mass, friction, wind, sensor noise) — add in future phase
- ❌ SolverKamino migration — evaluate after MuJoCo/XPBD validation
- ❌ Sim-to-real transfer pipeline
- ❌ Hierarchical RL / options framework — upgrade path if curriculum insufficient
- ❌ Advanced manipulation scenarios (valve turning, sliding door, microwave) — Phase 4
- ❌ Pre-allocated fixed joint grasping fallback — only if contact-force grasping fails
- ❌ RNN/transformer observation history — start with MLP policy
- ❌ Multi-GPU training
- ❌ URDF/MJCF asset support (USD only for MVP)
- ❌ Custom Warp kernels for observation computation — use Python first, optimize later

---

## 5. User Stories

### Simulation & Physics

1. **As a researcher, I want to load my aerial manipulator USD model into Newton and see it rendered in the viewer**, so that I can visually verify the model is correct (joints, shapes, mass distribution).

2. **As a researcher, I want to fly the drone with my INDI controller and compare hover thrust / attitude response against analytical expectations**, so that I can trust the simulation physics before training any RL policy.

3. **As a researcher, I want to move the arm joints while hovering and observe the coupling disturbance on the drone**, so that I can verify the INDI controller compensates correctly.

### RL Training

4. **As a researcher, I want to define a new task (observation, action, reward, termination) by creating a task config file**, so that I can iterate on task design without modifying framework code.

5. **As a researcher, I want to launch training with 512+ parallel environments and see live TensorBoard metrics (reward, episode length, success rate)**, so that I can monitor training progress and tune hyperparameters.

6. **As a researcher, I want curriculum stages to automatically advance when the policy achieves a success threshold**, so that training progresses without manual intervention.

7. **As a researcher, I want periodic video recordings of the best-performing policy in the viewer**, so that I can visually assess learned behavior without pausing training.

### Evaluation

8. **As a researcher, I want to load a trained policy checkpoint and evaluate it in a single-environment viewer mode**, so that I can demonstrate results and debug failure cases interactively.

---

## 6. Core Architecture & Patterns

### High-Level Architecture

```
newton-rl/
├── controllers/              # Inner-loop controllers
│   ├── indi.py              # INDI attitude controller (port from existing project)
│   └── pd_joint.py          # PD joint position controller for arm/gripper
├── envs/                    # RL environment layer
│   ├── base_env.py          # Base vectorized Gymnasium env wrapping Newton
│   ├── aerial_manipulator.py # Aerial manipulator specific env
│   └── tasks/               # Task definitions (obs, reward, termination)
│       ├── hover.py
│       ├── navigate.py
│       ├── grasp.py
│       └── base_task.py
├── training/                # Training infrastructure
│   ├── train.py             # Main training script (skrl)
│   ├── evaluate.py          # Policy evaluation + viewer
│   ├── curriculum.py        # Curriculum stage manager
│   └── configs/             # Hyperparameter configs
├── models/                  # Neural network architectures
│   └── mlp_policy.py        # MLP actor-critic
├── assets/                  # USD models, meshes
│   └── aerial_manipulator.usd
├── testing/                 # Development scripts, demos
├── submodules/newton/       # Physics engine (git submodule)
├── main.py                  # Entry point
└── pyproject.toml
```

### Key Design Patterns

1. **Task abstraction:** Each task (hover, navigate, grasp) defines its own `compute_observations()`, `compute_rewards()`, `check_termination()`, and `reset_task()`. The base environment handles simulation stepping, multi-world management, and controller integration.

2. **Controller-in-the-loop:** The INDI controller and PD joint controllers run *inside* the simulation substep loop, not as part of the RL policy. The environment's `step()` method:
   - Receives RL action (attitude commands + joint targets)
   - Runs N substeps, each calling INDI at sim frequency
   - Returns observation after all substeps complete

3. **Zero-copy GPU pipeline:** Observations and actions stay as GPU tensors throughout. Newton state → `wp.to_torch()` → skrl policy → `wp.from_torch()` → Newton control. No CPU roundtrip.

4. **Curriculum as state machine:** The curriculum manager tracks per-metric thresholds and advances difficulty when conditions are met. Task parameters (e.g., target distance, object friction) are functions of the current curriculum stage.

---

## 7. Core Features

### F1: Drone Physics Model

**Purpose:** Accurate quadcopter flight dynamics matching real platform.

- Propeller force model: `F = C_T * rho * n^2 * d^4` per rotor
- Reaction torque model: `tau = C_P * rho * n^2 * d^5 / (2*pi)` per rotor
- Alternating turning directions for yaw authority
- Moment arm cross product for roll/pitch from offset thrust
- Configurable parameters from real hardware specs

### F2: INDI Inner-Loop Controller

**Purpose:** Attitude stabilization at high frequency, decoupling flight stability from RL task learning.

- Input: desired angular rates (p, q, r) + collective thrust
- Output: 4 motor commands (RPM or normalized thrust)
- Runs at sim frequency (250-1000 Hz, to be benchmarked)
- Ported from existing codebase with parameter matching

### F3: Arm & Gripper Control

**Purpose:** Position-controlled 2-DOF arm and 1-DOF gripper.

- PD controller with configurable gains (stiffness `ke`, damping `kd`)
- Joint limits enforced by Newton solver
- Gripper open/close as continuous position target (not binary)
- Contact force feedback via `SensorContact`

### F4: Vectorized RL Environment

**Purpose:** Gymnasium-compatible multi-world environment for GPU-batched training.

- `gymnasium.Env` interface: `reset()`, `step()`, `observation_space`, `action_space`
- Batched across N worlds (target: 512+ on RTX 4070, 2048+ on RTX 5090)
- Per-world reset with `world_mask`
- Configurable observation and action spaces per task

### F5: Curriculum Manager

**Purpose:** Automated difficulty progression during training.

- Stage definitions: list of (metric_name, threshold, task_params) tuples
- Auto-promotion when success_rate > threshold over rolling window
- Task parameter interpolation between stages
- Logging of stage transitions to TensorBoard

### F6: Task Framework

**Purpose:** Modular task definitions that plug into the base environment.

Each task defines:
- `compute_observations(state, contacts) -> tensor` — what the policy sees
- `compute_rewards(obs, action, next_obs) -> tensor` — per-world reward
- `check_termination(state) -> (terminated, truncated)` — episode end conditions
- `reset_task(world_ids)` — task-specific reset (e.g., randomize object position)
- `curriculum_params(stage) -> dict` — how task difficulty scales with curriculum

---

## 8. Technology Stack

### Core

| Component | Technology | Version/Notes |
|-----------|-----------|---------------|
| Physics engine | Newton (submodule) | GPU-accelerated, NVIDIA Warp |
| GPU compute | NVIDIA Warp | Auto-compiled CUDA kernels |
| ML framework | PyTorch | CUDA 12/13, GPU tensors |
| RL library | skrl | GPU-native vectorized env support |
| RL algorithm | PPO (primary), SAC (secondary) | Via skrl |
| Env interface | Gymnasium | Standard `Env` API |
| Python | 3.12 | Required by Newton |
| Package manager | uv | Workspace with Newton submodule |

### Infrastructure

| Component | Technology |
|-----------|-----------|
| Experiment tracking | TensorBoard |
| Video recording | Newton ViewerGL (offscreen) |
| Logging | Python `logging` + TensorBoard |
| Config management | YAML or dataclasses |
| Linting/formatting | ruff (via pre-commit) |

### Hardware

| Phase | GPU | Parallel Envs |
|-------|-----|:-------------:|
| Development | RTX 4070 Laptop (12 GB) | 512-1024 |
| Training (future) | RTX 5090 (32 GB) | 2048-4096 |

### Dependencies

```toml
[project]
dependencies = [
    "torch",
    "gymnasium",
    "skrl",
    "tensorboard",
    "numpy",
    "pyyaml",
]
```

---

## 9. Security & Configuration

### Configuration Management

- **Simulation config:** `configs/sim.yaml` — dt, solver, substeps, num_envs
- **Task config:** `configs/tasks/{task_name}.yaml` — obs/action dims, reward weights, curriculum stages
- **Training config:** `configs/train.yaml` — PPO hyperparams, learning rate, batch size, epochs
- **Platform config:** `configs/platform.yaml` — drone mass, prop coefficients, arm dimensions, joint limits

### Environment Variables

- `NEWTON_RL_DEVICE` — `cuda:0` (default) or `cpu`
- `NEWTON_RL_NUM_ENVS` — override parallel env count
- `NEWTON_RL_LOG_DIR` — TensorBoard log directory (default: `runs/`)

### Security Scope

Not applicable — this is a local research tool, not a deployed service.

---

## 10. Observation & Action Specification

### Observation Space (Rich)

| Component | Dimensions | Description |
|-----------|:----------:|-------------|
| Drone position | 3 | World-frame (x, y, z) [m] |
| Drone orientation | 9 | Rotation matrix (flattened 3x3) |
| Drone linear velocity | 3 | Body-frame (vx, vy, vz) [m/s] |
| Drone angular velocity | 3 | Body-frame (p, q, r) [rad/s] |
| Drone angular acceleration | 3 | Body-frame [rad/s^2] |
| IMU accelerometer | 3 | Body-frame acceleration [m/s^2] |
| Arm joint positions | 2 | (pitch, roll) [rad] |
| Arm joint velocities | 2 | [rad/s] |
| Gripper position | 1 | Open/close [m or rad] |
| Gripper velocity | 1 | [m/s or rad/s] |
| Target object pose (relative) | 7 | Relative position (3) + quaternion (4) |
| Finger contact forces | 6 | Left (3) + right (3) force vectors [N] |
| Previous action | N_act | For smoothness |
| **Total** | **~43+** | Exact count depends on task |

### Action Space (Hybrid)

| Component | Dimensions | Range | Controller |
|-----------|:----------:|:-----:|-----------|
| Roll command | 1 | [-30, +30] deg | INDI → motors |
| Pitch command | 1 | [-30, +30] deg | INDI → motors |
| Yaw rate command | 1 | [-180, +180] deg/s | INDI → motors |
| Thrust command | 1 | [0, 1] normalized | INDI → motors |
| Arm pitch target | 1 | joint limits [rad] | PD controller |
| Arm roll target | 1 | joint limits [rad] | PD controller |
| Gripper target | 1 | [open, closed] [rad] | PD controller |
| **Total** | **7** | | |

---

## 11. Success Criteria

### MVP Success Definition

The MVP is successful when a trained RL policy can:
1. Stabilize hover at a target position (< 5cm position error)
2. Navigate to waypoints in 3D space
3. Approach and grasp a static object on a pedestal

### Functional Requirements

- ✅ Drone hovers stably with INDI controller (no RL) in Newton viewer
- ✅ Hover thrust matches analytical expectation (mg = sum of motor forces)
- ✅ Arm movement causes visible but INDI-compensated disturbance
- ✅ Gripper can hold a simple object via contact forces
- ✅ 512+ parallel environments run on RTX 4070 without OOM
- ✅ PPO training converges on hover task within 1 hour
- ✅ Trained policy generalizes across initial conditions within task distribution
- ✅ TensorBoard shows reward curves, episode lengths, success rates
- ✅ Video recordings show qualitatively reasonable behavior

### Quality Indicators

- Simulation runs at > 10,000 env-steps/second on RTX 4070
- Policy evaluation success rate > 90% on hover task
- No NaN/Inf in observations or rewards during training
- Reproducible results with fixed random seed

---

## 12. Implementation Phases

### Phase 0: Environment Validation

**Goal:** Verify drone flight physics in Newton match reality.

**Deliverables:**
- ✅ Load aerial manipulator USD model into Newton
- ✅ Implement propeller force model with real parameters
- ✅ Port INDI controller from existing project
- ✅ Fly drone with manual commands in Newton viewer
- ✅ Compare hover thrust, attitude response, yaw dynamics against expectations
- ✅ Tune parameters until physics feel right

**Validation:** Side-by-side comparison of simulated vs. expected hover thrust, step response, and attitude dynamics. Visual inspection in Newton viewer.

**Depends on:** USD model file, INDI controller code, platform specs from Fabian.

---

### Phase 1: Aerial Manipulator Assembly

**Goal:** Full platform (drone + arm + gripper) flying and grasping in simulation.

**Deliverables:**
- ✅ Joint hierarchy: FREE root → drone body → arm pitch → arm roll → gripper
- ✅ PD-controlled arm and gripper with appropriate gains
- ✅ Contact shapes on finger surfaces with tuned friction (mu >= 0.5)
- ✅ Test arm coupling effects during hover
- ✅ Demonstrate grasping a static cube/sphere

**Validation:** Drone hovers while arm moves. Gripper picks up object via contact forces. Object doesn't phase through or fly away.

---

### Phase 2: RL Environment & Training Pipeline

**Goal:** Gymnasium-compatible vectorized environment + working skrl training loop.

**Deliverables:**
- ✅ `AerialManipulatorEnv(gymnasium.Env)` with multi-world batching
- ✅ Controller-in-the-loop substepping (INDI at sim freq, policy at control freq)
- ✅ Task framework: hover task as first implementation
- ✅ skrl PPO integration with TensorBoard logging
- ✅ Periodic video recording of policy behavior
- ✅ Per-world reset working correctly
- ✅ Benchmark simulation frequency for stability

**Validation:** `random_policy → env.step() → obs` loop runs without crashes. PPO training on hover task shows improving reward curve. 512+ envs on RTX 4070.

---

### Phase 3: Task Training (Curriculum)

**Goal:** Trained policies for progressively harder manipulation tasks.

**Deliverables:**
- ✅ **3a — Hover:** Policy stabilizes at target position (< 5cm error)
- ✅ **3b — Navigate:** Policy flies to waypoints in 3D
- ✅ **3c — Approach & Grasp:** Policy approaches object, grasps it, holds it
- ✅ **3d — Manipulation:** Policy performs pick-and-place or simple manipulation
- ✅ Curriculum manager with automated stage advancement
- ✅ Per-task reward functions with documented reward components

**Validation:** Success rate > 80% on each sub-task before advancing. Video demonstrations of learned behaviors. TensorBoard training curves.

---

### Phase 4: Advanced Manipulation Scenarios (Stretch)

**Goal:** Complex, contact-rich manipulation tasks.

**Deliverables:**
- ✅ Gate valve turning (360 deg rotation, custom torque curve)
- ✅ Sliding door interaction
- ✅ Microwave door opening
- ✅ Domain randomization for robustness
- ✅ Sim-to-real gap analysis

**Validation:** Policy handles variable difficulty (curriculum over friction, resistance). Success rate > 70% on hardest difficulty.

---

## 13. Future Considerations

### Post-MVP Enhancements

- **Domain randomization:** Randomize mass (+-10%), friction (+-20%), wind, motor delay, sensor noise for robust policies
- **SolverKamino migration:** Switch to GPU-optimized solver for 2-4x training throughput
- **Hierarchical RL:** If curriculum learning plateaus, upgrade to options framework with learned skill selection
- **Observation history:** RNN or transformer policy for partially observable tasks (e.g., estimating object properties)
- **Pre-allocated joint grasping:** Fallback if contact-force grasping proves unreliable for complex objects

### Integration Opportunities

- **Isaac Lab interop:** Newton's architecture is similar; policies may transfer
- **Real hardware deployment:** INDI controller already runs on real drone; bridge RL policy outputs to real flight controller
- **Multi-agent:** Multiple drones cooperating on manipulation tasks (Newton supports multi-world with distinct agents)

### Advanced Features

- Deformable object manipulation (Newton supports soft bodies via VBD solver)
- Dynamic grasping (catching moving objects)
- Long-horizon task planning with learned sub-skill libraries

---

## 14. Risks & Mitigations

### R1: Simulated physics don't match reality

**Risk:** Propeller model, contact dynamics, or arm coupling behave differently in sim vs. real. Policies trained in sim fail on real hardware.

**Mitigation:** Phase 0 is entirely dedicated to physics validation. Compare hover thrust, step responses, and attitude dynamics against analytical models and real data. Iterate on parameters before any RL training. Domain randomization in later phases adds robustness.

### R2: Contact-force grasping is too hard to learn

**Risk:** Friction-based grasping is physically realistic but requires precise finger positioning and force control, which may be too high-dimensional for PPO to discover.

**Mitigation:** Start with simple objects (cubes, spheres) with high friction. Use dense reward shaping (approach → contact → force → stability). If learning fails, fall back to pre-allocated fixed joints for grasp stabilization. Hydroelastic contacts provide richer contact patches as an upgrade path.

### R3: RTX 4070 memory limits constrain training

**Risk:** 12 GB VRAM may limit parallel environments to < 512, slowing training iteration.

**Mitigation:** Profile memory per environment early in Phase 2. Optimize observation/state buffers. Reduce mesh complexity if needed. RTX 5090 (32 GB) available as upgrade path. SolverKamino with CUDA graphs reduces per-env overhead.

### R4: Curriculum learning doesn't converge on complex tasks

**Risk:** The policy learns hover and navigation but fails to bridge to contact-rich grasping. The exploration problem at the grasp stage is too hard.

**Mitigation:** Dense reward shaping at each curriculum stage. Introduce grasping gradually (pre-positioned gripper → close approach → full approach). If curriculum insufficient, upgrade to hierarchical RL with separate grasp specialist. Pre-train grasp policy in isolation, then fine-tune end-to-end.

### R5: Newton API changes break the framework

**Risk:** Newton is actively developed; API changes in the submodule could break our wrapper.

**Mitigation:** Pin Newton submodule to a known-good commit. Abstract Newton-specific calls behind a thin simulation interface. Run Newton's own test suite (`uv run -m newton.tests`) as a smoke test before updating the submodule.

---

## 15. Appendix

### Key Dependencies

| Dependency | Purpose | Link |
|-----------|---------|------|
| Newton | GPU physics simulator | `submodules/newton/` (git submodule) |
| NVIDIA Warp | GPU kernel compiler | Bundled with Newton |
| PyTorch | ML framework | `torch` (CUDA 12/13) |
| skrl | RL training library | `skrl` (PyPI) |
| Gymnasium | Environment interface | `gymnasium` (PyPI) |
| TensorBoard | Experiment tracking | `tensorboard` (PyPI) |

### Research Files

Detailed technical research from codebase exploration is stored in `.claude/research/`:

| File | Contents |
|------|----------|
| `01-newton-core-api.md` | ModelBuilder, joints, state, solvers, multi-world |
| `02-drone-modeling.md` | Propeller physics, existing Crazyflie example |
| `03-rl-integration.md` | Existing RL infrastructure, Kamino, RigidBodySim |
| `04-contacts-and-grasping.md` | Contacts, friction, SensorContact, grasping strategies |
| `05-model-loading.md` | USD/URDF/MJCF loading APIs |
| `06-project-decomposition.md` | Phase breakdown with task lists |
| `07-design-questions.md` | 24 design questions (resolved) |
| `08-design-decisions.md` | All resolved design decisions |

### Repository Structure

```
newton-rl/
├── .claude/
│   ├── CLAUDE.md          # Project instructions for Claude Code
│   ├── PRD.md             # This document
│   └── research/          # Technical research files (8 files)
├── controllers/           # INDI + PD controllers (Phase 0-1)
├── envs/                  # Gymnasium environments (Phase 2)
├── training/              # skrl training scripts (Phase 2-3)
├── models/                # Neural network architectures (Phase 2)
├── assets/                # USD models (Phase 0)
├── configs/               # YAML configuration files
├── testing/               # Development demos
├── submodules/newton/     # Physics engine (git submodule, editable)
├── main.py                # Entry point
├── pyproject.toml         # Package config (uv workspace)
└── README.md
```
