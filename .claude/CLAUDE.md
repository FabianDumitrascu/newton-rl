# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**newton-rl** is a thesis research project for aerial manipulation using reinforcement learning. It wraps the [Newton physics engine](https://github.com/newton-physics/newton) (GPU-accelerated, built on NVIDIA Warp) as a git submodule in editable mode.

The goal is to train RL policies for a real aerial manipulator: a quadcopter with a 2-DOF arm (pitch + roll) and a 1-DOF parallel gripper (rack-and-pinion, both fingers move together). The system uses a hybrid control architecture — an INDI inner-loop controller handles flight stability at high frequency while the RL policy outputs collective thrust + body rate commands and arm joint targets at a lower frequency.

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Newton | GPU-accelerated physics simulator (git submodule, editable) |
| NVIDIA Warp | GPU kernel compiler (bundled with Newton) |
| PyTorch | ML framework, GPU tensors, policy networks |
| skrl | RL training library (GPU-native vectorized env support) |
| Gymnasium | Standard RL environment interface |
| TensorBoard | Experiment tracking and visualization |
| Python 3.12 | Runtime (required by Newton) |
| uv | Package manager (workspace with Newton submodule) |
| ruff | Linting/formatting (via pre-commit) |

## Setup

Requires Python 3.12, an NVIDIA GPU (Maxwell+, driver 545+), and CUDA 12 or 13. Check GPU with `nvidia-smi`.

```bash
# Clone with submodule
git clone --recursive git@github.com:FabianDumitrascu/newton-rl.git
cd newton-rl

# Sync environment — pick the CUDA version matching your hardware
uv sync --extra torch-cu12   # CUDA 12.x
uv sync --extra torch-cu13   # CUDA 13.x

# Verify
uv run python -c "import torch; import newton; print(torch.cuda.is_available())"
```

The submodule at `submodules/newton/` is installed in editable mode via a `uv` workspace — changes to Newton source take effect immediately.

## Common Commands

```bash
# Run main entry point
uv run python main.py

# Run the simulation test/demo
uv run python testing/test.py

# Run Newton unit tests (uses unittest, not pytest)
uv run -m newton.tests

# Run a single Newton test
uv run -m newton.tests -k test_name

# Run a Newton example
uv run -m newton.examples basic_pendulum

# Lint/format (runs ruff via pre-commit)
uvx pre-commit run -a

# Launch TensorBoard (once training infra exists)
uv run tensorboard --logdir runs/
```

## Project Structure

```
newton-rl/
├── main.py                  # Entry point
├── controllers/             # Inner-loop controllers
│   ├── indi.py              # INDI attitude controller (ported from real hardware)
│   └── pd_joint.py          # PD joint position controller for arm/gripper
├── envs/                    # RL environment layer
│   ├── base_env.py          # Base vectorized Gymnasium env wrapping Newton
│   ├── aerial_manipulator.py # Aerial manipulator specific env
│   └── tasks/               # Task definitions (obs, reward, termination per task)
│       ├── base_task.py
│       ├── hover.py
│       ├── navigate.py
│       └── grasp.py
├── training/                # Training infrastructure
│   ├── train.py             # Main training script (skrl PPO)
│   ├── evaluate.py          # Policy evaluation + Newton viewer
│   └── curriculum.py        # Curriculum stage manager
├── models/                  # Neural network architectures
│   └── mlp_policy.py        # MLP actor-critic
├── configs/                 # YAML configuration files
│   ├── sim.yaml             # dt, solver, substeps, num_envs
│   ├── platform.yaml        # Drone mass, prop coefficients, arm specs
│   └── tasks/               # Per-task reward weights, curriculum stages
├── assets/                  # USD models, meshes
│   ├── flattened-osprey.usd # Primary drone model (embedded meshes)
│   └── osprey-correct-usd.usd # Unflattened version
├── reference_code/          # Old Isaac Lab codebase (not runnable, for reference only)
│   └── osprey_rl/           # INDI controller, rewards, task configs, physical params
├── testing/                 # Development scripts, simulation demos
├── submodules/newton/       # Physics engine (git submodule, editable)
│   └── newton/
│       ├── _src/            # Internal implementation — never import directly
│       ├── examples/        # 60+ runnable examples across 16 categories
│       └── tests/           # 100+ unit tests (unittest framework)
└── pyproject.toml           # uv workspace config
```

## Architecture

### Control Flow (Training)

```
RL Policy (configurable frequency, skrl PPO)
  │
  ├─ Collective thrust + body rates (T, p, q, r)
  │     → INDI controller (sim frequency, configurable)
  │         → 4 motor force/torque applied to drone body
  │
  └─ Joint position targets (arm pitch, arm roll, gripper)
        → PD controller (sim frequency)
            → Joint torques applied to arm/gripper
```

### Simulation Loop (per env.step())

1. Receive RL action (7D: 4 drone [thrust, p, q, r] + 3 joint targets)
2. Run N substeps at sim frequency:
   a. INDI computes motor commands from body rate error
   b. PD computes joint torques from position error
   c. Apply forces → `model.collide()` → `solver.step()`
3. Extract observations from final state
4. Compute reward and check termination
5. Return (obs, reward, terminated, truncated, info)

### GPU Pipeline

All data stays on GPU throughout: Newton state → `wp.to_torch()` → skrl policy → `wp.from_torch()` → Newton control. No CPU roundtrip in the training loop.

## Code Patterns

### newton-rl Code (this repo)

- **Python style**: Follow ruff defaults, PEP 604 unions (`x | None`)
- **Type hints**: All public functions must be typed
- **Config-driven**: Task parameters, rewards, curriculum stages, simulation frequencies — all defined in config, not hardcoded
- **Easily iterable**: Observations, rewards, and action spaces must be easy to add/remove/adjust for rapid experimentation
- **Task abstraction**: Each task defines `compute_observations()`, `compute_rewards()`, `check_termination()`, `reset_task()`
- **No premature optimization**: Start with Python loops, move to Warp kernels only when profiling shows bottlenecks

### Newton API Conventions (submodule)

When modifying Newton source in `submodules/newton/`:

- **Public API only** via top-level `newton/` modules — never expose `newton._src` internals
- **Prefix-first naming**: `ActuatorPD` not `PDActuator`; `add_shape_sphere()` not `add_sphere_shape()`
- **Type hints**: PEP 604 unions (`x | None`, not `Optional[x]`)
- **Docstrings**: Google style with SI units specified
- **Tests**: Add to `newton/tests/` using `unittest` (not pytest)
- **Examples**: Must implement a `test_final()` method (used by CI)
- Breaking changes require deprecation first; use feature branches (no direct commits to main)

### Newton Core Pattern

All simulation follows this pattern:

```python
import newton

builder = newton.ModelBuilder()
builder.add_body(...)
builder.add_shape_sphere(...)
model = builder.finalize()  # Model is IMMUTABLE after this

state_0, state_1 = model.state(), model.state()
control = model.control()
contacts = model.contacts()

solver = newton.solvers.SolverXPBD(model)
while running:
    state_0.clear_forces()
    model.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0  # swap states
```

**Critical constraint:** Newton models are immutable after `finalize()`. No joints, bodies, or shapes can be added/removed at runtime. Only state values (positions, velocities, forces, joint targets) can change.

## Key Files

| File | Purpose |
|------|---------|
| `.claude/PRD.md` | Full Product Requirements Document |
| `.claude/research/` | 8 research files from Newton codebase exploration |
| `.claude/research/08-design-decisions.md` | All resolved design decisions |
| `reference_code/osprey_rl/` | Old Isaac Lab codebase — INDI controller, rewards, task configs (not runnable) |
| `reference_code/osprey_rl/osprey_rl/mdp/controller/indi.py` | Reference INDI implementation with real drone parameters |
| `reference_code/osprey_rl/osprey_rl/mdp/controller/motor_model.py` | Reference motor dynamics model |
| `reference_code/osprey_rl/osprey_rl/assets/` | USD models and manipulation assets |
| `submodules/newton/newton/examples/diffsim/example_diffsim_drone.py` | Reference drone example (Crazyflie) |
| `submodules/newton/newton/examples/robot/example_robot_panda_hydro.py` | Reference grasping example |
| `submodules/newton/newton/_src/solvers/kamino/examples/rl/` | Newton's existing RL infrastructure |

## References (API & Library Docs)

Comprehensive reference documents with API signatures, examples, and common pitfalls. Use these when implementing features or interacting with Newton/skrl.

| Topic | File | Description |
|-------|------|-------------|
| **ModelBuilder** | `.claude/references/model_builder.md` | Bodies, all shape types, ShapeConfig, finalize(), Model arrays |
| **Joints** | `.claude/references/joints.md` | All 8 joint types, JointDofConfig, FK/IK, runtime control |
| **Solvers** | `.claude/references/solvers.md` | All 7 solvers, decision matrix, CUDA graphs, differentiable sim |
| **USD Loading** | `.claude/references/usd.md` | add_usd() full API, schema resolvers, mesh utilities |
| **Contacts & Sensors** | `.claude/references/contacts_and_sensors.md` | Collision pipeline, material properties, SensorIMU, SensorContact |
| **Multi-World & RL** | `.claude/references/multi_world_and_rl.md` | Parallel envs, RigidBodySim, State/Control, wp.to_torch() |
| **skrl** | `.claude/references/skrl.md` | PPO config, GaussianMixin models, custom env wrapper, training |
| **Gymnasium GPU Envs** | `.claude/references/gymnasium_gpu_envs.md` | Auto-reset pattern, action scaling, reward shaping, curriculum |

## Research Notes (Early Exploration)

Earlier research summaries from initial codebase exploration. The references above are more detailed and authoritative.

| Topic | File |
|-------|------|
| Newton core API | `.claude/research/01-newton-core-api.md` |
| Drone physics model | `.claude/research/02-drone-modeling.md` |
| RL integration patterns | `.claude/research/03-rl-integration.md` |
| Contacts & grasping | `.claude/research/04-contacts-and-grasping.md` |
| USD/URDF model loading | `.claude/research/05-model-loading.md` |
| Project phases | `.claude/research/06-project-decomposition.md` |
| Design questions | `.claude/research/07-design-questions.md` |
| Design decisions | `.claude/research/08-design-decisions.md` |

## Important Notes

- **Solver choice**: Start with SolverMuJoCo or SolverXPBD (mature). Switch to SolverKamino for training throughput if needed.
- **Force application**: Apply rotor thrust/torque as external forces on bodies (not via spinning joints). Spinning rotors at thousands of RPM may destabilize the solver. Rotor joints can spin visually at capped speed for rendering only.
- **Grasping**: Use contact-force grasping (friction-based) as primary approach. Pre-allocated fixed joints as fallback.
- **Gripper**: Rack-and-pinion mechanism — both fingers open/close together as a single DOF.
- **Multi-world**: Newton supports 512-4096 parallel environments on GPU via `builder.replicate()` or `begin_world()`/`end_world()`.
- **No domain randomization** in first version — add later for sim-to-real transfer.
- **Curriculum learning**: Single policy trained with progressive difficulty (hover → navigate → grasp → manipulate). Auto-advance when success_rate > threshold.
- **Configurability**: All frequencies (sim, control, policy), parameters, observation/reward terms must be config-driven for rapid iteration.
- **USD model**: `flattened-osprey.usd` is the primary model. Joint names use `dof_*` prefix (`dof_differential`, `dof_arm`, `dof_finger_left`, `dof_finger_right`).
- **Reference code**: `reference_code/osprey_rl/` contains the old Isaac Lab-based project. Not runnable with Newton but useful for INDI logic, reward design, and parameter reference.
