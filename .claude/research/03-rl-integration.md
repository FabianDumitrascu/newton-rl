# RL Integration with Newton

## Current RL Infrastructure (Already Exists!)

Newton has **production-level RL foundations** in the Kamino solver:

**Location:** `submodules/newton/newton/_src/solvers/kamino/examples/rl/`

| File | Purpose |
|------|---------|
| `simulation.py` | `RigidBodySim` — reusable RL simulator wrapper |
| `observations.py` | Warp-kernel observation builders (batched) |
| `example_rl_drlegs.py` | DR-Legs walking policy playback (94D obs) |
| `example_rl_bipedal.py` | Bipedal robot RL control |
| `test_multi_env_dr_legs.py` | Multi-env test (2048-4096 parallel worlds) |

## What Already Works

| Capability | Status | Details |
|------------|--------|---------|
| Multi-world simulation | **Production** | Tested at 4096+ worlds on single GPU |
| Zero-copy state extraction | **Production** | `wp.to_torch()` / `wp.from_torch()` |
| Batched observations | **Production** | Warp kernels compute obs per-world |
| Action application | **Production** | Implicit PD: `control.q_j_ref`, direct torque: `control.tau_j` |
| Per-world reset | **Working** | `sim.reset(world_mask=mask, joint_q=..., base_q=...)` |
| CUDA graph capture | **Working** | `use_cuda_graph=True` for full step replay |

## What Needs to Be Built

| Component | Priority | Description |
|-----------|----------|-------------|
| Gymnasium wrapper | High | `gymnasium.Env` interface for standard RL libraries |
| Vectorized env | High | Per-env reset within single `.step()` call |
| Reward framework | High | Warp kernels for task-specific rewards |
| Domain randomization | Medium | Per-world parameter randomization |
| Training loop | Medium | Integration with PyTorch RL (rsl_rl, CleanRL, etc.) |

## RigidBodySim Wrapper Pattern

```python
from newton._src.solvers.kamino.examples.rl.simulation import RigidBodySim

# Build model
sim = RigidBodySim(model, use_cuda_graph=True)

# Step
sim.step()

# Get observations (zero-copy to PyTorch)
q_i = wp.to_torch(sim.state.q_i).reshape(num_worlds, num_bodies, 7)
joint_pos = wp.to_torch(sim.state.q_j).reshape(num_worlds, num_joint_coords)

# Apply actions
sim.q_j_ref[actuated_dof_indices] = action_scale * policy_output

# Reset specific worlds
sim.reset(world_mask=done_mask)
```

## Kamino Solver Control Interface

```python
# Implicit PD control (most common for RL)
control.q_j_ref      # Joint position targets
control.dq_j_ref     # Joint velocity targets  
control.tau_j_ref    # Feedforward torques
# Solver computes: tau = kp*(q_ref - q) + kd*(dq_ref - dq) + tau_ref

# Direct torque
control.tau_j        # Direct actuation forces
```

## Differentiability

- **Forward-only for RL**: Kamino solver has `enable_backward: False`
- **Policy training is external**: Use PyTorch PPO/SAC, not backprop through sim
- **Trajectory optimization**: SolverSemiImplicit supports `wp.Tape()` gradients (used in drone diffsim example)

## Performance Benchmarks

- 2048 envs x 10 steps ~ 0.1s (100+ steps/sec throughput)
- Scales linearly to 4096+ with GPU memory
- CUDA graph capture eliminates kernel launch overhead
- Warp auto-compiles to CUDA (5-10s first run, microseconds after)

## Recommended RL Stack

For aerial manipulation RL:
1. **Simulator**: Newton with SolverKamino (GPU-optimized for RL)
2. **Observations**: Custom Warp kernels (drone state + arm joints + contacts)
3. **Actions**: Propeller commands (4) + arm joint targets (N)
4. **Training**: PyTorch-based PPO (e.g., rsl_rl, rl_games, or CleanRL)
5. **Vectorization**: Newton's built-in multi-world (2048+ parallel envs)
