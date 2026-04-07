# Newton Multi-World Simulation and RL Reference

This document covers multi-world construction, state/control data structures,
reinforcement-learning integration patterns, and zero-copy PyTorch interop in the
Newton physics engine (including the Kamino solver).

---

## Table of Contents

1. [builder.replicate()](#1-builderreplicate)
2. [begin_world() / end_world()](#2-begin_world--end_world)
3. [World Indexing](#3-world-indexing)
4. [Per-World Gravity](#4-per-world-gravity)
5. [Per-World State Access](#5-per-world-state-access)
6. [Per-World Reset (world_mask)](#6-per-world-reset-world_mask)
7. [RigidBodySim Wrapper](#7-rigidbodysim-wrapper)
8. [Observation Building Patterns](#8-observation-building-patterns)
9. [Action Application](#9-action-application)
10. [Zero-Copy PyTorch Integration](#10-zero-copy-pytorch-integration)
11. [CUDA Graph Capture for RL](#11-cuda-graph-capture-for-rl)
12. [State Object Fields](#12-state-object-fields)
13. [Control Object Fields](#13-control-object-fields)
14. [ControlKamino Fields](#14-controlkamino-fields)
15. [External Force Application](#15-external-force-application)
16. [Complete Working Examples](#16-complete-working-examples)
17. [Common Pitfalls](#17-common-pitfalls)

---

## 1. builder.replicate()

Clones a source builder N times, creating one world per copy with optional spatial
offsets.

### Signature

```python
def replicate(
    self,
    builder: ModelBuilder,
    world_count: int,
    spacing: tuple[float, float, float] = (0.0, 0.0, 0.0),
)
```

### Parameters

| Parameter     | Type                         | Description |
|--------------|------------------------------|-------------|
| `builder`    | `ModelBuilder`               | Source builder whose entities are cloned. |
| `world_count`| `int`                        | Number of worlds to create. |
| `spacing`    | `tuple[float, float, float]` | XYZ offset between copies. `(5.0, 5.0, 0.0)` arranges in a 2D grid. Defaults to `(0.0, 0.0, 0.0)`. |

### How It Works

Internally, `replicate` computes a grid of offsets and calls `add_world` for each:

```python
offsets = compute_world_offsets(world_count, spacing, self.up_axis)
xform = wp.transform_identity()
for i in range(world_count):
    xform[:3] = offsets[i]
    self.add_world(builder, xform=xform)
```

For visual separation, keep `spacing=(0.0, 0.0, 0.0)` and use
`viewer.set_world_offsets()` instead. Physical spacing reduces numerical stability.

---

## 2. begin_world() / end_world()

Manual world construction for heterogeneous scenes where each world may have
different entities.

### begin_world()

```python
def begin_world(
    self,
    label: str | None = None,
    attributes: dict[str, Any] | None = None,
    gravity: Vec3 | None = None,
)
```

Opens a new world scope. All entities added after this call are assigned to the
new world until `end_world()` is called.

- `label`: Optional unique identifier (defaults to `"world_{index}"`).
- `gravity`: Per-world gravity vector. If `None`, inherits from `builder.gravity` and `builder.up_axis`.
- Worlds **cannot be nested**. Calling `begin_world()` twice without `end_world()` raises `RuntimeError`.

### end_world()

```python
def end_world(self)
```

Closes the current world scope. Subsequent entities go to the global scope (world -1).

### add_world() (convenience)

`add_world(builder, xform=None, label_prefix=None)` -- equivalent to
`begin_world()` + `add_builder(builder, xform, label_prefix)` + `end_world()`.

### Example

```python
builder = newton.ModelBuilder()
builder.begin_world(label="robot_0")                        # world 0
builder.add_body(...); builder.add_shape_box(...)
builder.end_world()
builder.begin_world(label="robot_1", gravity=(0,0,0))       # world 1, zero-g
builder.add_body(...); builder.add_shape_box(...)
builder.end_world()
builder.add_ground_plane()                                   # global (world -1)
model = builder.finalize()
```

---

## 3. World Indexing

### Flat Tensor Layout

All entities (bodies, joints, shapes, particles) are stored in flat contiguous
arrays. Entities are ordered by world: all world-0 entities first, then world-1,
etc. Global entities (world -1) appear at the beginning or end.

### world_start Arrays

The `Model` object provides per-entity-type start-index arrays of shape
`(world_count + 2,)`:

| Array                              | Entities Indexed     |
|------------------------------------|---------------------|
| `model.body_world_start`          | Rigid bodies        |
| `model.shape_world_start`         | Collision shapes    |
| `model.joint_world_start`         | Joints              |
| `model.joint_dof_world_start`     | Joint DOFs          |
| `model.joint_coord_world_start`   | Joint coordinates   |
| `model.particle_world_start`      | Particles           |
| `model.articulation_world_start`  | Articulations       |

### Layout Convention

For an array of length `world_count + 2`:
- Indices `0` to `world_count - 1`: start index for each world.
- Index `-2` (i.e. `world_count`): start of global entities (world -1).
- Index `-1` (i.e. `world_count + 1`): total entity count (sentinel).

### Computing Per-World Counts

```python
# Number of bodies in world w
n = model.body_world_start[w + 1] - model.body_world_start[w]

# Number of global bodies
n_global = (model.body_world_start[-1] - model.body_world_start[-2]
            + model.body_world_start[0])

# Slice of joint_q belonging to world w
start = model.joint_coord_world_start[w]
end   = model.joint_coord_world_start[w + 1]
world_joint_q = state.joint_q.numpy()[start:end]
```

Each entity also has a direct world-index array (`model.body_world`, `model.shape_world`,
`model.joint_world`, `model.particle_world`) with value -1 for global entities.

---

## 4. Per-World Gravity

**Build time:** `builder.begin_world(gravity=(0.0, 0.0, -3.0))`. With `add_world`, gravity comes from the source builder.

**Runtime:** `model.gravity` is `wp.array (world_count,) dtype vec3`.

```python
model.set_gravity((0.0, 0.0, -9.81))           # all worlds
model.set_gravity((0.0, 0.0, -3.0), world=2)   # single world
solver.notify_model_changed(newton.SolverNotifyFlags.MODEL_PROPERTIES)  # required after
```

Global entities (world -1) use gravity from world 0.

---

## 5. Per-World State Access

### Using world_start Arrays (Standard Newton)

```python
ws = model.body_world_start.numpy()
# Body transforms for world w
body_q_world_w = state.body_q.numpy()[ws[w]:ws[w+1]]
```

### Using Kamino Reshaped Tensors

In the Kamino RL wrapper, state arrays are stored in a `(num_worlds, max_entities, ...)`
layout, making per-world access trivial:

```python
# sim_wrapper is a RigidBodySim instance
q_j = sim_wrapper.q_j        # (num_worlds, num_joint_dofs)
q_i = sim_wrapper.q_i        # (num_worlds, num_bodies, 7)
u_i = sim_wrapper.u_i        # (num_worlds, num_bodies, 6)

# Access world 5
world_5_joints = q_j[5]      # (num_joint_dofs,)
world_5_poses  = q_i[5]      # (num_bodies, 7)
```

---

## 6. Per-World Reset (world_mask)

The RL wrapper supports resetting specific worlds without affecting others.

```python
# Stage resets for terminated worlds
sim.set_dof(dof_positions=init_pos[ids], dof_velocities=zeros, env_ids=ids)
sim.set_root(root_positions=spawn_pos[ids], root_orientations=spawn_quat[ids], env_ids=ids)
sim.apply_resets()   # applies staged resets, then clears mask and flags
sim.reset()          # full reset of all worlds
```

**set_dof(dof_positions, dof_velocities, env_ids)** -- shapes `(len(env_ids), num_joint_dofs)`.

**set_root(root_positions, root_orientations, root_linear_velocities, root_angular_velocities, env_ids)** -- shapes `(len(env_ids), 3)` or `(len(env_ids), 4)` for quaternions. `env_ids=None` resets all.

---

## 7. RigidBodySim Wrapper

`RigidBodySim` from `kamino/examples/rl/simulation.py` consolidates ~300 lines of
RL simulation boilerplate into a single class.

### Constructor

```python
RigidBodySim(usd_model_path, num_worlds=1, sim_dt=0.01, device=None,
             headless=False, body_pose_offset=None, add_ground=True,
             enable_gravity=True, settings=None, use_cuda_graph=False,
             record_video=False, max_contacts_per_pair=None,
             render_config=None, collapse_fixed_joints=False)
```

### Key Properties (all zero-copy PyTorch tensors unless noted)

**State (read-only):** `q_j (nw, njd)`, `dq_j (nw, njd)`, `q_i (nw, nb, 7)`, `u_i (nw, nb, 6)`

**Control (writable):** `q_j_ref (nw, njd)`, `dq_j_ref (nw, njd)`, `tau_j_ref (nw, njd)`

**Contacts:** `contact_flags (nw, nb)`, `ground_contact_flags (nw, nb)`, `net_contact_forces (nw, nb, 3)`

**Other:** `world_mask (nw,) int32`, `external_wrenches (nw, nb, 6)`, `body_masses (nw, nb)`,
`default_q_j (nw, njd)`, `joint_limits list[[lo, hi]]`, `env_origins (nw, 3)`

**Scalars:** `num_worlds`, `num_joint_dofs`, `num_bodies`, `num_actuated`

**Name lookups:** `joint_names`, `body_names`, `actuated_joint_names`, `actuated_dof_indices`,
`actuated_dof_indices_tensor (torch.long)`

### Key Methods

`step()`, `reset()`, `apply_resets()`, `set_dof(...)`, `set_root(...)`,
`render()`, `is_running()`, `find_body_index(name)`, `default_settings(dt)` (static)

### Construction Flow

Load USD -> `add_world()` per world -> `add_ground_plane()` -> `finalize()` ->
`SimulatorFromNewton` -> wire zero-copy PyTorch views -> extract metadata ->
optionally create viewer -> optionally capture CUDA graphs -> warm-up.

---

## 8. Observation Building Patterns

### Warp Kernel Approach

Observations are computed in a single Warp kernel that runs on GPU, one thread
per world. The kernel reads from flat Warp arrays and writes into a flat
`obs` output array.

### Key Pattern: Flat Array Indexing by World

```python
@wp.kernel
def _compute_obs(obs, q_i, u_i, q_j, dq_j, ..., num_bodies, num_obs):
    w = wp.tid()  # world index
    qi_base  = w * num_bodies * 7   # 7 floats per body (pos + quat)
    ui_base  = w * num_bodies * 6   # 6 floats per body (lin + ang vel)
    o        = w * num_obs           # output offset

    root_quat = wp.quat(q_i[qi_base+3], q_i[qi_base+4], q_i[qi_base+5], q_i[qi_base+6])
    obs[o + offset] = ...            # write per-world observations
```

### DR-Legs Observation Space (94D)

`ori_root_to_path(9)` + `path_deviation(2)` + `path_dev_heading(2)` + `path_cmd(3)` +
`cmd_linvel_in_root(3)` + `cmd_angvel_in_root(3)` + `phase_encoding(4)` +
`root_linvel_in_root(3)` + `root_angvel_in_root(3)` + `cmd_height(1)` + `height_error(1)` +
`normalized_joint_positions(36)` + `action_history(24)` = **94D**

Launched with `wp.launch(_compute_bipedal_obs_core, dim=num_worlds, inputs=[...])`.

---

## 9. Action Application

### Implicit PD Control (Kamino)

The Kamino solver supports implicit PD joint control. Actions from the policy
are mapped to joint position references:

```python
def _apply_actions(self):
    """Convert policy actions to implicit PD joint position references."""
    sim_wrapper.q_j_ref.zero_()
    sim_wrapper.q_j_ref[:, sim_wrapper.actuated_dof_indices_tensor] = (
        action_scale * self.actions
    )
    sim_wrapper.dq_j_ref.zero_()
```

The solver computes torques as:

```
tau = k_p * (q_j_ref - q_j) + k_d * (dq_j_ref - dq_j) + tau_j_ref
```

### Control Decimation

Typically the control frequency is lower than the physics frequency:

```python
for _ in range(control_decimation):
    sim_wrapper.step()  # e.g. 5 physics steps per control step
```

For direct torque control, write `sim_wrapper.tau_j_ref[:, actuated_indices] = torques`.
In standard Newton (without Kamino), use `control.joint_target_pos`, `control.joint_target_vel`,
and `control.joint_f` directly.

---

## 10. Zero-Copy PyTorch Integration

### wp.to_torch()

Creates a PyTorch tensor that shares memory with a Warp array (no copy):

```python
import warp as wp

# Warp array on GPU
warp_array = wp.zeros(100, dtype=wp.float32, device="cuda:0")

# Zero-copy view as PyTorch tensor
torch_tensor = wp.to_torch(warp_array)

# Reshape for multi-world layout
nw, njd = num_worlds, num_joint_dofs
q_j_torch = wp.to_torch(state.q_j).reshape(nw, njd)
```

### wp.from_torch()

Creates a Warp array that shares memory with a PyTorch tensor (no copy):

```python
import torch

torch_tensor = torch.zeros(100, device="cuda:0")
warp_array = wp.from_torch(torch_tensor)
```

### Typical RL Wiring Pattern

```python
# State (read from simulator) -- zero-copy views
q_j  = wp.to_torch(sim.state.q_j).reshape(nw, njd)    # joint positions
q_i  = wp.to_torch(sim.state.q_i).reshape(nw, nb, 7)   # body poses

# Control (write to simulator) -- zero-copy views
q_j_ref = wp.to_torch(sim.control.q_j_ref).reshape(nw, njd)

# World mask and external wrenches
mask_wp = wp.zeros((nw,), dtype=wp.int32, device=device)
mask    = wp.to_torch(mask_wp)
w_e_i   = wp.to_torch(sim.solver.data.bodies.w_e_i).reshape(nw, nb, 6)
```

Both `wp.to_torch()` and `wp.from_torch()` are zero-copy (shared GPU memory).
Writes to one are immediately visible in the other. Same-device requirement applies.

---

## 11. CUDA Graph Capture for RL

CUDA graphs eliminate kernel launch overhead by recording a sequence of GPU
operations and replaying them as a single unit.

### Requirements

- CUDA device with memory pool enabled: `wp.is_mempool_enabled(device)`
- All operations in the graph must be deterministic (no host-side branching).

### Capture and Replay in RigidBodySim

```python
# Capture
with wp.ScopedCapture(device=device) as step_capture:
    sim.step()
    contact_aggregation.compute()
step_graph = step_capture.graph

# Replay (in the step loop)
if step_graph:
    wp.capture_launch(step_graph)
else:
    sim.step()
    contact_aggregation.compute()
```

### State Swapping with CUDA Graphs

When using CUDA graphs, you cannot swap Python references. Instead, copy arrays:

```python
# Inside a CUDA graph, if sim_substeps is odd:
if sim_substeps % 2 == 1 and i == sim_substeps - 1:
    state_0.assign(state_1)  # array-level copy (graph-safe)
else:
    state_0, state_1 = state_1, state_0  # reference swap (not in graph)
```

---

## 12. State Object Fields

The `newton.State` object holds all time-varying simulation quantities.
Created via `model.state()`.

### Particle Fields

| Field            | Dtype      | Shape               | Units    | Description |
|-----------------|------------|----------------------|----------|-------------|
| `particle_q`   | `vec3`     | `(particle_count,)`  | m        | Particle positions |
| `particle_qd`  | `vec3`     | `(particle_count,)`  | m/s      | Particle velocities |
| `particle_f`   | `vec3`     | `(particle_count,)`  | N        | Particle forces (external) |

### Rigid Body Fields

| Field           | Dtype            | Shape            | Units         | Description |
|----------------|------------------|------------------|---------------|-------------|
| `body_q`       | `transform`      | `(body_count,)`  | m, quaternion | Body transforms (7-DOF: `[x,y,z, qx,qy,qz,qw]`) |
| `body_qd`      | `spatial_vector` | `(body_count,)`  | m/s, rad/s    | Body velocities: `[vx,vy,vz, wx,wy,wz]` (COM frame, world coords) |
| `body_f`       | `spatial_vector` | `(body_count,)`  | N, N*m        | External wrenches: `[fx,fy,fz, tx,ty,tz]` (world frame, COM ref) |
| `body_q_prev`  | `transform`      | `(body_count,)`  | m, quaternion | Previous transforms (for finite-difference velocity) |

**Extended (opt-in via `request_state_attributes`):** `body_qdd (spatial_vector)`,
`body_parent_f (spatial_vector)` -- both shape `(body_count,)`.

**Joint:** `joint_q (joint_coord_count,) float [m or rad]`,
`joint_qd (joint_dof_count,) float [m/s or rad/s]`

**Methods:** `clear_forces()` -- zeros `particle_f` and `body_f`.
`assign(other)` -- deep-copy all arrays (CUDA-graph safe).

---

## 13. Control Object Fields

The `newton.Control` object holds time-varying control inputs.
Created via `model.control()`.

| Field              | Dtype   | Shape                  | Units             | Description |
|-------------------|---------|------------------------|-------------------|-------------|
| `joint_f`         | `float` | `(joint_dof_count,)`   | N or N*m          | Generalized joint forces. Free joints: 6D wrench `(fx,fy,fz, tx,ty,tz)` in world frame at COM. |
| `joint_target_pos`| `float` | `(joint_dof_count,)`   | m or rad          | PD position targets |
| `joint_target_vel`| `float` | `(joint_dof_count,)`   | m/s or rad/s      | PD velocity targets |
| `joint_act`       | `float` | `(joint_dof_count,)`   | (varies)          | Feedforward actuation input (additive, applied before PD correction) |
| `tri_activations` | `float` | `(tri_count,)`         | dimensionless     | Triangle element activations |
| `tet_activations` | `float` | `(tet_count,)`         | dimensionless     | Tetrahedral element activations |
| `muscle_activations`| `float`| `(muscle_count,)`     | dimensionless 0-1 | Muscle activations |

### Methods

| Method    | Description |
|----------|-------------|
| `clear()` | Zero out all control arrays |

---

## 14. ControlKamino Fields

`ControlKamino` is the Kamino solver's control container, mapping to the
standard Newton `Control` fields with different naming conventions.

| ControlKamino Field | Newton Control Equivalent | Shape                 | Description |
|--------------------|---------------------------|-----------------------|-------------|
| `tau_j`            | `joint_f`                 | `(sum(d_j),)`        | Direct joint torques |
| `q_j_ref`          | `joint_target_pos`        | `(sum(c_j),)`        | PD position reference |
| `dq_j_ref`         | `joint_target_vel`        | `(sum(d_j),)`        | PD velocity reference |
| `tau_j_ref`        | `joint_act`               | `(sum(d_j),)`        | Feed-forward reference torque |

Where `d_j` = number of DOFs per joint, `c_j` = number of coordinates per joint.

### Conversion

```python
# Newton -> Kamino (zero-copy alias)
kamino_ctrl = ControlKamino.from_newton(newton_control)

# Kamino -> Newton (zero-copy alias)
newton_ctrl = ControlKamino.to_newton(kamino_control)
```

Both conversions create aliases without copying data. Modifying one modifies the other.

---

## 15. External Force Application

### Standard Newton: state.body_f

`body_f` is a `spatial_vector` array of shape `(body_count,)`.
Each entry is `(fx, fy, fz, tx, ty, tz)` -- force and torque in world frame,
applied at the body's center of mass.

```python
state = model.state()

# Apply a 10N upward force to body 0
import numpy as np
forces = np.zeros((model.body_count, 6), dtype=np.float32)
forces[0] = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]  # (fx, fy, fz, tx, ty, tz)
state.body_f.assign(forces)
```

Forces are cleared each step via `state.clear_forces()` and must be re-applied.

### Kamino RL: external_wrenches

In `RigidBodySim`, external wrenches are exposed as a zero-copy torch tensor:

```python
# Shape: (num_worlds, num_bodies, 6)
sim_wrapper.external_wrenches[:, body_idx, :3] = force_vector  # linear force
sim_wrapper.external_wrenches[:, body_idx, 3:] = torque_vector # torque

# These are zero-copy views into sim.solver.data.bodies.w_e_i
# They persist across steps (not auto-cleared like body_f)
```

For thread-safe force accumulation in custom Warp kernels, use `wp.atomic_add(body_f, idx, wrench)`.

---

## 16. Complete Working Examples

### Example: Kamino RL Pipeline

```python
import torch, warp as wp
from newton._src.solvers.kamino.examples.rl.simulation import RigidBodySim

wp.set_module_options({"enable_backward": False})

sim = RigidBodySim(
    usd_model_path="/path/to/robot.usda",
    num_worlds=64, sim_dt=0.004, device="cuda:0",
    headless=True, use_cuda_graph=True,
    body_pose_offset=(0.0, 0.0, 0.265, 0.0, 0.0, 0.0, 1.0),
)

for step in range(1000):
    obs = torch.cat([sim.q_j, sim.dq_j, sim.q_i[:, 0, :3]], dim=-1)
    actions = policy(obs)

    sim.q_j_ref.zero_()
    sim.q_j_ref[:, sim.actuated_dof_indices_tensor] = 0.4 * actions
    sim.dq_j_ref.zero_()

    for _ in range(5):  # control decimation
        sim.step()

    done = check_termination(sim)
    if done.any():
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        sim.set_dof(dof_positions=sim.default_q_j[ids], env_ids=ids)
        sim.apply_resets()

    sim.render()
```

---

## 17. Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Physical spacing in `replicate()` hurts numerical stability | Use `spacing=(0,0,0)` and `viewer.set_world_offsets()` for visuals |
| Nesting `begin_world()` calls | Always pair with `end_world()`, or use `add_world()` |
| `body_f` accumulates across steps | Call `state.clear_forces()` at start of each step |
| Host-side branching inside CUDA graph capture | Keep all conditional logic outside captured regions |
| Python reference swap invisible to CUDA graphs | Use `state_0.assign(state_1)` instead of swapping references |
| Changing model properties without notifying solver | Call `solver.notify_model_changed(SolverNotifyFlags.MODEL_PROPERTIES)` |
| `wp.to_torch()` tensor outlives source Warp array | Keep a reference to the Warp array as long as the tensor is used |
| Global entities (world -1) use world-0 gravity | Add ground planes *after* all worlds, outside world context |
| Kamino `q_i` is COM frame, not body origin | `RigidBodySim.render()` handles conversion; be aware when reading directly |
| Warp arrays are flat, RL needs per-world shape | Reshape immediately: `wp.to_torch(arr).reshape(nw, dofs_per_world)` |

---

## Source File Locations

| File | Description |
|------|-------------|
| `submodules/newton/newton/_src/sim/builder.py` | `ModelBuilder`, `replicate()`, `begin_world()`, `end_world()`, `add_world()` |
| `submodules/newton/newton/_src/sim/model.py` | `Model` class, world_start arrays, `set_gravity()` |
| `submodules/newton/newton/_src/sim/state.py` | `State` class with all state fields |
| `submodules/newton/newton/_src/sim/control.py` | `Control` class with all control fields |
| `submodules/newton/newton/_src/solvers/kamino/_src/core/control.py` | `ControlKamino` dataclass |
| `submodules/newton/newton/_src/solvers/kamino/_src/core/state.py` | `StateKamino` dataclass |
| `submodules/newton/newton/_src/solvers/kamino/examples/rl/simulation.py` | `RigidBodySim`, `SimulatorFromNewton` |
| `submodules/newton/newton/_src/solvers/kamino/examples/rl/observations.py` | Warp observation kernels |
| `submodules/newton/newton/_src/solvers/kamino/examples/rl/example_rl_drlegs.py` | Full DR-Legs RL example |
