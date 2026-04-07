# Newton Simulator — Core API Reference

## Builder Pattern

All simulation construction follows: **ModelBuilder → finalize() → Model → State/Control/Contacts → Solver.step()**

```python
import newton
import warp as wp

builder = newton.ModelBuilder()
builder.add_body(xform=wp.transform(...), label="drone")
builder.add_shape_box(body_id, hx=0.1, hy=0.1, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=1750.0))
model = builder.finalize()

state_0, state_1 = model.state(), model.state()
control = model.control()
contacts = model.contacts()
solver = newton.solvers.SolverXPBD(model)

for frame in range(N):
    state_0.clear_forces()
    model.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt=0.01)
    state_0, state_1 = state_1, state_0
```

## Bodies & Shapes

```python
body_id = builder.add_body(xform=wp.transform(pos, quat), label="name", mass=1.0)

# Primitives
builder.add_shape_sphere(body, radius=0.1)
builder.add_shape_box(body, hx=0.1, hy=0.2, hz=0.05)
builder.add_shape_capsule(body, radius=0.1, half_height=0.3)
builder.add_shape_cylinder(body, radius=0.1, half_height=0.2)
builder.add_shape_mesh(body, mesh=newton.Mesh(...))

# Shape config (contact properties, density, collision groups)
cfg = newton.ModelBuilder.ShapeConfig(
    density=1750.0,       # kg/m³
    ke=2.5e3,             # contact stiffness [N/m]
    kd=100.0,             # contact damping [N·s/m]
    mu=1.0,               # friction coefficient
    collision_group=0,    # collision group ID
)
```

## Joint Types

| Type | DOF | Method | Use Case |
|------|-----|--------|----------|
| FREE | 6 | `add_joint_free()` | Floating bodies (drones) |
| FIXED | 0 | `add_joint_fixed()` | Rigid connections |
| REVOLUTE | 1 | `add_joint_revolute()` | Hinges, motors |
| PRISMATIC | 1 | `add_joint_prismatic()` | Linear actuators |
| BALL | 3 | `add_joint_ball()` | Spherical joints |
| D6 | 0-6 | `add_joint_d6()` | Generic configurable |
| DISTANCE | 6 | `add_joint_distance()` | Distance constraint |
| CABLE | 2 | `add_joint_cable()` | Tendons |

```python
joint_id = builder.add_joint_revolute(
    parent=body_a, child=body_b,
    axis=newton.Axis.Z,
    limit_lower=-3.14, limit_upper=3.14,
    target_ke=100.0,   # PD stiffness [N·m/rad]
    target_kd=10.0,    # PD damping [N·m·s/rad]
)
```

## Joint Actuation Modes

| Mode | Enum | Description |
|------|------|-------------|
| NONE | 0 | Passive (no actuation) |
| POSITION | 1 | Tracks `control.joint_target_pos` |
| VELOCITY | 2 | Tracks `control.joint_target_vel` |
| POSITION_VELOCITY | 3 | Both position and velocity |
| EFFORT | 4 | Direct torque via `control.joint_f` |

## State Access (Observation for RL)

```python
state.body_q      # Body transforms (pos + quat), shape (body_count,)
state.body_qd     # Spatial velocities (lin + ang), shape (body_count,)
state.body_f      # Forces/torques, shape (body_count,)
state.joint_q     # Joint positions [m or rad], shape (joint_coord_count,)
state.joint_qd    # Joint velocities, shape (joint_dof_count,)
```

Extended attributes (must be requested):
```python
model.request_state_attributes("body_qdd", "body_parent_f")
state.body_qdd       # Accelerations
state.body_parent_f  # Joint reaction forces
```

## Control Inputs (Actions for RL)

```python
control = model.control()
control.joint_f           # Direct generalized forces [N or N·m]
control.joint_target_pos  # Position targets for PD control
control.joint_target_vel  # Velocity targets
control.joint_act         # Feedforward actuation
```

## Available Solvers

| Solver | Coordinates | Differentiable | Best For |
|--------|------------|----------------|----------|
| **SolverXPBD** | Maximal | Limited | General rigid + particles |
| **SolverFeatherstone** | Generalized | Basic | Articulated robots |
| **SolverMuJoCo** | Generalized | No | Full MuJoCo compatibility |
| **SolverSemiImplicit** | Maximal | Yes | Differentiable sim |
| **SolverKamino** | Maximal | No (backward disabled) | RL training (GPU-optimized) |
| **SolverVBD** | — | — | Cloth + deformable |

## Multi-World (Parallel Environments)

```python
# Approach 1: replicate a template
template = newton.ModelBuilder()
# ... add single robot ...

builder = newton.ModelBuilder()
builder.replicate(template, world_count=2048, spacing=(2.0, 2.0, 0.0))
model = builder.finalize()

# Approach 2: begin_world / end_world
builder = newton.ModelBuilder()
for i in range(N):
    builder.begin_world(label=f"env_{i}")
    # ... add robot ...
    builder.end_world()
```

All worlds simulated in parallel on GPU. State arrays are flat tensors indexed by `(world_id, entity_id)`.

## Model Loading

```python
# USD (preferred for Newton-native assets)
result = builder.add_usd("robot.usda", floating=True, collapse_fixed_joints=True)

# URDF (common in robotics)
builder.add_urdf("robot.urdf", floating=True, enable_self_collisions=False)

# MJCF (MuJoCo XML)
builder.add_mjcf("robot.xml", parse_mujoco_options=True)
```

`result` dict contains: `path_body_map`, `path_joint_map`, `path_shape_map`, `fps`, `physics_dt`, etc.

## Key Constraint: Immutability After finalize()

Once `builder.finalize()` is called, the Model structure is **read-only**. All structural changes (bodies, joints, shapes) must happen before finalization. Only state values (positions, velocities, forces) and some model arrays (joint targets, joint_enabled) can change at runtime.
