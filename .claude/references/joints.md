# Newton Joints Reference

Comprehensive reference for joints in the Newton physics engine. Joints connect bodies
in kinematic trees (articulations) and constrain their relative motion.

---

## Table of Contents

1. [Joint Types Overview](#joint-types-overview)
2. [Joint Type Details](#joint-type-details)
3. [JointDofConfig](#jointdofconfig)
4. [JointTargetMode](#jointtargetmode)
5. [Joint Configuration Parameters](#joint-configuration-parameters)
6. [Articulations](#articulations)
7. [Forward and Inverse Kinematics](#forward-and-inverse-kinematics)
8. [Runtime Joint Control via Control Object](#runtime-joint-control-via-control-object)
9. [Toggling joint_enabled](#toggling-joint_enabled)
10. [Collapsing Fixed Joints](#collapsing-fixed-joints)
11. [Complete Working Examples](#complete-working-examples)
12. [Common Pitfalls](#common-pitfalls)

---

## Joint Types Overview

| Joint Type   | Enum Value | Velocity DOFs | Position Coords | Constraints | Description |
|-------------|-----------|--------------|----------------|------------|-------------|
| `PRISMATIC` | 0         | 1            | 1              | 5          | Translation along one axis |
| `REVOLUTE`  | 1         | 1            | 1              | 5          | Rotation about one axis |
| `BALL`      | 2         | 3            | 4 (quaternion) | 3          | Rotation about all three axes |
| `FIXED`     | 3         | 0            | 0              | 6          | Locks all relative motion |
| `FREE`      | 4         | 6            | 7 (3 pos + 4 quat) | 0     | Full 6-DOF motion |
| `DISTANCE`  | 5         | 6            | 7 (3 pos + 4 quat) | 0     | Keeps bodies within distance range |
| `D6`        | 6         | N (user-defined) | N           | 6 - N      | Generic configurable joint |
| `CABLE`     | 7         | 2            | 2              | 4          | Stretch + bend/twist |

The `JointType` enum lives in the public API as `newton.JointType`.

---

## Joint Type Details

### REVOLUTE (Hinge)

Single rotational DOF about a specified axis.

```python
builder.add_joint_revolute(
    parent: int,                              # Parent body index (-1 = world)
    child: int,                               # Child body index
    parent_xform: Transform | None = None,    # Joint anchor in parent frame
    child_xform: Transform | None = None,     # Joint anchor in child frame
    axis: AxisType | Vec3 | JointDofConfig | None = None,  # Rotation axis
    target_pos: float | None = None,          # Target angle [rad]
    target_vel: float | None = None,          # Target angular velocity [rad/s]
    target_ke: float | None = None,           # Position gain (stiffness)
    target_kd: float | None = None,           # Velocity gain (damping)
    limit_lower: float | None = None,         # Lower angle limit [rad]
    limit_upper: float | None = None,         # Upper angle limit [rad]
    limit_ke: float | None = None,            # Limit stiffness
    limit_kd: float | None = None,            # Limit damping
    armature: float | None = None,            # Artificial rotational inertia [kg*m^2]
    effort_limit: float | None = None,        # Max torque [N*m]
    velocity_limit: float | None = None,      # Max angular velocity [rad/s]
    friction: float | None = None,            # Joint friction
    actuator_mode: JointTargetMode | None = None,
    label: str | None = None,
    collision_filter_parent: bool = True,
    enabled: bool = True,
) -> int
```

**Use case:** Hinges, single-axis rotation (e.g., elbow, knee, door hinge).

### PRISMATIC (Slider)

Single translational DOF along a specified axis.

```python
builder.add_joint_prismatic(
    parent: int,
    child: int,
    parent_xform: Transform | None = None,
    child_xform: Transform | None = None,
    axis: AxisType | Vec3 | JointDofConfig = Axis.X,  # Translation axis
    target_pos: float | None = None,          # Target position [m]
    target_vel: float | None = None,          # Target velocity [m/s]
    target_ke: float | None = None,           # Position gain (stiffness)
    target_kd: float | None = None,           # Velocity gain (damping)
    limit_lower: float | None = None,         # Lower position limit [m]
    limit_upper: float | None = None,         # Upper position limit [m]
    limit_ke: float | None = None,            # Limit stiffness
    limit_kd: float | None = None,            # Limit damping
    armature: float | None = None,            # Artificial inertia [kg]
    effort_limit: float | None = None,        # Max force [N]
    velocity_limit: float | None = None,      # Max velocity [m/s]
    friction: float | None = None,
    actuator_mode: JointTargetMode | None = None,
    label: str | None = None,
    collision_filter_parent: bool = True,
    enabled: bool = True,
) -> int
```

**Use case:** Linear actuators, sliders, telescoping arms.

### BALL (Spherical)

3 rotational DOFs. Position parameterized as quaternion (4 coords, xyzw), velocity as 3D angular vector.

```python
builder.add_joint_ball(
    parent: int,
    child: int,
    parent_xform: Transform | None = None,
    child_xform: Transform | None = None,
    armature: float | None = None,            # Artificial inertia [kg*m^2]
    friction: float | None = None,
    label: str | None = None,
    collision_filter_parent: bool = True,
    enabled: bool = True,
    actuator_mode: JointTargetMode | None = None,
) -> int
```

**Use case:** Shoulder joints, universal joints, any unconstrained rotation about a point.

**Note:** Target position/velocity control for ball joints is currently only supported in `SolverMuJoCo`.

### FIXED (Weld)

0 DOFs. Locks all relative motion between parent and child. Useful for anchoring bodies to the world or to other bodies.

```python
builder.add_joint_fixed(
    parent: int,                              # -1 = world
    child: int,
    parent_xform: Transform | None = None,
    child_xform: Transform | None = None,
    label: str | None = None,
    collision_filter_parent: bool = True,
    enabled: bool = True,
) -> int
```

**Use case:** Anchoring a body to the world, rigidly connecting two bodies (can later be collapsed via `collapse_fixed_joints()`).

### FREE (Floating Base)

6 DOFs (3 translational + 3 rotational). Position is 7 coords (3 translation + 4 quaternion xyzw). Velocity is 6D (3 linear + 3 angular).

```python
builder.add_joint_free(
    child: int,                               # Child body index
    parent_xform: Transform | None = None,
    child_xform: Transform | None = None,
    parent: int = -1,                         # Usually world (-1)
    label: str | None = None,
    collision_filter_parent: bool = True,
    enabled: bool = True,
) -> int
```

**Use case:** Floating-base robots (quadrupeds, drones), any body that moves freely in space. The positional DOFs are initialized from the child body's transform.

### DISTANCE

6 DOFs (same as FREE) but with a distance constraint between anchor points. Keeps bodies within `[min_distance, max_distance]`.

```python
builder.add_joint_distance(
    parent: int,
    child: int,
    parent_xform: Transform | None = None,
    child_xform: Transform | None = None,
    min_distance: float = -1.0,               # No limit if negative
    max_distance: float = 1.0,                # No limit if negative
    collision_filter_parent: bool = True,
    enabled: bool = True,
) -> int
```

**Use case:** Ropes, chains, tethered objects.

**Note:** Only supported in `SolverXPBD`.

### D6 (Generic)

User-defined DOFs via lists of `JointDofConfig` for linear and angular axes. The total DOF count equals `len(linear_axes) + len(angular_axes)`.

```python
builder.add_joint_d6(
    parent: int,
    child: int,
    linear_axes: Sequence[JointDofConfig] | None = None,
    angular_axes: Sequence[JointDofConfig] | None = None,
    label: str | None = None,
    parent_xform: Transform | None = None,
    child_xform: Transform | None = None,
    collision_filter_parent: bool = True,
    enabled: bool = True,
) -> int
```

**Use case:** Custom joint configurations (e.g., 2-DOF gimbal, planar joint). Gives full control over which axes are free and their individual parameters.

### CABLE

2 DOFs: one linear (stretch) and one angular (isotropic bend/twist). Designed for cable/rope simulation.

```python
builder.add_joint_cable(
    parent: int,
    child: int,
    parent_xform: Transform | None = None,
    child_xform: Transform | None = None,
    stretch_stiffness: float | None = None,   # Default: 1e9 [N/m]
    stretch_damping: float | None = None,     # Default: 0.0 (dimensionless Rayleigh)
    bend_stiffness: float | None = None,      # Default: 0.0 [N*m]
    bend_damping: float | None = None,        # Default: 0.0 (dimensionless Rayleigh)
    label: str | None = None,
    collision_filter_parent: bool = True,
    enabled: bool = True,
) -> int
```

**Use case:** Cables, wires, flexible rods.

**Note:** Supported by `SolverVBD` (uses AVBD backend for rigid bodies).

---

## JointDofConfig

`newton.ModelBuilder.JointDofConfig` configures a single degree of freedom for a joint. Used directly by `add_joint()` and `add_joint_d6()`, and implicitly by convenience methods.

```python
class ModelBuilder.JointDofConfig:
    def __init__(
        self,
        axis: AxisType | Vec3 = Axis.X,        # 3D axis direction (auto-normalized)
        limit_lower: float = -MAXVAL,           # Lower position limit [m or rad]
        limit_upper: float = MAXVAL,            # Upper position limit [m or rad]
        limit_ke: float = 1e4,                  # Limit stiffness [N/m or N*m/rad]
        limit_kd: float = 1e1,                  # Limit damping
        target_pos: float = 0.0,                # Target position [m or rad]
        target_vel: float = 0.0,                # Target velocity [m/s or rad/s]
        target_ke: float = 0.0,                 # PD position gain (stiffness)
        target_kd: float = 0.0,                 # PD velocity gain (damping)
        armature: float = 0.0,                  # Artificial inertia [kg*m^2 or kg]
        effort_limit: float = 1e6,              # Max force/torque
        velocity_limit: float = 1e6,            # Max velocity
        friction: float = 0.0,                  # Joint friction
        actuator_mode: JointTargetMode | None = None,  # Explicit mode (inferred if None)
    )
```

### Key behaviors

- **`target_pos` auto-clamping:** If `target_pos` is outside `[limit_lower, limit_upper]`, it defaults to the midpoint of the limits.
- **Axis normalization:** The `axis` vector is automatically normalized via `wp.normalize()`.
- **Unlimited factory:** `JointDofConfig.create_unlimited(axis)` creates a config with no limits (`limit_lower=-MAXVAL`, `limit_upper=MAXVAL`) and zero gains/stiffness.

### D6 joint example with JointDofConfig

```python
import newton
from newton import ModelBuilder

builder = newton.ModelBuilder()

# 2-DOF planar joint: translate along X, rotate around Z
linear_x = ModelBuilder.JointDofConfig(
    axis=(1.0, 0.0, 0.0),
    limit_lower=-1.0,
    limit_upper=1.0,
    target_ke=100.0,
    target_kd=10.0,
)
angular_z = ModelBuilder.JointDofConfig(
    axis=(0.0, 0.0, 1.0),
    limit_lower=-3.14,
    limit_upper=3.14,
    target_ke=50.0,
    target_kd=5.0,
)

body = builder.add_link()
builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
j = builder.add_joint_d6(
    parent=-1,
    child=body,
    linear_axes=[linear_x],
    angular_axes=[angular_z],
    label="planar_joint",
)
builder.add_articulation([j])
```

---

## JointTargetMode

`newton.JointTargetMode` determines which actuators are active for a joint DOF.

| Mode                | Value | Description |
|---------------------|-------|-------------|
| `NONE`              | 0     | No actuators. Joint is passive/unactuated. |
| `POSITION`          | 1     | Position actuator only. Tracks `joint_target_pos`. |
| `VELOCITY`          | 2     | Velocity actuator only. Tracks `joint_target_vel`. |
| `POSITION_VELOCITY` | 3     | Both position and velocity actuators. Tracks both targets. |
| `EFFORT`            | 4     | Direct force/torque control via `joint_f`. No PD gains. |

### Automatic inference from gains

If `actuator_mode` is not explicitly set, Newton infers it:

```python
JointTargetMode.from_gains(target_ke, target_kd, has_drive=True)
# target_ke > 0 and target_kd > 0  ->  POSITION (not POSITION_VELOCITY, unless forced)
# target_ke > 0 and target_kd == 0 ->  POSITION
# target_ke == 0 and target_kd > 0 ->  VELOCITY
# target_ke == 0 and target_kd == 0 ->  EFFORT (if has_drive=True), NONE (if has_drive=False)
```

To get `POSITION_VELOCITY`, either set `actuator_mode` explicitly or use `force_position_velocity=True`.

### When to use each mode

- **NONE:** Passive joints (e.g., a pendulum swinging freely under gravity).
- **POSITION:** Servo-like behavior where you set target angles/positions and the PD controller drives the joint there.
- **VELOCITY:** Motor-like behavior where you set target velocities (e.g., wheels).
- **POSITION_VELOCITY:** Full PD control with both position and velocity tracking. Best for precise trajectory following.
- **EFFORT:** Direct torque/force control. You supply forces via `control.joint_f`. Useful for RL policies that output raw torques.

---

## Joint Configuration Parameters

### Drive parameters (PD controller)

The joint drive applies a PD control law:

```
F = target_ke * (target_pos - q) + target_kd * (target_vel - qd)
```

| Parameter    | Description | Units |
|-------------|-------------|-------|
| `target_ke` | Proportional gain (stiffness) | N/m or N*m/rad |
| `target_kd` | Derivative gain (damping) | N*s/m or N*m*s/rad |
| `target_pos`| Desired position | m or rad |
| `target_vel`| Desired velocity | m/s or rad/s |

### Limit parameters

| Parameter     | Description | Default |
|--------------|-------------|---------|
| `limit_lower` | Lower position limit | -MAXVAL (unlimited) |
| `limit_upper` | Upper position limit | MAXVAL (unlimited) |
| `limit_ke`    | Limit contact stiffness | 1e4 |
| `limit_kd`    | Limit contact damping | 1e1 |

### Other parameters

| Parameter        | Description | Default |
|-----------------|-------------|---------|
| `armature`       | Artificial inertia added to the DOF | 0.0 |
| `effort_limit`   | Maximum force/torque the DOF can exert | 1e6 |
| `velocity_limit` | Maximum velocity the DOF can achieve | 1e6 |
| `friction`       | Coulomb friction on the DOF | 0.0 |

### Builder defaults

`ModelBuilder` has a `default_joint_cfg` attribute of type `JointDofConfig`. When a parameter is passed as `None` to a joint method, the corresponding value from `default_joint_cfg` is used.

---

## Articulations

An articulation is a set of contiguous joints forming a kinematic tree. Functions like `eval_fk()` parallelize over articulations.

```python
builder.add_articulation(
    joints: list[int],              # Joint indices (must be contiguous and monotonically increasing)
    label: str | None = None,
)
```

### Rules

1. **Contiguous indices:** Joint indices must be consecutive (e.g., `[3, 4, 5]`, not `[3, 5]`).
2. **Monotonically increasing:** Joints must be in order. Create all joints for one articulation before creating joints for another.
3. **No sharing:** Each joint belongs to exactly one articulation.
4. **Same world:** All joints in an articulation must belong to the same world.

### Pattern

```python
# Create bodies
base = builder.add_link(xform=wp.transform(p=wp.vec3(0, 0, 1)))
link1 = builder.add_link(xform=wp.transform(p=wp.vec3(0, 0, 0.5)))
link2 = builder.add_link(xform=wp.transform(p=wp.vec3(0, 0, 0)))

# Create joints (order matters - contiguous indices)
j0 = builder.add_joint_fixed(parent=-1, child=base)
j1 = builder.add_joint_revolute(parent=base, child=link1, axis=(1, 0, 0))
j2 = builder.add_joint_revolute(parent=link1, child=link2, axis=(1, 0, 0))

# Register articulation
builder.add_articulation([j0, j1, j2], label="my_arm")
```

---

## Forward and Inverse Kinematics

### Forward Kinematics: `newton.eval_fk()`

Computes body poses (`body_q`) and velocities (`body_qd`) from joint coordinates.

```python
newton.eval_fk(
    model: Model,
    joint_q: wp.array(dtype=float),     # Joint positions, shape [joint_coord_count]
    joint_qd: wp.array(dtype=float),    # Joint velocities, shape [joint_dof_count]
    state: State | Model,               # Target to update (body_q, body_qd)
    mask: wp.array(dtype=bool) | None = None,     # Per-articulation enable mask
    indices: wp.array(dtype=int) | None = None,   # Articulation indices to update
    body_flag_filter: int = BodyFlags.ALL,
)
```

**Typical usage after finalize:**

```python
model = builder.finalize()
state = model.state()
newton.eval_fk(model, model.joint_q, model.joint_qd, state)
```

### Inverse Kinematics: `newton.eval_ik()`

Computes joint coordinates from body poses. Useful for going from Cartesian body states back to joint space.

```python
newton.eval_ik(
    model: Model,
    state: State | Model,               # Source of body_q, body_qd
    joint_q: wp.array(dtype=float),     # Output joint positions
    joint_qd: wp.array(dtype=float),    # Output joint velocities
    mask: wp.array(dtype=bool) | None = None,
    indices: wp.array(dtype=int) | None = None,
    body_flag_filter: int = BodyFlags.ALL,
)
```

### IK Solver (Optimization-Based)

For task-space IK (e.g., reaching a target position), use `newton.ik`:

```python
import newton.ik as ik

solver = ik.IKSolver(model)
solver.add_objective(ik.IKObjectivePosition(
    body_index=end_effector_idx,
    target_pos=wp.vec3(0.5, 0.0, 0.3),
))
solver.add_objective(ik.IKObjectiveRotation(
    body_index=end_effector_idx,
    target_quat=wp.quat_identity(),
))
solver.solve(joint_q)
```

Available objectives: `IKObjectivePosition`, `IKObjectiveRotation`, `IKObjectiveJointLimit`.
Optimizers: `IKOptimizerLM` (Levenberg-Marquardt), `IKOptimizerLBFGS`.

---

## Runtime Joint Control via Control Object

The `Control` object holds time-varying inputs that change each simulation step.

```python
control = model.control()
```

### Control attributes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `control.joint_f` | `(joint_dof_count,)` | Generalized forces/torques [N or N*m] |
| `control.joint_target_pos` | `(joint_dof_count,)` | Position targets [m or rad] |
| `control.joint_target_vel` | `(joint_dof_count,)` | Velocity targets [m/s or rad/s] |
| `control.joint_act` | `(joint_dof_count,)` | Feedforward actuation input |

### Setting targets at runtime

```python
import warp as wp

model = builder.finalize()
state_0 = model.state()
state_1 = model.state()
control = model.control()

# Find the DOF index for a specific joint
joint_idx = model.joint_label.index("elbow")
dof_start = model.joint_qd_start.numpy()[joint_idx]

# Set a position target for a revolute joint (1 DOF)
control.joint_target_pos.numpy()[dof_start] = 1.57  # 90 degrees
control.joint_target_vel.numpy()[dof_start] = 0.0

# Or apply a direct torque
control.joint_f.numpy()[dof_start] = 10.0  # 10 N*m

# Step simulation
solver.step(state_0, state_1, control, contacts, dt)
```

### Free joint forces

For free joints, `joint_f` contains a 6D wrench `(f_x, f_y, f_z, t_x, t_y, t_z)` applied in world frame at the body's center of mass.

### Clearing control inputs

```python
control.clear()  # Resets all arrays to zero
```

---

## Toggling joint_enabled

The `model.joint_enabled` array (shape `[joint_count]`, dtype `bool`) controls whether each joint is active. When disabled, the connected bodies become disconnected.

**Supported solvers:** `SolverXPBD`, `SolverVBD`, `SolverSemiImplicit`. Not supported by `SolverFeatherstone`.

```python
model = builder.finalize()

# Disable a joint at runtime
joint_idx = model.joint_label.index("breakable_connection")
model.joint_enabled.numpy()[joint_idx] = False

# Re-enable it
model.joint_enabled.numpy()[joint_idx] = True
```

You can also set the initial enabled state at build time:

```python
builder.add_joint_revolute(
    parent=-1, child=body,
    axis=(0, 0, 1),
    enabled=False,  # Starts disabled
)
```

---

## Collapsing Fixed Joints

`builder.collapse_fixed_joints()` removes fixed joints and merges connected bodies into one. This simplifies the model for faster and more stable simulation.

```python
builder.collapse_fixed_joints(
    verbose: bool = wp.config.verbose,
    joints_to_keep: list[str] | None = None,  # Joint labels to preserve
) -> dict[str, Any]
```

**When to use:** After loading a URDF/MJCF that has many fixed joints connecting visual/collision geometry to link bodies. Collapsing them reduces body count and removes unnecessary constraints.

```python
builder = newton.ModelBuilder()
# ... load a robot with many fixed joints ...
builder.collapse_fixed_joints(joints_to_keep=["world_fixed"])
model = builder.finalize()
```

---

## Complete Working Examples

### Example 1: Double Pendulum (Revolute Joints)

```python
import warp as wp
import newton

builder = newton.ModelBuilder()

# Create links
link_a = builder.add_link(
    xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.0)),
)
builder.add_shape_box(link_a, hx=0.05, hy=0.05, hz=0.5)

link_b = builder.add_link(
    xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0)),
)
builder.add_shape_box(link_b, hx=0.05, hy=0.05, hz=0.5)

# Anchor link_a to world with a fixed joint
j0 = builder.add_joint_fixed(
    parent=-1,
    child=link_a,
    parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.5)),
    label="anchor",
)

# Revolute joint between link_a and link_b
j1 = builder.add_joint_revolute(
    parent=link_a,
    child=link_b,
    axis=wp.vec3(1.0, 0.0, 0.0),  # Rotate around X
    parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.5)),
    child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5)),
    limit_lower=-1.57,
    limit_upper=1.57,
    label="elbow",
)

# Register articulation
builder.add_articulation([j0, j1], label="pendulum")

# Finalize and simulate
model = builder.finalize()
state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.contacts()

newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

solver = newton.solvers.SolverXPBD(model)

dt = 1.0 / 1000.0
for step in range(1000):
    state_0.clear_forces()
    model.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0
```

### Example 2: Driven Prismatic Joint

```python
import warp as wp
import newton

builder = newton.ModelBuilder()

body = builder.add_link(
    xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0)),
    mass=1.0,
    inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
)
builder.add_shape_box(
    body,
    hx=0.1, hy=0.1, hz=0.1,
    cfg=newton.ModelBuilder.ShapeConfig(density=0.0),  # Use explicit mass above
)

j = builder.add_joint_prismatic(
    parent=-1,
    child=body,
    axis=wp.vec3(0.0, 0.0, 1.0),    # Slide along Z
    target_pos=0.5,                   # Target position: 0.5m
    target_ke=200.0,                  # Stiffness
    target_kd=20.0,                   # Damping
    limit_lower=-1.0,
    limit_upper=1.0,
    actuator_mode=newton.JointTargetMode.POSITION,
    label="slider",
)

builder.add_articulation([j])
model = builder.finalize()

state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.contacts()
newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

solver = newton.solvers.SolverXPBD(model)

dt = 1.0 / 1000.0
for step in range(500):
    state_0.clear_forces()
    model.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0
```

### Example 3: Ball Joint with Initial Orientation

```python
import warp as wp
import newton

builder = newton.ModelBuilder()

anchor = builder.add_link(
    xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.0)),
)
cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
builder.add_shape_sphere(anchor, radius=0.1, cfg=cfg)

arm = builder.add_link(
    xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.25)),
)
builder.add_shape_box(arm, hx=0.05, hy=0.05, hz=0.5)

j0 = builder.add_joint_fixed(
    parent=-1, child=anchor,
    parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.0)),
    label="anchor_fixed",
)
j1 = builder.add_joint_ball(
    parent=anchor, child=arm,
    parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0)),
    child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5)),
    label="shoulder",
)
builder.add_articulation([j0, j1])

# Set initial quaternion for the ball joint (xyzw format)
builder.joint_q[-4:] = wp.quat_rpy(0.3, 0.0, 0.5)

model = builder.finalize()
state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.contacts()
newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

solver = newton.solvers.SolverXPBD(model)

dt = 1.0 / 1000.0
for step in range(500):
    state_0.clear_forces()
    model.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0
```

### Example 4: Free Joint (Floating Body)

```python
import warp as wp
import newton

builder = newton.ModelBuilder()

body = builder.add_link(
    xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.0)),
)
builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)

j = builder.add_joint_free(child=body, label="floating")
builder.add_articulation([j])

model = builder.finalize()
state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.contacts()
newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

# Apply an upward force through the free joint DOFs
# Free joint DOFs: [f_x, f_y, f_z, t_x, t_y, t_z]
dof_start = model.joint_qd_start.numpy()[0]
control.joint_f.numpy()[dof_start + 2] = 20.0  # Force along Z

solver = newton.solvers.SolverXPBD(model)

dt = 1.0 / 1000.0
for step in range(200):
    state_0.clear_forces()
    model.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0
```

### Example 5: D6 Joint (Custom 2-DOF)

```python
import warp as wp
import newton
from newton import ModelBuilder

builder = newton.ModelBuilder()

body = builder.add_link(
    xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0)),
)
builder.add_shape_box(body, hx=0.1, hy=0.3, hz=0.1)

# Create a joint with 1 linear (X) + 1 angular (Z) DOF
lin_x = ModelBuilder.JointDofConfig(
    axis=(1.0, 0.0, 0.0),
    limit_lower=-0.5,
    limit_upper=0.5,
    target_ke=100.0,
    target_kd=10.0,
)
ang_z = ModelBuilder.JointDofConfig(
    axis=(0.0, 0.0, 1.0),
    limit_lower=-1.57,
    limit_upper=1.57,
    target_ke=50.0,
    target_kd=5.0,
)

j = builder.add_joint_d6(
    parent=-1,
    child=body,
    linear_axes=[lin_x],
    angular_axes=[ang_z],
    parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0)),
    label="planar",
)
builder.add_articulation([j])

model = builder.finalize()
state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.contacts()
newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

solver = newton.solvers.SolverXPBD(model)

dt = 1.0 / 1000.0
for step in range(500):
    state_0.clear_forces()
    model.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0
```

### Example 6: Runtime Target Control (RL-style)

```python
import warp as wp
import numpy as np
import newton

builder = newton.ModelBuilder()

# 3-link arm
links = []
joints = []

for i in range(3):
    z = 2.0 - i * 0.6
    link = builder.add_link(xform=wp.transform(p=wp.vec3(0, 0, z)))
    builder.add_shape_box(link, hx=0.04, hy=0.04, hz=0.25)
    links.append(link)

# Fix base to world
j0 = builder.add_joint_fixed(parent=-1, child=links[0],
    parent_xform=wp.transform(p=wp.vec3(0, 0, 2.0)))
joints.append(j0)

for i in range(1, 3):
    j = builder.add_joint_revolute(
        parent=links[i-1], child=links[i],
        axis=(1, 0, 0),
        parent_xform=wp.transform(p=wp.vec3(0, 0, -0.25)),
        child_xform=wp.transform(p=wp.vec3(0, 0, 0.25)),
        target_ke=200.0,
        target_kd=20.0,
        actuator_mode=newton.JointTargetMode.POSITION,
        label=f"joint_{i}",
    )
    joints.append(j)

builder.add_articulation(joints)
model = builder.finalize()

state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.contacts()
newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

solver = newton.solvers.SolverXPBD(model)
dt = 1.0 / 1000.0

# Simulate with changing targets (e.g., from an RL policy)
for episode_step in range(1000):
    # RL policy outputs target angles
    target_angles = np.sin(episode_step * 0.01) * np.array([0.5, 0.8])

    # Write targets to control object
    for i, jname in enumerate(["joint_1", "joint_2"]):
        jidx = model.joint_label.index(jname)
        dof_start = model.joint_qd_start.numpy()[jidx]
        control.joint_target_pos.numpy()[dof_start] = target_angles[i]

    state_0.clear_forces()
    model.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0
```

---

## Common Pitfalls

### 1. Forgetting to call `add_articulation()`

Joints alone do not form a kinematic tree. Without `add_articulation()`, `eval_fk()` has nothing to process and body poses will not be updated from joint coordinates.

### 2. Non-contiguous or out-of-order joint indices

`add_articulation()` requires joint indices to be contiguous and monotonically increasing. Create all joints for one articulation before starting another.

```python
# WRONG: joints created interleaved
j_arm_0 = builder.add_joint_revolute(...)
j_leg_0 = builder.add_joint_revolute(...)  # Now arm joints are non-contiguous
j_arm_1 = builder.add_joint_revolute(...)
builder.add_articulation([j_arm_0, j_arm_1])  # Fails: gap between indices
```

### 3. Not calling `eval_fk()` after finalize

After `model = builder.finalize()`, the `state.body_q` is not yet populated from joint coordinates. Always call `eval_fk()` before the first simulation step:

```python
newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
```

### 4. Quaternion format (xyzw)

Newton uses `xyzw` quaternion convention (scalar last), matching Warp. When setting ball/free joint coordinates directly, ensure the quaternion is in the correct format:

```python
# Ball joint: last 4 coords of joint_q are quaternion (x, y, z, w)
builder.joint_q[-4:] = wp.quat_rpy(roll, pitch, yaw)  # Returns xyzw
builder.joint_q[-1] = 1.0  # w component (identity rotation)
```

### 5. State swapping

Always swap states after each step. The solver reads from `state_0` and writes to `state_1`:

```python
solver.step(state_0, state_1, control, contacts, dt)
state_0, state_1 = state_1, state_0  # Critical!
```

### 6. Mixing solvers and joint types

Not all solvers support all joint types:
- `DISTANCE` joints: only `SolverXPBD`
- `CABLE` joints: only `SolverVBD`
- Ball joint target control: only `SolverMuJoCo`
- `joint_enabled` toggling: `SolverXPBD`, `SolverVBD`, `SolverSemiImplicit` (not `SolverFeatherstone`)

### 7. Parent body index -1 means world

When `parent=-1`, the joint connects the child to the fixed world frame. The `parent_xform` then specifies the joint anchor in world coordinates.

### 8. Transforms define the joint frame, not the body position

`parent_xform` and `child_xform` define where the joint anchor sits relative to each body's local origin, not the body's world position. For a revolute joint at the tip of a link:

```python
# parent_xform: anchor at the bottom of parent body
parent_xform=wp.transform(p=wp.vec3(0, 0, -half_length))
# child_xform: anchor at the top of child body
child_xform=wp.transform(p=wp.vec3(0, 0, +half_length))
```

### 9. Default gains are zero

By default, `target_ke=0.0` and `target_kd=0.0`. Joints are passive unless you explicitly set gains or use `actuator_mode=EFFORT` with `joint_f`.

### 10. `collapse_fixed_joints()` must be called before `finalize()`

It modifies the builder's internal data structures. Calling it after `finalize()` has no effect.

---

## Key Source Files

| File | Contents |
|------|----------|
| `newton/__init__.py` | Public API exports (`JointType`, `JointTargetMode`, `ModelBuilder`, `eval_fk`, `eval_ik`) |
| `newton/_src/sim/enums.py` | `JointType` and `JointTargetMode` enum definitions |
| `newton/_src/sim/builder.py` | `ModelBuilder` with all `add_joint_*` methods, `JointDofConfig`, `add_articulation`, `collapse_fixed_joints` |
| `newton/_src/sim/control.py` | `Control` class definition |
| `newton/_src/sim/articulation.py` | `eval_fk()` and `eval_ik()` implementations |
| `newton/_src/sim/model.py` | `Model` class with joint arrays (`joint_enabled`, etc.) |
| `newton/ik.py` | Public IK API (`IKSolver`, objectives, optimizers) |
| `newton/examples/basic/example_basic_joints.py` | Working example with revolute, prismatic, and ball joints |
| `newton/tests/test_joint_drive.py` | Test for PD drive control |
| `newton/tests/test_joint_limits.py` | Test for joint limits |
