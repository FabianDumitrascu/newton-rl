# Newton Contacts, Collision & Sensors Reference

Comprehensive reference for the contact/collision system and sensor APIs in the Newton
physics engine (`submodules/newton/`).

---

## Table of Contents

1. [CollisionPipeline](#1-collisionpipeline)
2. [model.collide() Usage](#2-modelcollide-usage)
3. [Contact Data Fields](#3-contact-data-fields)
4. [Soft Contacts (Particle-Shape)](#4-soft-contacts-particle-shape)
5. [Contact Material Properties](#5-contact-material-properties)
6. [Collision Filtering](#6-collision-filtering)
7. [Hydroelastic Contacts](#7-hydroelastic-contacts)
8. [Contact Force Extraction](#8-contact-force-extraction)
9. [SensorIMU](#9-sensorimu)
10. [SensorContact](#10-sensorcontact)
11. [Other Sensors](#11-other-sensors)
12. [Complete Working Examples](#12-complete-working-examples)
13. [Common Pitfalls](#13-common-pitfalls)

---

## 1. CollisionPipeline

**Source:** `newton/_src/sim/collide.py` -- class `CollisionPipeline`

The `CollisionPipeline` coordinates collision detection with a pluggable broad phase and
a GJK/MPR narrow phase. It supports mesh-mesh collision via SDF with contact reduction
and optional hydroelastic contacts.

### Broad Phase Options

| Mode | String | Class | Complexity | Best For |
|------|--------|-------|------------|----------|
| All-Pairs | `"nxn"` | `BroadPhaseAllPairs` | O(N^2) AABB overlap | Small scenes (< 100 shapes) |
| Sweep-and-Prune | `"sap"` | `BroadPhaseSAP` | O(N log N) per-axis sort | Moderately dynamic scenes |
| Explicit | `"explicit"` | `BroadPhaseExplicit` | O(P) where P = pair count | Tightly bounded pair sets |

**Requirements:**
- `"nxn"` and `"sap"` require `model.shape_world` to be set (always true after `finalize()`).
- `"explicit"` requires precomputed `model.shape_contact_pairs` or a `shape_pairs_filtered` argument.

### Construction

```python
import newton

# Simple -- default is "explicit" (uses model.shape_contact_pairs)
pipeline = newton.CollisionPipeline(model)

# Specify broad phase mode
pipeline = newton.CollisionPipeline(model, broad_phase="sap")
pipeline = newton.CollisionPipeline(model, broad_phase="nxn")

# With additional options
pipeline = newton.CollisionPipeline(
    model,
    broad_phase="sap",
    reduce_contacts=True,          # contact reduction for mesh-mesh (default True)
    rigid_contact_max=50000,       # override auto-estimated capacity
    max_triangle_pairs=1000000,    # buffer for mesh triangle pair tests
    soft_contact_margin=0.01,      # margin for particle-shape contacts [m]
)
```

### Expert API -- Prebuilt Components

```python
from newton._src.geometry.broad_phase_nxn import BroadPhaseAllPairs
from newton._src.geometry.narrow_phase import NarrowPhase

broad = BroadPhaseAllPairs(model.shape_world, shape_flags=model.shape_flags, device=model.device)
narrow = NarrowPhase(
    max_candidate_pairs=10000,
    max_triangle_pairs=1000000,
    reduce_contacts=True,
    device=model.device,
)

pipeline = newton.CollisionPipeline(model, broad_phase=broad, narrow_phase=narrow)
```

> When providing prebuilt components, both `broad_phase` and `narrow_phase` must be supplied.

### Narrow Phase

The narrow phase uses:
- **GJK/MPR** for convex-convex shape pairs (sphere, box, capsule, cylinder, ellipsoid, cone, convex mesh).
- **SDF queries** for mesh-mesh and mesh-primitive pairs, with optional contact reduction.
- **Hydroelastic SDF** when both shapes have `is_hydroelastic=True`.

A "lean" GJK/MPR kernel is automatically selected when the scene has no capsules, ellipsoids,
cylinders, or cones (avoiding extra axial rolling post-processing).

---

## 2. model.collide() Usage

**Source:** `newton/_src/sim/model.py` -- methods `Model.collide()` and `Model.contacts()`

### Allocating Contacts

```python
model = builder.finalize()

# Option A: Let model allocate with its default pipeline (broad_phase="explicit")
contacts = model.contacts()

# Option B: Provide a custom pipeline
pipeline = newton.CollisionPipeline(model, broad_phase="sap")
contacts = model.contacts(collision_pipeline=pipeline)
```

### Running Collision Detection

```python
# Run collision and get populated contacts (allocates if needed)
contacts = model.collide(state)

# Or reuse an existing contacts buffer
contacts = model.contacts()
for step in range(num_steps):
    model.collide(state, contacts)       # clears and repopulates
    solver.step(state, state_next, control, contacts, dt)
    state, state_next = state_next, state
```

### What model.collide() Does Internally

1. Clears the contacts buffer (resets counts; optionally zeros all arrays).
2. Computes world-space AABBs for all shapes.
3. Runs the broad phase to find candidate pairs.
4. Prepares geometry data (transforms, scales, margins).
5. Runs the narrow phase (GJK/MPR or SDF) -- writes contacts directly.
6. Generates soft contacts for particles near shapes.
7. Returns the populated `Contacts` object.

---

## 3. Contact Data Fields

**Source:** `newton/_src/sim/contacts.py` -- class `Contacts`

### Rigid Contact Arrays

All arrays have shape `(rigid_contact_max,)`. Only the first `rigid_contact_count` entries
are valid after collision detection.

| Field | dtype | Description |
|-------|-------|-------------|
| `rigid_contact_count` | `int32` | Number of active rigid contacts (scalar view) |
| `rigid_contact_shape0` | `int32` | Shape index of first body in pair |
| `rigid_contact_shape1` | `int32` | Shape index of second body in pair |
| `rigid_contact_point0` | `vec3` | Contact point on shape 0 in body frame [m] |
| `rigid_contact_point1` | `vec3` | Contact point on shape 1 in body frame [m] |
| `rigid_contact_normal` | `vec3` | Unit normal from shape 0 toward shape 1 (A-to-B) |
| `rigid_contact_offset0` | `vec3` | Friction anchor offset for shape 0 in body frame [m] |
| `rigid_contact_offset1` | `vec3` | Friction anchor offset for shape 1 in body frame [m] |
| `rigid_contact_margin0` | `float` | Surface thickness: effective radius + margin for shape 0 [m] |
| `rigid_contact_margin1` | `float` | Surface thickness: effective radius + margin for shape 1 [m] |
| `rigid_contact_force` | `vec3` | Contact force [N] (filled by solver) |
| `rigid_contact_tids` | `int32` | Triangle ID (for mesh contacts) |
| `rigid_contact_point_id` | `int32` | Contact point ID |

### Per-Contact Shape Properties (Optional)

Allocated when `per_contact_shape_properties=True` (auto-enabled for hydroelastic pairs):

| Field | dtype | Description |
|-------|-------|-------------|
| `rigid_contact_stiffness` | `float` | Per-contact stiffness [N/m] |
| `rigid_contact_damping` | `float` | Per-contact damping [N*s/m] |
| `rigid_contact_friction` | `float` | Per-contact friction coefficient |

### Reading Contact Data

```python
contacts = model.collide(state)
count = int(contacts.rigid_contact_count.numpy()[0])

shapes0  = contacts.rigid_contact_shape0.numpy()[:count]
shapes1  = contacts.rigid_contact_shape1.numpy()[:count]
points0  = contacts.rigid_contact_point0.numpy()[:count]   # body-frame [m]
points1  = contacts.rigid_contact_point1.numpy()[:count]   # body-frame [m]
normals  = contacts.rigid_contact_normal.numpy()[:count]    # A-to-B unit normal
margins0 = contacts.rigid_contact_margin0.numpy()[:count]   # [m]
margins1 = contacts.rigid_contact_margin1.numpy()[:count]   # [m]
forces   = contacts.rigid_contact_force.numpy()[:count]     # [N] (after solver)
```

---

## 4. Soft Contacts (Particle-Shape)

Soft contacts represent collisions between particles (cloth, soft bodies, deformable objects)
and rigid shapes. They are generated automatically during `collide()` when the model contains
both particles and shapes.

### Soft Contact Arrays

All arrays have shape `(soft_contact_max,)`. Only the first `soft_contact_count` entries are valid.

| Field | dtype | Description |
|-------|-------|-------------|
| `soft_contact_count` | `int32` | Number of active soft contacts (scalar view) |
| `soft_contact_particle` | `int` | Particle index |
| `soft_contact_shape` | `int` | Shape index |
| `soft_contact_body_pos` | `vec3` | Contact position on body [m] (supports `requires_grad`) |
| `soft_contact_body_vel` | `vec3` | Contact velocity on body [m/s] (supports `requires_grad`) |
| `soft_contact_normal` | `vec3` | Contact normal direction (supports `requires_grad`) |

### Configuration

```python
pipeline = newton.CollisionPipeline(
    model,
    soft_contact_margin=0.01,    # detection margin [m], default 0.01
    soft_contact_max=10000,      # buffer capacity; default = shape_count * particle_count
)

# Override margin per-call
pipeline.collide(state, contacts, soft_contact_margin=0.02)
```

### Differentiable Soft Contacts

Soft contact arrays (`body_pos`, `body_vel`, `normal`) support gradient computation when
`requires_grad=True`. Rigid contact arrays never require gradients because the narrow phase
kernels have `enable_backward=False`.

---

## 5. Contact Material Properties

**Source:** `newton/_src/sim/builder.py` -- class `ModelBuilder.ShapeConfig`

Material properties are set per-shape via `ShapeConfig` and combined during contact resolution.

### Property Reference

| Property | Default | Units | Solvers | Description |
|----------|---------|-------|---------|-------------|
| `ke` | 2500.0 | N/m | MuJoCo, Featherstone, SemiImplicit | Contact elastic stiffness |
| `kd` | 100.0 | N*s/m | MuJoCo, Featherstone, SemiImplicit | Contact damping coefficient |
| `kf` | 1000.0 | N*s/m | SemiImplicit, Featherstone | Friction damping coefficient |
| `mu` | 1.0 | -- | All | Coulomb friction coefficient |
| `mu_torsional` | 0.005 | -- | XPBD, MuJoCo | Torsional friction (spinning resistance) |
| `mu_rolling` | 0.0001 | -- | XPBD, MuJoCo | Rolling friction |
| `restitution` | 0.0 | -- | XPBD | Coefficient of restitution (0=inelastic, 1=elastic). Requires `enable_restitution=True` in solver. |
| `ka` | 0.0 | m | SemiImplicit, Featherstone | Adhesion distance |
| `kh` | 1e10 | Pa | All (hydroelastic) | Hydroelastic contact stiffness |
| `margin` | 0.0 | m | All | Outward offset from shape surface for collision |
| `gap` | None | m | All | Additional AABB expansion for pair filtering. If None, uses `builder.rigid_gap`. |

### Setting Material Properties

```python
cfg = newton.ModelBuilder.ShapeConfig(
    ke=5000.0,
    kd=200.0,
    mu=0.5,
    mu_torsional=0.01,
    mu_rolling=0.001,
    restitution=0.3,
    margin=0.001,
)

body = builder.add_body()
builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
```

### Physical Interpretation

- **ke (stiffness):** Higher values produce stiffer contacts with less penetration.
  Only used by spring-damper solvers (SemiImplicit, Featherstone, MuJoCo).
- **kd (damping):** Removes kinetic energy during contact. Prevents bouncing in
  spring-damper solvers.
- **mu (friction):** Limits tangential force to `mu * normal_force` (Coulomb model).
- **mu_torsional:** Resists spinning (drill-like rotation) at the contact point.
- **mu_rolling:** Resists rolling motion at the contact point.
- **restitution:** Controls bounce. Only XPBD supports it, and only when
  `enable_restitution=True` is passed to the solver constructor.
- **ka (adhesion):** Creates attractive force when the gap between shapes is less than `ka`.
- **kh (hydroelastic stiffness):** Controls the force-to-penetration ratio for
  hydroelastic contacts. For MuJoCo, values are internally scaled by masses.

---

## 6. Collision Filtering

### collision_group

Each shape has a `collision_group` integer (default: 1). Shapes with `collision_group=0`
are excluded from collision. Only shapes with compatible groups collide.

```python
# Normal colliding shape
cfg_solid = newton.ModelBuilder.ShapeConfig(collision_group=1)

# Non-colliding shape (e.g. sensor volume)
cfg_no_collide = newton.ModelBuilder.ShapeConfig(collision_group=0)

# Disable collision via has_shape_collision
cfg_visual = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, density=0)
```

### Collision Filter Pairs

Explicitly exclude specific shape pairs from collision detection:

```python
shape_a = builder.add_shape_sphere(body_a, radius=0.1)
shape_b = builder.add_shape_box(body_b, hx=0.1, hy=0.1, hz=0.1)

# Prevent collision between shape_a and shape_b
builder.add_shape_collision_filter_pair(shape_a, shape_b)
```

Pairs are stored in canonical order `(min, max)` and used by NXN/SAP broad phases to
skip excluded pairs. For the `"explicit"` broad phase, only precomputed pairs are tested,
so filter pairs act as an exclusion list for the other modes.

### collision_filter_parent

When `True` (default), shapes on the same body inherit collision filtering from their
parent body -- preventing self-collision between shapes attached to the same rigid body.

```python
cfg = newton.ModelBuilder.ShapeConfig(
    collision_filter_parent=True   # default; shapes on same body won't collide
)
```

### Sites (Non-Colliding Reference Points)

Sites are special shapes used as reference frames (e.g. for IMU sensors). They are created
with `collision_group=0` and `has_shape_collision=False` automatically:

```python
imu_site = builder.add_site(body, label="imu_0")
```

---

## 7. Hydroelastic Contacts

Hydroelastic contacts use Signed Distance Fields (SDFs) to compute distributed contact
patches instead of single-point contacts. They produce smoother, more stable contacts for
complex geometry interactions.

### Requirements

- Both shapes in a contact pair must have `is_hydroelastic=True`.
- The shapes must have SDF data configured (either via `configure_sdf()` or mesh `build_sdf()`).
- Does not work with planes, heightfields, flat meshes, or cloth.

### Configuration

```python
cfg = newton.ModelBuilder.ShapeConfig(
    is_hydroelastic=True,
    kh=1e10,          # hydroelastic stiffness [Pa]
    ke=1e7,           # elastic stiffness for solver
    kd=1e4,           # damping
    mu=0.01,          # friction
)

# Configure SDF resolution -- choose one method:
cfg.configure_sdf(
    max_resolution=256,         # grid resolution (must be divisible by 8)
    is_hydroelastic=True,
    kh=1e10,
)
# OR
cfg.configure_sdf(
    target_voxel_size=0.005,    # voxel size [m] (takes precedence)
    is_hydroelastic=True,
    kh=5e9,
)
```

### SDF Texture Formats

| Format | Memory | Precision |
|--------|--------|-----------|
| `"uint16"` (default) | 2 bytes/voxel | 16-bit normalized |
| `"float32"` | 4 bytes/voxel | Full precision |
| `"uint8"` | 1 byte/voxel | 8-bit normalized |

### Mesh SDF for Hydroelastic

For mesh shapes, build the SDF explicitly:

```python
mesh = newton.Mesh(vertices, indices)
mesh.build_sdf(
    max_resolution=256,
    narrow_band_range=(-0.005, 0.005),   # inner/outer band [m]
    margin=0.005,
)

body = builder.add_body()
builder.add_shape_mesh(body, mesh=mesh, cfg=cfg)
```

### When to Use Hydroelastic Contacts

- Complex mesh-mesh interactions (nuts, bolts, gears, interlocking parts).
- When point contacts cause instability or jitter.
- When combined with `reduce_contacts=True` for stable, reduced contact sets.

### Example

See `newton/examples/contacts/example_nut_bolt_hydro.py` for a hydroelastic nut-bolt assembly.

---

## 8. Contact Force Extraction

### Requesting the Force Attribute

The `"force"` extended attribute must be requested before creating `Contacts`:

```python
# Request via model (preferred -- propagates to all Contacts created later)
model.request_contact_attributes("force")

# Then create contacts -- force array is automatically allocated
contacts = model.contacts()
```

Or request manually when constructing `Contacts` directly:

```python
contacts = newton.Contacts(
    rigid_contact_max=10000,
    soft_contact_max=0,
    requested_attributes=model.get_requested_contact_attributes(),
)
```

### The Force Array

`contacts.force` has shape `(rigid_contact_max + soft_contact_max,)` with dtype
`wp.spatial_vector` (6 floats: 3 linear + 3 angular).

Layout:
- Indices `[0, rigid_contact_max)` -- rigid contact forces.
- Indices `[rigid_contact_max, rigid_contact_max + soft_contact_max)` -- soft contact forces.

### Reading Forces

Forces are populated by the solver. Call `solver.update_contacts()` before reading:

```python
solver.step(state, state_next, control, contacts, dt)
solver.update_contacts(contacts, state)

if contacts.force is not None:
    forces = contacts.force.numpy()
    count = int(contacts.rigid_contact_count.numpy()[0])

    for i in range(count):
        linear_force = forces[i][:3]    # [N] in world frame
        torque = forces[i][3:]          # [N*m] in world frame
```

### Force Semantics

Forces are expressed in the **world frame**, referenced to the center of mass (COM) of
shape0's body:

```
force_on_body0_from_body1  = contacts.force[i][:3]     # [N]
torque_on_body0_from_body1 = contacts.force[i][3:]     # [N*m]

# Newton's third law:
force_on_body1_from_body0  = -force_on_body0_from_body1
torque_on_body1_from_body0 = -torque_on_body0_from_body1
```

---

## 9. SensorIMU

**Source:** `newton/_src/sensors/sensor_imu.py`

Measures linear acceleration (specific force) and angular velocity at sensor sites.
Outputs are expressed in the sensor's local frame.

### Setup

```python
import warp as wp
import newton
from newton.sensors import SensorIMU

builder = newton.ModelBuilder()
builder.add_ground_plane()

body = builder.add_body(xform=wp.transform((0, 0, 1), wp.quat_identity()))
builder.add_shape_sphere(body, radius=0.1)

# Add a site (non-colliding reference frame for the IMU)
imu_site = builder.add_site(body, label="imu_0")

model = builder.finalize()

# Create sensor -- automatically requests body_qdd state attribute
imu = SensorIMU(model, sites="imu_*")

# Create state AFTER sensor so body_qdd is allocated
state = model.state()
```

### Site Selection

```python
# Glob pattern
imu = SensorIMU(model, sites="imu_*")

# Multiple patterns
imu = SensorIMU(model, sites=["imu_*", "gyro_*"])

# Explicit site indices
imu = SensorIMU(model, sites=[5, 10, 15])
```

### Reading Sensor Data

```python
solver = newton.solvers.SolverMuJoCo(model)

for step in range(num_steps):
    solver.step(state, state, None, None, dt=1.0 / 60.0)
    imu.update(state)

    acc  = imu.accelerometer.numpy()   # shape (n_sensors, 3) [m/s^2]
    gyro = imu.gyroscope.numpy()       # shape (n_sensors, 3) [rad/s]
```

### Requirements

- The solver must compute `body_qdd` (body accelerations). `SolverMuJoCo` supports this.
- `SensorIMU` automatically calls `model.request_state_attributes("body_qdd")` during
  construction (unless `request_state_attributes=False`).
- Create the sensor **before** calling `model.state()` so that `body_qdd` is allocated.

### Output Reference

| Attribute | dtype | Shape | Units | Frame |
|-----------|-------|-------|-------|-------|
| `accelerometer` | `vec3` | `(n_sensors,)` | m/s^2 | Sensor local |
| `gyroscope` | `vec3` | `(n_sensors,)` | rad/s | Sensor local |

The accelerometer measures specific force (includes gravity compensation): a sensor at
rest in a gravitational field reads `+g` along the gravity axis.

---

## 10. SensorContact

**Source:** `newton/_src/sensors/sensor_contact.py`

Monitors contact forces on a set of **sensing objects** (bodies or shapes), with optional
per-counterpart force breakdown.

### Basic Usage -- Total Force

```python
import warp as wp
import newton
from newton.sensors import SensorContact

builder = newton.ModelBuilder()
builder.add_ground_plane()

body = builder.add_body(xform=wp.transform((0, 0, 0.1), wp.quat_identity()))
builder.add_shape_sphere(body, radius=0.1, label="ball")

model = builder.finalize()

# Monitor total contact force on the ball
sensor = SensorContact(model, sensing_obj_shapes="ball")

solver = newton.solvers.SolverMuJoCo(model)
state = model.state()
contacts = model.contacts()

solver.step(state, state, None, None, dt=1.0 / 60.0)
solver.update_contacts(contacts)
sensor.update(state, contacts)

force = sensor.total_force.numpy()   # shape (1, 3) [N]
```

### Per-Counterpart Forces (force_matrix)

```python
sensor = SensorContact(
    model,
    sensing_obj_shapes=["*Plate1", "*Plate2"],
    counterpart_shapes=["*Cube*", "*Sphere*"],
    measure_total=False,             # skip total_force (optional)
)

# After solver step and update_contacts:
sensor.update(state, contacts)

# force_matrix shape: (n_sensing_objs, max_counterparts) [N]
fm = sensor.force_matrix.numpy()
force_plate1_from_cube   = fm[0, 0]
force_plate1_from_sphere = fm[0, 1]
```

### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sensing_obj_bodies` | `str \| list[str] \| list[int]` | Body indices or label patterns for sensing objects |
| `sensing_obj_shapes` | `str \| list[str] \| list[int]` | Shape indices or label patterns for sensing objects |
| `counterpart_bodies` | `str \| list[str] \| list[int]` | Body indices or label patterns for counterparts |
| `counterpart_shapes` | `str \| list[str] \| list[int]` | Shape indices or label patterns for counterparts |
| `measure_total` | `bool` | Allocate `total_force` (default `True`) |
| `verbose` | `bool \| None` | Print debug info (default uses `wp.config.verbose`) |

Exactly one of `sensing_obj_bodies` / `sensing_obj_shapes` must be specified.
At most one of `counterpart_bodies` / `counterpart_shapes` may be specified.

### Output Attributes

| Attribute | dtype | Shape | Description |
|-----------|-------|-------|-------------|
| `total_force` | `vec3` | `(n_sensing_objs,)` | Total contact force [N] per sensing object. `None` if `measure_total=False`. |
| `force_matrix` | `vec3` | `(n_sensing_objs, max_counterparts)` | Per-counterpart forces [N]. `None` if no counterparts specified. |
| `sensing_obj_transforms` | `transform` | `(n_sensing_objs,)` | World-frame transforms of sensing objects |
| `sensing_obj_idx` | `list[int]` | -- | Body/shape indices matching output rows |
| `sensing_obj_type` | `"body" \| "shape"` | -- | Whether indices are bodies or shapes |
| `counterpart_type` | `"body" \| "shape" \| None` | -- | Type of counterpart indices |
| `counterpart_indices` | `list[list[int]]` | -- | Per-sensing-object counterpart indices |

### Update Order

```
solver.step(...)
solver.update_contacts(contacts, state)   # populates contacts.force
sensor.update(state, contacts)            # reads contacts.force
```

---

## 11. Other Sensors

Newton provides five sensor types in `newton.sensors`:

| Sensor | Description |
|--------|-------------|
| `SensorIMU` | Accelerometer + gyroscope at site frames |
| `SensorContact` | Contact force monitoring on bodies/shapes |
| `SensorFrameTransform` | Tracks world-frame pose of bodies/shapes |
| `SensorRaycast` | GPU-accelerated raycasting |
| `SensorTiledCamera` | Tiled camera rendering (RGB, depth, segmentation) |

All sensors follow the same pattern: construct with a model reference, then call
`sensor.update(state, ...)` each step.

---

## 12. Complete Working Examples

### Example A: Falling Sphere with Contact Force Monitoring

```python
import warp as wp
import newton
from newton.sensors import SensorContact

builder = newton.ModelBuilder()
builder.add_ground_plane()

body = builder.add_body(xform=wp.transform((0, 0, 1.0), wp.quat_identity()))
builder.add_shape_sphere(body, radius=0.1, label="ball",
    cfg=newton.ModelBuilder.ShapeConfig(ke=5000, kd=200, mu=0.5, density=1000))

model = builder.finalize()

# Setup sensor (auto-requests "force" attribute)
sensor = SensorContact(model, sensing_obj_shapes="ball")

solver = newton.solvers.SolverMuJoCo(model)
state = model.state()
contacts = model.contacts()

dt = 1.0 / 60.0
for step in range(300):
    model.collide(state, contacts)
    solver.step(state, state, None, contacts, dt)
    solver.update_contacts(contacts)
    sensor.update(state, contacts)

    total = sensor.total_force.numpy()[0]
    if step % 60 == 0:
        print(f"Step {step}: force = {total}")
```

### Example B: IMU on Falling Cubes

```python
import warp as wp
import newton
from newton.sensors import SensorIMU

builder = newton.ModelBuilder()
builder.add_ground_plane()

for i in range(3):
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0, 0.5 * i, 1.0), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=200))
    builder.add_site(body, label=f"imu_{i}")

model = builder.finalize()
imu = SensorIMU(model, sites="imu_*")

solver = newton.solvers.SolverMuJoCo(model)
state = model.state()

dt = 1.0 / 200.0
for step in range(1000):
    solver.step(state, state, None, None, dt)
    imu.update(state)

    if step % 200 == 0:
        acc = imu.accelerometer.numpy()
        gyro = imu.gyroscope.numpy()
        for i in range(3):
            print(f"  IMU {i}: acc={acc[i]}, gyro={gyro[i]}")
```

### Example C: Per-Counterpart Contact Sensor

```python
import warp as wp
import newton
from newton.sensors import SensorContact

builder = newton.ModelBuilder()
builder.add_ground_plane()

# Two objects falling onto a plate
plate_body = builder.add_body(
    xform=wp.transform((0, 0, 0.05), wp.quat_identity()))
builder.add_shape_box(plate_body, hx=0.5, hy=0.5, hz=0.05, label="plate",
    cfg=newton.ModelBuilder.ShapeConfig(density=0))  # static

ball_body = builder.add_body(
    xform=wp.transform((0.1, 0, 0.5), wp.quat_identity()))
builder.add_shape_sphere(ball_body, radius=0.05, label="ball")

cube_body = builder.add_body(
    xform=wp.transform((-0.1, 0, 0.5), wp.quat_identity()))
builder.add_shape_box(cube_body, hx=0.05, hy=0.05, hz=0.05, label="cube")

model = builder.finalize()

sensor = SensorContact(
    model,
    sensing_obj_shapes="plate",
    counterpart_shapes=["ball", "cube"],
    measure_total=True,
)

solver = newton.solvers.SolverMuJoCo(model)
state = model.state()
contacts = model.contacts()

dt = 1.0 / 60.0
for step in range(120):
    model.collide(state, contacts)
    solver.step(state, state, None, contacts, dt)
    solver.update_contacts(contacts)
    sensor.update(state, contacts)

    total = sensor.total_force.numpy()[0]
    fm = sensor.force_matrix.numpy()

    if step % 30 == 0:
        print(f"Step {step}:")
        print(f"  Total force on plate: {total}")
        print(f"  Force from ball: {fm[0, 0]}")
        print(f"  Force from cube: {fm[0, 1]}")
```

---

## 13. Common Pitfalls

### Ordering: Create Sensors Before State/Contacts

Sensors register extended attributes on the model during construction. If you create
`State` or `Contacts` before the sensor, those attributes will not be allocated:

```python
# WRONG -- body_qdd not allocated in state
state = model.state()
imu = SensorIMU(model, sites="imu_*")   # requests body_qdd, but state already exists
imu.update(state)                        # raises ValueError: body_qdd is None

# CORRECT
imu = SensorIMU(model, sites="imu_*")   # requests body_qdd
state = model.state()                    # now includes body_qdd
```

Similarly for `SensorContact`:

```python
# CORRECT ORDER
sensor = SensorContact(model, sensing_obj_shapes="ball")  # requests "force"
contacts = model.contacts()                                # includes force array
```

### Call solver.update_contacts() Before sensor.update()

`SensorContact.update()` reads from `contacts.force`. The solver must populate it first:

```python
solver.step(state, state_next, control, contacts, dt)
solver.update_contacts(contacts)        # populates contacts.force
sensor.update(state, contacts)          # reads contacts.force
```

### contacts.force is None

If `contacts.force` is `None`, the `"force"` attribute was not requested before the
`Contacts` object was created. Either:
- Create `SensorContact` before `model.contacts()`, or
- Call `model.request_contact_attributes("force")` before creating contacts, or
- Pass `requested_attributes={"force"}` to the `Contacts` constructor.

### rigid_contact_max Overflow

If the auto-estimated `rigid_contact_max` is too small for dense scenes, contacts will
be silently dropped. Override explicitly:

```python
pipeline = newton.CollisionPipeline(model, rigid_contact_max=100000)
```

### Broad Phase "explicit" Requires shape_contact_pairs

The default `model.collide()` uses `broad_phase="explicit"` internally. This requires
`model.shape_contact_pairs` to be set (populated automatically by `finalize()` from
collision filter pairs and shape geometry). If you get errors about missing pairs, switch
to `"sap"` or `"nxn"`:

```python
pipeline = newton.CollisionPipeline(model, broad_phase="sap")
contacts = model.contacts(collision_pipeline=pipeline)
```

### Hydroelastic: Both Shapes Must Opt In

Hydroelastic contacts only activate when **both** shapes in a contact pair have
`is_hydroelastic=True`. If only one shape opts in, standard point contacts are used.

### Hydroelastic: Primitives Need configure_sdf()

For primitive shapes (sphere, box, capsule, etc.) to use hydroelastic contacts, you must
call `configure_sdf()` to generate the SDF volume:

```python
cfg = newton.ModelBuilder.ShapeConfig()
cfg.configure_sdf(max_resolution=64, is_hydroelastic=True, kh=1e10)
builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
```

### SDF Resolution Must Be Divisible by 8

SDF volumes are allocated in 8x8x8 tiles. `sdf_max_resolution` must be a multiple of 8
or `ValueError` is raised.

### IMU Requires a Solver That Computes body_qdd

Not all solvers compute body accelerations. `SolverMuJoCo` does. If using another solver,
check that `state.body_qdd` is populated after `solver.step()`.

### Global Bodies Cannot Be Sensing Objects

`SensorContact` raises `ValueError` if sensing objects are global (world=-1). Only
per-world entities can be sensing objects. Global entities (e.g. ground plane) can be
counterparts.

### Contacts.clear() Behavior

By default, `clear()` only resets counts (1 kernel launch). This is safe because solvers
only read up to `contact_count`. Use `clear_buffers=True` for debugging if you suspect
stale data issues:

```python
contacts = newton.Contacts(10000, 0, clear_buffers=True)  # zeros all arrays on clear
```
