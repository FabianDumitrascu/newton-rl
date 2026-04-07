# Newton ModelBuilder API Reference

Complete reference for `newton.ModelBuilder` -- the declarative API for constructing rigid-body
simulation scenes in the Newton physics engine.

---

## Overview

`ModelBuilder` uses standard Python lists internally. Call `finalize()` to transfer
everything to GPU memory and obtain a simulation-ready `Model`.

```python
import newton

builder = newton.ModelBuilder()
# ... add bodies, shapes, joints ...
model = builder.finalize(device="cuda")
```

---

## add_body()

Creates a standalone free-floating rigid body (single-body articulation with a free joint).

```python
def add_body(
    self,
    xform: Transform | None = None,
    armature: float | None = None,
    com: Vec3 | None = None,
    inertia: Mat33 | None = None,
    mass: float = 0.0,
    label: str | None = None,
    lock_inertia: bool = False,
    is_kinematic: bool = False,
    custom_attributes: dict[str, Any] | None = None,
) -> int:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `xform` | `Transform \| None` | `None` | Location of the body in world frame. Identity if `None`. |
| `armature` | `float \| None` | `None` | Artificial inertia added to the body. Uses `default_body_armature` if `None`. |
| `com` | `Vec3 \| None` | `None` | Center of mass relative to body origin. Origin if `None`. |
| `inertia` | `Mat33 \| None` | `None` | 3x3 inertia tensor relative to COM. Zero if `None`. |
| `mass` | `float` | `0.0` | Mass of the body [kg]. |
| `label` | `str \| None` | `None` | Label for the body. Auto-labels the free joint as `{label}_free_joint` and articulation as `{label}_articulation`. |
| `lock_inertia` | `bool` | `False` | If `True`, subsequent shape additions will not modify this body's mass, COM, or inertia. Does not affect `collapse_fixed_joints`. |
| `is_kinematic` | `bool` | `False` | If `True`, body does not respond to forces (see [Body Flags](#body-flags-dynamic-vs-kinematic)). |
| `custom_attributes` | `dict[str, Any] \| None` | `None` | Dictionary of custom attribute names to values. |

**Returns:** `int` -- the body index in the model.

**Internally calls:**
1. `add_link()` -- creates the body
2. `add_joint_free()` -- adds a 6-DOF free joint connecting to world
3. `add_articulation()` -- creates a new articulation

For multi-body articulations (robots, mechanisms), use `add_link()`, joint methods
(`add_joint_revolute`, `add_joint_prismatic`, etc.), and `add_articulation()` directly.

---

## Body Flags: DYNAMIC vs KINEMATIC

| Flag | `is_kinematic` | Behavior |
|---|---|---|
| **DYNAMIC** | `False` | Body responds to gravity, contact forces, joint forces. The solver integrates its motion. This is the default. |
| **KINEMATIC** | `True` | Body follows a prescribed trajectory. It is not affected by forces but still participates in collision (other bodies bounce off it). Mass and inertia are ignored by the solver. |

**When to use each:**

- **DYNAMIC**: Any body whose motion should be determined by physics (falling objects, robot links, projectiles).
- **KINEMATIC**: Moving platforms, animated obstacles, end-effector targets, or any body whose trajectory you control directly (e.g., via `state.body_q`).

---

## add_ground_plane()

Convenience method that adds an infinite static ground plane.

```python
def add_ground_plane(
    self,
    height: float = 0.0,
    cfg: ShapeConfig | None = None,
    color: Vec3 | None = (0.125, 0.125, 0.15),
    label: str | None = None,
) -> int:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `height` | `float` | `0.0` | Vertical offset along the up-vector axis [m]. Positive raises, negative lowers. |
| `cfg` | `ShapeConfig \| None` | `None` | Shape configuration. Uses `default_shape_cfg` if `None`. |
| `color` | `Vec3 \| None` | `(0.125, 0.125, 0.15)` | Display RGB color in [0, 1]. Pass `None` for palette color. |
| `label` | `str \| None` | `None` | Defaults to `"ground_plane"`. |

**Returns:** `int` -- the shape index.

Internally calls `add_shape_plane()` with `width=0.0, length=0.0` (infinite collision extent)
and the plane equation derived from `builder.up_vector`.

---

## Shape Types

All shape methods attach geometry to a body (or world with `body=-1`). They share common
parameters: `body`, `xform`, `cfg`, `color`, `label`, `custom_attributes`. Many also
support `as_site=True` to create non-colliding reference points.

All shape methods share common parameters: `body` (parent body index, `-1` for world),
`xform` (transform in parent frame), `cfg` (ShapeConfig), `color` (RGB in [0,1]),
`label`, `custom_attributes`. Primitive shapes also support `as_site=True` for non-colliding
reference points. All return `int` (the shape index).

### add_shape_sphere()

```python
add_shape_sphere(body, xform=None, radius=1.0, cfg=None, as_site=False, ...)
```

- `radius`: Sphere radius [m] (default `1.0`).

### add_shape_box()

```python
add_shape_box(body, xform=None, hx=0.5, hy=0.5, hz=0.5, cfg=None, as_site=False, ...)
```

- `hx`, `hy`, `hz`: Half-extents along local X, Y, Z [m]. Full width = `2 * hx`.

### add_shape_capsule()

```python
add_shape_capsule(body, xform=None, radius=1.0, half_height=0.5, cfg=None, as_site=False, ...)
```

- `radius`: Radius of hemispherical caps and cylindrical segment [m].
- `half_height`: Half-length of cylindrical segment (excluding caps) [m].
- Extends along the **Z-axis**. Total length = `2 * half_height + 2 * radius`.

### add_shape_cylinder()

```python
add_shape_cylinder(body, xform=None, radius=1.0, half_height=0.5, cfg=None, as_site=False, ...)
```

- `radius` [m], `half_height` [m] along the Z-axis.

### add_shape_cone()

```python
add_shape_cone(body, xform=None, radius=1.0, half_height=0.5, cfg=None, as_site=False, ...)
```

- `radius`: Base radius [m]. `half_height`: Half the total height [m].
- Base at `-half_height`, apex at `+half_height` along Z.
- Center of mass is at `-half_height/2` from origin (1/4 height from base).

### add_shape_ellipsoid()

```python
add_shape_ellipsoid(body, xform=None, a=1.0, b=0.75, c=0.5, cfg=None, as_site=False, ...)
```

- `a`, `b`, `c`: Semi-axes along local X, Y, Z [m]. A sphere when `a == b == c`.
- Collision uses GJK/MPR pipeline.

### add_shape_mesh()

```python
add_shape_mesh(body, xform=None, mesh=None, scale=None, cfg=None, ...)
```

- `mesh`: A `newton.Mesh` object with vertex and triangle data.
- `scale`: 3D scale `(sx, sy, sz)`, defaults to `(1, 1, 1)`.

### add_shape_convex_hull()

Same signature as `add_shape_mesh()`. Uses the convex hull of the mesh vertices for collision.

### add_shape_heightfield()

```python
add_shape_heightfield(xform=None, heightfield=None, scale=None, cfg=None, ...)
```

- `heightfield`: A `newton.Heightfield` 2D elevation grid. **Required**.
- Always static (body=-1). More memory-efficient than triangle meshes for terrain.

### add_shape_plane()

```python
add_shape_plane(plane=(0,0,1,0), xform=None, width=10.0, length=10.0, body=-1, cfg=None, ...)
```

- `plane`: Equation `(a, b, c, d)` where `ax + by + cz + d = 0`. Used when `xform` is `None`.
- `width`, `length`: Extents along local X, Y [m]. `0.0` = infinite.
- Always static (massless). Shorthand: `(0, 0, 1, -h)` defines the plane `z = h`.

---

## ShapeConfig

`ModelBuilder.ShapeConfig` is a dataclass controlling physical and collision properties of shapes.

### Field Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `density` | `float` | `1000.0` | Material density [kg/m^3]. Used to compute mass/inertia from shape volume. |
| `ke` | `float` | `2500.0` | Contact elastic stiffness [N/m]. Used by SemiImplicit, Featherstone, MuJoCo solvers. |
| `kd` | `float` | `100.0` | Contact damping coefficient [N*s/m]. Used by SemiImplicit, Featherstone, MuJoCo. |
| `kf` | `float` | `1000.0` | Friction damping coefficient [N*s/m]. Used by SemiImplicit, Featherstone. |
| `ka` | `float` | `0.0` | Contact adhesion distance [m]. Used by SemiImplicit, Featherstone. |
| `mu` | `float` | `1.0` | Coefficient of friction (Coulomb). Used by all solvers. |
| `restitution` | `float` | `0.0` | Coefficient of restitution (bounciness). Used by XPBD. Requires `enable_restitution=True` in solver. |
| `mu_torsional` | `float` | `0.005` | Torsional friction coefficient (resistance to spinning at contact). Used by XPBD, MuJoCo. |
| `mu_rolling` | `float` | `0.0001` | Rolling friction coefficient (resistance to rolling). Used by XPBD, MuJoCo. |
| `margin` | `float` | `0.0` | Outward surface offset [m] for collision. Both margins are summed for a pair. Also used for hollow shape inertia. |
| `gap` | `float \| None` | `None` | Additional contact detection gap [m]. If `None`, uses `builder.rigid_gap`. Broadphase uses `margin + gap` for AABB expansion. |
| `is_solid` | `bool` | `True` | Whether shape is solid or hollow (affects inertia computation). |
| `collision_group` | `int` | `1` | Collision group ID. Set to `0` to disable collisions entirely for this shape. |
| `collision_filter_parent` | `bool` | `True` | Whether to inherit collision filtering from parent body. |
| `has_shape_collision` | `bool` | `True` | Whether this shape collides with other shapes. |
| `has_particle_collision` | `bool` | `True` | Whether this shape collides with particles. |
| `is_visible` | `bool` | `True` | Whether the shape is rendered in the viewer. |
| `is_site` | `bool` | `False` | Whether this is a site (non-colliding reference point). Use `mark_as_site()` to set properly. |
| `is_hydroelastic` | `bool` | `False` | Use SDF-based hydroelastic contacts. Both shapes in a pair must have this enabled. Not supported for planes/heightfields. |
| `kh` | `float` | `1e10` | Hydroelastic contact stiffness [N/m^3]. Effective spring = `area * kh`. Used by MuJoCo, Featherstone, SemiImplicit. |
| `sdf_narrow_band_range` | `tuple` | `(-0.1, 0.1)` | Inner/outer distance range for SDF computation. |
| `sdf_target_voxel_size` | `float \| None` | `None` | Target voxel size for SDF grid. Enables SDF if set. Requires GPU. |
| `sdf_max_resolution` | `int \| None` | `None` | Max SDF grid dimension (must be divisible by 8). Enables SDF if set. Requires GPU. |
| `sdf_texture_format` | `str` | `"uint16"` | SDF subgrid texture format: `"uint16"`, `"float32"`, or `"uint8"`. |

### Key Methods

- **`mark_as_site()`** -- Sets `is_site=True`, `has_shape_collision=False`, `has_particle_collision=False`, `density=0.0`, `collision_group=0`.
- **`configure_sdf(max_resolution=None, target_voxel_size=None, is_hydroelastic=False, kh=1e10, texture_format=None)`** -- Convenience to enable SDF-based collision and hydroelastics in one call.
- **`copy()`** -- Returns a shallow copy of the config.
- **`validate(shape_type=None)`** -- Validates parameters (e.g., SDF resolution divisibility).

---

## finalize()

Transfers all builder data to device memory and returns a simulation-ready `Model`.

```python
def finalize(
    self,
    device: Devicelike | None = None,
    *,
    requires_grad: bool = False,
    skip_all_validations: bool = False,
    skip_validation_worlds: bool = False,
    skip_validation_joints: bool = False,
    skip_validation_shapes: bool = False,
    skip_validation_structure: bool = False,
    skip_validation_joint_ordering: bool = True,
) -> Model:
```

- `device`: Target device (`"cpu"`, `"cuda"`, `"cuda:0"`). Uses current Warp device if `None`.
- `requires_grad`: Enable gradient computation for differentiable simulation (default `False`).
- `skip_all_validations`: Skip all checks for max performance (default `False`).
- `skip_validation_worlds` / `skip_validation_joints` / `skip_validation_shapes` / `skip_validation_structure`: Selective validation skips (all default `False`).
- `skip_validation_joint_ordering`: Skip DFS topological ordering check (default `True`, opt-in).

**Returns:** `Model` -- fully constructed simulation model on the specified device.

**What finalize() does:**
1. Validates model structure (unless skipped)
2. Builds world-start index arrays
3. Computes particle inverse masses
4. Transfers all arrays to device memory (bodies, shapes, joints, particles, springs, etc.)
5. Finalizes geometry objects (meshes, heightfields, SDFs)
6. Sets up collision data structures (contact pairs, hash grids, AABBs)
7. Corrects rigid body inertia and mass properties
8. Creates and returns the `Model` object

---

## Model Properties

After `finalize()`, the `Model` object exposes these count properties:

| Property | Type | Description |
|---|---|---|
| `body_count` | `int` | Total number of rigid bodies. |
| `shape_count` | `int` | Total number of collision shapes. |
| `joint_count` | `int` | Total number of joints. |
| `joint_dof_count` | `int` | Total velocity degrees of freedom (number of joint axes). |
| `joint_coord_count` | `int` | Total position degrees of freedom. |
| `articulation_count` | `int` | Total number of articulations. |
| `particle_count` | `int` | Total number of particles. |
| `tri_count` | `int` | Total number of triangle elements. |
| `tet_count` | `int` | Total number of tetrahedral elements. |
| `spring_count` | `int` | Total number of springs. |
| `muscle_count` | `int` | Total number of muscles. |
| `world_count` | `int` | Number of simulation worlds. |
| `requires_grad` | `bool` | Whether gradient computation is enabled. |

---

## Model Arrays

### Body Arrays (`[body_count]`)

`body_q` (transform), `body_qd` (spatial_vector), `body_com` (vec3), `body_mass` (float32),
`body_inv_mass` (float32), `body_inertia` (mat33), `body_inv_inertia` (mat33),
`body_flags` (int32), `body_label` (list[str]), `body_world` (int32).

### Shape Arrays (`[shape_count]`)

**Material:** `shape_material_ke`, `shape_material_kd`, `shape_material_kf`, `shape_material_ka`,
`shape_material_mu`, `shape_material_restitution`, `shape_material_mu_torsional`,
`shape_material_mu_rolling`, `shape_material_kh`, `shape_gap` -- all float32, correspond to
ShapeConfig fields.

**Geometry:** `shape_transform` (transform), `shape_body` (int32), `shape_type` (int32),
`shape_scale` (vec3), `shape_margin` (float32), `shape_is_solid` (bool), `shape_flags` (int32),
`shape_collision_group` (int32), `shape_color` (vec3).

### Joint Arrays

**Per-joint** (`[joint_count]`): `joint_type`, `joint_parent`, `joint_child`, `joint_X_p`,
`joint_X_c`, `joint_articulation`, `joint_enabled`, `joint_q_start`, `joint_qd_start`.

**Per-DOF** (`[joint_dof_count]`): `joint_axis` (vec3), `joint_armature`, `joint_target_ke`,
`joint_target_kd`, `joint_limit_lower`, `joint_limit_upper`, `joint_effort_limit`,
`joint_velocity_limit`, `joint_friction`.

**Per-coord** (`[joint_coord_count]`): `joint_q` (initial positions).
**Per-DOF** also: `joint_qd` (initial velocities).

---

## Extended Attributes

### request_state_attributes()

```python
builder.request_state_attributes("body_qdd", "body_parent_f")
```

Call **before** `finalize()`. When a `State` object is created from the finalized model,
these additional arrays will be allocated:

- `"body_qdd"` -- body accelerations
- `"body_parent_f"` -- forces transmitted to parent body

### request_contact_attributes()

```python
builder.request_contact_attributes("force")
```

Call **before** `finalize()`. When a `Contacts` object is created from the finalized model,
the requested arrays will be allocated:

- `"force"` -- per-contact force vectors

---

## add_urdf()

Parses a URDF file and adds bodies, joints, and shapes to the builder.

```python
def add_urdf(
    self, source: str, *, xform=None, floating=None, base_joint=None,
    parent_body=-1, scale=1.0, hide_visuals=False,
    parse_visuals_as_colliders=False, up_axis=Axis.Z,
    force_show_colliders=False, enable_self_collisions=True,
    ignore_inertial_definitions=False, joint_ordering="dfs",
    bodies_follow_joint_ordering=True, collapse_fixed_joints=False,
    mesh_maxhullvert=None, force_position_velocity_actuation=False,
    override_root_xform=False,
):
```

**Key parameters:**

- `source` (required): URDF filename or XML string content.
- `xform`: Transform for the root body. Identity if `None`.
- `floating`: `None` = FIXED joint (URDF default), `True` = FREE joint (6 DOF), `False` = FIXED. Cannot combine with `base_joint`.
- `base_joint`: Custom joint dict for root-to-world connection. Cannot combine with `floating`.
- `parent_body`: Attach to existing body (`-1` = world frame).
- `scale`: Scaling factor for the mechanism (default `1.0`).
- `up_axis`: Up axis of the URDF (default `Axis.Z`), affects capsule/cylinder orientation.
- `joint_ordering`: `"bfs"`, `"dfs"` (default), or `None` (original URDF order).
- `collapse_fixed_joints`: Merge bodies connected by fixed joints.
- `ignore_inertial_definitions`: Recompute inertia from shape geometry.
- `enable_self_collisions`: Enable self-collisions (default `True`).
- `parse_visuals_as_colliders`: Use `<visual>` geometry for collision.
- `override_root_xform`: Replace (not compose) root transform with `xform`.

**Key differences from add_usd():** URDF takes a filename or XML string (USD takes filename or `UsdStage`). URDF defaults `floating` to FIXED; URDF has `joint_ordering` and `force_position_velocity_actuation` parameters not present in USD.

---

## add_mjcf()

Parses MuJoCo XML (MJCF) and adds bodies, joints, shapes. Automatically registers
MuJoCo-specific custom attributes on the builder.

```python
def add_mjcf(
    self, source: str, *, xform=None, floating=None, base_joint=None,
    parent_body=-1, armature_scale=1.0, scale=1.0, hide_visuals=False,
    parse_visuals_as_colliders=False, parse_meshes=True, parse_sites=True,
    parse_visuals=True, parse_mujoco_options=True, up_axis=Axis.Z,
    ignore_names=(), ignore_classes=(), visual_classes=("visual",),
    collider_classes=("collision",), no_class_as_colliders=True,
    force_show_colliders=False, enable_self_collisions=True,
    ignore_inertial_definitions=False, collapse_fixed_joints=False,
    verbose=False, skip_equality_constraints=False,
    convert_3d_hinge_to_ball_joints=False, mesh_maxhullvert=None,
    ctrl_direct=False, path_resolver=None, override_root_xform=False,
):
```

**Key parameters (shared with URDF):** `source`, `xform`, `floating`, `base_joint`, `parent_body`, `scale`, `enable_self_collisions`, `collapse_fixed_joints`, `ignore_inertial_definitions`, `override_root_xform`.

**MuJoCo-specific parameters:**

- `floating`: `None` = honors `<freejoint>` tags (unlike URDF which defaults FIXED).
- `armature_scale`: Scaling factor for MJCF-defined joint armature values (default `1.0`).
- `parse_meshes` / `parse_sites` / `parse_visuals`: Toggle loading of mesh geometries, sites, and visual shapes.
- `parse_mujoco_options`: Parse `<option>` tag solver options (default `True`).
- `ignore_names` / `ignore_classes`: Regex patterns for skipping bodies/joints.
- `visual_classes` / `collider_classes`: Regex patterns classifying visual vs collision geometry (defaults: `("visual",)`, `("collision",)`).
- `no_class_as_colliders`: Treat class-less geometries as collision shapes (default `True`).
- `skip_equality_constraints`: Ignore `<equality>` tags.
- `convert_3d_hinge_to_ball_joints`: Convert series of 3 hinge joints to a ball joint.
- `ctrl_direct`: If `True`, actuators use `CTRL_DIRECT` mode (MuJoCo-native). If `False` (default), position/velocity actuators use `JOINT_TARGET` mode.
- `path_resolver`: `Callable[[str | None, str], str]` to resolve file paths for `<include>` and assets.
- `verbose`: Print parsing diagnostics.

---

## Complete Working Examples

### Example 1: Falling Sphere

```python
import newton
from newton.solvers import SolverXPBD

builder = newton.ModelBuilder()

# Ground plane
builder.add_ground_plane()

# Dynamic sphere at height 2
body = builder.add_body(
    xform=newton.transform((0.0, 0.0, 2.0), newton.quat_identity()),
    label="sphere",
)
builder.add_shape_sphere(body=body, radius=0.1)

# Finalize and simulate
model = builder.finalize(device="cuda")
state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.contacts()
solver = SolverXPBD(model)

for _ in range(240):
    state_0.clear_forces()
    model.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt=1.0 / 60.0)
    state_0, state_1 = state_1, state_0
```

### Example 2: Stacked Boxes with Custom Material

```python
import newton
import warp as wp
from newton.solvers import SolverXPBD

builder = newton.ModelBuilder()
builder.add_ground_plane()

# Custom material: rubber-like
rubber = newton.ModelBuilder.ShapeConfig(
    density=1200.0,
    mu=0.8,
    restitution=0.5,
    ke=5000.0,
    kd=200.0,
)

# Stack 5 boxes
for i in range(5):
    body = builder.add_body(
        xform=wp.transform((0.0, 0.0, 0.15 + i * 0.3), wp.quat_identity()),
        label=f"box_{i}",
    )
    builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1, cfg=rubber)

model = builder.finalize(device="cuda")
```

### Example 3: Loading a URDF Robot

```python
import newton
import warp as wp

builder = newton.ModelBuilder()
builder.add_ground_plane()

builder.add_urdf(
    "path/to/robot.urdf",
    xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()),
    floating=True,           # free-floating base
    enable_self_collisions=False,
    collapse_fixed_joints=True,
)

model = builder.finalize(device="cuda")
```

### Example 4: Compound Body with Multiple Shapes

```python
import newton
import warp as wp

builder = newton.ModelBuilder()
builder.add_ground_plane()

body = builder.add_body(
    xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()),
    label="compound",
)
builder.add_shape_capsule(body=body, radius=0.1, half_height=0.3)
builder.add_shape_box(
    body=body, xform=wp.transform((-0.4, 0.0, 0.0), wp.quat_identity()),
    hx=0.2, hy=0.05, hz=0.1,
)
builder.add_shape_sphere(  # non-colliding sensor site
    body=body, xform=wp.transform((0.0, 0.0, 0.4), wp.quat_identity()),
    radius=0.02, as_site=True,
)

model = builder.finalize(device="cuda")
```

---

## Common Pitfalls

### 1. Zero mass with no shapes

If you call `add_body(mass=0.0)` and attach no shapes (or shapes with `density=0.0`),
the body has zero mass and zero inertia. The solver may produce NaN or skip the body.
Always attach at least one shape with nonzero density, or set mass/inertia explicitly.

### 2. Forgetting to call finalize()

The builder is not a simulation model. You must call `finalize()` before creating states,
contacts, or stepping the solver. All `model.state()`, `model.control()`, `model.contacts()`
calls require a finalized model.

### 3. Shape scale vs. dimensions

Shape dimensions are passed as named parameters (`radius`, `hx`, `half_height`, etc.),
**not** via the `scale` parameter. The `scale` parameter on `add_shape_mesh` and
`add_shape_heightfield` is a 3D scaling factor applied to the geometry source. For
primitive shapes, the named parameters are internally packed into a scale vector.

### 4. Capsule/cylinder/cone orientation

Capsules, cylinders, and cones extend along the **local Z-axis**. To orient them
differently, apply a rotation via the `xform` parameter.

### 5. Kinematic bodies still need shapes for collision

Setting `is_kinematic=True` makes a body ignore forces, but it still needs shapes
to participate in collision. Other dynamic bodies will collide with it.

### 6. ShapeConfig is shared by reference

`builder.default_shape_cfg` is shared across all shapes that don't provide an explicit
`cfg`. If you modify it, all subsequently added shapes inherit the change. Use
`cfg.copy()` to create independent configs.

### 7. request_*_attributes must be called before finalize

`request_state_attributes()` and `request_contact_attributes()` must be called on the
**builder** before `finalize()`. Calling them afterward has no effect -- the model is
already constructed.

### 8. URDF floating default is FIXED, MJCF honors freejoint

URDF: `floating=None` creates a FIXED joint (robot base is welded to world). Set
`floating=True` for a free-floating robot. MJCF: `floating=None` respects `<freejoint>`
tags in the XML. If no `<freejoint>` is present, it also defaults to FIXED.

### 9. Hydroelastic requires SDF configuration

If `is_hydroelastic=True` on a primitive shape (sphere, box, capsule, cylinder, ellipsoid,
cone), you must also set `sdf_max_resolution` or `sdf_target_voxel_size`. Use
`cfg.configure_sdf(max_resolution=64, is_hydroelastic=True)` for convenience. Both
shapes in a colliding pair must have hydroelastics enabled.
