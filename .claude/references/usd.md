# Newton USD Reference

Newton provides first-class support for loading articulated rigid-body models from Universal
Scene Description (USD) files. The primary entry point is `ModelBuilder.add_usd()` (internally
`parse_usd()`), which reads a USD stage containing UsdPhysics schema definitions and populates
a `ModelBuilder` with bodies, shapes, joints, and actuators.

**Key source files** (all under `submodules/newton/`):

| File | Purpose |
|------|---------|
| `newton/_src/utils/import_usd.py` | `parse_usd()` -- the engine behind `add_usd()` |
| `newton/_src/usd/utils.py` | Low-level USD helpers (`get_mesh`, `get_transform`, ...) |
| `newton/_src/usd/schema_resolver.py` | `SchemaResolver` base class + `SchemaResolverManager` |
| `newton/_src/usd/schemas.py` | Concrete resolvers: Newton, PhysX, MuJoCo |

---

## Table of Contents

1. [add_usd() Full Signature](#1-add_usd-full-signature)
2. [Return Dictionary](#2-return-dictionary)
3. [Schema Resolvers](#3-schema-resolvers)
4. [Mesh Approximation](#4-mesh-approximation)
5. [USD Utilities](#5-usd-utilities)
6. [add_urdf() and add_mjcf() Comparison](#6-add_urdf-and-add_mjcf-comparison)
7. [Complete Examples](#7-complete-examples)
8. [Common Pitfalls](#8-common-pitfalls)

---

## 1. add_usd() Full Signature

```python
builder.add_usd(
    source: str | UsdStage,
    *,
    xform: Transform | None = None,
    floating: bool | None = None,
    base_joint: dict | None = None,
    parent_body: int = -1,
    only_load_enabled_rigid_bodies: bool = False,
    only_load_enabled_joints: bool = True,
    joint_drive_gains_scaling: float = 1.0,
    verbose: bool = False,
    ignore_paths: list[str] | None = None,
    collapse_fixed_joints: bool = False,
    enable_self_collisions: bool = True,
    apply_up_axis_from_stage: bool = False,
    root_path: str = "/",
    joint_ordering: Literal["bfs", "dfs"] | None = "dfs",
    bodies_follow_joint_ordering: bool = True,
    skip_mesh_approximation: bool = False,
    load_sites: bool = True,
    load_visual_shapes: bool = True,
    hide_collision_shapes: bool = False,
    force_show_colliders: bool = False,
    parse_mujoco_options: bool = True,
    mesh_maxhullvert: int | None = None,
    schema_resolvers: list[SchemaResolver] | None = None,
    force_position_velocity_actuation: bool = False,
    override_root_xform: bool = False,
) -> dict[str, Any]
```

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str \| Usd.Stage` | required | File path/URL to `.usd`/`.usda`/`.usdc`, or an open `Usd.Stage` |
| `xform` | `Transform \| None` | `None` | World-space transform applied to the entire imported scene |
| `override_root_xform` | `bool` | `False` | Replace root transform instead of composing with `xform` |
| `floating` | `bool \| None` | `None` | Root joint type: `None`=format default, `True`=FREE, `False`=FIXED |
| `base_joint` | `dict \| None` | `None` | Custom joint spec for root (mutually exclusive with `floating`) |
| `parent_body` | `int` | `-1` | Parent body for hierarchical composition (-1 = world) |
| `only_load_enabled_rigid_bodies` | `bool` | `False` | Skip bodies with `physics:rigidBodyEnabled=False` |
| `only_load_enabled_joints` | `bool` | `True` | Skip joints with `physics:jointEnabled=False` |
| `joint_drive_gains_scaling` | `float` | `1.0` | Global multiplier for PD drive stiffness/damping |
| `verbose` | `bool` | `False` | Print parsing details |
| `ignore_paths` | `list[str] \| None` | `None` | Regex patterns for prim paths to skip |
| `collapse_fixed_joints` | `bool` | `False` | Merge bodies connected by fixed joints |
| `enable_self_collisions` | `bool` | `True` | Default self-collision within articulation |
| `apply_up_axis_from_stage` | `bool` | `False` | Adopt stage's up axis; otherwise rotate to match builder's |
| `root_path` | `str` | `"/"` | USD subtree to import |
| `joint_ordering` | `"bfs" \| "dfs" \| None` | `"dfs"` | Joint traversal order; `None` = USD file order |
| `bodies_follow_joint_ordering` | `bool` | `True` | Bodies added in same order as joints |
| `skip_mesh_approximation` | `bool` | `False` | Ignore `physics:approximation` on meshes |
| `load_sites` | `bool` | `True` | Import MjcSiteAPI prims as non-colliding reference points |
| `load_visual_shapes` | `bool` | `True` | Import non-physics visual geometry |
| `hide_collision_shapes` | `bool` | `False` | Hide colliders on bodies that have visual geometry |
| `force_show_colliders` | `bool` | `False` | Force VISIBLE flag on collision shapes |
| `parse_mujoco_options` | `bool` | `True` | Parse MuJoCo solver options from PhysicsScene |
| `mesh_maxhullvert` | `int \| None` | `None` | Max vertices for convex hull. Per-shape `newton:maxHullVertices` overrides |
| `schema_resolvers` | `list[SchemaResolver] \| None` | `None` | Resolver instances in priority order (default: Newton-only) |
| `force_position_velocity_actuation` | `bool` | `False` | Force `POSITION_VELOCITY` mode when both kp and kd are nonzero |

### floating / base_joint / parent_body Combinations

| `floating` | `base_joint` | `parent_body` | Result |
|------------|-------------|---------------|--------|
| `None` | `None` | `-1` | Format default (USD: FREE for unjointed bodies) |
| `True` | `None` | `-1` | FREE joint to world (6 DOF) |
| `False` | `None` | `-1` | FIXED joint to world (0 DOF) |
| `None` | `{dict}` | `-1` | Custom joint to world |
| `False` | `None` | `body_idx` | FIXED joint to parent body |
| `None` | `{dict}` | `body_idx` | Custom joint to parent body |
| *set* | *set* | *any* | Error -- mutually exclusive |
| `True` | `None` | `body_idx` | Error -- FREE requires world frame |

### Actuation Mode Inference

| Condition | Mode |
|-----------|------|
| `stiffness > 0` and `damping > 0` | `POSITION` (or `POSITION_VELOCITY` if `force_position_velocity_actuation=True`) |
| `stiffness > 0` only | `POSITION` |
| `damping > 0` only | `VELOCITY` |
| Drive present, both gains = 0 | `EFFORT` (direct torque) |
| No drive | `NONE` |

---

## 2. Return Dictionary

`add_usd()` returns a `dict[str, Any]` with these keys:

| Key | Type | Description |
|-----|------|-------------|
| `"fps"` | `float` | Stage frames per second |
| `"duration"` | `float` | `endTimeCode - startTimeCode` |
| `"up_axis"` | `Axis` | Stage up axis (`"X"`, `"Y"`, or `"Z"`) |
| `"path_body_map"` | `dict[str, int]` | USD prim path to body index |
| `"path_joint_map"` | `dict[str, int]` | USD prim path to joint index |
| `"path_shape_map"` | `dict[str, int]` | USD prim path to shape index |
| `"path_shape_scale"` | `dict[str, tuple]` | USD prim path to 3D world scale |
| `"mass_unit"` | `float` | Kilograms per unit (default 1.0) |
| `"linear_unit"` | `float` | Meters per unit (default 1.0) |
| `"scene_attributes"` | `dict` | All attributes on the PhysicsScene prim |
| `"physics_dt"` | `float \| None` | Resolved physics time step |
| `"schema_attrs"` | `dict` | Per-prim schema attributes collected by resolvers |
| `"max_solver_iterations"` | `int \| None` | Resolved max solver iterations |
| `"collapse_results"` | `dict \| None` | Result of `collapse_fixed_joints`, if enabled |
| `"path_body_relative_transform"` | `dict` | Relative transforms for collapsed bodies |
| `"path_original_body_map"` | `dict` | Prim path to original body index (before collapse) |
| `"actuator_count"` | `int` | Number of external actuators parsed |

### Inspecting the Return Dict

```python
result = builder.add_usd("robot.usda", verbose=True)

for path, body_idx in result["path_body_map"].items():
    print(f"Body {body_idx}: {path}")

for path, joint_idx in result["path_joint_map"].items():
    print(f"Joint {joint_idx}: {path}")

dt = result["physics_dt"]   # e.g. 0.001
print(f"Actuators: {result['actuator_count']}")
```

---

## 3. Schema Resolvers

Schema resolvers collect per-prim vendor-specific USD attributes (Newton, PhysX, MuJoCo)
and map them onto Newton builder attributes using a priority system.

### Architecture

```
SchemaResolver          (base class)
  |-- SchemaResolverNewton   namespace: "newton"
  |-- SchemaResolverPhysx    namespace: "physx"  (+ physxScene, physxRigidBody, ...)
  |-- SchemaResolverMjc      namespace: "mjc"

SchemaResolverManager   (resolves values in priority order)
```

### PrimType Enum

```python
class PrimType(IntEnum):
    SCENE = 0
    JOINT = 1
    SHAPE = 2
    BODY = 3
    MATERIAL = 4
    ACTUATOR = 5
    ARTICULATION = 6
```

### SchemaResolverNewton (default)

Namespace: `newton:*`. Key mappings:

| PrimType | Key | USD Attribute | Default |
|----------|-----|---------------|---------|
| SCENE | `max_solver_iterations` | `newton:maxSolverIterations` | -1 |
| SCENE | `time_steps_per_second` | `newton:timeStepsPerSecond` | 1000 |
| SCENE | `gravity_enabled` | `newton:gravityEnabled` | True |
| JOINT | `armature` | `newton:armature` | 0.0 |
| JOINT | `friction` | `newton:friction` | 0.0 |
| JOINT | `limit_linear_ke` | `newton:linear:limitStiffness` | 1.0e4 |
| JOINT | `limit_linear_kd` | `newton:linear:limitDamping` | 1.0e1 |
| SHAPE | `margin` | `newton:contactMargin` | 0.0 |
| SHAPE | `gap` | `newton:contactGap` | -inf |
| SHAPE | `max_hull_vertices` | `newton:maxHullVertices` | -1 |
| ARTICULATION | `self_collision_enabled` | `newton:selfCollisionEnabled` | True |
| MATERIAL | `mu_torsional` | `newton:torsionalFriction` | 0.25 |
| MATERIAL | `mu_rolling` | `newton:rollingFriction` | 0.0005 |

### SchemaResolverPhysx

Namespace: `physx` with extra namespaces: `physxScene`, `physxRigidBody`,
`physxCollision`, `physxConvexHullCollision`, `physxSDFMeshCollision`,
`physxMaterial`, `physxJoint`, `physxArticulation`, etc.

| PrimType | Key | USD Attribute | Default |
|----------|-----|---------------|---------|
| SCENE | `time_steps_per_second` | `physxScene:timeStepsPerSecond` | 60 |
| SCENE | `gravity_enabled` | `physxRigidBody:disableGravity` | False (inverted) |
| JOINT | `armature` | `physxJoint:armature` | 0.0 |
| SHAPE | `gap` | computed | -inf (`contactOffset - restOffset`) |
| SHAPE | `max_hull_vertices` | `physxConvexHullCollision:hullVertexLimit` | 64 |
| BODY | `linear_damping` | `physxRigidBody:linearDamping` | 0.0 |
| MATERIAL | `stiffness` | `physxMaterial:compliantContactStiffness` | 0.0 |

### SchemaResolverMjc

Namespace: `mjc:*`. Uses `solref` (timeconst, dampratio) conversion to stiffness/damping.

| PrimType | Key | USD Attribute | Default |
|----------|-----|---------------|---------|
| SCENE | `time_steps_per_second` | `mjc:option:timestep` | 0.002 (inverted: `1/ts`) |
| JOINT | `armature` | `mjc:armature` | 0.0 |
| JOINT | `friction` | `mjc:frictionloss` | 0.0 |
| SHAPE | `margin` | `mjc:margin` | 0.0 (computed: `margin - gap`) |
| SHAPE | `ke` | `mjc:solref` | [0.02, 1.0] (contact stiffness) |
| ACTUATOR | `gainPrm` | `mjc:gainPrm` | [1,0,...] (10-element vector) |

**Requirement**: Before using `SchemaResolverMjc`, call:

```python
newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
```

### Resolution Priority

When multiple resolvers are provided, the `SchemaResolverManager` evaluates in list order:

1. First authored value found (highest-priority resolver first)
2. Caller-provided `default` argument
3. First non-`None` mapping default from resolvers in priority order
4. `None`

```python
from newton.usd import SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

result = builder.add_usd(
    "robot.usda",
    schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton(), SchemaResolverPhysx()],
)

schema_attrs = result["schema_attrs"]
```

---

## 4. Mesh Approximation

When `skip_mesh_approximation=False` (default), Newton reads `physics:approximation`
from `UsdPhysicsMeshCollisionAPI` and remeshes accordingly.

| `physics:approximation` value | Internal method | Description |
|-------------------------------|-----------------|-------------|
| `convexDecomposition` | `coacd` | Approximate Convex Decomposition (CoACD) |
| `convexHull` | `convex_hull` | Single convex hull |
| `boundingSphere` | `bounding_sphere` | Minimum bounding sphere |
| `boundingCube` | `bounding_box` | Axis-aligned bounding box |
| `meshSimplification` | `quadratic` | Quadratic mesh decimation |
| (none / `"none"`) | (no remeshing) | Use original triangle mesh |

### Controlling Convex Hull Vertices

```python
# Global limit via parameter
result = builder.add_usd("robot.usd", mesh_maxhullvert=64)

# Per-shape override via USD attribute: newton:maxHullVertices = 128
```

### Example in USD

```usda
def Mesh "CollisionMesh" (
    prepend apiSchemas = ["PhysicsMeshCollisionAPI"]
)
{
    uniform token physics:approximation = "convexDecomposition"
}
```

---

## 5. USD Utilities

All importable from `newton.usd`.

### get_mesh()

```python
def get_mesh(
    prim: Usd.Prim,
    load_normals: bool = False,
    load_uvs: bool = False,
    maxhullvert: int | None = None,
    face_varying_normal_conversion: Literal[
        "vertex_averaging", "angle_weighted", "vertex_splitting"
    ] = "vertex_splitting",
    vertex_splitting_angle_threshold_deg: float = 25.0,
) -> Mesh | tuple[Mesh, np.ndarray | None]
```

| `face_varying_normal_conversion` | Behavior |
|----------------------------------|----------|
| `"vertex_averaging"` | Average all corner normals per vertex. Fastest |
| `"angle_weighted"` | Weight by corner angle. Better at sharp edges |
| `"vertex_splitting"` | Split vertices when normals differ > threshold. Preserves hard edges |

```python
from pxr import Usd
import newton, newton.usd

stage = Usd.Stage.Open("bunny.usd")
mesh = newton.usd.get_mesh(stage.GetPrimAtPath("/root/bunny"), load_normals=True)
```

### get_transform()

```python
def get_transform(
    prim: Usd.Prim,
    local: bool = True,
    xform_cache: UsdGeom.XformCache | None = None,
) -> wp.transform
```

- `local=True` -- local transform relative to parent
- `local=False` -- world-space transform

### get_scale()

```python
def get_scale(prim: Usd.Prim, local: bool = True) -> wp.vec3
```

### get_attribute() / has_attribute()

```python
def get_attribute(prim: Usd.Prim, name: str, default=None) -> Any | None
def has_attribute(prim: Usd.Prim, name: str) -> bool
```

Only returns authored values (ignores schema fallback defaults).

### get_attributes_in_namespace()

```python
def get_attributes_in_namespace(prim: Usd.Prim, namespace: str) -> dict[str, Any]
```

```python
attrs = newton.usd.get_attributes_in_namespace(prim, "newton")
# {"newton:contactMargin": 0.01, "newton:contactGap": 0.005, ...}
```

### Material Resolution

The parser resolves material properties from USD PhysicsMaterialAPI bindings:

```python
# Default material values when not authored in USD:
# staticFriction:    builder.default_shape_cfg.mu           (1.0)
# dynamicFriction:   builder.default_shape_cfg.mu           (1.0)
# torsionalFriction: builder.default_shape_cfg.mu_torsional (0.005)
# rollingFriction:   builder.default_shape_cfg.mu_rolling   (0.0001)
# restitution:       builder.default_shape_cfg.restitution  (0.0)
# density:           builder.default_shape_cfg.density      (1000.0)
```

---

## 6. add_urdf() and add_mjcf() Comparison

Newton also supports URDF and MJCF formats with a similar interface.

| Feature | `add_usd()` | `add_urdf()` | `add_mjcf()` |
|---------|-------------|--------------|--------------|
| Format | USD (OpenUSD) | URDF (ROS) | MJCF (MuJoCo XML) |
| Schema support | UsdPhysics + custom | Standard URDF | MuJoCo XML |
| Mesh approximation | 5 modes | Via collision tags | Via geom types |
| Articulation root | `floating` param | `floating` param | `floating` param |
| Sites/sensors | MjcSiteAPI prims | Not native | `<site>` elements |
| Joint ordering | `bfs`/`dfs`/`None` | As authored | As authored |
| Multi-articulation | Yes (single stage) | Single robot | Single model |

```python
import newton

builder = newton.ModelBuilder()

# USD (most feature-rich)
result = builder.add_usd("robot.usd", floating=False)

# URDF (common in ROS)
result = builder.add_urdf("robot.urdf", floating=True)

# MJCF (common in RL research)
result = builder.add_mjcf("robot.xml", floating=False)
```

---

## 7. Complete Examples

### Loading a Fixed-Base Robot

```python
import newton

builder = newton.ModelBuilder()
result = builder.add_usd(
    "franka_panda.usd",
    floating=False,
    enable_self_collisions=False,
    collapse_fixed_joints=True,
    joint_ordering="dfs",
)

builder.add_ground_plane()
model = builder.finalize("cuda:0")
state = model.state()
control = model.control()
contacts = model.contacts()

solver = newton.solvers.SolverXPBD(model)
for _ in range(1000):
    newton.sim.collide(model, state, contacts)
    solver.step(state, state, control, contacts, dt=1.0 / 60.0)
```

### Hierarchical Composition (Attaching Gripper to Arm)

```python
builder = newton.ModelBuilder()

base_result = builder.add_usd("arm.usda", floating=False)
ee_idx = base_result["path_body_map"]["/World/ee_link"]

builder.add_usd(
    "gripper.usda",
    parent_body=ee_idx,
    floating=False,
)

model = builder.finalize("cuda:0")
```

### Custom Root Joint (D6)

```python
import newton
from newton import Transform

builder = newton.ModelBuilder()
result = builder.add_usd(
    "mobile_robot.usd",
    base_joint={
        "type": "D6",
        "linear_axes": [
            newton.ModelBuilder.JointDofConfig(axis="X"),
            newton.ModelBuilder.JointDofConfig(axis="Y"),
        ],
        "angular_axes": [
            newton.ModelBuilder.JointDofConfig(axis="Z"),
        ],
    },
)

model = builder.finalize("cuda:0")
```

### Placing at a Specific Location

```python
builder = newton.ModelBuilder()
result = builder.add_usd(
    "robot.usd",
    xform=Transform(pos=(0.0, 0.0, 1.0)),
    floating=False,
)
model = builder.finalize("cuda:0")
```

### Loading with Schema Resolvers

```python
import newton
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx

builder = newton.ModelBuilder()
result = builder.add_usd(
    "scene.usd",
    schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
    verbose=True,
)

for path, attrs in result["schema_attrs"].items():
    print(f"{path}: {attrs}")

print(f"Physics dt: {result['physics_dt']}")
print(f"Max iterations: {result['max_solver_iterations']}")
```

### Ignoring Paths

```python
result = builder.add_usd(
    "scene.usd",
    ignore_paths=[
        r"/World/Camera.*",       # Skip camera prims
        r".*/visual_only/.*",     # Skip visual-only subtrees
    ],
)
```

---

## 8. Common Pitfalls

- **USD not installed**: All USD functions require `pxr` (`pip install usd-core`).
- **MuJoCo resolver without registration**: Using `SchemaResolverMjc` without calling
  `SolverMuJoCo.register_custom_attributes(builder)` first raises `RuntimeError`.
- **`floating=True` with `parent_body`**: FREE joints must connect to world frame.
- **`floating` and `base_joint` are mutually exclusive**: Setting both raises an error.
- **`get_attribute()` returns only authored values**: Schema fallback defaults are NOT returned.
- **Scale not in `get_transform()`**: Use `get_scale()` separately for scale components.
- **Quaternion conventions**: USD uses `(real, i, j, k)`, Warp uses `(i, j, k, real)`.
  `value_to_warp()` handles this automatically.
- **Joint ordering affects body indices**: Changing `joint_ordering` changes body indices.
- **PhysX `-inf` sentinel values**: Treated as unset (`None`), not literal negative infinity.
