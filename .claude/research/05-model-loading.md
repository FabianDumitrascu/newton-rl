# Model Loading in Newton (USD / URDF / MJCF)

## Supported Formats

| Format | Method | Default Root Joint | Best For |
|--------|--------|--------------------|----------|
| USD | `builder.add_usd()` | FREE (floating) | Newton-native, rich metadata |
| URDF | `builder.add_urdf()` | FIXED (grounded) | ROS ecosystem robots |
| MJCF | `builder.add_mjcf()` | Varies | MuJoCo-compatible models |

## USD Loading (Primary)

```python
result = builder.add_usd(
    "robot.usda",
    xform=wp.transform(pos, quat),   # World placement
    floating=True,                     # FREE root joint (for drones)
    collapse_fixed_joints=True,        # Merge fixed-joint bodies
    enable_self_collisions=False,      # Faster if not needed
    ignore_paths=[".*Dummy"],          # Skip visual-only prims
    schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
)

# Result contains mappings:
result["path_body_map"]   # USD prim path → body index
result["path_joint_map"]  # USD prim path → joint index
result["path_shape_map"]  # USD prim path → shape index
result["physics_dt"]      # Authored timestep
```

## URDF Loading

```python
builder.add_urdf(
    "robot.urdf",
    xform=wp.transform(pos, quat),
    floating=True,                        # Override default FIXED root
    scale=1.0,                            # Global geometry scale
    parse_visuals_as_colliders=True,      # Use visual meshes for collision
    enable_self_collisions=False,
    ignore_inertial_definitions=True,     # Recompute from geometry
)
```

Supports `package://`, `model://`, and `http://` URIs for mesh assets.

## MJCF Loading

```python
builder.add_mjcf(
    "robot.xml",
    parse_mujoco_options=True,  # Import solver settings
    parse_sites=True,           # Load reference points
    ctrl_direct=False,          # Use Newton joint targets (not MuJoCo ctrl)
)
```

## Composing Multiple Models

```python
builder = newton.ModelBuilder()

# Load drone
drone_result = builder.add_usd("drone.usda", floating=True)

# Load manipulator arm attached to drone body
arm_result = builder.add_urdf(
    "arm.urdf",
    parent_body=drone_result["path_body_map"]["/Drone/Body"],
    floating=False,  # FIXED to drone
)

# Add ground + objects
builder.add_ground_plane()
builder.add_body(label="target_object")
builder.add_shape_box(...)

model = builder.finalize()
```

## Post-Load Customization (Before finalize)

```python
# Adjust joint gains
builder.joint_target_ke[joint_id] = 500.0
builder.joint_target_kd[joint_id] = 50.0
builder.joint_target_mode[joint_id] = int(JointTargetMode.POSITION)
builder.joint_armature[joint_id] = 0.1

# Set initial joint positions
builder.joint_q[0:N] = initial_positions

# Add extra shapes/bodies programmatically
extra_body = builder.add_body(label="payload")
builder.add_shape_sphere(extra_body, radius=0.05)
```

## Schema Resolvers (Physics Properties from USD)

Priority-based resolution for PhysX/MuJoCo/Newton-specific attributes:

```python
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx, SchemaResolverMjc

# Newton attributes take priority
result = builder.add_usd("asset.usda",
    schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])
```

## Mesh Approximation Options

USD `physics:approximation` attribute controls collision geometry:
- `convexdecomposition` — COACD decomposition
- `convexhull` — Single convex hull
- `boundingsphere` — Bounding sphere
- `boundingcube` — Bounding box
- `meshsimplification` — Reduced triangle count

## Key Assets in Newton

| Asset | Path | Type |
|-------|------|------|
| Crazyflie | `examples/assets/crazyflie.usd` | Quadcopter USD |
| Cartpole | `examples/assets/cartpole.usda` | Cartpole USD |
| Cartpole URDF | `examples/assets/cartpole.urdf` | Cartpole URDF |
| Panda (download) | `newton.utils.download_asset("franka_emika_panda")` | URDF |
| UR10 (download) | `newton.utils.download_asset("universal_robots_ur10")` | USD |
