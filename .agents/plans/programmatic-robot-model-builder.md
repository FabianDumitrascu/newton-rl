# Feature: Programmatic Robot Model Construction

The following plan should be complete, but it's important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils, types, and models. Import from the right files etc.

## Feature Description

Replace the broken USD-based model loading with programmatic construction using Newton's `ModelBuilder` API. The current `flattened-osprey.usd` file has deep incompatibilities with Newton — collision shapes on Xform prims are silently skipped, PD gains are auto-scaled by 57.3x, all body masses are zero, and an extra `/World/Cube` body pollutes the model. Building the robot programmatically gives full control over every parameter, eliminates format-related bugs, and directly supports `replicate()` for future GPU-batched RL training.

## User Story

As a thesis researcher
I want the Osprey aerial manipulator to load correctly in Newton with proper joints, collision shapes, and mass properties
So that the INDI hover controller works stably and I can proceed to RL environment development

## Problem Statement

The USD file authored for Isaac Sim produces a broken Newton model:
1. **Zero collision shapes loaded** — all ~100 collision prims are on `Xform` types (not geometry), Newton skips them all with warnings
2. **PD gains auto-scaled by 57.3x** — Newton's USD importer applies `DegreesToRadian` scaling, producing `ke=57295` instead of ~500
3. **All body masses are zero** — no `MassAPI` in USD
4. **Extra `/World/Cube` body** loaded as body 9, shifting indices
5. **Current workarounds in `validate_hover.py`** fight the broken skeleton instead of fixing it

## Solution Statement

Build the Osprey model from scratch using Newton's `ModelBuilder.add_link()` + `add_joint_*()` API. Load OBJ meshes for visual rendering and add simple box collision shapes. All parameters come from the existing `OspreyConfig` dataclass. The builder function returns an unfinalized `ModelBuilder` so callers can use `scene.add_world(osprey)` or `scene.replicate(osprey, N)` for batched RL.

## Feature Metadata

**Feature Type**: Bug Fix / Refactor
**Estimated Complexity**: Medium
**Primary Systems Affected**: `controllers/`, `testing/validate_hover.py`
**Dependencies**: Newton ModelBuilder API (submodule), trimesh (for OBJ loading), warp

---

## CONTEXT REFERENCES

### Relevant Codebase Files — YOU MUST READ THESE BEFORE IMPLEMENTING

- `controllers/config.py` (full file) — All physical parameters: mass, inertia, COM, rotor geometry, PD gains, joint DOF indices, body indices. This is the source of truth for every numeric value.
- `testing/validate_hover.py` (lines 64-177, `_build_model()`) — Current broken implementation to replace. Also lines 229-276 (`_apply_rotor_forces()`) and 290-340 (`step()`) — the simulation loop that must continue working.
- `controllers/indi.py` (full file) — INDI controller interface. Uses `self.G1`, `self.G1_inv`, `self.thrust_coeff`. Inputs: `omega_body`, `rotor_speeds`, `collective_thrust`, `desired_rates`. Must not change.
- `controllers/motor_model.py` (full file) — Motor dynamics. `step()` returns `(thrusts, moments, omega)`. Must not change.
- `controllers/math_utils.py` (full file) — Quaternion ops (XYZW convention). Must not change.
- `submodules/newton/newton/examples/basic/example_basic_pendulum.py` (lines 47-64) — Reference pattern: `add_link()` + `add_joint_revolute()` + `add_articulation()`. This is the exact pattern to follow.
- `submodules/newton/newton/examples/robot/example_robot_ur10.py` (lines 66-84) — Reference for `register_custom_attributes()` and post-load PD gain setting.
- `submodules/newton/newton/examples/robot/example_robot_panda_hydro.py` (lines 287-298) — Reference for `SolverMuJoCo` constructor parameters.

### New Files to Create

- `controllers/osprey_model.py` — `build_osprey()` function that constructs the full articulated model programmatically

### Files to Modify

- `controllers/config.py` — Add `JointFrames` dataclass with joint anchor transforms extracted from USD
- `testing/validate_hover.py` — Replace `_build_model()` to use `build_osprey()`, optionally switch solver

### Relevant Documentation — READ BEFORE IMPLEMENTING

- `.claude/references/model_builder.md` — ModelBuilder API: `add_link`, shapes, ShapeConfig, finalize
- `.claude/references/joints.md` — Joint types, JointDofConfig, axes, limits, PD control
- `.claude/references/solvers.md` — Solver comparison, SolverMuJoCo vs SolverXPBD
- `.claude/references/contacts_and_sensors.md` — Collision groups, filtering, self-collision

### Patterns to Follow

**Articulated Robot Construction (from pendulum example):**
```python
builder = newton.ModelBuilder()

link_0 = builder.add_link()
builder.add_shape_box(link_0, hx=hx, hy=hy, hz=hz)

link_1 = builder.add_link()
builder.add_shape_box(link_1, hx=hx, hy=hy, hz=hz)

j0 = builder.add_joint_revolute(
    parent=-1, child=link_0,
    axis=wp.vec3(0.0, 1.0, 0.0),
    parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=rot),
    child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
)
j1 = builder.add_joint_revolute(
    parent=link_0, child=link_1,
    axis=wp.vec3(0.0, 1.0, 0.0),
    parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
    child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
)
builder.add_articulation([j0, j1], label="pendulum")
```

**Multi-world replication pattern:**
```python
robot = newton.ModelBuilder()
# ... build one robot ...
scene = newton.ModelBuilder()
scene.add_ground_plane()
scene.add_world(robot)  # or scene.replicate(robot, world_count=512)
model = scene.finalize()
```

**ShapeConfig for visual-only shapes (no collision):**
```python
visual_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, collision_group=0)
```

**ShapeConfig for collision shapes (no mass contribution):**
```python
collision_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.8, collision_group=1)
```

**Key API details:**
- `add_link()` creates a body WITHOUT a joint — you must add joints explicitly
- `add_body()` creates body + FREE joint + articulation in one call — do NOT use this for articulated robots
- `add_articulation(joints)` requires joint indices that are contiguous and monotonically increasing — add joints in order
- `parent=-1` in `add_joint_free()` connects to world frame (floating base)
- `collision_filter_parent=True` (default) auto-filters collisions between parent-child body pairs
- Joint `axis` is specified in the **parent anchor frame** (parent body frame transformed by `parent_xform`)
- `lock_inertia=True` on `add_link()` prevents shapes from modifying mass/inertia — use when setting explicit mass

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation — Add Joint Frame Data to Config

Add the joint frame transforms (extracted from the USD file) to `controllers/config.py` as a new dataclass. This keeps all physical parameters in one place.

### Phase 2: Core Implementation — Build Osprey Programmatically

Create `controllers/osprey_model.py` with a `build_osprey()` function that:
1. Creates 9 bodies with explicit mass/inertia
2. Connects them with the correct joint types, axes, limits, and PD gains
3. Adds visual mesh shapes from OBJ files
4. Adds simple collision shapes (boxes) on key bodies
5. Returns an unfinalized `ModelBuilder`

### Phase 3: Integration — Update Hover Validation

Replace the `_build_model()` method in `testing/validate_hover.py` to use the new builder. Keep all controller/simulation loop code unchanged.

### Phase 4: Testing & Validation

Run the hover validation viewer and verify stable hover, correct joint behavior, and absence of glitching.

---

## STEP-BY-STEP TASKS

### Task 1: UPDATE `controllers/config.py` — Add JointFrames dataclass

- **IMPLEMENT**: Add a `JointFrames` dataclass containing joint anchor positions and rotations for all 8 joints. Values were extracted from the USD file's `localPos0`, `localRot0`, `localPos1`, `localRot1` attributes. All positions in meters.
- **PATTERN**: Follow existing dataclass pattern in `config.py` (e.g., `RotorConfig`, `InertiaConfig`)
- **GOTCHA**: Quaternions in USD are WXYZ format `(w, x, y, z)`. Warp uses XYZW format `(x, y, z, w)`. Convert when creating `wp.quat()` values.

**Joint frame data (from USD inspection):**

| Joint | parent_pos | parent_rot (WXYZ from USD) | child_pos | child_rot | Axis | Limits |
|-------|-----------|---------------------------|----------|----------|------|--------|
| dof_differential | (0.06968, 0, 0.04771) | (1, 0, 0, 0) identity | (0.01458, 0, 0.00209) | identity | Y | [-π/2, π/2] |
| dof_arm | (-0.00049, 0, -0.00013) | identity | (0, 0, 0) | identity | X | unlimited |
| dof_finger_left | (0.15011, 0.04307, -0.00043) | (0.5, -0.5, 0.5, -0.5) WXYZ | (0, 0, 0) | identity | X | [0, 0.027] m |
| dof_finger_right | (0.15011, -0.04307, -0.00043) | (~0, 0.70711, 0.70711, ~0) WXYZ | (0, 0, 0) | identity | X | [0, 0.027] m |
| rotor_front_right | (0.11101, -0.11367, 0.09629) | identity | (0, 0, 0) | identity | Z | unlimited |
| rotor_front_left | (0.11101, 0.11367, 0.09629) | identity | (0, 0, 0) | identity | Z | unlimited |
| rotor_back_left | (-0.05812, 0.07620, 0.09629) | identity | (0, 0, 0) | identity | Z | unlimited |
| rotor_back_right | (-0.05812, -0.07620, 0.09629) | identity | (0, 0, 0) | identity | Z | unlimited |

- **IMPLEMENT**: Also add the `JointFrames` field to `OspreyConfig` dataclass
- **VALIDATE**: `uv run python controllers/config.py` — should print config validation and pass assertions

### Task 2: CREATE `controllers/osprey_model.py` — build_osprey() function

- **IMPLEMENT**: A function `build_osprey(cfg: OspreyConfig, spawn_pos: tuple[float, float, float] = (0.0, 0.0, 2.0)) -> newton.ModelBuilder`

**Body creation order (determines body indices):**

| Body Index | Label | Mass | COM | Inertia | Notes |
|-----------|-------|------|-----|---------|-------|
| 0 | base | falcon_mass - 4*rotor_mass (~0.620) | cfg.inertia.base_com | cfg.inertia.base_inertia_diag | Main drone body |
| 1 | differential | manipulator_mass * 0.3 - finger_mass (~0.050) | None | None | Arm pitch link |
| 2 | arm | manipulator_mass * 0.7 - finger_mass (~0.130) | None | None | Arm roll link |
| 3 | finger_left | 0.005 | None | None | Left gripper finger |
| 4 | finger_right | 0.005 | None | None | Right gripper finger |
| 5 | rotor_back_left | 0.01 | None | None | Rotor body (force-driven) |
| 6 | rotor_back_right | 0.01 | None | None | Rotor body (force-driven) |
| 7 | rotor_front_left | 0.01 | None | None | Rotor body (force-driven) |
| 8 | rotor_front_right | 0.01 | None | None | Rotor body (force-driven) |

- **GOTCHA**: Use `lock_inertia=True` on all bodies to prevent shape additions from overriding the explicit mass/inertia. Shapes use `density=0.0` for this reason.
- **GOTCHA**: The base body `xform` should be `wp.transform(p=wp.vec3(*spawn_pos), q=wp.quat_identity())`. Other bodies get `xform=None` — their world positions are determined by the joint chain from the root.

**Joint creation order (determines joint and DOF indices):**

| Joint Index | Type | Parent→Child | Axis | PD Gains | Limits | Label |
|------------|------|-------------|------|----------|--------|-------|
| 0 | FREE | world→base | — | ke=0, kd=0 | — | base_free |
| 1 | REVOLUTE | base→differential | Y | cfg.arm.arm_ke, arm_kd | [-π/2, π/2] | dof_differential |
| 2 | REVOLUTE | differential→arm | X | cfg.arm.arm_ke, arm_kd | unlimited | dof_arm |
| 3 | PRISMATIC | arm→finger_left | X | cfg.arm.gripper_ke, gripper_kd | [0, 0.027] | dof_finger_left |
| 4 | PRISMATIC | arm→finger_right | X | cfg.arm.gripper_ke, gripper_kd | [0, 0.027] | dof_finger_right |
| 5 | REVOLUTE | base→rotor_bl | Z | ke=0, kd=0 | unlimited | dof_rotor_back_left |
| 6 | REVOLUTE | base→rotor_br | Z | ke=0, kd=0 | unlimited | dof_rotor_back_right |
| 7 | REVOLUTE | base→rotor_fl | Z | ke=0, kd=0 | unlimited | dof_rotor_front_left |
| 8 | REVOLUTE | base→rotor_fr | Z | ke=0, kd=0 | unlimited | dof_rotor_front_right |

**DOF layout (resulting from joint order):**

| DOF Index | Joint | Meaning |
|----------|-------|---------|
| 0-5 | FREE (base) | tx, ty, tz, rx, ry, rz |
| 6 | dof_differential | arm pitch |
| 7 | dof_arm | arm roll |
| 8 | dof_finger_left | left finger slide |
| 9 | dof_finger_right | right finger slide |
| 10 | dof_rotor_back_left | rotor spin |
| 11 | dof_rotor_back_right | rotor spin |
| 12 | dof_rotor_front_left | rotor spin |
| 13 | dof_rotor_front_right | rotor spin |

- **IMPLEMENT**: After creating all joints, call `builder.add_articulation([j0, j1, j2, j3, j4, j5, j6, j7, j8], label="osprey")`
- **IMPORTS**: `import newton`, `import warp as wp`, `import numpy as np`, `import trimesh`, `from pathlib import Path`, `from controllers.config import OspreyConfig`

**Visual mesh shapes:**
- For each body, load the corresponding OBJ from `assets/meshes/` (e.g., `link_body.obj` for base)
- Scale vertices by 0.001 (STEP files are in mm)
- Use `builder.add_shape_mesh(body=idx, mesh=newton.Mesh(vertices=v, indices=i), cfg=visual_cfg)`
- `visual_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, collision_group=0)` — no collision, no mass

**Collision shapes:**
- Base body: `builder.add_shape_box(body=0, hx=0.10, hy=0.13, hz=0.05, cfg=collision_cfg)` — box approximation of drone frame. Offset the box center to match mesh center using `xform=wp.transform(p=wp.vec3(0.026, 0.0, 0.054), q=wp.quat_identity())`
- Finger bodies: Small boxes for future grasping contact. `add_shape_box(body=3, hx=0.024, hy=0.010, hz=0.035, cfg=collision_cfg)` (from mesh extents)
- Other bodies: No collision shapes needed (self-collisions disabled, rotors don't interact)
- `collision_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.8, collision_group=1)`

**Body-to-mesh mapping:**
```python
body_mesh_map = {
    0: "link_body",
    1: "link_differential",
    2: "link_arm",
    3: "link_finger_left",
    4: "link_finger_right",
    5: "link_rotor_back_left",
    6: "link_rotor_back_right",
    7: "link_rotor_front_left",
    8: "link_rotor_front_right",
}
```

- **PATTERN**: Follow OBJ loading pattern from current `validate_hover.py` lines 128-154
- **GOTCHA**: `newton.Mesh(vertices=v, indices=i)` — vertices must be `np.float32`, indices must be `np.int32`
- **GOTCHA**: Joint `parent_xform` and `child_xform` define the joint anchor position relative to the parent/child body origins. Use `wp.transform(p=wp.vec3(...), q=wp.quat(...))`. For identity rotation, use `wp.quat_identity()`.
- **VALIDATE**: `uv run python -c "from controllers.osprey_model import build_osprey; from controllers.config import default_osprey_config; b = build_osprey(default_osprey_config()); print(f'Bodies: {b.body_count}, Joints: {b.joint_count}, Shapes: {b.shape_count}')"` — should print `Bodies: 9, Joints: 9, Shapes: ~11` (9 visual meshes + 2-3 collision boxes)

### Task 3: UPDATE `controllers/config.py` — Verify body/DOF index consistency

- **IMPLEMENT**: Update `BodyIndices` to match the new build order. The current values should already be correct if we follow the same body creation order:
  - base=0, differential=1, arm=2, finger_left=3, finger_right=4, rotor_back_left=5, rotor_back_right=6, rotor_front_left=7, rotor_front_right=8
- **IMPLEMENT**: Update `ArmConfig` DOF indices to match:
  - differential_dof=6, arm_dof=7, finger_left_dof=8, finger_right_dof=9
- **IMPLEMENT**: Update `RotorConfig.body_indices` to match:
  - [8, 7, 5, 6] = [rotor_front_right, rotor_front_left, rotor_back_left, rotor_back_right] (reference order FR, FL, BL, BR)
- **GOTCHA**: The current `config.py` already has these exact values. Verify they match the build order in Task 2. If they differ, update them.
- **VALIDATE**: `uv run python controllers/config.py`

### Task 4: UPDATE `testing/validate_hover.py` — Replace _build_model()

- **IMPLEMENT**: Replace the entire `_build_model()` method (lines 64-177) with:

```python
def _build_model(self) -> None:
    from controllers.osprey_model import build_osprey

    osprey = build_osprey(self.cfg, spawn_pos=(0.0, 0.0, self.cfg.sim.spawn_height))

    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.add_world(osprey)

    self.model = scene.finalize()
    self.state_0 = self.model.state()
    self.state_1 = self.model.state()
    self.control = self.model.control()
    self.contacts = self.model.contacts()

    self.solver = newton.solvers.SolverXPBD(
        self.model, iterations=self.cfg.sim.solver_iterations
    )

    # Print mass verification
    masses = self.model.body_mass.numpy()
    total = sum(float(masses[i]) for i in range(9))
    print(f"Drone total mass: {total:.3f} kg (expected {self.cfg.total_mass:.3f})")
    print(f"Expected hover thrust: {self.cfg.hover_thrust:.2f} N")
```

- **IMPLEMENT**: Remove the `trimesh` and `numpy` imports from `validate_hover.py` if they are no longer used (they move into `osprey_model.py`)
- **IMPLEMENT**: Remove `import newton.usd` from `validate_hover.py` — no longer needed
- **GOTCHA**: The `scene.add_world(osprey)` call adds the osprey builder as world 0. Body indices within that world are preserved. However, the ground plane body (added by `add_ground_plane()`) is NOT in any world and gets a separate body index. Verify that `self.cfg.body.base` still maps to the correct body in the finalized model by checking `self.model.body_mass.numpy()[0]` matches expected base mass.
- **GOTCHA**: If body indices shift due to `add_world()`, you may need to adjust the body index used in `_extract_state()` and `_apply_rotor_forces()`. Test this carefully.
- **PATTERN**: The simulation loop (`step()`, `_apply_rotor_forces()`, `_extract_state()`, `_update_arm_targets()`, `gui()`) should remain unchanged — they reference body/DOF indices from `self.cfg` which we verified in Task 3.
- **VALIDATE**: `uv run python testing/validate_hover.py` — viewer opens, drone visible with correct mesh geometry

### Task 5: VERIFY — Run hover validation

- **IMPLEMENT**: Run the viewer and test all functionality:
  1. Drone hovers at ~2m altitude
  2. Total thrust reads ~8.4N in telemetry
  3. Thrust slider up → drone rises
  4. Thrust slider down → drone falls
  5. Roll/pitch rate sliders → drone tilts, INDI compensates
  6. Arm pitch slider → arm pitches (rotates around Y-axis of differential joint)
  7. Arm roll slider → arm rolls (rotates around X-axis of arm joint)
  8. Gripper slider → fingers open/close smoothly without jitter
  9. Press R → simulation resets
- **GOTCHA**: If body indices shifted due to `add_world()`, the force application and state extraction will read wrong bodies. The symptom would be: drone doesn't respond to thrust commands, or wrong body moves.
- **GOTCHA**: If the drone falls immediately, check that the spawn height is applied correctly. The base body xform in `build_osprey()` should place it at `spawn_pos`.
- **VALIDATE**: `uv run python testing/validate_hover.py` — interactive testing

### Task 6: LINT — Run pre-commit checks

- **VALIDATE**: `uvx pre-commit run -a` — all checks pass

---

## TESTING STRATEGY

### Unit Tests

No formal unit test framework (project uses manual validation scripts). Validate via:
1. Config validation script: `uv run python controllers/config.py`
2. Smoke test of model building:
```bash
uv run python -c "
from controllers.osprey_model import build_osprey
from controllers.config import default_osprey_config
cfg = default_osprey_config()
b = build_osprey(cfg)
assert b.body_count == 9, f'Expected 9 bodies, got {b.body_count}'
assert b.joint_count == 9, f'Expected 9 joints, got {b.joint_count}'
print('Model build OK')
"
```

### Integration Tests

3. Full INDI hover test:
```bash
uv run python -c "
from controllers.osprey_model import build_osprey
from controllers.config import default_osprey_config
from controllers.indi import IndiController
from controllers.motor_model import RotorMotor
import torch
import newton

cfg = default_osprey_config()
osprey = build_osprey(cfg)
scene = newton.ModelBuilder()
scene.add_ground_plane()
scene.add_world(osprey)
model = scene.finalize()

state = model.state()
print(f'Body 0 position: {state.body_q.numpy()[0][:3]}')
print(f'Body 0 mass: {model.body_mass.numpy()[0]:.3f}')

# Verify INDI controller works with new model
indi = IndiController(1, cfg, torch.device('cpu'))
init_omega = torch.tensor(cfg.control.init_omega).unsqueeze(0)
motor = RotorMotor(1, cfg.rotor, cfg.motor, init_omega, torch.device('cpu'))
for _ in range(50):
    target = indi.get_command(torch.zeros(1,3), motor.current_omega, torch.tensor([cfg.hover_thrust]), torch.zeros(1,3))
    thrusts, _, _ = motor.step(target, cfg.sim.sim_dt)
print(f'Hover thrust: {thrusts.sum():.2f}N (expected {cfg.hover_thrust:.2f}N)')
print('Integration test OK')
"
```

### Edge Cases

- Body with zero mass but non-zero inertia (rotors have small mass but no explicit inertia)
- Joint limits at boundaries (finger at 0 and 0.027m)
- Rotor joints with zero PD gains (should spin freely)

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
uvx pre-commit run -a
```

### Level 2: Unit Tests

```bash
uv run python controllers/config.py
uv run python -c "from controllers.osprey_model import build_osprey; from controllers.config import default_osprey_config; b = build_osprey(default_osprey_config()); assert b.body_count == 9; print('OK')"
```

### Level 3: Integration Tests

```bash
uv run python -c "
from controllers.osprey_model import build_osprey
from controllers.config import default_osprey_config
import newton
cfg = default_osprey_config()
osprey = build_osprey(cfg)
scene = newton.ModelBuilder()
scene.add_ground_plane()
scene.add_world(osprey)
model = scene.finalize()
state = model.state()
# Verify spawn height
pos = state.body_q.numpy()[0][:3]
assert abs(pos[2] - cfg.sim.spawn_height) < 0.01, f'Spawn height wrong: {pos[2]}'
# Verify mass
mass = float(model.body_mass.numpy()[0])
assert mass > 0.5, f'Base mass too low: {mass}'
print(f'All checks passed. Base at z={pos[2]:.2f}, mass={mass:.3f}')
"
```

### Level 4: Manual Validation

```bash
uv run python testing/validate_hover.py
```

Then in the viewer:
- [ ] Drone visible with correct mesh geometry at ~2m height
- [ ] Telemetry shows total thrust ~8.4N
- [ ] Thrust slider moves drone up/down
- [ ] Arm pitch slider pitches arm (Y-axis rotation)
- [ ] Arm roll slider rolls arm (X-axis rotation)
- [ ] Gripper slider opens/closes fingers smoothly (no jitter)
- [ ] Press R resets simulation

---

## ACCEPTANCE CRITERIA

- [ ] `build_osprey()` creates a 9-body, 9-joint articulated model without using `add_usd()`
- [ ] No Newton warnings about "CollisionAPI applied to an unknown UsdGeomGPrim type"
- [ ] PD gains are exactly as specified in config (500/50 for arm, 2000/200 for gripper) — no 57.3x scaling
- [ ] All body masses match config values (total ~0.86 kg)
- [ ] Visual mesh geometry renders correctly in viewer
- [ ] Base body has a collision box for ground interaction
- [ ] Drone hovers stably with INDI controller
- [ ] Arm joints move along correct axes (differential=pitch/Y, arm=roll/X)
- [ ] Gripper fingers slide along X-axis without jitter or collision with each other
- [ ] No extra `/World/Cube` body in the model
- [ ] `build_osprey()` returns unfinalized `ModelBuilder` (supports future `replicate()`)
- [ ] All validation commands pass
- [ ] Lint passes (`uvx pre-commit run -a`)

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order (1-6)
- [ ] Each task validation passed
- [ ] All validation commands executed successfully
- [ ] Manual testing in viewer confirms stable hover
- [ ] No linting errors
- [ ] Acceptance criteria all met

---

## NOTES

### Why Programmatic Build Over Fixed USD

The USD file from Isaac Sim has fundamental format incompatibilities with Newton's USD importer:
- Collision shapes on `Xform` prims (Newton expects `Mesh`, `Cube`, `Sphere`, etc.)
- PD gain scaling by `DegreesToRadian` factor
- No `MassAPI` on bodies
- Extra scene objects (`/World/Cube`, `/World/GroundPlane`)

Building programmatically is the approach used by Newton's own drone example and matches the `add_link()` + `add_joint_*()` + `add_articulation()` pattern used in all Newton articulation examples.

### Solver Choice

The plan keeps `SolverXPBD` for now to minimize changes. All Newton robot examples use `SolverMuJoCo` — switching to it is a natural follow-up that may also fix the altitude drift issue (XPBD doesn't propagate external forces through joints as well as MuJoCo's solver). When switching:
1. Call `newton.solvers.SolverMuJoCo.register_custom_attributes(builder)` before finalize
2. Replace `model.collide()` with either `use_mujoco_contacts=True` (simplest) or `CollisionPipeline`
3. Update solver constructor

### Joint Frame Transform Source

All joint transforms were extracted from the USD file using `UsdPhysics.Joint.GetLocalPos0Attr()` / `GetLocalRot0Attr()` etc. These encode where each joint anchor sits relative to its parent and child body origins. The values are in meters and match the CAD geometry.

### Future: Multi-World Batching

The `build_osprey()` function returns an unfinalized builder specifically so that the future RL environment can do:
```python
osprey = build_osprey(cfg)
scene = newton.ModelBuilder()
scene.add_ground_plane()
scene.replicate(osprey, world_count=512)
model = scene.finalize()
```

This is the standard Newton pattern for GPU-batched simulation.

### Risk: Body Index Shift from add_world()

When `scene.add_world(osprey)` is called, the osprey bodies are added to the scene builder. If `add_ground_plane()` is called first, it may add a ground body at index 0, shifting all osprey body indices by 1. The ground plane body index is -1 (static), so this should NOT shift indices. But verify by printing `model.body_mass.numpy()` after finalize.

### Confidence Score: 8/10

High confidence because:
- All joint transforms extracted from verified USD
- API signatures confirmed from source code
- Pattern matches working Newton examples
- Existing controller code doesn't need changes

Risk factors:
- Joint anchor transforms may need minor adjustment if body origins differ between USD and programmatic construction
- `add_world()` body index mapping needs runtime verification
