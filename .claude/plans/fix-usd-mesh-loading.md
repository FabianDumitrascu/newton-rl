# Feature: Fix USD Model — Load Actual Osprey Mesh Geometry

Read referenced files before implementing. Validate codebase patterns and task sanity first.

## Feature Description

The Osprey USD model (`assets/flattened-osprey.usd`) has the correct joint/body hierarchy and physics definitions but **zero mesh geometry**. All 107 component Xforms have `CollisionAPI` applied but contain no actual vertices or faces — the geometry was lost during the STEP→USD conversion. This means:
- The drone is invisible in the Newton viewer
- No collision shapes exist for interaction with objects
- We rely on manually-added primitive shapes as a workaround

This plan converts the STEP CAD files into mesh data and creates a proper asset pipeline that produces a USD file with embedded visual and collision geometry.

## User Story

As a thesis researcher
I want to see the actual Osprey drone model in the Newton viewer with proper collision shapes
So that I can visually verify drone behavior and test contact-rich manipulation tasks

## Problem Statement

The flattened USD file contains only Xform nodes with CollisionAPI metadata but no mesh vertices/faces. Newton's `add_usd()` correctly skips these empty Xforms (logging "CollisionAPI applied to an unknown UsdGeomGPrim type"). The component USD files in `reference_code/` are also empty — only the STEP files contain actual geometry.

## Solution Statement

Build an offline asset pipeline that:
1. Converts STEP files → OBJ meshes using `cadquery` (OpenCascade-based, Python-native)
2. Creates a Python script that loads the OBJ meshes and attaches them to the existing Newton model as visual + collision shapes
3. Updates `validate_hover.py` to use the real mesh geometry instead of primitive placeholders

The STEP→OBJ conversion is a one-time offline step. The mesh loading integrates into the existing model building code.

## Feature Metadata

**Feature Type**: Enhancement
**Estimated Complexity**: Medium
**Primary Systems Affected**: `testing/validate_hover.py`, new `scripts/convert_step_to_obj.py`, new `assets/meshes/`
**Dependencies**: `cadquery` (for STEP→mesh, install via `uv add --dev cadquery`), `trimesh` (already available)

---

## CONTEXT REFERENCES

### Relevant Codebase Files — MUST READ BEFORE IMPLEMENTING

- `testing/validate_hover.py` (lines 60-174) — Current `_build_model()` with primitive placeholder shapes. This is where mesh loading replaces the placeholders.
- `controllers/config.py` — `BodyIndices` class (lines ~148-175) maps body names to Newton indices. Used to attach meshes to correct bodies.
- `submodules/newton/newton/_src/geometry/types.py` — `newton.Mesh` class. Constructor: `Mesh(vertices, indices, normals=None, ...)`. This is how we create meshes from OBJ data.
- `submodules/newton/newton/examples/robot/example_robot_panda_hydro.py` (lines with `add_shape_mesh`) — Reference pattern for loading meshes and attaching to bodies.
- `.claude/references/model_builder.md` — `add_shape_mesh()` API and `ShapeConfig` reference.

### Source Geometry Files

All in `reference_code/osprey_rl/osprey_rl/assets/step_and_usd_of_parts/`:

| Component | STEP File | Size |
|-----------|-----------|------|
| link_body | `link_body.step` | 11 MB |
| link_arm | `link_arm.step` | 1.2 MB |
| link_differential | `link_differential.step` | 1.5 MB |
| link_finger_left | `link_finger_left.step` | 356 KB |
| link_finger_right | `link_finger_right.step` | 356 KB |
| link_rotor_front_left | `link_rotor_front_left.step` | 1.6 MB |
| link_rotor_front_right | `link_rotor_front_right.step` | 1.6 MB |
| link_rotor_back_left | `link_rotor_back_left.step` | 1.5 MB |
| link_rotor_back_right | `link_rotor_back_right.step` | 1.5 MB |

### Newton Mesh Loading Pattern (from panda_hydro example)

```python
import newton
import trimesh

# Load mesh from OBJ
tm = trimesh.load("assets/meshes/link_body.obj")
mesh = newton.Mesh(
    vertices=tm.vertices.astype(np.float32),
    indices=tm.faces.flatten().astype(np.int32),
)

# Attach to body with collision enabled
cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.8)
builder.add_shape_mesh(body=body_idx, mesh=mesh, cfg=cfg)
```

### Relevant Documentation

- `.claude/references/model_builder.md` — `add_shape_mesh()`, `ShapeConfig`, `add_shape_convex_hull()`
- `.claude/references/usd.md` — `newton.usd.get_mesh()` if loading from USD prim
- [cadquery docs](https://cadquery.readthedocs.io/en/latest/) — STEP import API
- [trimesh docs](https://trimesh.org/) — OBJ loading, mesh simplification

### Patterns to Follow

**Mesh attachment**: Use `builder.add_shape_mesh(body=idx, mesh=mesh, cfg=cfg)` — matches existing pattern from Newton examples.

**Visual vs collision**: Use two separate configs:
- Visual: `ShapeConfig(density=0.0, collision_group=0)` — visible, no collision
- Collision: `ShapeConfig(density=0.0, mu=0.8)` — simplified convex hull for collision

**File organization**: Converted OBJ files go in `assets/meshes/`. Conversion script in `scripts/`.

---

## IMPLEMENTATION PLAN

### Phase 1: STEP → OBJ Conversion (Offline)

Install `cadquery` as a dev dependency. Write a one-time conversion script that reads each STEP file and exports OBJ meshes. The meshes may need decimation since STEP files can produce very dense triangulations.

### Phase 2: Mesh Loading in Model Builder

Replace the primitive placeholder shapes in `validate_hover.py` with actual mesh loading. Load OBJ files via `trimesh`, create `newton.Mesh` objects, and attach them to the correct bodies using `add_shape_mesh()`.

For collision, use convex hulls of the meshes (simpler, faster) via `add_shape_convex_hull()`. For visual rendering, use the full meshes.

### Phase 3: Validation

Verify the drone is visible in the viewer with correct geometry, and that the base body collides with the ground plane.

---

## STEP-BY-STEP TASKS

### Task 1: ADD cadquery dev dependency

- **IMPLEMENT**: Add `cadquery` to dev dependencies in `pyproject.toml`
- **COMMAND**: `uv add --dev cadquery`
- **GOTCHA**: cadquery pulls in OpenCascade (~100MB). This is a dev-only dependency for asset conversion.
- **VALIDATE**: `uv run python -c "import cadquery; print('cadquery OK')"`

### Task 2: CREATE `scripts/convert_step_to_obj.py`

- **IMPLEMENT**: Script that converts all 9 STEP files to OBJ format in `assets/meshes/`
- **PATTERN**: Use cadquery to import STEP, tessellate, and export via trimesh
- **KEY CODE**:
```python
import cadquery as cq
import trimesh
import numpy as np
from pathlib import Path

STEP_DIR = Path("reference_code/osprey_rl/osprey_rl/assets/step_and_usd_of_parts")
OUT_DIR = Path("assets/meshes")

COMPONENTS = [
    "link_body", "link_arm", "link_differential",
    "link_finger_left", "link_finger_right",
    "link_rotor_front_left", "link_rotor_front_right",
    "link_rotor_back_left", "link_rotor_back_right",
]

for name in COMPONENTS:
    step_path = STEP_DIR / f"{name}.step"
    result = cq.importers.importStep(str(step_path))
    # Tessellate with tolerance (mm precision)
    verts, faces = result.tessellate(tolerance=0.5)
    vertices = np.array([(v.x, v.y, v.z) for v in verts], dtype=np.float32)
    # Scale from mm to meters if needed (check units)
    triangles = np.array(faces, dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    # Simplify if too many faces (>50k per component)
    if len(mesh.faces) > 50000:
        mesh = mesh.simplify_quadric_decimation(50000)
    mesh.export(OUT_DIR / f"{name}.obj")
    print(f"{name}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
```
- **GOTCHA**: STEP files might use mm units — check and convert to meters (Newton uses SI). cadquery returns coordinates in the STEP file's native units.
- **VALIDATE**: `uv run python scripts/convert_step_to_obj.py && ls -la assets/meshes/*.obj`

### Task 3: CREATE `assets/meshes/` directory and convert

- **IMPLEMENT**: Run the conversion script to populate `assets/meshes/`
- **VALIDATE**: All 9 OBJ files exist: `ls assets/meshes/link_*.obj | wc -l` → 9

### Task 4: UPDATE `testing/validate_hover.py` — Load real meshes

- **IMPLEMENT**: Replace the primitive placeholder shape section in `_build_model()` with mesh loading
- **PATTERN**: Mirror `example_robot_panda_hydro.py` mesh loading pattern
- **KEY CHANGES**:
  1. Add `import trimesh` and `import numpy as np` to imports
  2. Replace the "Add placeholder shapes" block (lines ~123-172) with:
```python
# Load actual mesh geometry for visual + collision
mesh_dir = Path("assets/meshes")
body_mesh_map = {
    bi.base: "link_body",
    bi.differential: "link_differential",
    bi.arm: "link_arm",
    bi.finger_left: "link_finger_left",
    bi.finger_right: "link_finger_right",
}
rotor_mesh_map = {
    bi.rotor_front_right: "link_rotor_front_right",
    bi.rotor_front_left: "link_rotor_front_left",
    bi.rotor_back_left: "link_rotor_back_left",
    bi.rotor_back_right: "link_rotor_back_right",
}

visual_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, collision_group=0)

# Base body: full mesh for visual + convex hull for collision
for body_idx, mesh_name in {**body_mesh_map, **rotor_mesh_map}.items():
    tm = trimesh.load(mesh_dir / f"{mesh_name}.obj")
    mesh = newton.Mesh(
        vertices=tm.vertices.astype(np.float32),
        indices=tm.faces.flatten().astype(np.int32),
    )
    # Visual shape (full detail)
    builder.add_shape_mesh(body=body_idx, mesh=mesh, cfg=visual_cfg)

# Collision shape on base body only (convex hull)
tm_base = trimesh.load(mesh_dir / "link_body.obj")
base_mesh = newton.Mesh(
    vertices=tm_base.vertices.astype(np.float32),
    indices=tm_base.faces.flatten().astype(np.int32),
)
collision_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.8)
builder.add_shape_convex_hull(body=bi.base, mesh=base_mesh, cfg=collision_cfg)
```
  3. Add `from pathlib import Path` to imports
- **GOTCHA**: Mesh coordinate frame must match the body-local frame in the USD. If meshes appear offset, check if STEP origin differs from USD body origin. May need a transform offset.
- **GOTCHA**: If mesh vertex count is very high (>100k per component), the viewer will be slow. Use decimated meshes or only load collision shapes initially.
- **VALIDATE**: `uv run --extra torch-cu12 python -c "from testing.validate_hover import HoverValidator; v = HoverValidator(); print(f'Shapes: {v.model.shape_count}')"` — should show >12 shapes

### Task 5: VERIFY mesh orientation and scale

- **IMPLEMENT**: Run the viewer and check if meshes are correctly positioned relative to bodies
- **GOTCHA**: Common issues:
  - Meshes offset from body origin → need to check if STEP files use a different origin than the USD body transforms
  - Meshes too large/small → STEP might be in mm, Newton expects meters (scale by 0.001)
  - Meshes rotated → STEP coordinate frame might differ from USD
- **VALIDATE**: `uv run --extra torch-cu13 python testing/validate_hover.py` — visually confirm drone shape is correct
- **FIX**: If meshes are offset/scaled wrong, add `xform` parameter to `add_shape_mesh()` and/or scale vertices during loading

### Task 6: UPDATE `.gitignore` for mesh artifacts

- **IMPLEMENT**: Add `assets/meshes/` to `.gitignore` if OBJ files should be generated (not committed), OR commit them if they should be in the repo (recommended for reproducibility)
- **DECISION**: Committing OBJ files is preferred — they're the canonical mesh representation and avoid requiring cadquery to build
- **VALIDATE**: `git status` shows mesh files correctly tracked or ignored

---

## TESTING STRATEGY

### Visual Validation (Primary)

- Launch viewer: `uv run --extra torch-cu13 python testing/validate_hover.py`
- Confirm drone shape matches the real Osprey platform
- Confirm drone doesn't fall through ground plane
- Confirm arm/rotor bodies move correctly relative to base

### Unit Validation

- Mesh file existence: `ls assets/meshes/link_*.obj | wc -l` → 9
- Shape count: model should have 9+ visual shapes plus collision shapes
- Mass unchanged: total mass still 0.86 kg

### Collision Validation

- With thrust=0: drone should fall and rest on ground plane (not pass through)
- With hover thrust: drone should hover above ground

---

## VALIDATION COMMANDS

### Level 1: Lint

```bash
uvx ruff check controllers/ testing/ scripts/ && uvx ruff format --check controllers/ testing/ scripts/
```

### Level 2: Asset Pipeline

```bash
# Convert STEP → OBJ (requires cadquery)
uv run python scripts/convert_step_to_obj.py

# Verify all meshes exist
ls assets/meshes/link_*.obj | wc -l  # expect 9
```

### Level 3: Integration

```bash
# Verify model builds with meshes
uv run --extra torch-cu12 python -c "
from testing.validate_hover import HoverValidator
v = HoverValidator()
print(f'Bodies: {v.model.body_count}, Shapes: {v.model.shape_count}')
assert v.model.shape_count > 12, 'Not enough shapes loaded'
print('OK')
"
```

### Level 4: Visual Validation

```bash
uv run --extra torch-cu13 python testing/validate_hover.py
# Check: drone visible, correct shape, doesn't fall through ground
```

---

## ACCEPTANCE CRITERIA

- [ ] All 9 STEP files converted to OBJ meshes in `assets/meshes/`
- [ ] Drone is visible in the Newton viewer with recognizable shape
- [ ] Base body collides with ground plane (doesn't fall through)
- [ ] Arm, rotors, fingers, differential visible as separate bodies
- [ ] INDI controller still stabilizes hover with mesh-loaded model
- [ ] No regressions in hover validation (thrust ~8.4N, rotor speeds stable)
- [ ] Lint passes

---

## RISKS AND MITIGATIONS

| Risk | Mitigation |
|------|------------|
| STEP units are mm, not meters | Check first mesh bounding box; scale by 0.001 if needed |
| Mesh origin doesn't match body origin | Compare mesh centroid to body position; add xform offset |
| Too many vertices → slow viewer | Decimate to 50k faces per component using trimesh |
| cadquery fails to import a STEP file | Try with lower tolerance; or use convex hull approximation |
| Convex hull collision is too coarse | Add more collision shapes (per-part convex decomposition) later |

---

## NOTES

- The STEP→OBJ conversion only needs to run once. The OBJ files should be committed to the repo so other machines don't need cadquery installed.
- For Phase 0, collision only on the base body is sufficient. Per-body collision shapes are a Phase 1 enhancement.
- The mesh coordinate frame issue (Task 5) is the highest-risk item. The STEP files may use different origins/orientations than the USD body transforms. This will be apparent immediately when viewing and can be fixed with transform offsets.
- If cadquery installation proves problematic (large dependency, build issues), an alternative is to use FreeCAD headless (`freecad-python3`) or ask the user to manually export OBJ files from their CAD tool.
