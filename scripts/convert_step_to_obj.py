"""Convert STEP CAD files to OBJ meshes for Newton physics engine.

One-time offline conversion. Run with:
    uv run --with cadquery python scripts/convert_step_to_obj.py

The STEP files are in the reference_code directory (original CAD exports).
OBJ files are written to assets/meshes/ and should be committed to the repo.
"""

from __future__ import annotations

from pathlib import Path

import cadquery as cq
import trimesh

STEP_DIR = Path("reference_code/osprey_rl/osprey_rl/assets/step_and_usd_of_parts")
OUT_DIR = Path("assets/meshes")

COMPONENTS = [
    "link_body",
    "link_arm",
    "link_differential",
    "link_finger_left",
    "link_finger_right",
    "link_rotor_front_left",
    "link_rotor_front_right",
    "link_rotor_back_left",
    "link_rotor_back_right",
]

# Tessellation tolerance in mm (lower = more detail, more faces)
TESSELLATION_TOLERANCE = 0.5

# Maximum faces per component (decimate if exceeded)
MAX_FACES = 50_000


def convert_step_to_obj(
    step_path: Path, obj_path: Path, tolerance: float, max_faces: int
) -> trimesh.Trimesh:
    """Convert a single STEP file to OBJ.

    Uses cadquery to import STEP → export STL → load with trimesh → save OBJ.
    cadquery.importStep returns a Workplane; we export to STL as intermediate.

    Args:
        step_path: Path to input STEP file.
        obj_path: Path to output OBJ file.
        tolerance: Tessellation tolerance (mm).
        max_faces: Maximum face count (decimates if exceeded).

    Returns:
        The resulting trimesh.
    """
    import tempfile

    result = cq.importers.importStep(str(step_path))

    # Export to STL via cadquery (handles tessellation internally)
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = tmp.name
    cq.exporters.export(result, tmp_path, exportType="STL", tolerance=tolerance)

    # Load STL with trimesh
    mesh = trimesh.load(tmp_path)
    Path(tmp_path).unlink()

    if len(mesh.faces) > max_faces:
        ratio = 1.0 - max_faces / len(mesh.faces)
        mesh = mesh.simplify_quadric_decimation(ratio)

    mesh.export(str(obj_path))
    return mesh


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(COMPONENTS)} STEP files to OBJ...")
    print(f"  Source: {STEP_DIR}")
    print(f"  Output: {OUT_DIR}")
    print(f"  Tolerance: {TESSELLATION_TOLERANCE}mm, max faces: {MAX_FACES}")
    print()

    for name in COMPONENTS:
        step_path = STEP_DIR / f"{name}.step"
        obj_path = OUT_DIR / f"{name}.obj"

        if not step_path.exists():
            print(f"  SKIP {name}: {step_path} not found")
            continue

        mesh = convert_step_to_obj(
            step_path, obj_path, TESSELLATION_TOLERANCE, MAX_FACES
        )

        # Report mesh stats and bounding box (to check units)
        bbox = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        extent = bbox[1] - bbox[0]
        print(
            f"  {name}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
            f"extent=[{extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f}]"
        )

    print()
    print(
        "Done. Check extent values — if they're ~100-1000, units are mm (scale by 0.001)."
    )
    print("If they're ~0.1-1.0, units are meters (no scaling needed).")


if __name__ == "__main__":
    main()
