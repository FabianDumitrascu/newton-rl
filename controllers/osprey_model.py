"""Programmatic Osprey aerial manipulator model construction.

Builds the full 9-body articulated robot using Newton's ModelBuilder API,
replacing the broken USD-based loading. All parameters come from OspreyConfig.

The builder is returned unfinalized so callers can use scene.add_world(osprey)
or scene.replicate(osprey, N) for GPU-batched RL training.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

import newton

from controllers.config import JointFrameData, OspreyConfig


def _make_xform(frame: JointFrameData, is_parent: bool) -> wp.transform:
    """Create a Warp transform from joint frame data.

    Args:
        frame: Joint frame data with positions and XYZW quaternions.
        is_parent: If True, use parent_pos/parent_rot; otherwise child.

    Returns:
        Warp transform for the joint anchor.
    """
    if is_parent:
        pos = frame.parent_pos
        rot = frame.parent_rot_xyzw
    else:
        pos = frame.child_pos
        rot = frame.child_rot_xyzw
    return wp.transform(
        p=wp.vec3(*pos),
        q=wp.quat(*rot),
    )


def _load_visual_meshes(
    builder: newton.ModelBuilder,
    body_mesh_map: dict[int, str],
    mesh_dir: Path,
) -> None:
    """Load OBJ visual meshes and attach to bodies.

    STEP files are in mm, so vertices are scaled by 0.001 to meters.

    Args:
        builder: ModelBuilder to add shapes to.
        body_mesh_map: Mapping of body index to mesh filename (without .obj).
        mesh_dir: Directory containing OBJ files.
    """
    visual_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, collision_group=0)

    for body_idx, mesh_name in body_mesh_map.items():
        obj_path = mesh_dir / f"{mesh_name}.obj"
        if not obj_path.exists():
            print(f"  Warning: {obj_path} not found, skipping visual mesh")
            continue
        tm = trimesh.load(str(obj_path))
        vertices = (tm.vertices * 0.001).astype(np.float32)
        indices = tm.faces.flatten().astype(np.int32)
        mesh = newton.Mesh(vertices=vertices, indices=indices)
        builder.add_shape_mesh(body=body_idx, mesh=mesh, cfg=visual_cfg)


def build_osprey(
    cfg: OspreyConfig,
    spawn_pos: tuple[float, float, float] = (0.0, 0.0, 2.0),
) -> newton.ModelBuilder:
    """Build the Osprey aerial manipulator as an articulated Newton model.

    Creates 9 bodies connected by 9 joints (1 free + 4 arm/gripper + 4 rotor),
    with visual meshes from OBJ files and collision boxes on key bodies.

    The builder is returned unfinalized to support scene.add_world() or
    scene.replicate() for multi-world batching.

    Args:
        cfg: Full Osprey platform configuration.
        spawn_pos: Initial world position of the base body (x, y, z) [m].

    Returns:
        Unfinalized ModelBuilder with the complete Osprey model.
    """
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    jf = cfg.joints
    arm = cfg.arm

    rotor_mass = 0.01
    finger_mass = 0.005

    # ── Bodies ──────────────────────────────────────────────────────────

    # Body 0: base drone body
    base_mass = cfg.inertia.falcon_mass - 4 * rotor_mass
    base_inertia = tuple(
        cfg.inertia.base_inertia_diag[i] if i == j else 0.0
        for i in range(3)
        for j in range(3)
    )
    link_base = builder.add_link(
        xform=wp.transform(p=wp.vec3(*spawn_pos), q=wp.quat_identity()),
        mass=base_mass,
        com=wp.vec3(*cfg.inertia.base_com),
        inertia=wp.mat33(*base_inertia),
        lock_inertia=True,
        label="base",
    )

    # Body 1: differential (arm pitch link)
    arm_body_mass = cfg.inertia.manipulator_mass - 2 * finger_mass
    link_differential = builder.add_link(
        mass=arm_body_mass * 0.3,
        lock_inertia=True,
        label="differential",
    )

    # Body 2: arm (arm roll link)
    link_arm = builder.add_link(
        mass=arm_body_mass * 0.7,
        lock_inertia=True,
        label="arm",
    )

    # Body 3: finger left
    link_finger_left = builder.add_link(
        mass=finger_mass,
        lock_inertia=True,
        label="finger_left",
    )

    # Body 4: finger right
    link_finger_right = builder.add_link(
        mass=finger_mass,
        lock_inertia=True,
        label="finger_right",
    )

    # Body 5: rotor back left
    link_rotor_bl = builder.add_link(
        mass=rotor_mass,
        lock_inertia=True,
        label="rotor_back_left",
    )

    # Body 6: rotor back right
    link_rotor_br = builder.add_link(
        mass=rotor_mass,
        lock_inertia=True,
        label="rotor_back_right",
    )

    # Body 7: rotor front left
    link_rotor_fl = builder.add_link(
        mass=rotor_mass,
        lock_inertia=True,
        label="rotor_front_left",
    )

    # Body 8: rotor front right
    link_rotor_fr = builder.add_link(
        mass=rotor_mass,
        lock_inertia=True,
        label="rotor_front_right",
    )

    # ── Joints ─────────────────────────────────────────────────────────

    # Joint 0: FREE — world → base (floating base, 6 DOFs: tx,ty,tz,rx,ry,rz)
    j0 = builder.add_joint_free(
        parent=-1,
        child=link_base,
        label="base_free",
    )

    # Joint 1: REVOLUTE — base → differential (arm pitch, Y-axis)
    j1 = builder.add_joint_revolute(
        parent=link_base,
        child=link_differential,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=_make_xform(jf.dof_differential, is_parent=True),
        child_xform=_make_xform(jf.dof_differential, is_parent=False),
        target_ke=arm.arm_ke,
        target_kd=arm.arm_kd,
        limit_lower=-math.pi / 2,
        limit_upper=math.pi / 2,
        label="dof_differential",
    )

    # Joint 2: REVOLUTE — differential → arm (arm roll, X-axis)
    j2 = builder.add_joint_revolute(
        parent=link_differential,
        child=link_arm,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=_make_xform(jf.dof_arm, is_parent=True),
        child_xform=_make_xform(jf.dof_arm, is_parent=False),
        target_ke=arm.arm_ke,
        target_kd=arm.arm_kd,
        label="dof_arm",
    )

    # Joint 3: PRISMATIC — arm → finger_left (X-axis slide)
    j3 = builder.add_joint_prismatic(
        parent=link_arm,
        child=link_finger_left,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=_make_xform(jf.dof_finger_left, is_parent=True),
        child_xform=_make_xform(jf.dof_finger_left, is_parent=False),
        target_ke=arm.gripper_ke,
        target_kd=arm.gripper_kd,
        limit_lower=0.0,
        limit_upper=0.027,
        label="dof_finger_left",
    )

    # Joint 4: PRISMATIC — arm → finger_right (X-axis slide)
    j4 = builder.add_joint_prismatic(
        parent=link_arm,
        child=link_finger_right,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=_make_xform(jf.dof_finger_right, is_parent=True),
        child_xform=_make_xform(jf.dof_finger_right, is_parent=False),
        target_ke=arm.gripper_ke,
        target_kd=arm.gripper_kd,
        limit_lower=0.0,
        limit_upper=0.027,
        label="dof_finger_right",
    )

    # Joint 5: REVOLUTE — base → rotor_back_left (Z-axis spin, no PD)
    j5 = builder.add_joint_revolute(
        parent=link_base,
        child=link_rotor_bl,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=_make_xform(jf.rotor_back_left, is_parent=True),
        child_xform=_make_xform(jf.rotor_back_left, is_parent=False),
        target_ke=0.0,
        target_kd=0.0,
        label="dof_rotor_back_left",
    )

    # Joint 6: REVOLUTE — base → rotor_back_right (Z-axis spin, no PD)
    j6 = builder.add_joint_revolute(
        parent=link_base,
        child=link_rotor_br,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=_make_xform(jf.rotor_back_right, is_parent=True),
        child_xform=_make_xform(jf.rotor_back_right, is_parent=False),
        target_ke=0.0,
        target_kd=0.0,
        label="dof_rotor_back_right",
    )

    # Joint 7: REVOLUTE — base → rotor_front_left (Z-axis spin, no PD)
    j7 = builder.add_joint_revolute(
        parent=link_base,
        child=link_rotor_fl,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=_make_xform(jf.rotor_front_left, is_parent=True),
        child_xform=_make_xform(jf.rotor_front_left, is_parent=False),
        target_ke=0.0,
        target_kd=0.0,
        label="dof_rotor_front_left",
    )

    # Joint 8: REVOLUTE — base → rotor_front_right (Z-axis spin, no PD)
    j8 = builder.add_joint_revolute(
        parent=link_base,
        child=link_rotor_fr,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=_make_xform(jf.rotor_front_right, is_parent=True),
        child_xform=_make_xform(jf.rotor_front_right, is_parent=False),
        target_ke=0.0,
        target_kd=0.0,
        label="dof_rotor_front_right",
    )

    # ── Articulation ───────────────────────────────────────────────────

    builder.add_articulation([j0, j1, j2, j3, j4, j5, j6, j7, j8], label="osprey")

    # ── Visual Meshes ──────────────────────────────────────────────────

    mesh_dir = Path("assets/meshes")
    body_mesh_map = {
        link_base: "link_body",
        link_differential: "link_differential",
        link_arm: "link_arm",
        link_finger_left: "link_finger_left",
        link_finger_right: "link_finger_right",
        link_rotor_bl: "link_rotor_back_left",
        link_rotor_br: "link_rotor_back_right",
        link_rotor_fl: "link_rotor_front_left",
        link_rotor_fr: "link_rotor_front_right",
    }
    _load_visual_meshes(builder, body_mesh_map, mesh_dir)

    # ── Collision Shapes ───────────────────────────────────────────────

    collision_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0, mu=0.8, collision_group=1, is_visible=False
    )

    # Base body: box approximation of drone frame
    builder.add_shape_box(
        body=link_base,
        xform=wp.transform(p=wp.vec3(0.026, 0.0, 0.054), q=wp.quat_identity()),
        hx=0.10,
        hy=0.13,
        hz=0.05,
        cfg=collision_cfg,
    )

    # Finger collision boxes for future grasping contact
    builder.add_shape_box(
        body=link_finger_left,
        hx=0.024,
        hy=0.010,
        hz=0.035,
        cfg=collision_cfg,
    )
    builder.add_shape_box(
        body=link_finger_right,
        hx=0.024,
        hy=0.010,
        hz=0.035,
        cfg=collision_cfg,
    )

    return builder
