"""Interactive hover validation script for the Osprey aerial manipulator.

Loads the USD model into Newton, runs the INDI flight controller, and provides
an interactive viewer with GUI sliders for thrust, body rates, and arm positions.
Use this to verify that:
  - Hover thrust matches mg (~8.4N)
  - INDI controller stabilizes the drone
  - Arm movement causes visible but compensated coupling
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh
import warp as wp

import newton
import newton.usd

from controllers.config import OspreyConfig, default_osprey_config
from controllers.indi import IndiController
from controllers.math_utils import quat_rotate, quat_rotate_inverse
from controllers.motor_model import RotorMotor


class HoverValidator:
    """Interactive drone hover validation with Newton viewer.

    Loads the Osprey USD model, initializes controllers, and runs a simulation
    loop with GUI controls for manual flight testing.
    """

    def __init__(self, config: OspreyConfig | None = None) -> None:
        self.cfg = config or default_osprey_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build Newton model
        self._build_model()

        # Initialize controllers
        self._init_controllers()

        # GUI state
        self.cmd_thrust = self.cfg.hover_thrust
        self.cmd_roll_rate = 0.0
        self.cmd_pitch_rate = 0.0
        self.cmd_yaw_rate = 0.0
        self.cmd_arm_pitch = 0.0
        self.cmd_arm_roll = 0.0
        self.cmd_gripper = 0.0

        # Telemetry
        self.altitude = 0.0
        self.total_thrust = 0.0
        self.rotor_speeds_display = [0.0, 0.0, 0.0, 0.0]
        self.sim_time = 0.0

        # Reset flag
        self.reset_requested = False

    def _build_model(self) -> None:
        """Load USD and build Newton model with correct mass/inertia."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        spawn_xform = wp.transform(
            p=wp.vec3(0.0, 0.0, self.cfg.sim.spawn_height),
            q=wp.quat_identity(),
        )
        self.usd_result = builder.add_usd(
            "assets/flattened-osprey.usd",
            floating=True,
            xform=spawn_xform,
            enable_self_collisions=False,
        )

        # Set mass properties — USD has zero mass on all drone bodies
        bi = self.cfg.body
        rotor_mass = 0.01  # Small mass per rotor body
        finger_mass = 0.005  # Small mass per finger

        # Base drone body gets the bulk of falcon_mass
        base_mass = self.cfg.inertia.falcon_mass - 4 * rotor_mass
        builder.body_mass[bi.base] = base_mass
        builder.body_inertia[bi.base] = tuple(
            self.cfg.inertia.base_inertia_diag[i] if i == j else 0.0
            for i in range(3)
            for j in range(3)
        )
        builder.body_com[bi.base] = tuple(self.cfg.inertia.base_com)

        # Arm bodies split the manipulator mass
        arm_body_mass = self.cfg.inertia.manipulator_mass - 2 * finger_mass
        builder.body_mass[bi.differential] = arm_body_mass * 0.3
        builder.body_mass[bi.arm] = arm_body_mass * 0.7

        # Fingers
        builder.body_mass[bi.finger_left] = finger_mass
        builder.body_mass[bi.finger_right] = finger_mass

        # Rotors
        for ridx in bi.rotor_indices_ref_order:
            builder.body_mass[ridx] = rotor_mass

        # Set arm/gripper PD gains
        arm = self.cfg.arm
        # dof_differential (arm pitch)
        builder.joint_target_ke[arm.differential_dof] = arm.arm_ke
        builder.joint_target_kd[arm.differential_dof] = arm.arm_kd
        # dof_arm (arm roll)
        builder.joint_target_ke[arm.arm_dof] = arm.arm_ke
        builder.joint_target_kd[arm.arm_dof] = arm.arm_kd
        # Fingers
        builder.joint_target_ke[arm.finger_left_dof] = arm.gripper_ke
        builder.joint_target_kd[arm.finger_left_dof] = arm.gripper_kd
        builder.joint_target_ke[arm.finger_right_dof] = arm.gripper_ke
        builder.joint_target_kd[arm.finger_right_dof] = arm.gripper_kd

        # Disable PD on rotor joints (force-driven)
        for dof in range(10, 14):  # rotor DOFs
            builder.joint_target_ke[dof] = 0.0
            builder.joint_target_kd[dof] = 0.0

        # Load actual mesh geometry from OBJ files (converted from STEP).
        # STEP files are in mm, so we scale vertices by 0.001 to meters.
        mesh_dir = Path("assets/meshes")
        visual_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, collision_group=0)

        body_mesh_map = {
            bi.base: "link_body",
            bi.differential: "link_differential",
            bi.arm: "link_arm",
            bi.finger_left: "link_finger_left",
            bi.finger_right: "link_finger_right",
            bi.rotor_front_right: "link_rotor_front_right",
            bi.rotor_front_left: "link_rotor_front_left",
            bi.rotor_back_left: "link_rotor_back_left",
            bi.rotor_back_right: "link_rotor_back_right",
        }

        for body_idx, mesh_name in body_mesh_map.items():
            obj_path = mesh_dir / f"{mesh_name}.obj"
            if not obj_path.exists():
                print(f"  Warning: {obj_path} not found, skipping")
                continue
            tm = trimesh.load(str(obj_path))
            # Scale mm → meters
            vertices = (tm.vertices * 0.001).astype(np.float32)
            indices = tm.faces.flatten().astype(np.int32)
            mesh = newton.Mesh(vertices=vertices, indices=indices)
            builder.add_shape_mesh(body=body_idx, mesh=mesh, cfg=visual_cfg)

        # Collision shape on base body (convex hull for ground interaction)
        base_obj = mesh_dir / "link_body.obj"
        if base_obj.exists():
            tm_base = trimesh.load(str(base_obj))
            base_verts = (tm_base.vertices * 0.001).astype(np.float32)
            base_indices = tm_base.faces.flatten().astype(np.int32)
            base_mesh = newton.Mesh(vertices=base_verts, indices=base_indices)
            collision_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.8)
            builder.add_shape_convex_hull(
                body=bi.base, mesh=base_mesh, cfg=collision_cfg
            )

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.solver = newton.solvers.SolverXPBD(
            self.model, iterations=self.cfg.sim.solver_iterations
        )

        # Print mass verification
        masses = self.model.body_mass.numpy()
        total = sum(masses[i] for i in range(9))  # Bodies 0-8 are drone
        print(f"Drone total mass: {total:.3f} kg (expected {self.cfg.total_mass:.3f})")
        print(f"Expected hover thrust: {self.cfg.hover_thrust:.2f} N")

    def _init_controllers(self) -> None:
        """Initialize INDI controller and motor model."""
        init_omega = torch.tensor(
            self.cfg.control.init_omega, device=self.device, dtype=torch.float32
        ).unsqueeze(0)

        self.indi = IndiController(1, self.cfg, self.device)
        self.motor = RotorMotor(
            1, self.cfg.rotor, self.cfg.motor, init_omega, self.device
        )

    def _extract_state(self) -> dict:
        """Extract drone state from Newton simulation state.

        Returns body-frame angular velocity and position/orientation.
        Newton body_qd is (vx, vy, vz, wx, wy, wz) in world frame.
        Newton body_q is (px, py, pz, qx, qy, qz, qw) — XYZW quaternion.
        """
        bi = self.cfg.body.base

        # Position and orientation from body_q
        body_q = self.state_0.body_q.numpy()
        pos = body_q[bi][:3]  # (px, py, pz)
        quat_xyzw = body_q[bi][3:]  # (qx, qy, qz, qw)

        # Angular velocity: Newton stores world-frame, need body-frame
        body_qd = self.state_0.body_qd.numpy()
        # body_qd layout: (vx, vy, vz, wx, wy, wz) — linear first, angular second
        omega_world = body_qd[bi][3:]  # (wx, wy, wz)

        # Convert world-frame omega to body-frame using quat inverse rotation
        q_t = torch.tensor(
            quat_xyzw, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        omega_w_t = torch.tensor(
            omega_world, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        omega_body = quat_rotate_inverse(q_t, omega_w_t)

        return {
            "pos": pos,
            "quat_xyzw": quat_xyzw,
            "omega_body": omega_body,  # (1, 3) torch tensor on device
        }

    def _apply_rotor_forces(self, thrusts: torch.Tensor, moments: torch.Tensor) -> None:
        """Apply rotor thrust forces and torques to the base body.

        Uses the G1 matrix to compute the actual control wrench from per-rotor
        thrusts, then applies force + torque to the base body. This avoids
        computing moment arms from rotor positions (which requires precise COM
        knowledge) and instead relies on the G1 geometry that's already
        validated against the real platform.

        Args:
            thrusts: Per-rotor thrust (1, 4) [N] in reference order [FR, FL, BL, BR].
            moments: Per-rotor yaw moment (1, 4) [N*m].
        """
        bi_base = self.cfg.body.base
        body_q = self.state_0.body_q.numpy()
        body_f = self.state_0.body_f.numpy()

        # Compute actual control wrench: [total_thrust, tau_x, tau_y, tau_z]
        wrench = self.indi.G1.cpu().matmul(thrusts.squeeze(0).cpu())  # (4,)

        total_thrust = wrench[0].item()
        tau_body = (
            wrench[1:].detach().clone()
        )  # body-frame torques [tau_x, tau_y, tau_z]

        # Add yaw reaction moments to body-frame torque
        tau_body[2] += float(moments.sum())

        # Transform force and torque from body frame to world frame
        base_quat = body_q[bi_base][3:]  # xyzw
        q_base_t = torch.tensor(base_quat, dtype=torch.float32).unsqueeze(0)

        # Force: total thrust along body z-axis → world frame
        local_z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).unsqueeze(0)
        thrust_dir_world = quat_rotate(q_base_t, local_z).squeeze(0).numpy()
        force_world = thrust_dir_world * total_thrust

        # Torque: body-frame tau → world frame
        torque_world = quat_rotate(q_base_t, tau_body.unsqueeze(0)).squeeze(0).numpy()

        # Apply to base body
        body_f[bi_base][0] += force_world[0]
        body_f[bi_base][1] += force_world[1]
        body_f[bi_base][2] += force_world[2]
        body_f[bi_base][3] += torque_world[0]
        body_f[bi_base][4] += torque_world[1]
        body_f[bi_base][5] += torque_world[2]

        # Write back to Warp array
        self.state_0.body_f.assign(body_f)

    def _update_arm_targets(self) -> None:
        """Set arm and gripper joint targets from GUI commands."""
        arm = self.cfg.arm
        joint_target = self.control.joint_target_pos.numpy()
        joint_target[arm.differential_dof] = self.cmd_arm_pitch
        joint_target[arm.arm_dof] = self.cmd_arm_roll
        joint_target[arm.finger_left_dof] = self.cmd_gripper
        joint_target[arm.finger_right_dof] = self.cmd_gripper
        self.control.joint_target_pos.assign(joint_target)

    def step(self) -> None:
        """Run one frame of simulation (multiple physics substeps)."""
        dt = self.cfg.sim.sim_dt

        for _ in range(self.cfg.sim.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # Extract state
            state = self._extract_state()
            self.altitude = float(state["pos"][2])

            # Get current rotor speeds from motor model
            rotor_speeds = self.motor.current_omega.clone()

            # Compute INDI command
            collective_thrust = torch.tensor(
                [self.cmd_thrust], device=self.device, dtype=torch.float32
            )
            desired_rates = torch.tensor(
                [[self.cmd_roll_rate, self.cmd_pitch_rate, self.cmd_yaw_rate]],
                device=self.device,
                dtype=torch.float32,
            )

            target_speeds = self.indi.get_command(
                state["omega_body"],
                rotor_speeds,
                collective_thrust,
                desired_rates,
            )

            # Motor dynamics
            thrusts, moments, omega = self.motor.step(target_speeds, dt)

            # Update telemetry
            self.total_thrust = float(thrusts.sum())
            self.rotor_speeds_display = omega.squeeze(0).tolist()

            # Apply forces to Newton
            self._apply_rotor_forces(thrusts, moments)

            # Update arm joint targets
            self._update_arm_targets()

            # Physics step
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.cfg.sim.frame_dt

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.sim_time = 0.0

        # Reset controllers
        init_omega = torch.tensor(
            self.cfg.control.init_omega, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        self.motor.current_omega = init_omega.clone()
        self.indi.reset(torch.tensor([0], device=self.device), self.cfg)

        # Reset GUI commands to hover defaults
        self.cmd_thrust = self.cfg.hover_thrust
        self.cmd_roll_rate = 0.0
        self.cmd_pitch_rate = 0.0
        self.cmd_yaw_rate = 0.0
        self.cmd_arm_pitch = 0.0
        self.cmd_arm_roll = 0.0
        self.cmd_gripper = 0.0

        self.reset_requested = False
        print("Simulation reset.")

    def gui(self, ui) -> None:
        """Imgui callback for viewer GUI."""
        ui.text("=== Flight Commands ===")

        changed, self.cmd_thrust = ui.slider_float(
            "Thrust [N]", self.cmd_thrust, 0.0, 25.0
        )
        changed, self.cmd_roll_rate = ui.slider_float(
            "Roll rate [rad/s]", self.cmd_roll_rate, -5.0, 5.0
        )
        changed, self.cmd_pitch_rate = ui.slider_float(
            "Pitch rate [rad/s]", self.cmd_pitch_rate, -5.0, 5.0
        )
        changed, self.cmd_yaw_rate = ui.slider_float(
            "Yaw rate [rad/s]", self.cmd_yaw_rate, -3.0, 3.0
        )

        ui.separator()
        ui.text("=== Arm Commands ===")

        changed, self.cmd_arm_pitch = ui.slider_float(
            "Arm Pitch [rad]", self.cmd_arm_pitch, -1.57, 1.57
        )
        changed, self.cmd_arm_roll = ui.slider_float(
            "Arm Roll [rad]", self.cmd_arm_roll, -1.57, 1.57
        )
        changed, self.cmd_gripper = ui.slider_float(
            "Gripper [m]", self.cmd_gripper, 0.0, 0.027
        )

        ui.separator()
        ui.text("=== Telemetry ===")

        ui.text(f"Altitude: {self.altitude:.3f} m")
        ui.text(f"Total thrust: {self.total_thrust:.3f} N")
        ui.text(f"Expected hover: {self.cfg.hover_thrust:.2f} N")
        ui.text(
            f"Rotor speeds: [{self.rotor_speeds_display[0]:.0f}, "
            f"{self.rotor_speeds_display[1]:.0f}, "
            f"{self.rotor_speeds_display[2]:.0f}, "
            f"{self.rotor_speeds_display[3]:.0f}]"
        )
        ui.text(f"Sim time: {self.sim_time:.1f} s")

        if ui.button("Reset"):
            self.reset_requested = True

    def run(self) -> None:
        """Main viewer loop."""
        self.viewer = newton.viewer.ViewerGL()
        self.viewer.set_model(self.model)
        self.viewer.register_ui_callback(self.gui, position="side")

        # Initial camera: behind and above the drone, looking down
        self.viewer.set_camera(pos=wp.vec3(-1.0, 0.0, 3.0), pitch=-30.0, yaw=0.0)

        while self.viewer.is_running():
            if self.viewer.is_key_down("r") or self.reset_requested:
                self.reset()
            elif not self.viewer.is_paused():
                self.step()

            # Camera follows drone position
            body_q = self.state_0.body_q.numpy()
            drone_pos = body_q[self.cfg.body.base][:3]
            cam_offset = wp.vec3(-0.8, 0.0, 0.5)
            self.viewer.set_camera(
                pos=wp.vec3(
                    drone_pos[0] + cam_offset[0],
                    drone_pos[1] + cam_offset[1],
                    drone_pos[2] + cam_offset[2],
                ),
                pitch=-20.0,
                yaw=0.0,
            )

            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.log_contacts(self.contacts, self.state_0)
            self.viewer.end_frame()

        print(f"Simulation ended. Total time: {self.sim_time:.2f} s")


if __name__ == "__main__":
    validator = HoverValidator()
    validator.run()
