"""Interactive hover validation script for the Osprey aerial manipulator.

Builds the model programmatically via build_osprey(), runs the INDI flight
controller, and provides an interactive viewer with GUI sliders for thrust,
body rates, and arm positions. Use this to verify that:
  - Hover thrust matches mg (~8.4N)
  - INDI controller stabilizes the drone
  - Arm movement causes visible but compensated coupling
"""

from __future__ import annotations

import math

import torch
import warp as wp

import newton

from controllers.config import OspreyConfig, default_osprey_config
from controllers.indi import IndiController
from controllers.math_utils import quat_rotate, quat_rotate_inverse
from controllers.motor_model import RotorMotor


class HoverValidator:
    """Interactive drone hover validation with Newton viewer.

    Builds the Osprey model programmatically, initializes controllers, and runs
    a simulation loop with GUI controls for manual flight testing.
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

        # RC controller input (optional -- falls back to GUI-only)
        from controllers.rc_input import RCInput, RCInputConfig

        self.rc = RCInput(RCInputConfig(), frame_dt=self.cfg.sim.frame_dt)

    @staticmethod
    def _add_obstacles(scene: newton.ModelBuilder) -> None:
        """Add an obstacle course to the scene for RC flying."""
        obstacle_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, collision_group=0
        )

        def _pillar(x: float, y: float, height: float, radius: float = 0.15) -> None:
            b = scene.add_body(
                xform=wp.transform(p=wp.vec3(x, y, height / 2), q=wp.quat_identity()),
                is_kinematic=True,
                label=f"pillar_{x:.0f}_{y:.0f}",
            )
            scene.add_shape_cylinder(
                body=b, radius=radius, half_height=height / 2, cfg=obstacle_cfg
            )

        def _box(x: float, y: float, z: float, hx: float, hy: float, hz: float) -> None:
            b = scene.add_body(
                xform=wp.transform(p=wp.vec3(x, y, z), q=wp.quat_identity()),
                is_kinematic=True,
                label=f"box_{x:.0f}_{y:.0f}_{z:.0f}",
            )
            scene.add_shape_box(body=b, hx=hx, hy=hy, hz=hz, cfg=obstacle_cfg)

        # Gate 1: fly-through archway at x=3
        _pillar(3.0, -1.0, 4.0)
        _pillar(3.0, 1.0, 4.0)
        _box(3.0, 0.0, 3.5, 0.15, 1.15, 0.15)  # lintel

        # Gate 2: offset archway at x=6
        _pillar(6.0, 0.5, 3.5)
        _pillar(6.0, 2.5, 3.5)
        _box(6.0, 1.5, 3.0, 0.15, 1.15, 0.15)

        # Slalom pillars
        _pillar(1.5, 1.5, 3.0, 0.2)
        _pillar(4.5, -1.5, 3.0, 0.2)
        _pillar(7.5, 1.0, 3.5, 0.2)

        # Stacked crates cluster
        _box(9.0, 0.0, 0.4, 0.4, 0.4, 0.4)
        _box(9.0, 0.0, 1.2, 0.35, 0.35, 0.35)
        _box(9.0, 0.8, 0.3, 0.3, 0.3, 0.3)

        # Floating ring approximation (4 capsules in a square) at x=5, z=2.5
        ring_x, ring_y, ring_z = 5.0, -1.0, 2.5
        ring_r = 0.08
        ring_span = 0.8
        for dx, dy, rot in [
            (0, ring_span, (0.0, 0.0, 0.0, 1.0)),       # top bar (along Y)
            (0, -ring_span, (0.0, 0.0, 0.0, 1.0)),      # bottom bar
            (ring_span, 0, (0.0, 0.0, 0.7071, 0.7071)),  # right bar (along X, rotated 90 around Z)
            (-ring_span, 0, (0.0, 0.0, 0.7071, 0.7071)), # left bar
        ]:
            b = scene.add_body(
                xform=wp.transform(
                    p=wp.vec3(ring_x, ring_y + dy, ring_z + dx),
                    q=wp.quat(*rot),
                ),
                is_kinematic=True,
            )
            scene.add_shape_capsule(
                body=b, radius=ring_r, half_height=ring_span - ring_r, cfg=obstacle_cfg
            )

        # Low wall to hop over
        _box(2.0, -3.0, 0.5, 0.15, 1.5, 0.5)

        # Tall narrow gap
        _pillar(8.0, -2.0, 5.0, 0.12)
        _pillar(8.0, -1.2, 5.0, 0.12)

    def _build_model(self) -> None:
        """Build Newton model programmatically via build_osprey()."""
        from controllers.osprey_model import build_osprey

        osprey = build_osprey(self.cfg, spawn_pos=(0.0, 0.0, self.cfg.sim.spawn_height))

        scene = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
        scene.add_ground_plane()
        scene.add_world(osprey)
        self._add_obstacles(scene)

        self.model = scene.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None  # MuJoCo uses its own contact pipeline

        # Compute initial body positions from joint chain
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_0
        )

        self.solver = newton.solvers.SolverMuJoCo(
            self.model, iterations=self.cfg.sim.solver_iterations
        )

        # Print mass verification
        masses = self.model.body_mass.numpy()
        total = sum(float(masses[i]) for i in range(9))
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

            # Physics step (MuJoCo handles contacts internally)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.cfg.sim.frame_dt

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_0
        )
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
        if self.rc.connected:
            ui.text("RC: Connected (USB)")
        else:
            ui.text("RC: Not connected (GUI only)")
        ui.separator()
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
            # Poll RC controller (non-blocking, no-op if disconnected)
            if self.rc.connected:
                self.rc.poll()
                self.cmd_thrust = self.rc.thrust
                self.cmd_roll_rate = self.rc.roll_rate
                self.cmd_pitch_rate = self.rc.pitch_rate
                self.cmd_yaw_rate = self.rc.yaw_rate

            if self.viewer.is_key_down("r") or self.reset_requested:
                self.reset()
            elif not self.viewer.is_paused():
                self.step()

            # Camera follows drone position and yaw (skip on NaN)
            body_q = self.state_0.body_q.numpy()
            drone_pos = body_q[self.cfg.body.base][:3]
            if not math.isnan(drone_pos[0]):
                qx, qy, qz, qw = body_q[self.cfg.body.base][3:]
                drone_yaw = math.atan2(
                    2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz),
                )

                # Orbit camera behind the drone
                dist_back = 0.8
                dist_up = 0.5
                cam_x = drone_pos[0] - dist_back * math.cos(drone_yaw)
                cam_y = drone_pos[1] - dist_back * math.sin(drone_yaw)
                cam_z = drone_pos[2] + dist_up
                self.viewer.set_camera(
                    pos=wp.vec3(cam_x, cam_y, cam_z),
                    pitch=-20.0,
                    yaw=math.degrees(drone_yaw),
                )

            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()

        print(f"Simulation ended. Total time: {self.sim_time:.2f} s")


if __name__ == "__main__":
    validator = HoverValidator()
    validator.run()
