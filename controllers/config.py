"""Structured platform configuration for the Osprey aerial manipulator.

All physical parameters are defined here as dataclasses. Values are sourced from
the real platform specs and the reference codebase (reference_code/osprey_rl/).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RotorConfig:
    """Per-rotor thrust and geometry parameters.

    Rotor order: [front_right, front_left, back_left, back_right].
    Front rotors are larger to compensate for manipulator payload.
    """

    thrust_coeff: list[float] = field(
        default_factory=lambda: [3.1e-6, 3.1e-6, 1.5e-6, 1.5e-6]
    )
    """Thrust coefficient C_T per rotor [N/(rad/s)^2]."""

    moment_constant: float = 0.022
    """Ratio of reaction torque to thrust (kappa)."""

    directions: list[float] = field(default_factory=lambda: [1.0, -1.0, 1.0, -1.0])
    """Spin direction per rotor (+1 CW, -1 CCW)."""

    tilt_angles_deg: list[float] = field(default_factory=lambda: [53.53, 41.42])
    """Tilt angles [front_beta, back_beta] in degrees."""

    arm_lengths: list[float] = field(default_factory=lambda: [0.143, 0.1133])
    """Motor arm lengths [front_l, back_l] in meters."""

    omega_min: float = 150.0
    """Minimum rotor speed [rad/s]."""

    omega_max: float = 2800.0
    """Maximum rotor speed [rad/s]."""

    thrust_max: float = 6.25
    """Maximum thrust per rotor [N]."""

    # Body indices in Newton model — mapped from USD discovery
    # Order: [front_right, front_left, back_left, back_right]
    body_indices: list[int] = field(default_factory=lambda: [8, 7, 5, 6])


@dataclass
class MotorConfig:
    """First-order motor dynamics parameters."""

    tau_up: float = 0.033
    """Time constant for spin-up [s]."""

    tau_down: float = 0.033
    """Time constant for spin-down [s]."""

    motor_inertia_z: float = 9.3575e-6
    """Motor rotor inertia about spin axis [kg*m^2]."""


@dataclass
class InertiaConfig:
    """Mass and inertia properties of the platform."""

    falcon_mass: float = 0.660
    """Base drone mass (without manipulator) [kg]."""

    manipulator_mass: float = 0.200
    """Manipulator arm + gripper mass [kg]."""

    base_inertia_diag: list[float] = field(
        default_factory=lambda: [0.00254, 0.00271, 0.00515]
    )
    """Diagonal inertia tensor of the base drone [kg*m^2]."""

    manipulator_inertia_diag: list[float] = field(
        default_factory=lambda: [0.0001, 0.0001, 0.0001]
    )
    """Diagonal inertia tensor of the manipulator [kg*m^2]."""

    manipulator_cog: list[float] = field(default_factory=lambda: [0.075, 0.0, 0.0])
    """Manipulator center of gravity in arm frame [m]."""

    manipulator_root: list[float] = field(default_factory=lambda: [0.0557, 0.0, -0.016])
    """Manipulator attachment point in body frame [m]."""

    base_com: list[float] = field(default_factory=lambda: [0.0506, 0.0, 0.0963])
    """Base body center of mass in body-local frame [m].

    Computed so that rotor thrust moment arms produce zero net pitch/roll torque
    at hover. The front rotors produce more thrust and are farther from center,
    so the COM is shifted forward and upward to compensate.
    """

    @property
    def total_mass(self) -> float:
        return self.falcon_mass + self.manipulator_mass


@dataclass
class FilterConfig:
    """Low-pass filter parameters for INDI controller."""

    cutoff_freq: float = 12.0
    """Butterworth filter cutoff frequency [Hz]."""


@dataclass
class ControlConfig:
    """INDI control gains."""

    k_alpha_cmd: list[float] = field(default_factory=lambda: [100.0, 100.0, 10.0])
    """Angular acceleration command gains [roll, pitch, yaw]."""

    init_omega: list[float] = field(
        default_factory=lambda: [935.0, 935.0, 1002.0, 1002.0]
    )
    """Initial rotor speeds for hover [rad/s]. Order: FR, FL, BL, BR."""


@dataclass
class SimConfig:
    """Simulation timing and solver parameters."""

    fps: int = 60
    """Viewer frame rate [Hz]."""

    sim_substeps: int = 5
    """Physics substeps per frame. sim_freq = fps * sim_substeps."""

    solver_iterations: int = 10
    """Constraint solver iterations per substep."""

    gravity: float = 9.81
    """Gravitational acceleration [m/s^2]."""

    spawn_height: float = 2.0
    """Initial drone height above ground [m]."""

    @property
    def frame_dt(self) -> float:
        return 1.0 / self.fps

    @property
    def sim_dt(self) -> float:
        return self.frame_dt / self.sim_substeps

    @property
    def sim_freq(self) -> float:
        """Physics simulation frequency [Hz]."""
        return self.fps * self.sim_substeps


@dataclass
class ArmConfig:
    """PD gains for arm and gripper joints."""

    arm_ke: float = 500.0
    """Arm joint position stiffness [N*m/rad]."""

    arm_kd: float = 50.0
    """Arm joint velocity damping [N*m*s/rad]."""

    gripper_ke: float = 2000.0
    """Gripper joint position stiffness [N/m]."""

    gripper_kd: float = 200.0
    """Gripper joint velocity damping [N*s/m]."""

    # DOF indices in Newton model (from joint_qd_start)
    differential_dof: int = 6
    """qd_start index for dof_differential (arm pitch)."""

    arm_dof: int = 7
    """qd_start index for dof_arm (arm roll)."""

    finger_left_dof: int = 8
    """qd_start index for dof_finger_left."""

    finger_right_dof: int = 9
    """qd_start index for dof_finger_right."""


@dataclass
class BodyIndices:
    """Newton model body indices from USD discovery."""

    base: int = 0
    """Drone base body index."""

    differential: int = 1
    """Differential link body index."""

    arm: int = 2
    """Arm link body index."""

    finger_left: int = 3
    """Left finger body index."""

    finger_right: int = 4
    """Right finger body index."""

    rotor_back_left: int = 5
    """Back-left rotor body index."""

    rotor_back_right: int = 6
    """Back-right rotor body index."""

    rotor_front_left: int = 7
    """Front-left rotor body index."""

    rotor_front_right: int = 8
    """Front-right rotor body index."""

    @property
    def rotor_indices_ref_order(self) -> list[int]:
        """Rotor body indices in reference code order: FR, FL, BL, BR."""
        return [
            self.rotor_front_right,
            self.rotor_front_left,
            self.rotor_back_left,
            self.rotor_back_right,
        ]


@dataclass
class JointFrameData:
    """Anchor transform for a single joint.

    Positions in meters, quaternions in XYZW format (Warp convention).
    """

    parent_pos: tuple[float, float, float]
    parent_rot_xyzw: tuple[float, float, float, float]
    child_pos: tuple[float, float, float]
    child_rot_xyzw: tuple[float, float, float, float]


@dataclass
class JointFrames:
    """Joint anchor transforms for all Osprey joints.

    Values extracted from the USD file's localPos0/localRot0/localPos1/localRot1.
    Quaternions converted from USD WXYZ to Warp XYZW format.
    """

    dof_differential: JointFrameData = field(
        default_factory=lambda: JointFrameData(
            parent_pos=(0.06968, 0.0, 0.04771),
            parent_rot_xyzw=(0.0, 0.0, 0.0, 1.0),
            child_pos=(0.01458, 0.0, 0.00209),
            child_rot_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
    )
    dof_arm: JointFrameData = field(
        default_factory=lambda: JointFrameData(
            parent_pos=(-0.00049, 0.0, -0.00013),
            parent_rot_xyzw=(0.0, 0.0, 0.0, 1.0),
            child_pos=(0.0, 0.0, 0.0),
            child_rot_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
    )
    dof_finger_left: JointFrameData = field(
        default_factory=lambda: JointFrameData(
            parent_pos=(0.15011, 0.04307, -0.00043),
            parent_rot_xyzw=(-0.5, 0.5, -0.5, 0.5),  # USD WXYZ (0.5,-0.5,0.5,-0.5)
            child_pos=(0.0, 0.0, 0.0),
            child_rot_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
    )
    dof_finger_right: JointFrameData = field(
        default_factory=lambda: JointFrameData(
            parent_pos=(0.15011, -0.04307, -0.00043),
            parent_rot_xyzw=(
                0.70711,
                0.70711,
                0.0,
                0.0,
            ),  # USD WXYZ (~0,0.70711,0.70711,~0)
            child_pos=(0.0, 0.0, 0.0),
            child_rot_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
    )
    rotor_front_right: JointFrameData = field(
        default_factory=lambda: JointFrameData(
            parent_pos=(0.11101, -0.11367, 0.09629),
            parent_rot_xyzw=(0.0, 0.0, 0.0, 1.0),
            child_pos=(0.0, 0.0, 0.0),
            child_rot_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
    )
    rotor_front_left: JointFrameData = field(
        default_factory=lambda: JointFrameData(
            parent_pos=(0.11101, 0.11367, 0.09629),
            parent_rot_xyzw=(0.0, 0.0, 0.0, 1.0),
            child_pos=(0.0, 0.0, 0.0),
            child_rot_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
    )
    rotor_back_left: JointFrameData = field(
        default_factory=lambda: JointFrameData(
            parent_pos=(-0.05812, 0.07620, 0.09629),
            parent_rot_xyzw=(0.0, 0.0, 0.0, 1.0),
            child_pos=(0.0, 0.0, 0.0),
            child_rot_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
    )
    rotor_back_right: JointFrameData = field(
        default_factory=lambda: JointFrameData(
            parent_pos=(-0.05812, -0.07620, 0.09629),
            parent_rot_xyzw=(0.0, 0.0, 0.0, 1.0),
            child_pos=(0.0, 0.0, 0.0),
            child_rot_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
    )


@dataclass
class OspreyConfig:
    """Top-level configuration container for the Osprey aerial manipulator."""

    rotor: RotorConfig = field(default_factory=RotorConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    inertia: InertiaConfig = field(default_factory=InertiaConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    arm: ArmConfig = field(default_factory=ArmConfig)
    body: BodyIndices = field(default_factory=BodyIndices)
    joints: JointFrames = field(default_factory=JointFrames)

    @property
    def total_mass(self) -> float:
        return self.inertia.total_mass

    @property
    def hover_thrust(self) -> float:
        """Expected total thrust for hover [N]."""
        return self.total_mass * self.sim.gravity


def default_osprey_config() -> OspreyConfig:
    """Create the default Osprey configuration with real platform parameters.

    Returns:
        OspreyConfig with all parameters set to values from the real platform.
    """
    return OspreyConfig()


if __name__ == "__main__":
    config = default_osprey_config()
    hover = config.hover_thrust
    print(f"Total mass: {config.total_mass:.3f} kg")
    print(f"Expected hover thrust: {hover:.2f} N")
    # Verify: 2 * 3.1e-6 * 935^2 + 2 * 1.5e-6 * 1002^2 ≈ 8.43 N
    ct = config.rotor.thrust_coeff
    omega = config.control.init_omega
    computed = sum(c * w**2 for c, w in zip(ct, omega))
    print(f"Thrust from init_omega: {computed:.2f} N")
    print(f"Sim freq: {config.sim.sim_freq:.0f} Hz")
    assert abs(hover - 8.43) < 0.1, f"Hover thrust mismatch: {hover}"
    print("Config validation passed.")
