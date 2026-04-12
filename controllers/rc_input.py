"""RC transmitter input for flight control via USB HID.

Reads a RadioMaster Pocket (or any EdgeTX radio) connected via USB-C
in joystick mode. Uses evdev for non-blocking Linux input.
"""

from __future__ import annotations

import math
import select
from dataclasses import dataclass

try:
    from evdev import InputDevice, ecodes, list_devices

    _HAS_EVDEV = True
except ImportError:
    _HAS_EVDEV = False


@dataclass
class RCInputConfig:
    """Configuration for RC transmitter input processing.

    All axis codes reference Linux evdev ABS_* constants.
    """

    # EdgeTX USB HID identifiers
    vendor_id: int = 0x1209
    """EdgeTX USB vendor ID."""

    product_id: int = 0x4F54
    """EdgeTX USB product ID."""

    # Axis mapping (evdev ABS_* codes) -- Mode 2 AETR
    axis_roll: int = 0x00  # ABS_X -- CH1 Aileron
    """evdev axis code for roll (aileron)."""

    axis_pitch: int = 0x01  # ABS_Y -- CH2 Elevator
    """evdev axis code for pitch (elevator)."""

    axis_throttle: int = 0x02  # ABS_Z -- CH3 Throttle
    """evdev axis code for throttle."""

    axis_yaw: int = 0x03  # ABS_RX -- CH4 Rudder
    """evdev axis code for yaw (rudder)."""

    # Input processing
    deadzone: float = 0.05
    """Deadzone radius in normalized [-1, 1] space."""

    expo: float = 0.3
    """Expo curve factor. 0.0 = linear, 1.0 = full cubic."""

    smoothing_tau: float = 0.05
    """First-order low-pass time constant [s]. 0 = no smoothing."""

    # Output scaling -- command ranges matching validate_hover.py sliders
    thrust_min: float = 0.0
    """Minimum thrust output [N]."""

    thrust_max: float = 25.0
    """Maximum thrust output [N]."""

    roll_rate_max: float = 5.0
    """Maximum roll rate magnitude [rad/s]."""

    pitch_rate_max: float = 5.0
    """Maximum pitch rate magnitude [rad/s]."""

    yaw_rate_max: float = 3.0
    """Maximum yaw rate magnitude [rad/s]."""

    # Axis inversion (True = negate raw axis)
    invert_pitch: bool = False
    """Invert pitch axis (pull back = positive pitch rate)."""

    invert_roll: bool = False
    """Invert roll axis."""

    invert_yaw: bool = True
    """Invert yaw axis."""

    invert_throttle: bool = False
    """Invert throttle axis."""


class RCInput:
    """Non-blocking RC transmitter input reader.

    Discovers an EdgeTX USB joystick by VID:PID, reads axis events
    via evdev, applies expo + deadzone + smoothing, and outputs
    flight commands (thrust, roll rate, pitch rate, yaw rate).

    Falls back gracefully when evdev is unavailable or no device found.

    Args:
        config: Input processing configuration.
        frame_dt: Expected time between poll() calls [s]. Used for smoothing.
    """

    def __init__(
        self, config: RCInputConfig | None = None, frame_dt: float = 1 / 60
    ) -> None:
        self.cfg = config or RCInputConfig()
        self.frame_dt = frame_dt
        self.connected = False

        # Raw axis state (normalized to [-1, 1] or [0, 1] for throttle)
        self._raw: dict[int, float] = {}
        # Smoothed output values
        self._smoothed_roll = 0.0
        self._smoothed_pitch = 0.0
        self._smoothed_yaw = 0.0
        self._smoothed_throttle = 0.0
        # Axis normalization ranges from evdev absinfo
        self._axis_ranges: dict[int, tuple[int, int]] = {}

        self._device: InputDevice | None = None  # type: ignore[name-defined]
        self._try_connect()

    def _try_connect(self) -> None:
        """Attempt to discover and connect to an EdgeTX USB joystick."""
        if not _HAS_EVDEV:
            print("RC: evdev not installed -- falling back to GUI-only")
            return

        cfg = self.cfg
        for path in list_devices():
            dev = InputDevice(path)
            if dev.info.vendor == cfg.vendor_id and dev.info.product == cfg.product_id:
                self._device = dev
                self.connected = True

                # Read axis ranges from capabilities
                caps = dev.capabilities(absinfo=True)
                abs_caps = caps.get(ecodes.EV_ABS, [])
                for code, absinfo in abs_caps:
                    self._axis_ranges[code] = (absinfo.min, absinfo.max)

                print(f"RC: Connected to {dev.name} ({dev.path})")
                return

        print("RC: No EdgeTX controller found -- falling back to GUI-only")

    def poll(self) -> None:
        """Read all pending events and update command state.

        Non-blocking: returns immediately if no events are available.
        Should be called once per frame (~60Hz).
        """
        if not self.connected or self._device is None:
            return

        # Non-blocking check for events
        try:
            readable, _, _ = select.select([self._device], [], [], 0)
        except Exception:
            # Device disconnected
            self.connected = False
            print("RC: Controller disconnected")
            return

        if not readable:
            return

        # Drain all pending events
        try:
            while True:
                event = self._device.read_one()
                if event is None:
                    break
                if event.type == ecodes.EV_ABS and event.code in self._axis_ranges:
                    axis_min, axis_max = self._axis_ranges[event.code]
                    # Normalize to [-1, 1] for sticks, [0, 1] for throttle
                    if event.code == self.cfg.axis_throttle:
                        self._raw[event.code] = (event.value - axis_min) / (
                            axis_max - axis_min
                        )
                    else:
                        self._raw[event.code] = (
                            2.0 * (event.value - axis_min) / (axis_max - axis_min) - 1.0
                        )
        except OSError:
            # Device disconnected mid-read
            self.connected = False
            print("RC: Controller disconnected")
            return

        # Process axes: deadzone -> expo -> inversion
        cfg = self.cfg

        roll_raw = self._raw.get(cfg.axis_roll, 0.0)
        pitch_raw = self._raw.get(cfg.axis_pitch, 0.0)
        yaw_raw = self._raw.get(cfg.axis_yaw, 0.0)
        throttle_raw = self._raw.get(cfg.axis_throttle, 0.0)

        roll = self._process_stick(roll_raw, cfg.invert_roll)
        pitch = self._process_stick(pitch_raw, cfg.invert_pitch)
        yaw = self._process_stick(yaw_raw, cfg.invert_yaw)

        # Throttle: [0, 1] range, only apply inversion (no deadzone/expo)
        if cfg.invert_throttle:
            throttle_raw = 1.0 - throttle_raw

        # Apply first-order smoothing
        if cfg.smoothing_tau > 0:
            alpha = math.exp(-self.frame_dt / cfg.smoothing_tau)
        else:
            alpha = 0.0

        self._smoothed_roll = alpha * self._smoothed_roll + (1.0 - alpha) * roll
        self._smoothed_pitch = alpha * self._smoothed_pitch + (1.0 - alpha) * pitch
        self._smoothed_yaw = alpha * self._smoothed_yaw + (1.0 - alpha) * yaw
        self._smoothed_throttle = (
            alpha * self._smoothed_throttle + (1.0 - alpha) * throttle_raw
        )

    def _process_stick(self, value: float, invert: bool) -> float:
        """Apply deadzone, expo, and inversion to a stick axis.

        Args:
            value: Normalized input in [-1, 1].
            invert: Whether to negate the axis.

        Returns:
            Processed value in [-1, 1].
        """
        if invert:
            value = -value
        value = self._apply_deadzone(value, self.cfg.deadzone)
        return self._apply_expo(value, self.cfg.expo)

    @property
    def thrust(self) -> float:
        """Collective thrust command [N]. Range: [thrust_min, thrust_max]."""
        cfg = self.cfg
        return cfg.thrust_min + self._smoothed_throttle * (
            cfg.thrust_max - cfg.thrust_min
        )

    @property
    def roll_rate(self) -> float:
        """Roll rate command [rad/s]. Range: [-roll_rate_max, roll_rate_max]."""
        return self._smoothed_roll * self.cfg.roll_rate_max

    @property
    def pitch_rate(self) -> float:
        """Pitch rate command [rad/s]. Range: [-pitch_rate_max, pitch_rate_max]."""
        return self._smoothed_pitch * self.cfg.pitch_rate_max

    @property
    def yaw_rate(self) -> float:
        """Yaw rate command [rad/s]. Range: [-yaw_rate_max, yaw_rate_max]."""
        return self._smoothed_yaw * self.cfg.yaw_rate_max

    @staticmethod
    def _apply_deadzone(value: float, deadzone: float) -> float:
        """Remove deadzone and rescale remaining range to [-1, 1].

        Args:
            value: Input in [-1, 1].
            deadzone: Deadzone radius.

        Returns:
            Processed value in [-1, 1] with deadzone removed.
        """
        if abs(value) < deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - deadzone) / (1.0 - deadzone)

    @staticmethod
    def _apply_expo(value: float, expo: float) -> float:
        """Apply expo curve: output = (1 - expo) * value + expo * value^3.

        Standard RC expo formula. expo=0 is linear, expo=1 is full cubic.

        Args:
            value: Input in [-1, 1] (already deadzone-processed).
            expo: Expo factor in [0, 1].

        Returns:
            Curved value in [-1, 1].
        """
        return (1.0 - expo) * value + expo * value**3
