# Feature: RadioMaster Pocket RC Controller Input

The following plan should be complete, but validate documentation and codebase patterns before implementing.

Pay special attention to naming of existing utils, types, and models. Import from the right files.

## Feature Description

Connect a RadioMaster Pocket RC transmitter to the PC via USB-C and use it as a flight controller for the Osprey drone in the Newton simulator. The 4 stick axes (throttle, roll, pitch, yaw) control flight via the existing INDI controller pipeline. Arm and gripper remain on GUI sliders. The implementation creates a reusable `controllers/rc_input.py` module that gracefully falls back to GUI-only when no controller is connected.

## User Story

As a thesis researcher
I want to fly my simulated drone with my RadioMaster Pocket RC transmitter
So that I can intuitively validate flight dynamics and INDI controller behavior using real RC stick feel

## Problem Statement

The current hover validation script (`testing/validate_hover.py`) only accepts input via imgui sliders, which are imprecise and unintuitive for evaluating flight dynamics. A real RC transmitter provides analog stick input with proper feel, enabling meaningful pilot-in-the-loop validation before RL training.

## Solution Statement

Use the `evdev` Python library to read the RadioMaster Pocket's USB HID joystick input (4 flight axes) in a non-blocking poll loop at 60Hz. Apply expo curves and deadzone filtering, then write the resulting commands to the existing `cmd_thrust`, `cmd_roll_rate`, `cmd_pitch_rate`, `cmd_yaw_rate` variables in `HoverValidator`. The GUI sliders update to reflect joystick values, providing visual feedback. When no controller is detected, the system prints a warning and falls back to GUI-only operation.

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: Medium
**Primary Systems Affected**: `controllers/` module, `testing/validate_hover.py`
**Dependencies**: `evdev` (Python package, Linux-only)

---

## CONTEXT REFERENCES

### Relevant Codebase Files — MUST READ BEFORE IMPLEMENTING

- `testing/validate_hover.py` (full file) — Why: Primary integration target. Command variables at lines 42-48, step() at lines 196-247, gui() at lines 277-323, run() loop at lines 324-357. Joystick polling inserts into run() before step().
- `controllers/config.py` (full file) — Why: OspreyConfig dataclass pattern to follow for RCInputConfig. SimConfig at lines 126-155 shows property pattern. Command ranges defined by slider bounds in validate_hover.py gui().
- `controllers/low_pass_filter.py` (full file) — Why: Existing Butterworth IIR filter. Do NOT reuse for joystick — it's designed for INDI's 300Hz signal processing. The joystick needs a simpler first-order filter at 60Hz.
- `controllers/motor_model.py` (lines 54-91) — Why: Shows the project's pattern for first-order dynamics (exponential smoothing). Mirror this alpha = exp(-dt/tau) pattern for joystick smoothing.
- `submodules/newton/newton/_src/solvers/kamino/examples/rl/joystick.py` (full file) — Why: Newton's existing joystick reference. Shows deadband, low-pass filtering, keyboard fallback, and reset edge-detection patterns. Adapt architecture but use evdev instead of xbox360controller.

### New Files to Create

- `controllers/rc_input.py` — Reusable RC transmitter input module (RCInput class + RCInputConfig dataclass)

### Files to Modify

- `testing/validate_hover.py` — Add joystick initialization and polling to the run() loop
- `pyproject.toml` — Add `evdev` as optional dependency

### Relevant Documentation — READ BEFORE IMPLEMENTING

- [python-evdev tutorial](https://python-evdev.readthedocs.io/en/latest/tutorial.html) — Core API: InputDevice, list_devices(), read_one(), ecodes
- [python-evdev apidoc](https://python-evdev.readthedocs.io/en/latest/apidoc.html) — InputDevice.info (vendor, product), capabilities(absinfo=True) for axis ranges
- [EdgeTX USB Joystick docs](https://manual.edgetx.org/color-radios/model-settings/model-setup/usb-joystick) — EdgeTX joystick mode setup, axis mapping
- [EdgeTX Joystick Mapping for Developers](https://manual.edgetx.org/edgetx-how-to/joystick-mapping-information-for-game-developers) — HID axis codes, 11-bit resolution, channel-to-axis mapping

### Patterns to Follow

**Dataclass Config Pattern** (from `controllers/config.py`):
```python
@dataclass
class SomeConfig:
    """Docstring with SI units."""
    param: float = 0.5
    """Inline docstring for the field."""
```

**Type Hints**: PEP 604 unions (`X | None`), all public functions typed.

**Naming**: snake_case for files/functions, CamelCase for classes. Prefix descriptive: `RCInput` not `InputRC`.

**No premature optimization**: Plain Python, no Warp kernels.

**Import style**: Absolute imports from `controllers.` package.

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation — RCInputConfig dataclass + evdev dependency

Add evdev as an optional dependency and create the configuration dataclass with all tunable parameters (deadzone, expo, axis mapping, scaling ranges).

### Phase 2: Core — RCInput class

Implement the reusable RC input reader: device discovery, non-blocking polling, expo + deadzone processing, first-order smoothing, and graceful fallback.

### Phase 3: Integration — Wire into validate_hover.py

Initialize RCInput in HoverValidator, poll in the run() loop, update cmd_* variables, and let GUI sliders reflect joystick state.

### Phase 4: Validation

Test with actual hardware, verify axis mapping, confirm fallback behavior.

---

## STEP-BY-STEP TASKS

### Task 1: UPDATE `pyproject.toml` — Add evdev optional dependency

Add `evdev` as an optional dependency under a new `rc` extra. Keep it optional since it's Linux-only and not needed for training.

- **IMPLEMENT**: Add `[project.optional-dependencies]` entry: `rc = ["evdev>=1.7"]`
- **PATTERN**: Follow existing optional deps pattern at lines 22-24 (torch-cu12, torch-cu13)
- **VALIDATE**: `uv sync --extra rc && uv run python -c "import evdev; print(evdev.__version__)"`

### Task 2: CREATE `controllers/rc_input.py` — RCInputConfig dataclass

Create the config dataclass with all tunable parameters.

```python
"""RC transmitter input for flight control via USB HID.

Reads a RadioMaster Pocket (or any EdgeTX radio) connected via USB-C
in joystick mode. Uses evdev for non-blocking Linux input.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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

    # Axis mapping (evdev ABS_* codes) — Mode 2 AETR
    axis_roll: int = 0x00        # ABS_X — CH1 Aileron
    """evdev axis code for roll (aileron)."""

    axis_pitch: int = 0x01       # ABS_Y — CH2 Elevator
    """evdev axis code for pitch (elevator)."""

    axis_throttle: int = 0x02    # ABS_Z — CH3 Throttle
    """evdev axis code for throttle."""

    axis_yaw: int = 0x03         # ABS_RX — CH4 Rudder
    """evdev axis code for yaw (rudder)."""

    # Input processing
    deadzone: float = 0.05
    """Deadzone radius in normalized [-1, 1] space."""

    expo: float = 0.3
    """Expo curve factor. 0.0 = linear, 1.0 = full cubic."""

    smoothing_tau: float = 0.05
    """First-order low-pass time constant [s]. 0 = no smoothing."""

    # Output scaling — command ranges matching validate_hover.py sliders
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
    invert_pitch: bool = True
    """Invert pitch axis (pull back = positive pitch rate)."""

    invert_roll: bool = False
    """Invert roll axis."""

    invert_yaw: bool = False
    """Invert yaw axis."""

    invert_throttle: bool = False
    """Invert throttle axis."""
```

- **PATTERN**: Mirror `controllers/config.py` dataclass style (inline docstrings, SI units)
- **GOTCHA**: evdev `ABS_X=0x00`, `ABS_Y=0x01`, `ABS_Z=0x02`, `ABS_RX=0x03` — these are integer codes, not string names. Use int literals so the config works without importing evdev.
- **VALIDATE**: `uv run python -c "from controllers.rc_input import RCInputConfig; c = RCInputConfig(); print(c)"`

### Task 3: CREATE `controllers/rc_input.py` — RCInput class (device discovery + polling)

Add the main RCInput class to the same file, below RCInputConfig.

```python
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
```

Key implementation details:

**Device discovery** (`__init__`):
```python
def __init__(self, config: RCInputConfig | None = None, frame_dt: float = 1/60) -> None:
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

    self._device = None
    self._try_connect()
```

**`_try_connect()`**: Iterate `evdev.list_devices()`, match VID:PID, read `capabilities(absinfo=True)` to get min/max per axis. Store in `_axis_ranges`. Print connection status.

**`poll()`**: The main per-frame method:
1. Non-blocking read via `select.select([self._device], [], [], 0)`
2. Drain all pending events in a while loop using `read_one()`
3. For `EV_ABS` events with matching axis codes, normalize to [-1, 1] using stored ranges
4. Apply `_process_axis()` (deadzone → expo → inversion) to each flight axis
5. Apply first-order smoothing: `smoothed = alpha * smoothed + (1 - alpha) * processed` where `alpha = exp(-frame_dt / tau)`
6. Return nothing — callers read properties

**Properties** (read by HoverValidator):
```python
@property
def thrust(self) -> float:
    """Collective thrust command [N]. Range: [thrust_min, thrust_max]."""

@property
def roll_rate(self) -> float:
    """Roll rate command [rad/s]. Range: [-roll_rate_max, roll_rate_max]."""

@property
def pitch_rate(self) -> float:
    """Pitch rate command [rad/s]. Range: [-pitch_rate_max, pitch_rate_max]."""

@property
def yaw_rate(self) -> float:
    """Yaw rate command [rad/s]. Range: [-yaw_rate_max, yaw_rate_max]."""
```

**Static helpers**:

```python
@staticmethod
def _apply_deadzone(value: float, deadzone: float) -> float:
    """Remove deadzone and rescale remaining range to [0, 1].

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
    return (1.0 - expo) * value + expo * value ** 3
```

- **PATTERN**: Mirror `controllers/motor_model.py` class structure (docstring, __init__, step/poll, reset)
- **IMPORTS**: `from __future__ import annotations`, `import math`, `import select` at top. `evdev` imported inside `_try_connect()` with try/except ImportError for graceful fallback.
- **GOTCHA**: evdev import must be guarded — the module should be importable even without evdev installed (fallback mode). Use lazy import pattern:
  ```python
  try:
      from evdev import InputDevice, ecodes, list_devices
      _HAS_EVDEV = True
  except ImportError:
      _HAS_EVDEV = False
  ```
  Put this at module level so the flag is available everywhere.
- **GOTCHA**: Throttle normalization is different from other axes. Throttle goes [0, 1] (bottom=0, top=1), not [-1, 1]. Normalize as: `(value - min) / (max - min)`. Other axes: `2 * (value - min) / (max - min) - 1`.
- **GOTCHA**: The `select` import is from the Python stdlib, not evdev.
- **VALIDATE**: `uv run --extra rc python -c "from controllers.rc_input import RCInput; rc = RCInput(); print(f'Connected: {rc.connected}')"` (should print False if no controller plugged in, no crash)

### Task 4: UPDATE `testing/validate_hover.py` — Initialize RCInput

Add joystick initialization to `HoverValidator.__init__()`.

- **IMPLEMENT**: After line 57 (`self.reset_requested = False`), add:
  ```python
  # RC controller input (optional — falls back to GUI-only)
  from controllers.rc_input import RCInput, RCInputConfig
  self.rc = RCInput(RCInputConfig(), frame_dt=self.cfg.sim.frame_dt)
  ```
- **GOTCHA**: Import inside __init__ to keep the import optional (validate_hover.py should work without evdev installed).
- **VALIDATE**: `uv run python testing/validate_hover.py` should still launch without evdev installed (graceful fallback).

### Task 5: UPDATE `testing/validate_hover.py` — Poll joystick in run() loop

Add joystick polling to the main viewer loop, before the step/reset logic.

- **IMPLEMENT**: In `run()` method, insert after `while self.viewer.is_running():` (line 333) and before the reset check (line 334):
  ```python
  # Poll RC controller (non-blocking, no-op if disconnected)
  if self.rc.connected:
      self.rc.poll()
      self.cmd_thrust = self.rc.thrust
      self.cmd_roll_rate = self.rc.roll_rate
      self.cmd_pitch_rate = self.rc.pitch_rate
      self.cmd_yaw_rate = self.rc.yaw_rate
  ```
- **GOTCHA**: Only update flight commands (thrust, rates). Arm/gripper stay on GUI sliders per user's choice.
- **GOTCHA**: The GUI sliders in `gui()` use `ui.slider_float()` which returns the current value. When joystick is active, the slider values will be overwritten each frame by joystick commands. The sliders will visually track the joystick position, providing feedback. This is the desired behavior — no code changes needed in `gui()`.
- **VALIDATE**: Launch with controller plugged in. Moving sticks should update sliders and drone behavior.

### Task 6: UPDATE `testing/validate_hover.py` — Add joystick status to GUI

Add a visual indicator in the GUI showing whether the RC controller is connected.

- **IMPLEMENT**: In `gui()` method, at the top (after line 279 `ui.text("=== Flight Commands ===")`), add:
  ```python
  if self.rc.connected:
      ui.text("RC: Connected (USB)")
  else:
      ui.text("RC: Not connected (GUI only)")
  ui.separator()
  ```
- **VALIDATE**: Visual inspection — GUI shows connection status.

### Task 7: UPDATE `controllers/__init__.py` — Export new module

The `__init__.py` is currently empty. Keep it minimal — just ensure the package is importable. No changes needed unless the file needs explicit exports. Since it's empty, leave it as-is.

- **VALIDATE**: `uv run python -c "from controllers.rc_input import RCInput, RCInputConfig"`

---

## TESTING STRATEGY

### Manual Hardware Testing

This feature is inherently hardware-dependent. The primary validation is manual:

1. **No controller**: Run without RadioMaster plugged in → warning printed, GUI works normally
2. **Controller connected**: Plug in RadioMaster, select "USB Joystick" in EdgeTX → sticks control drone
3. **Axis verification**: Each stick moves the correct command in the correct direction
4. **Deadzone**: Small stick deflections near center produce zero output
5. **Expo**: Fine control near center, full authority at extremes
6. **Smoothing**: No jitter on commands, smooth transitions
7. **Hot-plug**: Disconnecting controller mid-session doesn't crash (graceful degradation)

### Automated Smoke Tests

```bash
# Module imports without evdev
uv run python -c "from controllers.rc_input import RCInputConfig; print('OK')"

# Module imports with evdev
uv run --extra rc python -c "from controllers.rc_input import RCInput; rc = RCInput(); print(f'Connected: {rc.connected}')"

# Config validation
uv run python -c "
from controllers.rc_input import RCInputConfig
c = RCInputConfig()
assert 0 <= c.expo <= 1
assert c.deadzone >= 0
assert c.thrust_max > c.thrust_min
assert c.smoothing_tau >= 0
print('Config OK')
"

# Expo curve unit test
uv run python -c "
from controllers.rc_input import RCInput
# Linear at expo=0
assert RCInput._apply_expo(0.5, 0.0) == 0.5
assert RCInput._apply_expo(-1.0, 0.0) == -1.0
# Reduced sensitivity at center with expo
assert abs(RCInput._apply_expo(0.5, 0.3)) < 0.5
# Full authority at extremes
assert abs(RCInput._apply_expo(1.0, 0.3) - 1.0) < 1e-6
assert abs(RCInput._apply_expo(-1.0, 0.3) - (-1.0)) < 1e-6
# Zero stays zero
assert RCInput._apply_expo(0.0, 0.3) == 0.0
print('Expo OK')
"

# Deadzone unit test
uv run python -c "
from controllers.rc_input import RCInput
# Inside deadzone
assert RCInput._apply_deadzone(0.03, 0.05) == 0.0
assert RCInput._apply_deadzone(-0.04, 0.05) == 0.0
# Outside deadzone, rescaled
assert RCInput._apply_deadzone(1.0, 0.05) == 1.0
assert RCInput._apply_deadzone(-1.0, 0.05) == -1.0
# Continuity at deadzone edge
assert abs(RCInput._apply_deadzone(0.06, 0.05)) < 0.02
print('Deadzone OK')
"

# Lint
uvx pre-commit run -a
```

### Edge Cases

- Controller not plugged in at startup → graceful fallback, no crash
- Controller disconnected during flight → no crash (evdev read_one returns None or raises; handle gracefully)
- Multiple joysticks connected → pick the one matching VID:PID
- Axis values at exact min/max boundaries → proper normalization to 0.0/1.0
- Throttle at zero → thrust = 0.0 N (not negative)
- All sticks centered → all rates = 0.0, thrust = 0.0

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
uvx pre-commit run -a
```

### Level 2: Unit Tests (inline)

```bash
# Run all smoke tests from Testing Strategy section above
uv run python -c "from controllers.rc_input import RCInputConfig; print('Import OK')"
uv run --extra rc python -c "from controllers.rc_input import RCInput; print('Import OK')"
```

### Level 3: Integration Test

```bash
# Launch viewer — should not crash regardless of controller state
timeout 5 uv run python testing/validate_hover.py 2>&1 || true
# Expected: either opens viewer or exits cleanly (timeout kills it)
```

### Level 4: Manual Validation

1. Plug in RadioMaster Pocket via USB-C (use the port near the antenna)
2. On the radio: power on → select "USB Joystick" when prompted
3. Verify Linux sees it: `cat /proc/bus/input/devices | grep -A5 EdgeTX`
4. Run: `uv run --extra rc python testing/validate_hover.py`
5. Verify:
   - GUI shows "RC: Connected (USB)"
   - Left stick up = drone rises (thrust increases)
   - Left stick left/right = yaw
   - Right stick = roll/pitch
   - Sticks centered = drone hovers (rates near zero)
   - Fine movements near center are smooth (expo working)
   - GUI sliders track stick positions
   - Arm sliders still work independently
   - Press R or click Reset = simulation resets

### Level 5: Permissions Check

```bash
# If device not readable, add user to input group:
# sudo usermod -aG input $USER
# Then log out and back in

# Verify with:
ls -la /dev/input/event* | head -5
groups | grep input
```

---

## ACCEPTANCE CRITERIA

- [ ] `controllers/rc_input.py` exists with `RCInputConfig` and `RCInput` classes
- [ ] `evdev` is an optional dependency (`uv sync --extra rc`)
- [ ] Module imports successfully without evdev installed (graceful fallback)
- [ ] RadioMaster Pocket detected by VID:PID when connected
- [ ] 4 flight axes (throttle, roll, pitch, yaw) control drone in viewer
- [ ] Expo curve reduces sensitivity near stick center
- [ ] Deadzone eliminates jitter at stick center
- [ ] First-order smoothing produces clean command signals
- [ ] GUI sliders reflect joystick values in real-time
- [ ] GUI shows RC connection status
- [ ] Arm/gripper sliders remain functional (GUI-only)
- [ ] No crash when controller is not connected
- [ ] All code passes `uvx pre-commit run -a` (ruff lint/format)
- [ ] All static helpers have unit-testable validation

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order (1-7)
- [ ] Each task validation passed
- [ ] Pre-commit passes clean
- [ ] Manual flight test with actual RadioMaster Pocket
- [ ] Fallback test without controller
- [ ] Acceptance criteria all met

---

## NOTES

### Design Decisions

1. **evdev over pygame**: evdev is Linux-native, non-blocking, no SDL dependency, no conflict with Newton's pyglet/OpenGL viewer. pygame would drag in SDL video subsystem unnecessarily.

2. **Lazy evdev import**: The module must be importable without evdev installed. This keeps the training pipeline (which doesn't need RC input) free of unnecessary dependencies.

3. **First-order smoothing over Butterworth**: The existing `LowPassFilter` in `controllers/low_pass_filter.py` is a 2nd-order Butterworth designed for 300Hz INDI signals. Joystick input at 60Hz needs simpler smoothing. A first-order exponential filter (same pattern as `motor_model.py` line 79: `alpha = exp(-dt/tau)`) is sufficient.

4. **Properties for output**: Using `@property` for thrust/roll_rate/pitch_rate/yaw_rate makes the interface clean and read-only from the consumer side, matching Python conventions.

5. **No threading**: evdev non-blocking poll via `select()` runs in the main thread at 60Hz. No need for a separate thread — the viewer loop already runs at this rate.

6. **Throttle as [0,1] not [-1,1]**: RC throttle sticks are non-centering (spring-less on one axis). The raw value naturally maps 0→1 (bottom to top). This maps directly to [0, 25] N thrust.

### RadioMaster Pocket Setup

User must:
1. Connect via USB-C (port near antenna, NOT the charging port)
2. Turn on radio
3. Select "USB Joystick" on the EdgeTX prompt
4. Disable internal/external RF modules for best performance (1000Hz mixer rate)

### Linux Permissions

If `/dev/input/event*` is not readable, user needs to be in the `input` group:
```bash
sudo usermod -aG input $USER
# Log out and back in
```

### Future Extensions

- Hot-plug detection (periodic re-scan if disconnected)
- Button mapping for reset (aux switches → `reset_requested`)
- Arm/gripper on aux channels (CH5-CH7) if user adds EdgeTX mixer
- Rate/expo profiles switchable via aux switch
- Support for ELRS BLE and USB dongle (same evdev interface, different VID:PID)
