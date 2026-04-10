# Phase 0: Interactive Drone Flight Validation in Newton

## Context

This is the first implementation step for the newton-rl project. Before building any RL pipeline, we need to validate that the drone flies correctly in the Newton physics engine. The goal is an interactive script that loads the aerial manipulator USD model, runs the INDI flight controller, and lets the user fly the drone and move the arm via viewer GUI sliders — confirming that hover thrust matches mg (~8.4N) and that INDI compensates for arm coupling.

## User Story

As a thesis researcher, I want to fly my aerial manipulator in Newton's viewer with interactive controls, so that I can verify the flight physics are correct before building the RL training pipeline.

## Solution

Create 6 new files: a config dataclass, math utilities, low-pass filter, motor model, INDI controller, and an interactive validation script. All controller code is ported from the old Isaac Lab codebase (`reference_code/osprey_rl/`) and adapted for Newton's API conventions.

---

## FILES TO CREATE

| # | File | Purpose |
|---|------|---------|
| 1 | `controllers/__init__.py` | Package init (empty) |
| 2 | `controllers/config.py` | Structured platform config (all physical params) |
| 3 | `controllers/math_utils.py` | Quaternion/rotation helpers (PyTorch, XYZW convention) |
| 4 | `controllers/low_pass_filter.py` | 2nd-order Butterworth IIR filter |
| 5 | `controllers/motor_model.py` | First-order motor dynamics + thrust computation |
| 6 | `controllers/indi.py` | INDI attitude controller |
| 7 | `testing/validate_hover.py` | Interactive simulation with viewer GUI |

## CRITICAL REFERENCE FILES (must read before implementing)

| File | Why |
|------|-----|
| `reference_code/osprey_rl/osprey_rl/mdp/controller/indi.py` | INDI control law, G1 matrix, inertia coupling — port formulas |
| `reference_code/osprey_rl/osprey_rl/mdp/controller/motor_model.py` | Motor dynamics — port directly |
| `reference_code/osprey_rl/osprey_rl/mdp/controller/low_pass_filter.py` | Butterworth filter — port directly |
| `reference_code/osprey_rl/osprey_rl/mdp/controller/control.py` | Force application pattern, body index mapping |
| `submodules/newton/newton/examples/diffsim/example_diffsim_drone.py` | Newton force application pattern (`compute_prop_wrenches` kernel) |
| `testing/test.py` | Working Newton viewer + imgui GUI pattern |
| `.claude/references/joints.md` | Joint PD control API (ke, kd, target_pos) |
| `.claude/references/usd.md` | `add_usd()` API and return dict |
| `.claude/references/model_builder.md` | Body force arrays, state format |

## NEWTON API CONVENTIONS (verified from source)

| API | Format | Notes |
|-----|--------|-------|
| `body_q` | `wp.transform` = `(px, py, pz, qx, qy, qz, qw)` | **XYZW** quaternion |
| `body_qd` | `wp.spatial_vector` = `(vx, vy, vz, wx, wy, wz)` | Linear first, angular second. **World frame** at COM |
| `body_f` | `wp.spatial_vector` = `(fx, fy, fz, tx, ty, tz)` | Force first, torque second. **World frame** at COM |
| `wp.spatial_vector(a, b)` | Constructor takes two `vec3` | First = linear (force), Second = angular (torque) |
| `wp.transform_vector(tf, v)` | Rotates vector by transform's rotation | Body-to-world direction transform |
| `wp.transform_point(tf, p)` | Transforms point by full transform | Body-to-world point transform |
| `state.clear_forces()` | Zeros `body_f` | Must call every substep |
| Joint PD | `F = ke * (target - q) + kd * (target_vel - qd)` | Set `builder.joint_target_ke/kd` before finalize |
| `control.joint_target_pos` | Array of joint position targets | Update at runtime for arm control |

**Quaternion convention difference:** Old code (Isaac Lab) uses WXYZ. Newton uses XYZW. Every quaternion operation must use XYZW.

---

## IMPLEMENTATION PLAN

### Step 0: USD Discovery (do first, informs everything else)

Write a throwaway script to load the USD and print all body/joint names and indices:

```python
builder = newton.ModelBuilder()
result = builder.add_usd("assets/flattened-osprey.usd", floating=True, verbose=True)
for path, idx in sorted(result["path_body_map"].items(), key=lambda x: x[1]):
    print(f"Body {idx}: {path}")
for path, idx in sorted(result["path_joint_map"].items(), key=lambda x: x[1]):
    print(f"Joint {idx}: {path}")
model = builder.finalize()
print(f"Bodies: {model.body_count}, Joints: {model.joint_count}")
print(f"Joint coords: {model.joint_coord_count}, Joint DOFs: {model.joint_dof_count}")
print(f"Body labels: {model.body_label}")
print(f"Joint labels: {model.joint_label}")
```

This reveals: base body index, 4 rotor body indices, arm/gripper joint DOF indices. These indices are needed by every subsequent file. Record the results and use them to parameterize the config.

**VALIDATE:** Script runs without error, prints meaningful body/joint names matching `dof_*` convention.

---

### Step 1: `controllers/__init__.py`

Empty file. Makes `controllers` a package.

---

### Step 2: `controllers/config.py` — Platform Configuration

Structured dataclasses for all physical parameters. Single source of truth.

**Key dataclasses:**
- `RotorConfig` — thrust coefficients (per-rotor), moment constant, directions, tilt angles, arm lengths, omega limits
- `MotorConfig` — time constants (tau_up, tau_down), motor inertia
- `InertiaConfig` — base mass, arm mass, inertia tensors, arm CoG/root offsets
- `FilterConfig` — cutoff freq, sampling freq (NOTE: sampling freq must match actual sim freq, not hardcoded)
- `ControlConfig` — k_alpha_cmd gains
- `SimConfig` — fps, substeps, solver iterations, gravity, spawn height
- `ArmConfig` — PD gains for arm and gripper joints
- `OspreyConfig` — top-level container for all sub-configs

**Factory function:** `default_osprey_config() -> OspreyConfig` with all values from reference code.

**Key parameter values (from reference):**

| Parameter | Value | Source |
|-----------|-------|--------|
| `thrust_coeff` | `[3.1e-6, 3.1e-6, 1.5e-6, 1.5e-6]` | `indi.py:59` |
| `moment_constant` | `0.022` | `indi.py:23` |
| `directions` | `[1.0, -1.0, 1.0, -1.0]` | `motor_model.py:22` |
| `tilt_angles_deg` | `[53.53, 41.42]` | `indi.py:24-25` |
| `arm_lengths` | `[0.143, 0.1133]` | `indi.py:26-27` |
| `omega_min/max` | `150.0 / 2800.0` | `indi.py:53-54` |
| `tau_up/down` | `0.033 / 0.033` | `motor_model.py:8-9` |
| `base_mass` | `0.660` | `indi.py:62` |
| `arm_mass` | `0.2` | `indi.py:62` |
| `base_inertia` | `[0.00254, 0.00271, 0.00515]` | `indi.py:65` |
| `k_alpha_cmd` | `[100.0, 100.0, 10.0]` | `indi.py:67` |
| `init_omega` | `[935.0, 935.0, 1002.0, 1002.0]` | `control.py:111` |
| `cutoff_freq` | `12.0` | `indi.py:70` |

**VALIDATE:** `config = default_osprey_config(); assert abs(config.total_mass * config.sim.gravity - 8.43) < 0.1`

---

### Step 3: `controllers/math_utils.py` — Quaternion Utilities

Pure PyTorch. All functions work with **XYZW** convention (Newton's format).

**Functions:**
- `quat_rotate(q_xyzw, v)` — rotate vector by quaternion (body→world)
- `quat_rotate_inverse(q_xyzw, v)` — rotate by inverse (world→body). Needed to get body-frame angular velocity from Newton's world-frame `body_qd`.
- `euler_from_quat_xyzw(q)` — extract (roll, pitch, yaw)
- `skew_symmetric(v)` — 3x3 skew matrix from vector (for cross product as matrix multiply)

**Port notes:** The reference code uses `isaaclab.utils.math` functions which assume WXYZ. The formulas are standard quaternion rotation: `v' = q * v * q_inv` expanded to avoid full quaternion multiply. Just ensure the x,y,z,w indexing matches XYZW.

**VALIDATE:** `quat_rotate(identity_quat, v) == v`, `quat_rotate_inverse(q, quat_rotate(q, v)) == v` for random inputs.

---

### Step 4: `controllers/low_pass_filter.py` — Butterworth Filter

Direct port from `reference_code/osprey_rl/osprey_rl/mdp/controller/low_pass_filter.py`.

**Changes from reference:**
- Add type annotations
- Accept config instead of hardcoded values
- Fix potential bug: reference line 48 references `self.sampling_frequency` but attribute may be named differently

**Class: `LowPassFilter`**
- `__init__(fc, fs, initial_value)` — computes Butterworth coefficients via bilinear transform
- `add(sample)` — feed new sample, return filtered output
- `derivative()` — time derivative of filtered signal (fs * delta_output)
- `reset(env_ids, val)` — reset filter state for specific environments

**Butterworth coefficients:**
```
K = tan(pi * fc / fs)
poly = K^2 + sqrt(2)*K + 1
num[0] = K^2 / poly
num[1] = 2 * K^2 / poly
den[0] = 2*(K^2 - 1) / poly
den[1] = (K^2 - sqrt(2)*K + 1) / poly
```

**VALIDATE:** Filter a 1Hz + 50Hz mixed signal with fc=5Hz. Verify 50Hz component is attenuated >10x.

---

### Step 5: `controllers/motor_model.py` — Motor Dynamics

Direct port from `reference_code/osprey_rl/osprey_rl/mdp/controller/motor_model.py`.

**Class: `RotorMotor`**
- `__init__(num_envs, config)` — initialize with hover omega from config
- `step(target_speeds, dt)` — advance motor dynamics, return (thrusts, moments, omega)
- `reset(env_ids)` — reset to initial omega

**Motor dynamics:**
```python
tau = where(target > current, tau_up, tau_down)  # asymmetric
alpha = exp(-dt / tau)
omega = alpha * omega_old + (1 - alpha) * omega_target
omega = clamp(omega, omega_min, omega_max)
thrust = C_T * omega^2        # per-rotor, using per-rotor C_T
moment = kappa * thrust * dir  # per-rotor yaw moment
```

**VALIDATE:** Command constant hover omega [935, 935, 1002, 1002]. After convergence, total thrust should be ~8.4N: `2 * 3.1e-6 * 935^2 + 2 * 1.5e-6 * 1002^2 ≈ 5.42 + 3.01 = 8.43N`.

---

### Step 6: `controllers/indi.py` — INDI Controller

Port from `reference_code/osprey_rl/osprey_rl/mdp/controller/indi.py`. This is the most complex file.

**Class: `IndiController`**
- `__init__(num_envs, config)` — build G1 matrix, G1_inv, init filters
- `compute_g1_matrix(config)` — 4x4 mixing matrix from rotor geometry
- `get_command(omega_body, joint_pos, rotor_speeds, collective_thrust, alpha_cmd)` → target rotor speeds
- `reset(env_ids)` — reset filters

**G1 matrix construction:**
```
G1[0,:] = [1, 1, 1, 1]                                           # total thrust
G1[1,:] = [l1*sin(b1), -l1*sin(b1), -l2*sin(b2), l2*sin(b2)]    # roll torque
G1[2,:] = [-l1*cos(b1), -l1*cos(b1), l2*cos(b2), l2*cos(b2)]    # pitch torque
G1[3,:] = [kappa, -kappa, kappa, -kappa]                          # yaw torque
```

**INDI control law (from reference `getCommand()`):**
1. Filter angular velocity → `omega_filtered`, compute `omega_dot = filter_gyr.derivative()`
2. Filter rotor speeds → compute current thrusts: `thrusts_state = C_T * omega_filtered^2`
3. Current torques: `tau_f = G1[1:4, :] @ thrusts_state`
4. Desired control wrench:
   - `mu[0] = collective_thrust`
   - `mu[1:3] = tau_f + J @ (alpha_cmd - omega_dot)[roll,pitch]` (INDI for roll/pitch)
   - `mu[3]` = NDI for yaw: `(J @ alpha_cmd + omega x (J @ omega))[yaw]`
5. Solve: `thrusts = G1_inv @ mu`, clamp to [0, thrust_max]
6. `rotor_speeds = sqrt(thrusts / C_T)`, clamp to [omega_min, omega_max]

**Inertia coupling (`get_total_J`):** For Phase 0, start with static `inertia_mat` (no arm coupling). Add the dynamic `get_total_J(joint_pos)` as a follow-up once basic hover works. The formula uses parallel axis theorem with arm joint rotations — see reference `indi.py:97-128`.

**Key porting changes:**
- All quaternion ops use XYZW (our `math_utils`)
- Filter sampling frequency = actual sim frequency (config-driven, not hardcoded 300Hz)
- No Isaac Lab imports — use our own `LowPassFilter` and `math_utils`
- Config-driven parameters throughout

**VALIDATE:** Command hover thrust (mass * g) with zero body rates. INDI should output rotor speeds near [935, 935, 1002, 1002]. Total `C_T * omega^2` across 4 rotors ≈ 8.4N.

---

### Step 7: `testing/validate_hover.py` — Interactive Validation Script

Main simulation script tying everything together.

**Structure:**
```
class HoverValidator:
    __init__:  load config, build Newton model, init controllers, create viewer
    gui():     imgui callback — sliders for thrust/rates/arm, telemetry display
    step():    per-frame: extract state → INDI → motor → apply forces → solver.step
    reset():   re-init states and controllers
    run():     main viewer loop
```

**Model building:**
1. `builder = newton.ModelBuilder()`
2. `builder.add_ground_plane()`
3. `result = builder.add_usd("assets/flattened-osprey.usd", floating=True, xform=..., enable_self_collisions=False)`
4. Set arm/gripper joint PD gains: `builder.joint_target_ke[dof] = kp` etc.
5. Set rotor joint gains to 0 (no PD, force-driven)
6. `model = builder.finalize()`

**Per-substep simulation loop:**
1. `state_0.clear_forces()`
2. `viewer.apply_forces(state_0)` (for viewer picking)
3. Extract drone state: position, quat (XYZW), angular velocity (world→body transform)
4. Extract joint positions from `state_0.joint_q`
5. Compute `alpha_cmd = k_alpha * (desired_rate - current_rate)`
6. Call `indi.get_command(...)` → target rotor speeds
7. Call `motor.step(target_speeds, dt)` → thrusts, moments
8. Apply per-rotor forces to rotor bodies via `state_0.body_f`
9. Update arm joint targets via `control.joint_target_pos`
10. `model.collide(state_0, contacts)`
11. `solver.step(state_0, state_1, control, contacts, dt)`
12. Swap states

**Force application approach:**
Apply thrust to each rotor body (not base body). Each rotor's thrust acts along its local Z-axis. Newton handles transmitting forces through joints to the base body, creating correct torques from moment arms. Yaw reaction torques are summed and applied to the base body.

For each rotor body:
```python
tf = state.body_q[rotor_body_idx]
thrust_dir_world = wp.transform_vector(tf, wp.vec3(0, 0, 1))
force = thrust_dir_world * thrust_magnitude
# Write to body_f[rotor_body_idx] linear component
```

For yaw on base body:
```python
tf_base = state.body_q[base_body_idx]
yaw_axis_world = wp.transform_vector(tf_base, wp.vec3(0, 0, 1))
yaw_torque = yaw_axis_world * sum(moments)
# Write to body_f[base_body_idx] angular component
```

**Fallback:** If rotor bodies don't exist as separate rigid bodies in the USD, apply all forces to the base body with explicit moment arm computation (like Newton's `compute_prop_wrenches` kernel).

**GUI layout:**
```
=== Flight Commands ===
[Thrust slider]     0.0 — 25.0 N  (default: ~8.4N = mg)
[Roll rate slider]  -5.0 — 5.0 rad/s
[Pitch rate slider] -5.0 — 5.0 rad/s
[Yaw rate slider]   -3.0 — 3.0 rad/s

=== Arm Commands ===
[Arm Pitch slider]  -1.57 — 1.57 rad
[Arm Roll slider]   -1.57 — 1.57 rad
[Gripper slider]    0.0 — 0.027 rad (closed — open)

=== Telemetry ===
Altitude: X.XXX m
Total thrust: X.XXX N
Expected hover: 8.43 N
Rotor speeds: [XXX, XXX, XXX, XXX] rad/s
[Reset button]
```

**SimConfig defaults:** `fps=60`, `sim_substeps=5` → 300Hz physics (matches reference filter frequency). Solver: XPBD with 10 iterations. Spawn height: 2.0m.

**VALIDATE:**
| Check | Expected |
|-------|----------|
| Drone hovers at spawn height | Altitude stable ±0.1m for 10+ seconds |
| Total thrust at hover | ~8.4N (displayed in GUI) |
| Rotor speeds at hover | Front ~935, back ~1002 rad/s |
| Move arm pitch slider | Drone tilts briefly, INDI recovers |
| Move arm roll slider | Same — visible coupling, compensation |
| Reset button | Returns to initial state |
| No NaN/crash | Run 60+ seconds without divergence |

---

## IMPLEMENTATION ORDER

1. **Step 0** — Run USD discovery script, record body/joint names and indices
2. **Step 1** — Create `controllers/__init__.py`
3. **Step 2** — Create `controllers/config.py` with all parameters
4. **Step 3** — Create `controllers/math_utils.py` with quaternion helpers
5. **Step 4** — Create `controllers/low_pass_filter.py`
6. **Step 5** — Create `controllers/motor_model.py`
7. **Step 6** — Create `controllers/indi.py`
8. **Step 7** — Create `testing/validate_hover.py`, integrate everything

Each step builds on the previous. Steps 3-5 are independent of each other but all needed by step 6.

## VALIDATION COMMANDS

```bash
# Lint
uvx pre-commit run -a

# Run the validation script
uv run python testing/validate_hover.py

# Quick smoke test (headless, if supported)
uv run python -c "from controllers.config import default_osprey_config; c = default_osprey_config(); print(f'Total mass: {c.inertia.falcon_mass + c.inertia.manipulator_mass} kg')"
```

## RISKS AND MITIGATIONS

| Risk | Mitigation |
|------|------------|
| USD model has unexpected body/joint structure | Step 0 discovery runs first; adapt code to actual names |
| Rotor bodies not separate rigid bodies in USD | Fall back to single-body force application with explicit moment arms |
| Solver instability at 300Hz | Increase substeps (try 600Hz), increase solver iterations, or try SolverMuJoCo |
| NumPy roundtrip perf (GPU→CPU→GPU) | Acceptable for Phase 0 single-env. Replace with `wp.to_torch()` in Phase 2 |
| Filter frequency mismatch | Filter fs is derived from actual sim frequency (fps * substeps), not hardcoded |
| Quaternion convention bugs | All math_utils tested with XYZW; never mix conventions |
