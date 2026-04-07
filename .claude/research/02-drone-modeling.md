# Drone Modeling in Newton

## Existing Example: Crazyflie Quadcopter

**File:** `submodules/newton/newton/examples/diffsim/example_diffsim_drone.py`
**Asset:** `submodules/newton/newton/examples/assets/crazyflie.usd` (visual only, not loaded in sim yet)

## Propeller Physics Model

Newton uses a custom Warp struct + kernel for propeller forces:

```python
@wp.struct
class Propeller:
    body: int           # Body the propeller is attached to
    pos: wp.vec3        # Position relative to body frame
    dir: wp.vec3        # Thrust direction (typically (0,0,1) = up)
    thrust: float       # Thrust coefficient (dimensionless, ~0.109919)
    power: float        # Power coefficient (dimensionless, ~0.040164)
    diameter: float     # Propeller diameter [m] (0.2286 for Crazyflie)
    max_rpm: float      # Maximum RPM (6396.667)
    max_thrust: float   # = thrust_coeff * air_density * rps² * d⁴ [N]
    max_torque: float   # = power_coeff * air_density * rps² * d⁵ / 2π [N·m]
    turning_direction: float  # +1.0 or -1.0 (alternating for quad)
    max_speed_square: float
```

### Force Computation Kernel

```python
@wp.kernel
def compute_prop_wrenches(props, controls, body_q, body_com, body_f):
    tid = wp.tid()
    prop = props[tid]
    control = controls[tid]  # 0.0 to 1.0 normalized
    
    tf = body_q[prop.body]
    dir = wp.transform_vector(tf, prop.dir)
    
    # Thrust force (in world frame, along prop direction)
    force = dir * prop.max_thrust * control
    
    # Reaction torque (yaw moment from rotor)
    torque = dir * prop.max_torque * control * prop.turning_direction
    
    # Moment from offset thrust (roll/pitch moments)
    moment_arm = wp.transform_point(tf, prop.pos) - wp.transform_point(tf, body_com[prop.body])
    torque += wp.cross(moment_arm, force)
    
    # Angular damping
    torque *= 0.8
    
    wp.atomic_add(body_f, prop.body, wp.spatial_vector(force, torque))
```

### Key Physics: `force = thrust_coeff * air_density * (RPM/60)² * diameter⁴`

This is the standard aerodynamic thrust equation. Very similar to the INDI controller physics described by the user (ω² * parameter → force, another parameter → torque).

## Drone Body Structure

The Crazyflie is modeled as:
- **1 rigid body** (FREE joint = 6 DOF floating)
- **2 perpendicular crossbar box shapes** (carbon fiber, density 1750 kg/m³)
- **4 propellers** at arm tips with alternating turning directions:
  - Front (y+): CW (-1.0)
  - Back (y-): CCW (+1.0)
  - Right (x+): CCW (+1.0)
  - Left (x-): CW (-1.0)

## Control Interface

- **4 normalized inputs** [0.0, 1.0] per propeller
- Applied via custom `control.prop_controls` array
- Uses `SolverSemiImplicit` (differentiable, for trajectory optimization)

## Solver Used: SolverSemiImplicit

The drone example uses differentiable simulation for trajectory optimization (not RL), with:
- Control waypoints (3 control points, 10 steps apart)
- Cost function: position tracking + altitude constraints + orientation penalty
- Gradient descent on control inputs

## For Our Aerial Manipulator

We need to extend this with:
1. **Articulated arm** attached to the drone body (revolute joints for pitch, roll)
2. **Gripper fingers** at the end effector (revolute joints)
3. **Combined control**: propeller forces + joint targets
4. **Contact sensing** on fingers for grasp feedback

## Relevant Robot Arm Examples

| Example | File | Joints | Features |
|---------|------|--------|----------|
| Panda + Hydro | `robot/example_robot_panda_hydro.py` | 7 arm + 2 gripper | IK, hydroelastic contacts, 100 worlds |
| Allegro Hand | `robot/example_robot_allegro_hand.py` | 22 joints | USD loading, finger control |
| UR10 | `robot/example_robot_ur10.py` | 6 DOF | URDF loading, PD control |
| Soft Franka | `softbody/example_softbody_franka.py` | 7+2 | Deformable grasping |
