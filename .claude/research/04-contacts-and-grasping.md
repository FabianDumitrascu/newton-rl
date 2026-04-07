# Contacts, Collisions & Grasping in Newton

## Contact System

### Collision Detection Pipeline

```python
contacts = model.contacts()
model.collide(state, contacts)  # Broad + narrow phase
```

**Broad phase options:** `"explicit"` (pre-computed pairs), `"sap"` (sweep-and-prune), `"nxn"` (brute force)

### Contact Data Available

```python
contacts.rigid_contact_count       # Number of active contacts
contacts.rigid_contact_shape0/1    # Shape indices in contact
contacts.rigid_contact_point0/1    # Body-frame contact points [m]
contacts.rigid_contact_normal      # Unit normal (A → B)
contacts.rigid_contact_margin0/1   # Surface thickness per shape
contacts.rigid_contact_force       # Contact forces [N] (extended attribute)
```

### Contact Material Properties (Per-Shape)

| Property | Default | Description |
|----------|---------|-------------|
| `ke` | 2500.0 | Contact stiffness [N/m] |
| `kd` | 100.0 | Contact damping [N·s/m] |
| `mu` | 1.0 | Coulomb friction coefficient |
| `mu_torsional` | 0.005 | Torsional friction |
| `mu_rolling` | 0.0001 | Rolling friction |
| `ka` | 0.0 | Adhesion distance [m] |
| `kh` | 1e10 | Hydroelastic stiffness [N/m³] |
| `restitution` | 0.0 | Coefficient of restitution |

## Contact Filtering

```python
# Per-shape collision group
cfg = ShapeConfig(collision_group=1)

# Explicit pair filtering
builder.add_shape_collision_filter_pair(shape_a, shape_b)

# Joint-based (parent-child auto-filtered)
builder.add_joint_fixed(parent, child, collision_filter_parent=True)
```

## Force/Torque Sensing

```python
# Request contact force computation
model.request_contact_attributes("force")
contacts = model.contacts()

# Use SensorContact for structured feedback
from newton.sensors import SensorContact

sensor = SensorContact(
    model,
    sensing_obj_shapes="*Finger*",
    counterpart_shapes="*Object*",
    measure_total=True,
)
sensor.update(state, contacts)
total_force = sensor.total_force        # [N, N, N]
force_matrix = sensor.force_matrix      # Per-counterpart forces
```

## Grasping Approaches

### Approach 1: Contact-Force Grasping (RECOMMENDED)

Rely on friction and normal forces — no runtime joint creation needed.

**Already demonstrated in:**
- `example_robot_panda_hydro.py` — hydroelastic finger contacts
- `example_softbody_franka.py` — deformable object grasping

**Requirements:**
- High friction (μ ≥ 0.5) on finger surfaces
- Adequate contact stiffness/damping
- Multiple contact points (2+ fingers)
- Controlled grip force via joint PD gains

### Approach 2: Pre-Allocated Fixed Joints (WORKAROUND)

Pre-create disabled fixed joints during model building, enable at grasp time.

```python
# During build
grasp_joint = builder.add_joint_fixed(gripper_body, object_body, enabled=False)

# At runtime — toggle joint_enabled array
# NOTE: Limited support, solver-dependent
```

**Limitation:** Must know all possible grasp targets at build time.

### Approach 3: Hydroelastic Contacts

Advanced contact model using SDF representations:
```python
cfg = ShapeConfig(is_hydroelastic=True, kh=1e10)
```
Provides richer contact patches (not just point contacts), better for stable grasps.

## CRITICAL: No Runtime Joint Creation

**Newton models are immutable after `finalize()`.** You CANNOT:
- Add new joints during simulation
- Remove joints during simulation  
- Change joint topology

**You CAN:**
- Toggle `joint_enabled` (solver-dependent)
- Change joint targets, gains, limits
- Apply external forces/torques
- Read contact forces

## Implications for Aerial Manipulation

For the aerial manipulator grasping task:
1. **Primary strategy:** Contact-force grasping with tuned friction
2. **Finger control:** PD-controlled revolute joints with position targets
3. **Grasp detection:** Monitor `SensorContact` forces on fingers
4. **Grasp stability:** Reward shaping based on contact forces + object motion
5. **If contact grasping insufficient:** Pre-allocate fixed joints for known objects
