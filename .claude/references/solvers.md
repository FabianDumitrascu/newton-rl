# Newton Solvers Reference

All Newton solvers share a common interface via `SolverBase.step()`:

```python
solver.step(state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float) -> None
```

- `state_in` / `state_out`: Current and next simulation states (double-buffered).
- `control`: Joint feedforward forces and targets. `None` uses model defaults.
- `contacts`: Output of the collision pipeline. `None` skips contact handling.
- `dt`: Time step in seconds.

---

## Solver Feature Matrix

| Solver | Integration | Rigid Bodies | Articulations | Particles | Cloth | Soft Bodies | Differentiable |
|---|---|---|---|---|---|---|---|
| **SolverXPBD** | Implicit | Maximal coords | Maximal coords | Yes | Yes (no self-collision) | Experimental | No |
| **SolverMuJoCo** | Explicit / Semi-implicit / Implicit | Yes (own contacts) | Generalized coords | No | No | No | No |
| **SolverFeatherstone** | Explicit | Yes | Generalized coords | Yes | Yes (no self-collision) | Yes | Basic |
| **SolverSemiImplicit** | Semi-implicit | Yes | Maximal coords | Yes | Yes (no self-collision) | Yes | Basic (wp.Tape) |
| **SolverKamino** | Semi-implicit (Euler / Moreau) | Maximal coords | Maximal coords | No | No | No | No |
| **SolverVBD** | Implicit | Yes (AVBD) | Limited joint support | Yes | Yes | No | No |
| **SolverImplicitMPM** | Implicit | No | No | Yes | No | No | No |

---

## 1. SolverXPBD

eXtended Position-Based Dynamics. Iteratively solves positional constraints for rigid bodies, particles, cloth, and soft bodies.

### Constructor

```python
newton.solvers.SolverXPBD(
    model: Model,
    iterations: int = 2,                      # Constraint solver iterations per step
    soft_body_relaxation: float = 0.9,        # Relaxation for soft body constraints
    soft_contact_relaxation: float = 0.9,     # Relaxation for particle-shape contacts
    joint_linear_relaxation: float = 0.7,     # Linear joint constraint relaxation
    joint_angular_relaxation: float = 0.4,    # Angular joint constraint relaxation
    joint_linear_compliance: float = 0.0,     # Joint linear compliance (1/stiffness)
    joint_angular_compliance: float = 0.0,    # Joint angular compliance
    rigid_contact_relaxation: float = 0.8,    # Rigid body contact relaxation
    rigid_contact_con_weighting: bool = True, # Constraint-based weighting for contacts
    angular_damping: float = 0.0,             # Global angular damping
    enable_restitution: bool = False,         # Coefficient of restitution
)
```

### Performance

- Unconditionally stable for implicit constraints.
- Slower than semi-implicit due to multiple constraint iterations.
- Increase `iterations` for stiffer constraints (4-8 for stiff systems).

### Joint Support

| Feature | Supported |
|---|---|
| PRISMATIC, REVOLUTE, BALL, FIXED, FREE, DISTANCE, D6 | Yes |
| `joint_enabled` | Yes |
| `joint_target_ke` / `joint_target_kd` | Yes |
| `Control.joint_f` | Yes |
| Joint limits | Yes (hard constraints, ignores ke/kd) |
| `joint_armature`, `joint_friction`, `joint_effort_limit` | No |

### Working Example

```python
import warp as wp
import newton

builder = newton.ModelBuilder()

# Ground plane (static shape on body -1)
builder.add_shape_box(body=-1, hx=5.0, hy=0.1, hz=5.0)

# Dynamic sphere
builder.add_rigid_body(pos=wp.vec3(0.0, 2.0, 0.0), mass=1.0)
builder.add_shape_sphere(body=0, radius=0.3)

model = builder.finalize()
solver = newton.solvers.SolverXPBD(model, iterations=4)

state_0 = model.state()
state_1 = model.state()
control = model.control()
collision_pipeline = newton.CollisionPipeline(model)
contacts = collision_pipeline.contacts()

dt = 1.0 / 60.0
for _ in range(240):
    state_0.clear_forces()
    collision_pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0
```

---

## 2. SolverMuJoCo

Integrates the MuJoCo physics engine (GPU-accelerated via mujoco_warp). Converts Newton models to MuJoCo XML internally.

### Constructor

```python
newton.solvers.SolverMuJoCo(
    model: Model,
    *,
    integrator: int | str | None = None,      # "euler", "rk4", "implicitfast" (default)
    solver: int | str | None = None,          # "cg" or "newton" (default)
    cone: int | str | None = None,            # "pyramidal" (default) or "elliptic"
    iterations: int | None = None,            # Solver iterations (default 100)
    ls_iterations: int | None = None,         # Line search iterations (default 50)
    impratio: float | None = None,            # Friction-to-normal impedance ratio
    use_mujoco_contacts: bool = True,         # Use MuJoCo collision pipeline
    use_mujoco_cpu: bool = False,             # Run on CPU via MuJoCo (not mujoco_warp)
    disable_contacts: bool = False,           # Disable all contact handling
    save_to_mjcf: str | None = None,          # Debug: save MuJoCo XML to file
    separate_worlds: bool | None = None,      # Map Newton worlds to MuJoCo worlds
    njmax: int | None = None,                 # Max constraints per world
    nconmax: int | None = None,               # Max contacts per world
    update_data_interval: int = 1,            # How often to sync Newton <-> MuJoCo
    include_sites: bool = True,               # Include sites in MuJoCo model
    skip_visual_only_geoms: bool = True,      # Skip visual-only geometries
)
```

Option resolution priority: constructor argument > model custom attribute (`model.mujoco.<option>`) > MuJoCo default.

### Custom Attributes

Call `SolverMuJoCo.register_custom_attributes(builder)` before building the model to enable per-world MuJoCo options and equality/mimic constraints.

### Joint Support

| Feature | Supported |
|---|---|
| PRISMATIC, REVOLUTE, BALL, FIXED, FREE, D6 | Yes |
| DISTANCE | No |
| `joint_armature`, `joint_friction`, `joint_effort_limit` | Yes |
| `joint_target_ke` / `joint_target_kd` / `joint_target_mode` | Yes |
| `joint_limit_ke` / `joint_limit_kd` | Yes |
| `Control.joint_f` | Yes |
| Equality constraints (CONNECT, WELD, JOINT) | Yes |
| Mimic constraints (REVOLUTE, PRISMATIC only) | Yes |
| `joint_enabled`, `joint_velocity_limit` | No |

### Performance

- Most stable solver (implicit mode).
- Overhead from Newton-to-MuJoCo model conversion at construction time.
- Uses its own collision pipeline by default; set `use_mujoco_contacts=False` to use Newton's.

### Working Example

```python
import warp as wp
import newton

builder = newton.ModelBuilder()
newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

# Simple pendulum
builder.add_articulation()
b = builder.add_rigid_body(
    pos=wp.vec3(0.5, 1.0, 0.0),
    mass=1.0,
)
builder.add_shape_capsule(body=b, radius=0.05, half_height=0.25)
builder.add_joint_revolute(
    parent=-1,
    child=b,
    parent_xform=wp.transform(wp.vec3(0.0, 1.0, 0.0), wp.quat_identity()),
    child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
    axis=wp.vec3(0.0, 0.0, 1.0),
    target_ke=100.0,
    target_kd=10.0,
)

model = builder.finalize()
solver = newton.solvers.SolverMuJoCo(model, integrator="implicitfast")

state_0 = model.state()
state_1 = model.state()
control = model.control()

dt = 1.0 / 60.0
for _ in range(240):
    state_0.clear_forces()
    solver.step(state_0, state_1, control, None, dt)  # MuJoCo handles contacts internally
    state_0, state_1 = state_1, state_0
```

---

## 3. SolverFeatherstone

Reduced-coordinate solver using Featherstone's Composite Rigid Body Algorithm (CRBA). Operates on joint coordinates (generalized/reduced) rather than body poses (maximal), eliminating constraint drift.

### Constructor

```python
newton.solvers.SolverFeatherstone(
    model: Model,
    angular_damping: float = 0.05,            # Angular damping factor
    update_mass_matrix_interval: int = 1,     # Update inertia matrix every N steps
    friction_smoothing: float = 1.0,          # Huber norm delta for friction
    use_tile_gemm: bool = False,              # Warp Tile API for matrix ops
    fuse_cholesky: bool = True,               # Fuse Cholesky with inertia eval (Tile API only)
)
```

### Articulation Requirements

- Model **must** have a proper joint tree hierarchy (no kinematic loops).
- Each floating body needs an explicit FREE joint as root.
- Joints must form a tree rooted at the world (`parent=-1`).
- Also integrates particles/cloth/soft bodies using semi-implicit Euler (like SolverSemiImplicit).

### Joint Support

| Feature | Supported |
|---|---|
| PRISMATIC, REVOLUTE, BALL, FIXED, FREE, D6 | Yes |
| DISTANCE | Treated as FREE |
| `joint_armature` | Yes |
| `joint_target_ke` / `joint_target_kd` | Yes |
| `joint_limit_ke` / `joint_limit_kd` | Yes |
| `Control.joint_f` | Yes |
| `joint_friction`, `joint_effort_limit`, `joint_enabled` | No |

### Performance

- O(N) complexity for articulated chains.
- No constraint drift (reduced coordinates).
- Best for robot arms, humanoids, kinematic trees.

### Working Example

```python
import warp as wp
import newton

builder = newton.ModelBuilder()
builder.add_articulation()

# Link 0: attached to world via revolute joint
b0 = builder.add_rigid_body(pos=wp.vec3(0.0, 1.0, 0.0), mass=1.0)
builder.add_shape_capsule(body=b0, radius=0.05, half_height=0.25)
builder.add_joint_revolute(
    parent=-1, child=b0,
    parent_xform=wp.transform(wp.vec3(0.0, 1.5, 0.0), wp.quat_identity()),
    child_xform=wp.transform(wp.vec3(0.0, 0.25, 0.0), wp.quat_identity()),
    axis=wp.vec3(0.0, 0.0, 1.0),
)

# Link 1: attached to link 0
b1 = builder.add_rigid_body(pos=wp.vec3(0.0, 0.5, 0.0), mass=1.0)
builder.add_shape_capsule(body=b1, radius=0.05, half_height=0.25)
builder.add_joint_revolute(
    parent=b0, child=b1,
    parent_xform=wp.transform(wp.vec3(0.0, -0.25, 0.0), wp.quat_identity()),
    child_xform=wp.transform(wp.vec3(0.0, 0.25, 0.0), wp.quat_identity()),
    axis=wp.vec3(0.0, 0.0, 1.0),
)

model = builder.finalize()
solver = newton.solvers.SolverFeatherstone(model)

state_0 = model.state()
state_1 = model.state()
control = model.control()
collision_pipeline = newton.CollisionPipeline(model)
contacts = collision_pipeline.contacts()

dt = 1.0 / 60.0
for _ in range(240):
    state_0.clear_forces()
    collision_pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0
```

---

## 4. SolverSemiImplicit

Semi-implicit (symplectic) Euler integrator on maximal coordinates. The primary solver for differentiable simulation via `wp.Tape()`.

### Constructor

```python
newton.solvers.SolverSemiImplicit(
    model: Model,
    angular_damping: float = 0.05,            # Angular damping for rigid bodies
    friction_smoothing: float = 1.0,          # Huber norm delta for friction
    joint_attach_ke: float = 1.0e4,           # Joint attachment spring stiffness
    joint_attach_kd: float = 1.0e2,           # Joint attachment spring damping
    enable_tri_contact: bool = True,          # Enable triangle contact
)
```

### Performance

- Very fast (single-pass, no iterations).
- Symplectic (energy-preserving for conservative systems).
- Requires small `dt` for stiff springs; joints enforced via penalty springs, not hard constraints.

### Joint Support

| Feature | Supported |
|---|---|
| PRISMATIC, REVOLUTE, BALL, FIXED, FREE, D6 | Yes |
| DISTANCE | Treated as FREE |
| `joint_enabled` | Yes |
| `joint_target_ke` / `joint_target_kd` | Yes (not enforced for BALL) |
| `joint_limit_ke` / `joint_limit_kd` | Yes (not enforced for BALL) |
| `Control.joint_f` | Yes |
| `joint_armature`, `joint_friction`, `joint_effort_limit` | No |

### Differentiable Simulation with wp.Tape

This is the go-to solver for gradient-based optimization. Requires `builder.finalize(requires_grad=True)`.

```python
import warp as wp
import newton

@wp.kernel
def loss_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    target: wp.vec3,
    loss: wp.array(dtype=float),
):
    delta = particle_q[0] - target
    loss[0] = wp.dot(delta, delta)

builder = newton.ModelBuilder()
builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.0), vel=wp.vec3(1.0, 0.0, 0.0), mass=1.0)

model = builder.finalize(requires_grad=True)
solver = newton.solvers.SolverSemiImplicit(model)

state_in = model.state()
state_out = model.state()
control = model.control()
loss = wp.zeros(1, dtype=float, requires_grad=True)
target = wp.vec3(0.25, 0.0, 0.0)

# Forward pass inside tape
tape = wp.Tape()
with tape:
    state_in.clear_forces()
    solver.step(state_in, state_out, control, None, 1.0 / 60.0)
    wp.launch(loss_kernel, dim=1, inputs=[state_out.particle_q, target], outputs=[loss])

# Backward pass
tape.backward(loss)
grad = state_in.particle_qd.grad.numpy()
print(f"Gradient of loss w.r.t. initial velocity: {grad}")
```

Multi-step rollout -- just put the full loop inside the tape context:

```python
tape = wp.Tape()
with tape:
    for i in range(num_steps):
        states[i].clear_forces()
        solver.step(states[i], states[i + 1], control, None, dt)
    wp.launch(loss_kernel, dim=1, inputs=[states[-1].particle_q, target], outputs=[loss])
tape.backward(loss)
```

**Caveat**: Particle-particle contact gradient computation can cause issues. Disable the particle hash grid via `model.particle_grid = None` if you do not need particle-particle collisions during differentiation.

---

## 5. SolverKamino

GPU-optimized constrained multi-body solver using Proximal-ADMM. Designed for reinforcement learning with thousands of parallel environments. Supports kinematic loops, under/overactuation, hard frictional contacts, and restitutive impacts.

**Status**: Beta. API and internals may change in future releases.

### Constructor

```python
config = newton.solvers.SolverKamino.Config()
solver = newton.solvers.SolverKamino(model, config)
```

### Config

```python
@dataclass
class SolverKamino.Config:
    sparse_jacobian: bool = False              # Sparse Jacobian representations
    sparse_dynamics: bool = False              # Sparse dynamics representations
    use_collision_detector: bool = False       # Use Kamino's own collision detector
    use_fk_solver: bool = False                # Forward kinematics solver for consistent init
    integrator: Literal["euler", "moreau"] = "euler"  # Time integrator
    angular_velocity_damping: float = 0.0     # Angular velocity damping
    rotation_correction: Literal["twopi", "continuous", "none"] = "twopi"
    collect_solver_info: bool = False          # Convergence metrics (slower)
    compute_solution_metrics: bool = False     # Solution quality metrics (slower)
    collision_detector: CollisionDetectorConfig | None = None
    constraints: ConstraintStabilizationConfig | None = None
    dynamics: ConstrainedDynamicsConfig | None = None
    padmm: PADMMSolverConfig | None = None    # Proximal-ADMM solver settings
    fk: ForwardKinematicsSolverConfig | None = None
```

### Control Interface

Kamino converts Newton `Control` objects to its internal `ControlKamino`:

```
tau = tau_j_ff + ke * (q_j_ref - q_j) + kd * (dq_j_ref - dq_j)
```

Where `ke` = `joint_target_ke`, `kd` = `joint_target_kd`, and `q_j_ref` / `dq_j_ref` come from `Control.joint_target_pos` / `Control.joint_target_vel`.

### Joint Support

| Feature | Supported |
|---|---|
| PRISMATIC, REVOLUTE, BALL, FIXED, FREE | Yes |
| D6, DISTANCE, CABLE | No |
| `joint_target_ke` / `joint_target_kd` | Yes |
| `joint_target_mode` | Yes |
| `joint_armature` | Yes |
| `joint_limit_lower` / `joint_limit_upper` | Yes |
| `Control.joint_f` | Yes |
| `joint_friction`, `joint_enabled`, equality/mimic | No |

### Performance

- Optimal for GPU multi-world simulation (1000+ parallel environments).
- Handles hard contacts and kinematic loops.
- Proximal-ADMM converges to accurate constraint solutions.

### Working Example

```python
import warp as wp
import newton

builder = newton.ModelBuilder()
builder.add_articulation()

# Simple pendulum
b = builder.add_rigid_body(pos=wp.vec3(0.5, 1.0, 0.0), mass=1.0)
builder.add_shape_capsule(body=b, radius=0.05, half_height=0.25)
builder.add_joint_revolute(
    parent=-1, child=b,
    parent_xform=wp.transform(wp.vec3(0.0, 1.0, 0.0), wp.quat_identity()),
    child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
    axis=wp.vec3(0.0, 0.0, 1.0),
    target_ke=200.0,
    target_kd=20.0,
)

# Ground plane
builder.add_shape_box(body=-1, hx=5.0, hy=0.1, hz=5.0)

model = builder.finalize()

config = newton.solvers.SolverKamino.Config()
config.integrator = "moreau"
solver = newton.solvers.SolverKamino(model, config)

state_0 = model.state()
state_1 = model.state()
control = model.control()
collision_pipeline = newton.CollisionPipeline(model)
contacts = collision_pipeline.contacts()

dt = 1.0 / 240.0
for _ in range(960):
    state_0.clear_forces()
    collision_pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0
```

---

## 6. SolverVBD

Vertex Block Descent for particles and Augmented VBD (AVBD) for rigid bodies. Implicit solver using block Gauss-Seidel with adaptive penalty parameters.

### Constructor (key parameters)

```python
newton.solvers.SolverVBD(
    model: Model,
    iterations: int = 10,                      # VBD iterations per step
    friction_epsilon: float = 1e-2,            # Friction cone epsilon
    integrate_with_external_rigid_solver: bool = False,
    particle_enable_self_contact: bool = False,
    particle_self_contact_radius: float = 0.2,
    particle_enable_tile_solve: bool = True,   # Warp Tile API for particle solve
    rigid_avbd_beta: float = 1.0e5,            # AVBD penalty scaling
    rigid_avbd_gamma: float = 0.99,            # AVBD penalty update rate
    rigid_contact_k_start: float = 1.0e2,      # Initial contact stiffness
    rigid_joint_linear_k_start: float = 1.0e4, # Initial linear joint stiffness
)
```

### Graph Coloring Requirement

VBD requires graph coloring for safe parallel constraint solving. Call `builder.color()` **before** `builder.finalize()`:

```python
builder = newton.ModelBuilder()
# ... build model ...
builder.color()    # Required for SolverVBD
model = builder.finalize()
solver = newton.solvers.SolverVBD(model)
```

### Supported Entities

- Rigid bodies (AVBD soft constraints), particles, cloth, cables (CABLE joint type with plasticity).
- No volumetric soft bodies.

### Special Notes

- VBD interprets `joint_target_kd` and `joint_limit_kd` as **dimensionless Rayleigh damping coefficients**: `D = kd * ke`. This differs from all other solvers. Watch for excessive damping.
- Supports the unique CABLE joint type (no other solver does).

---

## 7. SolverImplicitMPM

Implicit Material Point Method for granular and elasto-plastic materials (snow, sand, clay, viscous fluids). Based on [Stomakhin et al. 2013](https://doi.org/10.1145/2897824.2925877).

### Constructor

```python
config = newton.solvers.SolverImplicitMPM.Config()
solver = newton.solvers.SolverImplicitMPM(
    model: Model,
    config: SolverImplicitMPM.Config = Config(),
    temporary_store=None,           # Warp FEM temp store for scratch reuse
    verbose: bool = wp.config.verbose,
    enable_timers: bool = False,
)
```

### Config

```python
@dataclass
class SolverImplicitMPM.Config:
    # Numerics
    max_iterations: int = 250          # Rheology solver max iterations
    tolerance: float = 1.0e-4          # Rheology solver tolerance
    solver: str = "gauss-seidel"       # "gauss-seidel", "jacobi", or "cg"
    warmstart_mode: str = "auto"       # "none", "auto", "particles", "grid", "smoothed"

    # Grid
    voxel_size: float = 0.1            # Grid voxel size [m]
    grid_type: str = "sparse"          # "sparse", "dense", "fixed"
    grid_padding: int = 0              # Empty cells around particles
    transfer_scheme: str = "apic"      # "apic" (more accurate) or "pic"
    integration_scheme: str = "pic"    # "pic" or "gimp"

    # Material
    critical_fraction: float = 0.0     # Fraction below which yield surface collapses
    air_drag: float = 1.0              # Background air drag

    # Basis functions (advanced)
    collider_basis: str = "Q1"
    strain_basis: str = "P0"
    velocity_basis: str = "Q1"
```

### Custom Attributes

Call `SolverImplicitMPM.register_custom_attributes(builder)` before building to enable per-particle material properties:

- `mpm:young_modulus` -- Young's modulus [Pa]
- `mpm:poisson_ratio` -- Poisson's ratio [-]
- `mpm:friction` -- Internal friction angle
- `mpm:cohesion` -- Cohesion [Pa]
- `mpm:viscosity` -- Viscosity [Pa*s]
- `mpm:particle_elastic_strain` -- Per-particle elastic strain state

### Supported Entities

- **Particles only**. No rigid bodies, no joints, no articulations.
- Rigid body shapes act as colliders.

### Performance

- Unconditionally stable w.r.t. time step (implicit).
- Suited for stiff granular materials and fully inelastic limit.
- Grid-based computation scales with active particle volume, not particle count.

---

## Decision Matrix: Which Solver to Use

| Use Case | Recommended Solver | Rationale |
|---|---|---|
| **RL with 1000+ parallel envs** | SolverKamino | GPU-optimized multi-world, CUDA graph support |
| **Robot manipulation / articulated bodies** | SolverFeatherstone or SolverMuJoCo | Reduced coordinates, no constraint drift |
| **Differentiable simulation / gradient optimization** | SolverSemiImplicit | Only solver with full wp.Tape() support |
| **Cloth / fabric simulation** | SolverXPBD or SolverVBD | Implicit constraints, good cloth handling |
| **Cable simulation** | SolverVBD | Only solver with CABLE joint type |
| **Granular materials (sand, snow)** | SolverImplicitMPM | Material point method, elasto-plastic models |
| **MuJoCo compatibility / existing MJCF assets** | SolverMuJoCo | Direct MuJoCo integration |
| **General-purpose rigid body** | SolverXPBD | Stable, handles mixed entity types |
| **Kinematic loops** | SolverKamino or SolverMuJoCo | Both handle closed kinematic chains |
| **Simple prototyping** | SolverSemiImplicit | Fast, minimal config |

### Quick Selection Flowchart

1. Need gradients? --> **SolverSemiImplicit** (or SolverFeatherstone for basic diffsim)
2. Need 1000+ parallel worlds for RL? --> **SolverKamino**
3. Need MuJoCo compatibility? --> **SolverMuJoCo**
4. Need reduced coordinates / no drift? --> **SolverFeatherstone** (tree) or **SolverMuJoCo** (general)
5. Cloth or soft bodies? --> **SolverXPBD** or **SolverVBD**
6. Granular / MPM materials? --> **SolverImplicitMPM**
7. Default choice --> **SolverXPBD**

---

## CUDA Graph Capture

CUDA graph capture records a sequence of GPU operations and replays them with minimal CPU overhead. This is critical for RL workloads.

### Pattern

```python
import warp as wp
import newton

# Build model and solver
builder = newton.ModelBuilder()
# ... build scene ...
model = builder.finalize()
solver = newton.solvers.SolverXPBD(model, iterations=4)

state_0 = model.state()
state_1 = model.state()
control = model.control()
collision_pipeline = newton.CollisionPipeline(model)
contacts = collision_pipeline.contacts()
dt = 1.0 / 60.0

# Define the simulation substep
def simulate():
    state_0.clear_forces()
    collision_pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    # Swap handled via aliasing -- state_0/state_1 are reused each capture replay

# Capture to a CUDA graph (only on CUDA devices)
device = wp.get_device()
if device.is_cuda:
    with wp.ScopedCapture() as capture:
        simulate()
    graph = capture.graph
else:
    graph = None

# Main loop -- replay the captured graph
for frame in range(1000):
    if graph:
        wp.capture_launch(graph)
    else:
        simulate()
```

### Multi-Substep Capture

Capture multiple substeps in a single graph for better throughput:

```python
num_substeps = 4
sub_dt = dt / num_substeps

if device.is_cuda:
    with wp.ScopedCapture() as capture:
        for _ in range(num_substeps):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sub_dt)
            state_0, state_1 = state_1, state_0
    graph = capture.graph
```

### Requirements and Constraints

- **CUDA device only** -- CPU fallback needed for non-CUDA.
- **Fixed control flow** -- No data-dependent branching, no resizing arrays.
- **Memory pool**: For solvers that allocate during step (e.g., Kamino), call `wp.set_mempool_enabled(device)` before capture.
- **State aliasing**: The graph replays the exact same GPU operations on the exact same memory. Updating `control` arrays between launches is fine (same memory, new values). Do not reallocate buffers.

---

## Common Pitfalls

### 1. Forgetting to swap states

The double-buffer pattern is mandatory. Without the swap, `state_out` overwrites fresh data each step while `state_in` never updates:

```python
# CORRECT
solver.step(state_0, state_1, control, contacts, dt)
state_0, state_1 = state_1, state_0

# WRONG -- state_0 never advances
solver.step(state_0, state_1, control, contacts, dt)
```

### 2. Forgetting `clear_forces()`

External forces accumulate. Call `state.clear_forces()` at the start of each step:

```python
state_0.clear_forces()
# apply any external forces here
collision_pipeline.collide(state_0, contacts)
solver.step(state_0, state_1, control, contacts, dt)
```

### 3. Missing `builder.color()` for SolverVBD

SolverVBD will produce incorrect results or crash without graph coloring:

```python
builder.color()       # Must be called before finalize()
model = builder.finalize()
```

### 4. SolverSemiImplicit joint stiffness

Joints are enforced as penalty springs (`joint_attach_ke`, `joint_attach_kd`), not hard constraints. Large masses or forces may cause joint drift. Increase `joint_attach_ke` or use a smaller `dt`.

### 5. SolverMuJoCo ignores Newton contacts

When `use_mujoco_contacts=True` (default), the `contacts` argument to `step()` is ignored. MuJoCo runs its own collision detection. Pass `None` to make this explicit:

```python
solver.step(state_0, state_1, control, None, dt)  # MuJoCo handles contacts
```

### 6. SolverFeatherstone requires tree topology

Kinematic loops cause the Featherstone solver to fail. Use SolverKamino or SolverMuJoCo for closed-chain mechanisms.

### 7. VBD damping interpretation

VBD interprets `kd` as a dimensionless Rayleigh damping ratio (`D = kd * ke`), not as an absolute damping coefficient. Values that work in other solvers may produce excessive damping in VBD. Start with small values (0.001-0.01).

### 8. CUDA graph capture with dynamic allocation

Solvers that perform dynamic memory allocation (e.g., collision pipeline resizing) will break CUDA graph capture. Enable Warp's memory pool first:

```python
device = wp.get_device("cuda:0")
wp.set_mempool_enabled(device)
```

### 9. Differentiable simulation memory

Multi-step rollouts with `wp.Tape()` store all intermediate states. For long rollouts, this can exhaust GPU memory. Consider checkpointing or shorter rollout windows.

### 10. SolverImplicitMPM custom attributes

Forgetting to call `SolverImplicitMPM.register_custom_attributes(builder)` before building means per-particle material properties will use defaults. The same applies to `SolverMuJoCo.register_custom_attributes(builder)`.
