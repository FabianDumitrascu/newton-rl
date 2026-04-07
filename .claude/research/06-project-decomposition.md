# Project Decomposition: Aerial Manipulation RL

## Phase 0 — Environment Validation
> Get a drone flying in Newton, verify physics are reasonable

- [ ] Load/build a quadcopter model in Newton
- [ ] Implement propeller force model (thrust + reaction torque from ω²)
- [ ] Port INDI inner-loop controller from existing project
- [ ] Fly the drone with manual commands, visually verify physics
- [ ] Compare hover thrust, yaw response, attitude dynamics against expectations
- [ ] Tune propeller coefficients, mass, inertia to match real platform

**Deliverable:** A standalone script that spawns a drone and flies it with the INDI controller in Newton's viewer.

---

## Phase 1 — Aerial Manipulator Model
> Build the full platform: quadcopter + arm + gripper

- [ ] Create/import the aerial manipulator USD/URDF (quad + 2-DOF arm + 2-finger gripper)
- [ ] Define joint hierarchy: FREE root → drone body → arm pitch → arm roll → finger joints
- [ ] Configure joint limits, PD gains, and actuator modes for arm/gripper
- [ ] Test arm movement while hovering (does the arm disturb flight? coupling effects?)
- [ ] Add contact shapes to fingers with appropriate friction
- [ ] Verify grasping: close fingers on a simple object, check if it holds

**Deliverable:** A flying aerial manipulator that can move its arm and grasp a static object.

---

## Phase 2 — RL Environment Scaffold
> Wrap the simulator for RL training

- [ ] Design observation space (drone state, arm joint states, target object pose, contact forces)
- [ ] Design action space (propeller commands + arm joint targets)
- [ ] Build Gymnasium-compatible environment wrapper
- [ ] Implement multi-world batching (1024+ parallel envs)
- [ ] Implement per-world reset with domain randomization
- [ ] Design reward function framework (modular, task-specific)
- [ ] Verify: random actions → observations flow correctly, resets work

**Deliverable:** A vectorized `gymnasium.Env` running 1024+ parallel aerial manipulators.

---

## Phase 3 — RL Training Pipeline
> Train policies for progressively harder tasks

### 3a: Hover Stabilization
- [ ] Train PPO policy to hover at target position
- [ ] Reward: position error + orientation penalty + energy penalty

### 3b: Waypoint Navigation
- [ ] Train policy to fly between waypoints
- [ ] Curriculum: increasing distance, wind disturbances

### 3c: Approach & Grasp
- [ ] Train policy to approach an object and grasp it
- [ ] Reward shaping: approach reward → contact reward → grasp stability reward
- [ ] Handle the contact-force grasping (no runtime joints)

### 3d: Aerial Manipulation
- [ ] Train policy for pick-and-place or object interaction tasks
- [ ] Handle coupled dynamics (grasped object changes flight dynamics)

**Deliverable:** Trained policies for each sub-task, evaluated in simulation.

---

## Phase 4 — Advanced Features (Stretch)
> Depending on thesis scope

- [ ] Hinge interaction (approach a door hinge, attach, manipulate)
- [ ] Dynamic object handoff
- [ ] Multi-agent coordination
- [ ] Sim-to-real transfer considerations

---

## Dependencies Between Phases

```
Phase 0 (validate drone physics)
    ↓
Phase 1 (build aerial manipulator)
    ↓
Phase 2 (RL environment)
    ↓
Phase 3a → 3b → 3c → 3d (progressive training)
    ↓
Phase 4 (stretch goals)
```

## Solver Recommendation

| Phase | Solver | Reason |
|-------|--------|--------|
| 0-1 (validation) | SolverXPBD or SolverMuJoCo | Stable, well-tested, good viewer support |
| 2-3 (RL training) | SolverKamino | GPU-optimized, multi-world, CUDA graphs |
| 0 (if tuning with gradients) | SolverSemiImplicit | Differentiable (for trajectory optimization) |
