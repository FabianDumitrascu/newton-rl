# Design Questions for Newton-RL

## Drone Platform

1. **Which drone model are we targeting?** The Newton example uses Crazyflie parameters (46g, 0.2286m props). Your aerial manipulator will be much larger/heavier. What are the real platform specs (mass, arm length, prop diameter, motor KV)?

2. **Do you have a USD/URDF of the aerial manipulator?** Or do we need to build it from scratch in the ModelBuilder? If you have CAD, we can export to USD/URDF.

3. **Propeller model fidelity:** The existing Newton example uses `force = coeff * ρ * n² * d⁴`. Your INDI controller uses `ω² * parameter`. Are these the same model? Can you share the exact parameter values from your other project?

4. **What's the INDI controller structure?** Specifically: what are the inputs (attitude commands? rate commands? motor RPMs?), what are the outputs (individual motor commands?), and what's the control frequency?

## Arm & Gripper

5. **Arm DOF:** You mentioned pitch + roll for the arm. Is it exactly 2 DOF? What are the joint limits and max torques?

6. **Gripper type:** 2 fingers that open/close — is this a parallel gripper (1 DOF) or independent fingers (2 DOF)? What grip force can it exert?

7. **Arm/gripper actuators:** Are these servos with position control, or do you want direct torque control? What are the response characteristics?

8. **Coupling effects:** How much does the arm movement disturb the drone's flight? Is the arm mass significant relative to the drone mass? This determines whether we need the INDI controller to compensate for arm dynamics.

## Grasping Strategy

9. **Contact-force vs. fixed-joint grasping:** Newton can't create joints at runtime. Are you okay with contact-force-only grasping (using friction)? This is physically more realistic but harder to learn. The alternative is pre-allocating fixed joints for known objects.

10. **What objects will be grasped?** Simple shapes (spheres, boxes) or complex objects? How heavy relative to the drone?

11. **Grasp success criteria:** What defines a successful grasp? Object held for N seconds? Object transported to target? Contact force above threshold?

## RL Training

12. **RL library preference:** Do you have experience with any specific RL library? Options:
    - **rsl_rl** (NVIDIA's, designed for Isaac-like GPU envs)
    - **rl_games** (fast GPU-based PPO/SAC)
    - **CleanRL** (simple, educational, single-file implementations)
    - **Stable Baselines 3** (most popular, but CPU-centric)
    - **Custom** (roll our own PPO with PyTorch)

13. **Training compute:** What GPU do you have? How much VRAM? This determines max parallel environments (each env uses ~1-5MB depending on complexity).

14. **Curriculum strategy:** Do you want to train end-to-end (hover → approach → grasp → manipulate) or separate policies per sub-task with handoff?

15. **Observation space design:** What should the agent observe?
    - Drone state (position, velocity, orientation, angular velocity)
    - Arm joint positions and velocities
    - Target object pose (relative to drone? absolute?)
    - Contact forces on fingers
    - Any proprioceptive sensors (IMU)?
    - History/stacking of past observations?

16. **Action space design:** What should the agent control?
    - Option A: Raw motor RPMs (4) + arm joint targets (2+) + gripper (1-2)
    - Option B: Attitude commands (roll, pitch, yaw, thrust) + arm + gripper (INDI handles low-level)
    - Option C: Velocity commands (vx, vy, vz, yaw_rate) + arm + gripper

## Simulation

17. **Solver choice for RL:** SolverKamino is GPU-optimized for multi-world RL but is relatively new. SolverXPBD/MuJoCo are more mature. Preference?

18. **Simulation frequency:** What dt should we use? Drone control typically needs 200-500 Hz. RL policies typically run at 20-50 Hz. Do we substep (e.g., 10 sim steps per policy step)?

19. **Domain randomization:** What parameters should we randomize for robustness? Mass, inertia, friction, wind, motor response delay, sensor noise?

20. **Termination conditions:** When should an episode end? Drone crash, timeout, task success, object dropped, excessive tilt angle?

## Integration & Architecture

21. **Monorepo vs. separate packages?** Should the RL environment wrapper live in this repo alongside Newton, or as a separate installable package?

22. **Do you want viewer/rendering during training?** Newton has `ViewerGL` for interactive visualization. Useful for debugging but slows training. Record videos periodically instead?

23. **Logging & experiment tracking:** Any preference? WandB, TensorBoard, MLflow?

24. **The hinge interaction task** (from your description): Can you elaborate? Is this like opening a door — fly to hinge, grasp handle, rotate? This would be a very different task from pick-and-place.
