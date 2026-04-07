# Gymnasium GPU-Batched Environment Patterns

> Best practices for wrapping a custom GPU physics simulator (Newton) as a Gymnasium-compatible RL environment with batched tensor I/O.

---

## Core Interface

### Required gymnasium.Env Methods

```python
import torch
import gymnasium
from gymnasium import spaces
import numpy as np

class VectorizedGPUEnv(gymnasium.Env):
    """Base class for GPU-batched RL environments."""
    
    def __init__(self, num_envs: int, device: str = "cuda:0"):
        super().__init__()
        self.num_envs = num_envs
        self.device = torch.device(device)
        
        # Define spaces (used by RL libraries for network sizing)
        obs_dim = 43  # Example: drone state + arm + contacts
        act_dim = 7   # Example: attitude + joint targets
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )
    
    def reset(self, seed=None, options=None) -> tuple[torch.Tensor, dict]:
        """Reset all environments. Returns (obs, info) as GPU tensors."""
        ...
    
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step all environments. Returns (obs, reward, terminated, truncated, info)."""
        ...
    
    def close(self):
        """Clean up resources."""
        ...
```

**Important:** Spaces are defined with numpy dtypes (Gymnasium convention) but actual data is GPU tensors. RL libraries use spaces only for dimension/range information.

---

## Auto-Reset Pattern

GPU-batched environments cannot stop individual envs. Instead, use **auto-reset**:

```python
def step(self, actions: torch.Tensor):
    # 1. Apply actions and simulate
    self._apply_actions(actions)
    self._simulate_substeps()
    
    # 2. Compute observations, rewards, dones
    obs = self._compute_observations()
    rewards = self._compute_rewards()
    terminated = self._check_termination()
    truncated = self._check_truncation()
    
    # 3. Auto-reset done environments
    done = terminated | truncated
    if done.any():
        # Reset done envs and get fresh observations
        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
        self._reset_envs(reset_ids)
        
        # CRITICAL: Return observations AFTER reset for done envs
        # This is what the next policy step sees
        obs[reset_ids] = self._compute_observations_for(reset_ids)
    
    info = {
        "time_outs": truncated,  # For value bootstrapping
    }
    
    return obs, rewards, terminated, truncated, info
```

**Why auto-reset matters:**
- In vectorized envs, you can't "pause" individual envs
- Done envs must immediately reset so the next `step()` gets valid observations
- The returned `obs` for done envs is the **first observation of the new episode**
- RL algorithms handle the transition correctly using `terminated` and `truncated` flags

### Termination vs Truncation

```python
def _check_termination(self) -> torch.Tensor:
    """True failure conditions — value should be 0."""
    excessive_tilt = self.tilt_angle > 1.0  # > ~57 degrees
    ground_crash = self.drone_z < 0.1
    return excessive_tilt | ground_crash

def _check_truncation(self) -> torch.Tensor:
    """Timeout — value should be bootstrapped."""
    return self.episode_step >= self.max_episode_steps
```

**Key distinction:**
- `terminated=True` → episode ended due to failure, V(s_terminal) = 0
- `truncated=True` → episode ended due to timeout, V(s_terminal) should be bootstrapped
- Use `time_limit_bootstrap=True` in skrl PPO config to handle this

---

## Action Scaling

Map RL policy outputs (typically [-1, 1]) to physical actuator ranges:

```python
class ActionScaler:
    """Maps normalized [-1,1] actions to physical ranges."""
    
    def __init__(self, device):
        # Drone attitude commands
        self.roll_range = torch.tensor([-30.0, 30.0], device=device)   # degrees
        self.pitch_range = torch.tensor([-30.0, 30.0], device=device)
        self.yaw_rate_range = torch.tensor([-180.0, 180.0], device=device)  # deg/s
        self.thrust_range = torch.tensor([0.0, 1.0], device=device)
        
        # Arm joint targets
        self.arm_pitch_range = torch.tensor([-1.57, 1.57], device=device)  # rad
        self.arm_roll_range = torch.tensor([-1.57, 1.57], device=device)
        self.gripper_range = torch.tensor([0.0, 0.08], device=device)  # m
    
    def scale(self, normalized_actions: torch.Tensor) -> dict:
        """Convert [-1,1] to physical units."""
        a = normalized_actions
        return {
            "roll": self._lerp(a[:, 0], self.roll_range),
            "pitch": self._lerp(a[:, 1], self.pitch_range),
            "yaw_rate": self._lerp(a[:, 2], self.yaw_rate_range),
            "thrust": self._lerp(a[:, 3], self.thrust_range),
            "arm_pitch": self._lerp(a[:, 4], self.arm_pitch_range),
            "arm_roll": self._lerp(a[:, 5], self.arm_roll_range),
            "gripper": self._lerp(a[:, 6], self.gripper_range),
        }
    
    @staticmethod
    def _lerp(x, range_tensor):
        """Linear interpolation from [-1,1] to [low,high]."""
        low, high = range_tensor[0], range_tensor[1]
        return low + (x + 1.0) * 0.5 * (high - low)
```

---

## Observation Normalization

Two approaches:

### Option A: In the RL Library (Recommended)

skrl's `RunningStandardScaler` handles this:

```python
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": obs_dim, "device": device}
```

### Option B: In the Environment

```python
class ObservationNormalizer:
    """Running mean/std normalization on GPU."""
    
    def __init__(self, obs_dim, device, clip=5.0):
        self.mean = torch.zeros(obs_dim, device=device)
        self.var = torch.ones(obs_dim, device=device)
        self.count = 0
        self.clip = clip
    
    def update(self, obs: torch.Tensor):
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0)
        batch_count = obs.shape[0]
        self._update_stats(batch_mean, batch_var, batch_count)
    
    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (obs - self.mean) / torch.sqrt(self.var + 1e-8),
            -self.clip, self.clip
        )
```

---

## Reward Shaping Patterns

### Common Components for Robotics

```python
def compute_rewards(self) -> torch.Tensor:
    reward = torch.zeros(self.num_envs, device=self.device)
    
    # Position tracking (negative L2 distance)
    pos_error = torch.norm(self.drone_pos - self.target_pos, dim=-1)
    reward += -1.0 * pos_error
    
    # Orientation penalty (upright bonus)
    # Assuming z-axis of drone should point up
    up_dot = self.drone_up_vector[:, 2]  # dot(drone_z, world_z)
    reward += 0.5 * up_dot
    
    # Energy penalty (discourage high motor output)
    motor_effort = torch.sum(self.motor_commands ** 2, dim=-1)
    reward += -0.01 * motor_effort
    
    # Action smoothness (penalize jerky actions)
    action_diff = torch.norm(self.current_action - self.prev_action, dim=-1)
    reward += -0.1 * action_diff
    
    # Contact reward (for grasping tasks)
    finger_force = torch.norm(self.finger_contact_forces, dim=-1)
    has_contact = (finger_force > 0.1).float()
    reward += 0.5 * has_contact
    
    # Alive bonus (encourage not crashing)
    reward += 0.1
    
    return reward
```

### Curriculum-Dependent Rewards

```python
def compute_rewards(self, curriculum_stage: int) -> torch.Tensor:
    reward = torch.zeros(self.num_envs, device=self.device)
    
    if curriculum_stage >= 0:  # Hover
        reward += self._hover_reward()
    
    if curriculum_stage >= 1:  # Navigation
        reward += self._navigation_reward()
    
    if curriculum_stage >= 2:  # Grasping
        reward += self._grasp_reward()
    
    if curriculum_stage >= 3:  # Manipulation
        reward += self._manipulation_reward()
    
    return reward
```

---

## Curriculum Learning Implementation

```python
class CurriculumManager:
    """Manages progressive difficulty during RL training."""
    
    def __init__(self, stages: list[dict], device: str = "cuda:0"):
        self.stages = stages
        self.current_stage = 0
        self.device = torch.device(device)
        self.success_buffer = []
        self.window_size = 100  # Rolling window for success rate
    
    def update(self, success_flags: torch.Tensor) -> bool:
        """Update with per-env success flags. Returns True if stage advanced."""
        self.success_buffer.extend(success_flags.cpu().tolist())
        self.success_buffer = self.success_buffer[-self.window_size:]
        
        if len(self.success_buffer) >= self.window_size:
            success_rate = sum(self.success_buffer) / len(self.success_buffer)
            threshold = self.stages[self.current_stage].get("advance_threshold", 0.8)
            
            if success_rate >= threshold and self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                self.success_buffer.clear()
                return True
        return False
    
    @property
    def params(self) -> dict:
        """Current stage parameters."""
        return self.stages[self.current_stage]

# Usage
curriculum = CurriculumManager([
    {"name": "hover", "target_height": 1.0, "advance_threshold": 0.9},
    {"name": "navigate", "max_distance": 2.0, "advance_threshold": 0.8},
    {"name": "approach", "object_distance": 0.5, "advance_threshold": 0.7},
    {"name": "grasp", "friction": 0.8, "advance_threshold": 0.6},
])
```

---

## Performance Tips

### Avoid CPU-GPU Transfers

```python
# BAD — transfers to CPU and back
obs_np = state.body_q.numpy()   # GPU → CPU
obs_tensor = torch.from_numpy(obs_np).cuda()  # CPU → GPU

# GOOD — zero-copy on GPU
import warp as wp
obs_tensor = wp.to_torch(state.body_q)  # Same GPU memory, no copy
```

### Minimize CUDA Synchronization

```python
# BAD — forces sync on every step
contact_count = contacts.rigid_contact_count.numpy()[0]  # Sync!
if contact_count > 0:
    ...

# GOOD — keep everything on GPU
has_contacts = contacts.rigid_contact_count > 0  # GPU comparison, no sync
```

### CUDA Graph Capture

```python
# Capture the simulation loop as a CUDA graph for replay
if torch.cuda.is_available():
    with wp.ScopedCapture() as capture:
        simulate_one_frame()
    graph = capture.graph
    
    # In the loop — replay instead of re-launching kernels
    wp.capture_launch(graph)
```

### Profile Your Environment

```python
import torch

# Identify bottlenecks
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(100):
        env.step(random_actions)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

---

## Isaac Lab Environment Pattern (Reference)

The industry-standard GPU env pattern from NVIDIA:

```python
class IsaacLabStyleEnv:
    """Simplified Isaac Lab env pattern for reference."""
    
    def step(self, actions):
        # 1. Apply actions
        self._apply_actions(actions)
        
        # 2. Simulate physics (multiple substeps)
        for _ in range(self.decimation):
            self.sim.step()
        
        # 3. Update buffers
        self._update_buffers()
        
        # 4. Compute dones
        self.terminated, self.truncated = self._get_dones()
        
        # 5. Compute rewards (BEFORE reset, using terminal state)
        self.rewards = self._get_rewards()
        
        # 6. Reset done envs
        done_ids = (self.terminated | self.truncated).nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            self._reset_idx(done_ids)
        
        # 7. Compute observations (AFTER reset, so done envs get fresh obs)
        self.obs = self._get_observations()
        
        return self.obs, self.rewards, self.terminated, self.truncated, self.info
```

**Key ordering:** Rewards computed BEFORE reset (using terminal state), observations computed AFTER reset (fresh state for next episode).

---

## Source References

- [Isaac Lab Environment Design](https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/technical_env_design.html)
- [Isaac Gym Paper](https://arxiv.org/abs/2108.10470)
- [Gymnasium API Docs](https://gymnasium.farama.org/api/env/)
- [skrl Environment Wrapping](https://skrl.readthedocs.io/en/latest/api/envs/wrapping.html)
