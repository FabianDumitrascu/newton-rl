# skrl — Reinforcement Learning Library Reference

> skrl is a modular RL library implemented in PyTorch, JAX, and NVIDIA Warp with native support for GPU-batched environments (Isaac Lab, Isaac Gym, custom simulators).

**Docs:** https://skrl.readthedocs.io | **GitHub:** https://github.com/Toni-SM/skrl | **Version:** 1.4.x+

## Installation

```bash
pip install skrl
# or with extras
pip install skrl[torch]  # PyTorch backend
```

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Wrapper** | Adapts an environment to skrl's interface (num_envs, device, tensor I/O) |
| **Model** | Neural network (policy, value) built with Mixins (Gaussian, Deterministic) |
| **Agent** | RL algorithm (PPO, SAC, TD3, etc.) that trains models |
| **Trainer** | Orchestrates the train/eval loop (Sequential, Parallel, Step) |
| **Memory** | Rollout buffer storing transitions |

---

## Environment Wrapper

### Base Wrapper Interface

All custom environments must expose this interface:

```python
class Wrapper:
    @property
    def device(self) -> torch.device: ...          # cuda:0 or cpu
    
    @property
    def num_envs(self) -> int: ...                  # Number of parallel envs
    
    @property
    def observation_space(self) -> gymnasium.Space: ...
    
    @property
    def action_space(self) -> gymnasium.Space: ...
    
    @property
    def state_space(self) -> gymnasium.Space | None: ...  # Optional (asymmetric AC)
    
    def reset(self) -> tuple[torch.Tensor, dict]: ...
    
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Returns: (obs, reward, terminated, truncated, info)"""
        ...
    
    def render(self, *args, **kwargs) -> Any: ...
    
    def close(self) -> None: ...
```

### Wrapping with wrap_env()

```python
from skrl.envs.wrappers.torch import wrap_env

# Wrap a Gymnasium environment
env = wrap_env(gym_env, wrapper="gymnasium")

# Wrap an Isaac Lab environment
env = wrap_env(isaac_env, wrapper="isaaclab")

# Auto-detect wrapper type
env = wrap_env(any_env, wrapper="auto")
```

### Custom Wrapper for Newton

For our GPU-batched Newton simulator, we write a custom wrapper:

```python
import torch
import gymnasium
from skrl.envs.wrappers.torch import Wrapper

class NewtonWrapper(Wrapper):
    """Wraps Newton multi-world simulation for skrl."""
    
    def __init__(self, env):
        super().__init__(env)
        # env is our AerialManipulatorEnv
        self._env = env
    
    @property
    def device(self) -> torch.device:
        return self._env.device
    
    @property
    def num_envs(self) -> int:
        return self._env.num_envs
    
    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        return self._env.observation_space
    
    @property
    def action_space(self) -> gymnasium.spaces.Box:
        return self._env.action_space
    
    def reset(self) -> tuple[torch.Tensor, dict]:
        obs, info = self._env.reset()
        return obs, info  # obs shape: (num_envs, obs_dim)
    
    def step(self, actions: torch.Tensor):
        obs, reward, terminated, truncated, info = self._env.step(actions)
        # skrl expects reward shape: (num_envs, 1)
        return obs, reward.unsqueeze(-1), terminated, truncated, info
    
    def render(self, *args, **kwargs):
        return None
    
    def close(self):
        self._env.close()
```

**Key:** skrl expects reward tensors shaped `(num_envs, 1)` — add `.unsqueeze(-1)` if your env returns `(num_envs,)`.

---

## Model Definition

### GaussianMixin (Stochastic Policy)

For continuous action spaces (our aerial manipulator):

```python
import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin

class Policy(GaussianMixin, Model):
    """Stochastic policy for PPO — outputs mean action + learned log_std."""
    
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std,
                               min_log_std, max_log_std, reduction)
        
        obs_dim = self.num_observations
        act_dim = self.num_actions
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.mean = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # Learned std
    
    def compute(self, inputs, role=""):
        """Called by GaussianMixin.act() to get mean and log_std."""
        x = self.net(inputs["states"])
        return self.mean(x), self.log_std, {}
```

**GaussianMixin.act() flow:**
1. Calls `self.compute(inputs)` → gets (mean, log_std, outputs)
2. Clamps log_std to [min_log_std, max_log_std]
3. Creates Normal distribution: `Normal(mean, log_std.exp())`
4. Samples action via `rsample()` (reparameterized)
5. Optionally clips actions to action_space bounds
6. Computes log_prob with specified reduction ("sum", "mean", "prod", "none")
7. Returns (actions, log_prob, outputs)

### DeterministicMixin (Value Function)

```python
from skrl.models.torch import Model, DeterministicMixin

class Value(DeterministicMixin, Model):
    """Value function for PPO — outputs scalar V(s)."""
    
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        
        obs_dim = self.num_observations
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )
    
    def compute(self, inputs, role=""):
        return self.net(inputs["states"]), {}
```

### Shared Actor-Critic (Optional)

```python
class SharedActorCritic(GaussianMixin, DeterministicMixin, Model):
    """Shared backbone with separate policy/value heads."""
    
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=False, reduction="sum")
        DeterministicMixin.__init__(self)
        
        obs_dim = self.num_observations
        act_dim = self.num_actions
        
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
        )
        self.policy_head = nn.Linear(128, act_dim)
        self.value_head = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def compute(self, inputs, role=""):
        features = self.backbone(inputs["states"])
        if role == "policy":
            return self.policy_head(features), self.log_std, {}
        elif role == "value":
            return self.value_head(features), {}
        return self.policy_head(features), self.log_std, {}
```

---

## PPO Agent

### Configuration

```python
PPO_DEFAULT_CONFIG = {
    # Rollout
    "rollouts": 16,                    # Steps per env before update
    "learning_epochs": 8,              # PPO epochs per update
    "mini_batches": 2,                 # Mini-batches per epoch
    
    # GAE
    "discount_factor": 0.99,           # Gamma
    "lambda": 0.95,                    # GAE lambda
    
    # Optimization
    "learning_rate": 1e-3,
    "learning_rate_scheduler": None,   # e.g., KLAdaptiveLR
    "learning_rate_scheduler_kwargs": {},
    "grad_norm_clip": 0.5,             # Max gradient norm
    
    # Clipping
    "ratio_clip": 0.2,                 # PPO clip ratio (epsilon)
    "value_clip": 0.2,                 # Value function clip range
    "clip_predicted_values": False,    # Whether to clip V predictions
    
    # Loss weights
    "entropy_loss_scale": 0.0,         # Entropy bonus coefficient
    "value_loss_scale": 1.0,           # Value loss coefficient
    
    # KL divergence
    "kl_threshold": 0,                 # Early stop if KL > threshold (0 = disabled)
    
    # Preprocessing
    "state_preprocessor": None,        # e.g., RunningStandardScaler
    "state_preprocessor_kwargs": {},
    "value_preprocessor": None,
    "value_preprocessor_kwargs": {},
    
    # Misc
    "random_timesteps": 0,             # Random actions at start
    "learning_starts": 0,              # Steps before training starts
    "rewards_shaper": None,            # Reward transformation function
    "time_limit_bootstrap": False,     # Bootstrap V on timeout (not termination)
    "mixed_precision": False,          # FP16 training
    
    # Logging
    "experiment": {
        "directory": "",               # Log directory
        "experiment_name": "",         # Experiment name
        "write_interval": "auto",      # TensorBoard write frequency
        "checkpoint_interval": "auto", # Checkpoint save frequency
        "store_separately": False,     # Separate files per checkpoint
        "wandb": False,                # Enable W&B logging
        "wandb_kwargs": {},
    },
}
```

### Setting Up PPO

```python
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory

# Configure
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 32                    # 32 steps per env between updates
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 4
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["entropy_loss_scale"] = 0.01        # Small entropy bonus
cfg["value_loss_scale"] = 0.5
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": obs_dim, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 5000

# Memory (rollout buffer)
memory = RandomMemory(memory_size=cfg["rollouts"], num_envs=env.num_envs, device=device)

# Models
models = {
    "policy": Policy(env.observation_space, env.action_space, device),
    "value": Value(env.observation_space, env.action_space, device),
}

# Agent
agent = PPO(
    models=models,
    memory=memory,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
    cfg=cfg,
)
```

---

## Trainer

### SequentialTrainer

```python
from skrl.trainers.torch import SequentialTrainer

trainer_cfg = {
    "timesteps": 1_000_000,          # Total training timesteps
    "headless": True,                 # No rendering during training
    "disable_progressbar": False,
    "close_environment_at_exit": True,
}

trainer = SequentialTrainer(
    env=env,           # Wrapped environment
    agents=agent,      # PPO agent (or list of agents)
    cfg=trainer_cfg,
)

# Train
trainer.train()

# Evaluate
trainer.eval()
```

### Training Loop Internals

Each timestep:
1. `agent.pre_interaction(timestep)` — update schedulers, etc.
2. `actions = agent.act(obs)` — policy forward pass
3. `next_obs, reward, terminated, truncated, info = env.step(actions)`
4. `agent.record_transition(...)` — store in memory
5. `agent.post_interaction(timestep)` — train if rollout complete

---

## Observation Preprocessing

```python
from skrl.resources.preprocessors.torch import RunningStandardScaler

cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": obs_dim, "device": device}

cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
```

RunningStandardScaler computes online mean/std and normalizes observations. Highly recommended for stable PPO training.

---

## Checkpointing & Evaluation

### Save/Load

```python
# During training — automatic via checkpoint_interval
# Manual save:
agent.save("path/to/checkpoint.pt")

# Load for evaluation:
agent.load("path/to/checkpoint.pt")
```

### Evaluate Trained Policy

```python
# Create agent with trained weights
agent = PPO(models=models, memory=None, cfg=eval_cfg, ...)
agent.load("best_checkpoint.pt")

trainer = SequentialTrainer(env=env, agents=agent, cfg={"timesteps": 1000, "headless": False})
trainer.eval()
```

---

## TensorBoard Integration

skrl automatically logs to TensorBoard when `experiment.directory` is set:

```python
cfg["experiment"]["directory"] = "runs"
cfg["experiment"]["experiment_name"] = "hover_ppo"
cfg["experiment"]["write_interval"] = 100      # Log every 100 steps
cfg["experiment"]["checkpoint_interval"] = 5000
```

**Auto-logged metrics:**
- Reward (mean, max, min per episode)
- Episode length
- Policy loss, value loss, entropy
- Learning rate
- KL divergence (if kl_threshold > 0)

View with: `tensorboard --logdir runs/`

---

## Complete Training Example

```python
import torch
import gymnasium
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

# --- 1. Define Models ---
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True, reduction="sum")
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(),
            torch.nn.Linear(128, self.num_actions),
        )
        self.log_std = torch.nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role=""):
        return self.net(inputs["states"]), self.log_std, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )

    def compute(self, inputs, role=""):
        return self.net(inputs["states"]), {}

# --- 2. Create Environment ---
# env = YourNewtonEnv(num_envs=1024, device="cuda:0")
# env = NewtonWrapper(env)  # Our custom wrapper from above

# --- 3. Configure PPO ---
device = torch.device("cuda:0")
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 32
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 4
cfg["learning_rate"] = 3e-4
cfg["entropy_loss_scale"] = 0.01
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space.shape[0], "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
cfg["experiment"]["directory"] = "runs"
cfg["experiment"]["experiment_name"] = "aerial_manipulator_hover"
cfg["experiment"]["checkpoint_interval"] = 5000

# --- 4. Create Agent ---
models = {
    "policy": Policy(env.observation_space, env.action_space, device),
    "value": Value(env.observation_space, env.action_space, device),
}
memory = RandomMemory(memory_size=cfg["rollouts"], num_envs=env.num_envs, device=device)
agent = PPO(models=models, memory=memory,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device, cfg=cfg)

# --- 5. Train ---
trainer = SequentialTrainer(env=env, agents=agent,
                            cfg={"timesteps": 1_000_000, "headless": True})
trainer.train()
```

---

## Recommended Hyperparameters for Aerial Manipulation

Based on Isaac Lab robotics tasks:

```python
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 24          # 24 steps per env (short rollouts for fast feedback)
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 4       # num_envs * rollouts / mini_batches = mini-batch size
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["entropy_loss_scale"] = 0.005  # Small but nonzero for exploration
cfg["value_loss_scale"] = 1.0
cfg["mixed_precision"] = True   # Faster on modern GPUs
```

---

## Source Files

- Wrapper base: `skrl/envs/wrappers/torch/base.py`
- IsaacLab wrapper: `skrl/envs/wrappers/torch/isaaclab_envs.py`
- PPO agent: `skrl/agents/torch/ppo/ppo.py`
- Gaussian model: `skrl/models/torch/gaussian.py`
- Deterministic model: `skrl/models/torch/deterministic.py`
- Sequential trainer: `skrl/trainers/torch/sequential.py`
