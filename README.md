## Installation

This project uses `uv` for dependency management and includes a forked version of the [Newton simulator](https://github.com/newton-physics/newton) as a git submodule.

### 1. Clone the repository
Because this project uses a submodule for the physics engine, you **must** use the `--recursive` flag so Git pulls the simulator source code:

```bash
git clone --recursive git@github.com:FabianDumitrascu/newton-rl.git
cd newton-rl
```
*If you already cloned without the flag, fix it by running `git submodule update --init --recursive` inside the folder.*

### 2. Check your GPU Hardware
Newton is GPU-accelerated via NVIDIA Warp. Before syncing, check your CUDA version:
```bash
nvidia-smi
```

### 3. Sync the Environment
This project is configured as a `uv` workspace. Running the sync command creates a virtual environment (`.venv`) and links the local Newton submodule in **editable mode** (any changes you make to the simulator are live).

Choose the command matching your CUDA version:

**For CUDA 12.x:**
```bash
uv sync --extra torch-cu12
```

**For CUDA 13.x (Newer Hardware):**
```bash
uv sync --extra torch-cu13
```

### 4. Verify the Setup
Run this one-liner to ensure that Python, Torch, and Newton are all successfully talking to your GPU:

```bash
uv run python -c "import torch; import newton; print(f'Torch CUDA: {torch.cuda.is_available()} | Newton Device: {newton.get_device()}')"
```

### 5. Run a Test Script
To see the simulator in action with the aerial manipulator environment:
```bash
uv run scripts/test_sim.py
```
