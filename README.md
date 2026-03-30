## Installation

This project uses `uv` for lightning-fast dependency management and includes a modified version of the Newton simulator as a git submodule.

**1. Clone the repository (Important: use `--recursive`)**
Because this project uses a submodule for the physics engine, you must use the `--recursive` flag so Git pulls the simulator code as well:
```bash
git clone --recursive git@github.com:FabianDumitrascu/newton-rl.git
cd newton-rl
```
*(If you forgot the flag, you can fix it by running `git submodule update --init --recursive` inside the folder).*

**2. Sync the environment**
Make sure you have [uv installed](https://docs.astral.sh/uv/getting-started/installation/). Then, run:
```bash
uv sync
```
This command automatically creates a virtual environment, installs all exact dependencies from `uv.lock`, and links the local Newton simulator in editable mode.

**3. Run the code**
You can now run any script seamlessly!
```bash
uv run scripts/train.py
```