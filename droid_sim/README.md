# Launch OpenPi on Droid Sim

A walkthrough for running an [openpi](https://github.com/Physical-Intelligence/openpi)
policy zero-shot on the [`arhanjain/sim-evals`](https://github.com/arhanjain/sim-evals)
Droid simulator: Isaac Sim renders a simulated DROID arm (Franka + Robotiq), and an
openpi policy server drives it from camera images over a websocket.

This guide uses the **`pi0_fast_droid_jointpos`** policy (pi0-FAST-DROID,
joint-position action space). A `pi05_droid_jointpos` checkpoint is also available.

## Requirements

- **GPU: NVIDIA RTX-class with RT cores** (RTX 4090 / A6000 / L40, etc.).
  - ❌ **V100 / A100 / H100 do NOT work** — Isaac Sim's Omniverse RTX renderer
    requires RT cores, which Volta (V100) and the A100/H100 compute cards lack.
- **VRAM ≥ 24 GB.** The policy (~16 GB) and Isaac Sim share one GPU; on a 24 GB
  card it fits with the JAX memory cap below (peaks ~23.4 GB).
- **Disk ~150 GB**, Linux x86_64, recent NVIDIA driver (tested on 560.35.03,
  CUDA 12.4, Ubuntu 22.04).

## Setup

### 1. System libraries (for Isaac Sim headless rendering)
```bash
apt-get update
apt-get install -y --no-install-recommends \
  libgl1 libglib2.0-0 libxrender1 libxext6 libsm6 libice6 libegl1 libgomp1 \
  libgl1-mesa-glx libvulkan1 vulkan-tools libglu1-mesa libnss3 libasound2 \
  libxt6 libxmu6 libxi6 libxrandr2 libxcursor1 libxinerama1 ffmpeg
vulkaninfo --summary   # should list the RTX GPU under deviceName
```

### 2. Install uv and clone sim-evals
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# clone over https without needing ssh keys
git config --global url."https://github.com/".insteadOf "git@github.com:"
git clone --recurse-submodules https://github.com/arhanjain/sim-evals.git
cd sim-evals
```

### 3. Patch pyproject (build fix), then sync
`flatdict` (a transitive Isaac Lab dep) fails under build isolation because it
needs `pkg_resources` but doesn't declare it, and modern setuptools dropped it.
Pin an older setuptools for its build:
```toml
# append to pyproject.toml
[tool.uv.extra-build-dependencies]
flatdict = ["setuptools<80"]
```
```bash
uv sync   # installs Isaac Sim 5.0 + deps (~10+ GB, takes a while)
```

### 4. Headless smoke test (verify the RTX renderer)
```bash
OMNI_KIT_ACCEPT_EULA=YES uv run python - <<'PY'
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app
print("ISAAC_SIM_LAUNCHED_OK"); app.close()
PY
```

### 5. Download simulation assets
```bash
uvx --from huggingface_hub hf download owhan/DROID-sim-environments \
  --repo-type dataset --local-dir assets
# -> assets/scene{1,2,3}.usd, franka_robotiq_2f_85_flattened.usd, table.usd, backgrounds/
```

## Launch the OpenPi policy server

openpi lives in its own repo with its own environment. The joint-position DROID
configs are on the `karl/droid_policies` branch.
```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi && git checkout karl/droid_policies
GIT_LFS_SKIP_SMUDGE=1 uv sync                    # JAX + CUDA, etc.

# Serve. Downloads the checkpoint anonymously from S3 and loads it into the GPU.
# XLA_PYTHON_CLIENT_MEM_FRACTION caps JAX so Isaac Sim can share the GPU.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_fast_droid_jointpos \
  --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos
# -> "server listening on 0.0.0.0:8000"
```
The sim-evals client connects to `localhost:8000` and sends DROID-format
observations (`exterior_image_1_left`, `wrist_image_left`, `joint_position`,
`gripper_position`, `prompt`); the server returns an action chunk.

## Run the evaluation (separate terminal)
```bash
cd sim-evals
OMNI_KIT_ACCEPT_EULA=YES uv run python run_eval.py --episodes 1 --scene 1 --headless
```
- Scenes: `1` = "put the cube in the bowl", `2` = "put the can in the mug",
  `3` = "put banana in the bin".
- **`OMNI_KIT_ACCEPT_EULA=YES` is required** for non-interactive/headless runs,
  otherwise Isaac Sim blocks on an EULA prompt (`EOFError` when detached).
- A full episode is 450 steps. For a quick behavior check, cap the step count.

## Viewing rollouts
Episodes are written to `runs/<date>/<time>/episode_<n>.mp4`. Each frame is the
two camera views the policy sees (exterior + wrist), side by side.
```bash
# montage grid of sampled frames
ffmpeg -i episode_0.mp4 -vf "select=not(mod(n\,10)),scale=320:-1,tile=4x3" -frames:v 1 montage.png
# animated gif
ffmpeg -i episode_0.mp4 -vf "fps=12,scale=560:-1" rollout.gif
```

## Interactive teleop + eval demo
`run_eval.py` above runs fixed autonomous rollouts. For the **interactive** version —
toggle policy ⇄ keyboard teleop, a live mode/timer/scoreboard, and screen recording —
see **[`../droid_isaac_live/`](../droid_isaac_live/)** (`README.md` there). It replaces
`run_eval.py` with `teleop_cam.py` (same sim-evals env + openpi server, plus an HTTP
control/stream server) and a native pygame client on your laptop that connects over a
Vast-mapped port (no SSH tunnel). Same box setup as above; same `run_serve.sh` policy server.

## Gotchas
- **V100 → dead end.** No RT cores; Isaac Sim's renderer won't init. Use RTX.
- **flatdict build error** (`No module named 'pkg_resources'`) → pin `setuptools<80`.
- **EULA `EOFError`** on headless/detached runs → set `OMNI_KIT_ACCEPT_EULA=YES`.
- **One GPU, two consumers** → keep `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` so the
  policy and Isaac Sim coexist in 24 GB (peaks ~23.4 GB).
- `scene*.usd` references a missing `my_droid.usdz` payload — harmless warning;
  the robot is spawned separately by the env config.
