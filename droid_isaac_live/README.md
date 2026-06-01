# DROID live demo — policy ⇄ teleop toggle (Isaac Sim)

Live, interactive DROID (Franka + Robotiq 2F-85) simulation in **Isaac Sim / Isaac Lab**
(`arhanjain/sim-evals`), driven by the **openpi `pi0_fast_droid_jointpos`** policy, with
keyboard end-effector teleop for resetting the scene. Press **`T`** to toggle between the
policy running the task and manual teleop.

This is the *Isaac* live demo (the better-looking renderer). The separate Genesis port lives
in `../droid_genesis/`.

## Files
- **`teleop_cam.py`** — runs on the GPU box (`sim-evals` venv). Builds the DROID env at
  480×270 cameras, disables the 30 s auto-reset, serves a web UI + MJPEG stream on `:8080`,
  and runs the main loop: in TELEOP mode it does Jacobian DLS IK from keyboard input; in
  POLICY mode it calls the openpi client (`sim_evals.inference.droid_jointpos.Client`) with
  the instruction `"put the cube in the bowl"` and applies the returned joint-position action.
- **`run_serve.sh`** — runs on the box. Starts the openpi policy server
  (`pi0_fast_droid_jointpos`) listening on `:8000`, capped at 50% GPU
  (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`) so it coexists with the sim.
- **`run_teleop.sh`** — runs on the box. Launches `teleop_cam.py` with
  `OMNI_KIT_ACCEPT_EULA=YES`.
- **`proxy.py`** — runs on the **Mac**. Pulls the box's MJPEG stream (over an SSH tunnel) and
  re-serves it as single-JPEG polling on `localhost:8090` (Safari can't render
  `multipart/x-mixed-replace`). Also forwards `/key`, `/grip`, `/mode` to the box.

## Controls (in the browser)
- **`T`** — toggle POLICY ⇄ TELEOP (current mode shown as an on-screen overlay).
- **Arrow keys** — move the end-effector in the table plane.
- **`R` / `F`** — move the end-effector up / down.
- **Space** — toggle the gripper open/closed.

The wrist (gripper) camera is flipped 180° in the UI only (right pane); the external camera
is the left pane.

## How to run
The box is a Vast.ai GPU instance with **CUDA 12.x driver** (Isaac Sim 5.0's RTX renderer
crashes on CUDA-13 drivers). `sim-evals` and `openpi` are installed under `/root`.

On the box (two processes — kill any stale `python3` first and confirm the GPU + ports
`:8000`/`:8080` are free, see `~/.claude/skills/simulation` §11):
```bash
setsid bash run_serve.sh  > /root/serve.log  2>&1   # policy server :8000  (~3-4 min: checkpoint dl + JAX load)
setsid bash run_teleop.sh > /root/teleop.log 2>&1   # sim + web/stream :8080 (~3-4 min: Isaac build)
```
Wait for `server listening on 0.0.0.0:8000` (serve.log) and `TELEOP_READY_8080` (teleop.log).

On the Mac:
```bash
# tunnel the box's :8080 to localhost
ssh -f -N -L 8080:127.0.0.1:8080 -p <PORT> root@<HOST>
python3 proxy.py            # serves http://localhost:8090
```
Open **http://localhost:8090**. Drive with the keys above; press `T` to hand control to the
policy and `T` again to teleop-reset.

## Notes / gotchas
- Camera resolution is dropped to 480×270 for speed; raise in `teleop_cam.py` (the two
  `_c.width/_c.height` lines) if you want fidelity over framerate.
- `cfg.episode_length_s = 100000.0` disables the env's 30 s auto-reset so it doesn't snap back
  mid-teleop.
- The policy client is created lazily on the first `T`→POLICY (logs `POLICY_CONNECTED`); an
  inference/connection failure logs `POLICY_INFER_ERR` / `POLICY_CONNECT_FAILED` and falls
  back to teleop instead of crashing.
- See `~/.claude/skills/simulation/SKILL.md` for the full set of sim gotchas (GPU-contention
  hangs, kill-before-relaunch, the `pkill -f` self-kill trap, CUDA-driver requirement, etc.).
