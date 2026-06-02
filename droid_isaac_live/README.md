# DROID live demo — policy ⇄ teleop, eval scoreboard, recording

Interactive DROID (Franka + Robotiq 2F-85) simulation in **Isaac Sim** (`arhanjain/sim-evals`),
driven by the **openpi `pi0_fast_droid_jointpos`** policy, with keyboard end-effector teleop for
resetting. Toggle between the policy running the task and manual teleop, score each policy run on
a 10-trial scoreboard, and record the panel to a video.

The separate Genesis port lives in `../droid_genesis/`.

## Architecture (what runs where)
```
  Mac (native_viewer.py, pygame)                Vast box (Korea, RTX 4090, CUDA 12.x)
  ┌───────────────────────────┐   direct TCP   ┌──────────────────────────────────────┐
  │ window: 2 cams + scoreboard│ ────────────►  │ teleop_cam.py  (HTTP :8080)            │
  │ keep-alive cmds, TCP_NODELAY│   116.127...:  │  - DROID Isaac env (sim-evals)         │
  │ /stream frames              │     31755      │  - 6-DOF IK teleop / policy client     │
  └───────────────────────────┘                │           │ localhost:8000 (websocket) │
                                                │           ▼                            │
                                                │ serve_policy.py (openpi pi0-FAST)      │
                                                └──────────────────────────────────────┘
```
- Container port **8080 is mapped to host 31755**, so the Mac connects **directly** to
  `116.127.115.27:31755` — no SSH tunnel (lower latency + no tunnel drops).
- Command latency ≈ **155 ms** (the US↔Korea network floor; SSH-tunnel overhead and TCP-Nagle
  were removed via the direct port + `TCP_NODELAY`). Video ≈ **6 fps** (the RTX render of two
  640×360 cameras — `step` dominates; see `/timing`). A US box would cut latency to ~20–40 ms.

## Files
- **`teleop_cam.py`** (runs on the box, in the `sim-evals` venv) — builds the DROID env at
  640×360, serves HTTP on `:8080` (`/stream`, `/key`, `/grip`, `/mode`, `/reset`, `/state`,
  `/timing`), HTTP/1.1 keep-alive + `TCP_NODELAY`. Teleop = 6-DOF Jacobian DLS IK (position +
  orientation locked to the home/downward pose). Policy mode calls
  `sim_evals.inference.droid_jointpos.Client` → `localhost:8000` with the instruction
  `"put the rubik's cube in the red bowl"`. `/reset` re-homes the arm to its default joint config.
- **`run_serve.sh`** (box) — starts the openpi policy server on `:8000`
  (`pi0_fast_droid_jointpos`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` to share the GPU).
- **`run_teleop.sh`** (box) — launches `teleop_cam.py` with `OMNI_KIT_ACCEPT_EULA=YES`.
- **`native_viewer.py`** (Mac, **the primary client**) — pygame window. One persistent `/stream`
  connection + one keep-alive command connection (`TCP_NODELAY`). Mode buttons (REMOTE TELEOP /
  PI0.5 POLICY), SUCCESS/FAIL/RESET, a policy timer, a **scoreboard** of trials T1–T10 (time +
  green ✓ / red ✗), and a **REC** button that records the composited panel to
  `/tmp/droid_panel_<ts>.mp4` (no OS screen-recording permission needed). Point it at the box by
  editing `HOST, PORT` at the top.
- **`proxy.py`** (Mac, browser fallback) — re-serves the MJPEG stream as single-JPEG polling on
  `localhost:8090` for Safari, with on-screen buttons. Needs an SSH tunnel `localhost:8080 → box`.
  Superseded by `native_viewer.py`; kept for browser viewing.

## Launch
On the box (kill stale `teleop_cam.py` first; leave the policy server up — it's slow to reload):
```bash
setsid bash run_serve.sh  > /root/serve.log  2>&1   # policy server :8000  (~3-4 min: ckpt + JAX)
setsid bash run_teleop.sh > /root/teleop.log 2>&1   # Isaac sim + :8080    (~3-4 min build)
```
Wait for `server listening on 0.0.0.0:8000` and `TELEOP_READY_8080`.

On the Mac (no tunnel needed — direct mapped port):
```bash
python3 native_viewer.py        # needs: pip install pygame numpy imageio imageio-ffmpeg
```

## Controls (native viewer)
- **Arrows / R / F** — move the EE (in-plane + up/down); gripper stays facing down.
- **Space** — toggle the gripper.
- **PI0.5 POLICY / REMOTE TELEOP** buttons (or **T**) — switch who's driving.
- **SUCCESS / FAIL** (or **S** / **X**) — score the current/last policy run (manual, nothing auto-counts).
- **RESET** (or **G**) — reset scene, arm to home.
- **REC** (or **C**) — start/stop recording the panel to mp4.
- **Esc** — quit.

## Notes / gotchas
- Isaac Sim 5.0's RTX renderer crashes on CUDA-13 drivers — the box needs a CUDA-12.x driver.
- The policy resizes both camera images to **224×224** (`resize_with_pad`), so anything ≥224 is
  downsampled — render resolution mainly affects *your* view, not the policy input.
- Assets come from the HF dataset `owhan/DROID-sim-environments` (see `../droid_genesis/ASSETS.md`).
- See `~/.claude/skills/simulation/SKILL.md` for the broader sim gotchas (GPU-contention hangs,
  kill-before-relaunch, the `pkill -f` self-kill trap, etc.).
