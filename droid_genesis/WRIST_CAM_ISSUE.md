# Open issue: Genesis wrist (gripper) camera is misaimed

## Goal
Make the **wrist camera** in the Genesis port (`droid_genesis/franka_droid_env.py`) look the
same way the real DROID / Isaac (`arhanjain/sim-evals`) wrist camera does — down past the
gripper at the workspace. Everything else in the port is verified good (arm at DROID home
pose, real `rubiks_cube` + `_24_bowl` meshes, wood floor + walls, external cam roughly OK).

## Reference: how the wrist cam is defined in sim-evals
`src/sim_evals/environments/droid_environment.py`:
```python
wrist_cam = CameraCfg(
    prim_path=".../robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",   # parent = Robotiq base_link
    offset=CameraCfg.OffsetCfg(
        pos=(0.011, -0.031, -0.074),
        rot=(-0.420, 0.570, 0.576, -0.409),   # quaternion (w,x,y,z)
        convention="opengl"))                  # camera views down local -Z
```

## Why it doesn't transfer directly
- That pose is **relative to the Robotiq 2F-85 `base_link`**. Our Genesis robot is the
  **bundled Franka with the Panda parallel-jaw hand** (`xml/franka_emika_panda/panda.xml`) —
  there is **no Robotiq `base_link`**, and the Panda `hand` frame is oriented differently.
- So the raw quaternion can't be copied. It must be **re-expressed relative to a link that
  exists in BOTH sims** — a Panda *arm* link, `panda_link7` (the arm is identical in both;
  only the gripper differs). Then attach the Genesis camera to `link7` with that transform.
  Genesis cameras also view down -Z, so the opengl convention maps directly.

## The transform we need
```
T_cam_in_link7 = inverse(T_link7_world) @ T_baselink_world @ T_cam_in_baselink
```
- `T_cam_in_baselink` = the config offset above (pos + quat, opengl).
- `T_baselink_world`, `T_link7_world` = from the robot model
  (`assets/franka_robotiq_2f_85_flattened.usd`) at any pose (it's a fixed transform).

## What's been tried (and how it failed)
1. **Full Isaac env probe** (`gym.make("DROID") + reset`, then read `wrist_cam.data.pos_w` +
   robot link poses): **HANGS** — the full env spins up the RTX renderer + 3 cameras and
   stalls (silently) under GPU contention with the policy server, and even alone it stalls on
   the 2nd reset's render cycle. Also hit `CameraData` has no `quat_w` (use `quat_w_opengl`).
2. **Lightweight USD read** (`/root/probe3.py` on the box: `Usd.Stage.Open(franka_robotiq...usd,
   LoadAll)` + `Usd.PrimRange(..., TraverseInstanceProxies())`, find `panda_link7` & `base_link`,
   compose with the config offset): this is the RIGHT approach (see skill §1), but the last run
   **produced no stdout** — needs debugging: capture full stdout+stderr to a file and inspect
   (likely an exception in prim lookup / matrix code, or AppLauncher init issue).

## Concrete next steps for whoever picks this up
- Box: `ssh -p 20722 root@ssh9.vast.ai` (Vast contract 38860722). GPU is currently **free**
  (policy server was killed). `OMNI_KIT_ACCEPT_EULA=YES` is required for any Isaac launch.
- Run `/root/probe3.py` but **tee full output to a file** and read it raw (don't grep) to find
  why it printed nothing. Confirm the exact prim names (`panda_link7`? `base_link`? maybe under
  a `Robotiq_2F_85` scope) — print all prim paths first.
- Once `T_cam_in_link7` is known: in `franka_droid_env.py`, attach `self.wrist_cam` to
  `self.robot.get_link("link7")` with that 4x4 transform (instead of the current `hand` +
  hand-coded rotation), then render the wrist view and eyeball vs `droid_sim/montage.png`.
- Fallback if USD read stays flaky: empirical tuning — attach to `link7`, aim the view by hand,
  render, verify-by-looking, iterate. (The exact numbers just get us there faster.)

## Notes / caveats
- The Genesis camera will see the **Panda parallel-jaw hand**, not the Robotiq gripper, so the
  gripper *appearance* in the wrist view won't match — only the camera **pose** will. Porting
  the Robotiq mesh (USD→OBJ) is a separate task.
- See `~/.claude/skills/simulation/SKILL.md` for the full set of gotchas (USD payloads/instances,
  GPU-contention hangs, Genesis rasterizer-only / no HDR, macOS viewer rules, etc.).
