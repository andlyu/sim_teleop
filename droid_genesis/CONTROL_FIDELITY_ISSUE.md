# [RESOLVED] Genesis Franka arm won't hold commanded joint pose

## RESOLUTION
Root cause was **bad collision geometry**, not gains/gravity/friction. The gripper
`base_link` convexified into a huge hull penetrating the floor (`Plane <-> base_link`,
pen 0.065 at z=-0.032); that ground-contact force shoved the arm, and the PD balanced
it at ~0.3 rad. Fix: `gs.morphs.USD(..., collision=False)` on the robot → arm now tracks
commanded joints with **0.0 error** (home and perturbed targets). `gravity_compensation=1.0`
is also kept (matches sim-evals disable_gravity).
**Caveat / follow-up:** with robot collision off, the gripper can't physically grasp the
cube. Re-enable collision with proper per-link convex decomposition (CoACD) when grasp
contact is needed — the single-hull convexify on the gripper is the thing to fix.

---
(original investigation below)

# Open issue for Codex input: Genesis Franka arm won't hold commanded joint pose

## Context
We ported the DROID (sim-evals) setup into Genesis at `sim_teleop/droid_genesis/`.
The robot is now the **real Franka + Robotiq 2F-85 USD** (`assets/franka_robotiq_2f_85_flattened.usd`,
loaded via `gs.morphs.USD`). Wrist cam is on the real `base_link`, cube/bowl colored. That all looks right.

**Problem:** the arm does **not** track its commanded joint-position targets. Commanded the
DROID home pose `[0, -0.628, 0, -2.513, 0, 1.885, 0]`; it settles to
`[0, -0.313, 0, -2.426, 0, 2.132, 0]` → **max error ~0.30 rad**, and it barely improves with
more steps (0.317 → 0.296 over ~750 steps). The arm looks limp ("torque disabled").

## What we've checked (all on the built robot, arm DOFs)
- `kp = 400`, `kv = 80` (set via `set_dofs_kp/kv`, confirmed applied)
- `stiffness = 0`, `damping = 0` (no passive joint spring)
- `force_range = ±100000` (torque NOT clamped)
- `armature = 0.1`
- `gravity_compensation`: set to **1.0** via `material=gs.materials.Rigid(gravity_compensation=1.0)`
  (confirmed `robot.gravity_compensation == 1.0`). **Error unchanged** → not a gravity problem.
- Control loop: `control_dofs_position(arm_target, arm_dofs)` each substep; decimation 8, sim dt 1/120.

## Leading hypothesis (untested — user paused before we ran it)
**Joint `frictionloss` imported from the USD.** A static-friction deadband gives steady-state
error ≈ `frictionloss / kp`. With kp=400 and ~0.30 rad error, that implies frictionloss ~120 —
consistent with the slow creep toward target. Planned test: `get_dofs_frictionloss(arm_dofs)`,
then `set_dofs_frictionloss(0, arm_dofs)` and re-drive to home; expect error → ~0.

## Questions for Codex
1. Is `frictionloss` the right culprit, or is something else likely (e.g. `control_dofs_position`
   semantics in Genesis, `act_gain`/`act_bias`, armature, an implicit-actuator import from USD)?
2. Right fix: zero `frictionloss`, or raise kp, or use `control_dofs_force` with explicit PD +
   the friction feedforward? What matches sim-evals' implicit actuators best (stiffness 400 /
   damping 80, effort 87/12) without making the sim stiff/unstable at dt=1/120?
3. Any Genesis-specific gotcha with USD-imported joint drives we should know?

## How to reproduce
```
cd /Users/andrewlyubovsky/Projects/Blupe/sim_teleop
.venv/bin/python -c "
import sys; sys.path.insert(0,'droid_genesis'); import numpy as np
from franka_droid_env import FrankaDroidEnv, HOME_POSE
env=FrankaDroidEnv(scene_id=1, viewer=False); o=env.reset()
print('err', np.abs(o['arm_joint_pos']-HOME_POSE).max())
print('frictionloss', np.asarray(env.robot.get_dofs_frictionloss(env.arm_dofs)))
"
```
Files: `droid_genesis/franka_droid_env.py` (env), `franka_droid_viewer.py` (viewer).
