"""Slowly move the real SO101 follower to a start/home pose (no model in loop).

Home is the MolmoAct2 model-card reference state mapped into the follower's
(v3.0 / use_degrees) frame via the official LEROBOT_V21_COMPAT_DEG inverse — i.e.
the configuration the policy expects to start from. Motion is incremental and
software-clamped (NO lerobot max_relative_target, which amplified a glitchy read
into a slam earlier). Reads + prints each step; stops within --tol.

    .venv-lerobot/bin/python scripts/move_home.py            # dry run (prints plan, no motion)
    .venv-lerobot/bin/python scripts/move_home.py --execute  # actually move, slowly
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # noqa: E402
from molmoact2 import adapter  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# MolmoAct2 HF readme reference robot_state (MODEL frame) -> follower frame.
HOME_MODEL = np.array(
    [-0.52734375, 189.140625, 181.40625, 60.64453125, -3.603515625, 1.0971786975860596],
    dtype=np.float32,
)
HOME = adapter.action_model_to_sim(HOME_MODEL, adapter.LEROBOT_V21_COMPAT_DEG)


def fmt(v):
    return "  ".join(f"{x:8.2f}" for x in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/tty.usbmodem58FD0169761")
    ap.add_argument("--id", default="blupe_follower")
    ap.add_argument("--max-step-deg", type=float, default=4.0, help="per-joint per-step cap")
    ap.add_argument("--tol", type=float, default=3.0, help="stop when all joints within this many deg")
    ap.add_argument("--max-iters", type=int, default=80)
    ap.add_argument("--hz", type=float, default=8.0)
    ap.add_argument("--hold", type=float, default=0.0, help="after reaching home, hold pose (torque on) this many seconds")
    ap.add_argument("--execute", action="store_true", help="actually move (default: dry run)")
    args = ap.parse_args()

    print(f"HOME (follower frame): {fmt(HOME)}")
    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, max_relative_target=None))
    robot.connect(calibrate=False)
    try:
        state = np.array([robot.get_observation()[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
        print(f"start:                 {fmt(state)}")
        print(f"initial |Δ| to home:   {fmt(HOME - state)}  (max {np.abs(HOME-state).max():.1f})")
        if not args.execute:
            print("\n[dry run — nothing moved. add --execute to move]")
            return
        period = 1.0 / args.hz if args.hz > 0 else 0.0
        for it in range(args.max_iters):
            state = np.array([robot.get_observation()[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
            err = HOME - state
            if np.abs(err).max() <= args.tol:
                print(f"reached home at iter {it} (max |Δ| {np.abs(err).max():.2f})")
                break
            target = state + np.clip(err, -args.max_step_deg, args.max_step_deg)
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, target)})
            if it % 5 == 0:
                print(f"it {it:>3}: max|Δ|={np.abs(err).max():5.1f}  send {fmt(target)}")
            time.sleep(period)
        final = np.array([robot.get_observation()[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
        print(f"final:                 {fmt(final)}")
        if args.hold > 0:
            print(f"holding at home (torque ON) for {args.hold:.0f}s...")
            time.sleep(args.hold)
            print("hold done.")
    finally:
        robot.disconnect()
        print("[disconnected — torque relaxed]")


if __name__ == "__main__":
    main()
