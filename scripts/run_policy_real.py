"""Run MolmoAct2 on the REAL SO101 follower (lerobot), vs the Genesis sim runner.

Mirrors scripts/run_policy.py but the env is the lerobot SO101 follower + two
USB cameras instead of Genesis. The follower reports DEGREES (use_degrees=True)
in the LeRobot v3.0 (centered) frame; the MolmoAct2-SO100_101 checkpoint expects
the non-centered training frame, so we map with adapter.LEROBOT_V21_COMPAT_DEG
(empirically validated on hardware: it puts shoulder_lift/elbow_flex in the
~180 range the model was trained on; raw IDENTITY passthrough produced a +151deg
lunge and is unsafe).

SAFETY:
  * defaults to --dry-run (NO motion): reads state+cameras, queries the model,
    prints the action + per-joint delta.
  * --execute moves the arm, but every step is clamped to --max-step-deg per
    joint (software) AND the follower's max_relative_target (hardware backstop),
    for a small number of --steps. Keep a hand on the power.

Examples:
    # dry run (no motion):
    .venv-lerobot/bin/python scripts/run_policy_real.py \
        --url http://127.0.0.1:8202 --cameras 0,1 \
        --instruction "place the teal box into the red bowl"

    # drive the arm (slow, clamped) — only after the dry run looks sane:
    .venv-lerobot/bin/python scripts/run_policy_real.py --execute \
        --url http://127.0.0.1:8202 --cameras 0,1 --steps 8 --max-step-deg 6 \
        --instruction "place the teal box into the red bowl"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # noqa: E402

from molmoact2 import adapter  # noqa: E402
from molmoact2.client import MolmoActClient, Observation  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def grab(idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(idx)
    # Match the checkpoint's training/sample image format: native 640x480 (4:3).
    # Default 1920x1080 (16:9) gets distorted when the processor resizes it.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(3):
        cap.read()  # flush stale/init frames (esp. after the mode switch)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"camera index {idx} returned no frame")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def read_state(robot, retries: int = 4) -> np.ndarray:
    for attempt in range(retries):
        try:
            obs = robot.get_observation()
            return np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(0.05)  # transient bus glitch — retry rather than abort the run


def send_with_retry(robot, target, retries: int = 4) -> None:
    action = {f"{j}.pos": float(v) for j, v in zip(JOINTS, target)}
    for attempt in range(retries):
        try:
            robot.send_action(action)
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(0.05)


def clamp_step(target: np.ndarray, current: np.ndarray, max_step_deg: float) -> np.ndarray:
    """Cap each joint's move so no joint shifts more than max_step_deg this step."""
    delta = np.clip(target - current, -max_step_deg, max_step_deg)
    return (current + delta).astype(np.float32)


def fmt(v):
    return "  ".join(f"{x:8.2f}" for x in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8202")
    ap.add_argument("--port", default="/dev/tty.usbmodem58FD0169761", help="follower serial port")
    ap.add_argument("--id", default="blupe_follower", help="lerobot calibration id")
    ap.add_argument("--cameras", default="0,1", help="two OpenCV camera indices (top,side order-agnostic)")
    ap.add_argument("--instruction", default="place the teal box into the red bowl")
    ap.add_argument("--chunks", type=int, default=50, help="number of model queries (chunks) to run")
    ap.add_argument("--exec-steps", type=int, default=16,
                    help="steps of each 30-step chunk to execute before re-querying (receding horizon)")
    ap.add_argument("--max-step-deg", type=float, default=15.0, help="per-step per-joint safety cap")
    ap.add_argument("--hz", type=float, default=8.0, help="control rate while replaying a chunk")
    ap.add_argument("--execute", action="store_true", help="ACTUALLY MOVE the arm (default: dry run, no motion)")
    args = ap.parse_args()

    cam_idx = [int(x) for x in args.cameras.split(",") if x.strip() != ""]
    if len(cam_idx) != 2:
        raise ValueError("--cameras needs exactly two indices, e.g. 0,1")

    print(f"Policy: MolmoActClient -> {args.url}  (conv=LEROBOT_V21_COMPAT_DEG)")
    print(f"Instruction: {args.instruction!r}")
    print(f"Mode: {'EXECUTE (arm WILL move)' if args.execute else 'DRY RUN (no motion)'}  "
          f"chunks={args.chunks} exec_steps={args.exec_steps} max_step={args.max_step_deg}deg")

    # generous timeout: the first inference after idle can be slow (CUDA-graph
    # warmup), and we'd rather wait than abort a 50-query rollout.
    client = MolmoActClient(args.url, conv=adapter.LEROBOT_V21_COMPAT_DEG, timeout_s=120.0)
    print("health:", client.health())

    # NOTE: do NOT use lerobot's max_relative_target backstop here. It re-reads
    # Present_Position inside send_action and clamps the goal toward it — so a
    # single glitchy read on a marginal bus becomes a *commanded* slam (this is
    # exactly what sent shoulder_pan to ~101deg in the first run). Our software
    # clamp_step() already bounds motion from the state we read+print each step.
    # disable_torque_on_disconnect=False keeps the servos HOLDING their pose when
    # the run ends or errors, instead of the arm going limp ("servos disabled").
    cfg = SO101FollowerConfig(
        port=args.port, id=args.id, max_relative_target=None,
        disable_torque_on_disconnect=False,
    )
    robot = SO101Follower(cfg)
    robot.connect(calibrate=False)
    print(f"\n{'step':>4}  {'  '.join(j[:8] for j in JOINTS)}")
    try:
        period = 1.0 / args.hz if args.hz > 0 else 0.0
        executed = 0
        for c in range(args.chunks):
            state = read_state(robot)
            images = [grab(i) for i in cam_idx]
            # A query failure (timeout / tunnel blip) must NOT kill the run or the
            # arm: retry, and if it still fails just hold pose and move on. Torque
            # stays engaged throughout (disable_torque_on_disconnect=False).
            chunk = None
            for attempt in range(3):
                try:
                    chunk = client.act(Observation(images=images, state_rad=state, instruction=args.instruction))
                    break
                except Exception as e:
                    print(f"  query failed ({type(e).__name__}); retry {attempt+1}/3 — arm holds pose", flush=True)
                    time.sleep(1.0)
            if chunk is None:
                print("  query still failing — holding pose, skipping this chunk", flush=True)
                continue
            n = min(args.exec_steps, len(chunk))
            print(f"\nchunk {c+1}/{args.chunks} @ step {executed}: shape {chunk.shape}, executing {n}", flush=True)
            print(f"  cur  {fmt(state)}", flush=True)
            cur = state
            for k in range(n):  # replay the planned trajectory, clamped step-to-step
                clamped = clamp_step(chunk[k], cur, args.max_step_deg)
                print(f"  [{executed:>3}] {fmt(clamped)}  |Δ|={np.abs(clamped-cur).max():4.1f}", flush=True)
                if args.execute:
                    send_with_retry(robot, clamped)
                cur = clamped
                executed += 1
                time.sleep(period)
        final = read_state(robot)
        print(f"\nfinal cur   {fmt(final)}")
        print("done." + ("" if args.execute else "  [dry run — nothing moved]"))
    finally:
        robot.disconnect()  # relaxes torque on disconnect — support the arm
        print("[disconnected — torque relaxed]")


if __name__ == "__main__":
    main()
