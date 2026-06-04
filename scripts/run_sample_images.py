"""Diagnostic: drive the REAL arm using MolmoAct2 fed the model-card SAMPLE images.

Feeds the checkpoint's own in-distribution lemon-scene images (which we verified
produce a real reaching trajectory) as the VISUAL input, but uses the REAL
follower's live state each query and executes the resulting actions on the arm.

If the arm moves coherently (the lemon-reach), that proves the control loop +
conversion + execution all work, and isolates the prior plateau to our scene's
pixels being out-of-distribution.

    .venv-lerobot/bin/python scripts/run_sample_images.py --execute --chunks 20

State each query is printed so you can see it updating as the arm moves.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # noqa: E402
from molmoact2 import adapter  # noqa: E402
from molmoact2.client import MolmoActClient, Observation  # noqa: E402

JOINTS = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
REPO_ID = "allenai/MolmoAct2-SO100_101"


def fmt(v): return "  ".join(f"{x:7.2f}" for x in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8202")
    ap.add_argument("--port", default="/dev/tty.usbmodem58FD0169761")
    ap.add_argument("--id", default="blupe_follower")
    ap.add_argument("--prompt", default="Move the arm towards the lemon, grasp it, lift it up, and drop it into the red bowl.")
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--exec-steps", type=int, default=16)
    ap.add_argument("--max-step-deg", type=float, default=8.0)
    ap.add_argument("--hz", type=float, default=8.0)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    # the card's OWN sample images (fixed every query)
    print("loading model-card sample images...")
    top = np.array(Image.open(hf_hub_download(REPO_ID, "assets/sample_realsense_top_rgb.png")).convert("RGB"))
    side = np.array(Image.open(hf_hub_download(REPO_ID, "assets/sample_realsense_side_rgb.png")).convert("RGB"))
    SAMPLE = [top, side]

    client = MolmoActClient(args.url, conv=adapter.LEROBOT_V21_COMPAT_DEG, timeout_s=120.0)
    print("health:", client.health())
    print(f"prompt: {args.prompt!r}   mode: {'EXECUTE' if args.execute else 'DRY RUN'}")

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id,
                                              max_relative_target=None, disable_torque_on_disconnect=False))
    robot.connect(calibrate=False)
    def read(): o=robot.get_observation(); return np.array([o[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
    try:
        period = 1.0/args.hz if args.hz>0 else 0.0
        for c in range(args.chunks):
            state = read()
            chunk = client.act(Observation(images=SAMPLE, state_rad=state, instruction=args.prompt))
            n = min(args.exec_steps, len(chunk))
            print(f"\nquery {c+1}/{args.chunks}  STATE (live): {fmt(state)}  | chunk {chunk.shape}, exec {n}", flush=True)
            cur = state
            for k in range(n):
                tgt = cur + np.clip(chunk[k]-cur, -args.max_step_deg, args.max_step_deg)
                if args.execute:
                    robot.send_action({f"{j}.pos": float(v) for j,v in zip(JOINTS, tgt)})
                cur = tgt
                time.sleep(period)
        print("\nfinal STATE:", fmt(read()))
    finally:
        robot.disconnect()
        print("[disconnected — torque held]")


if __name__ == "__main__":
    main()
