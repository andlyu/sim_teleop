"""Per-joint debug: what we feed MolmoAct2 vs. what it expects.

For a given sim pose (radians), prints each joint side by side:
  - our adapter output (degrees sent to the model)
  - the model's typical operating band (q01..q99 from norm_stats)
  - the model card's single sample state (a real calibrated arm)
  - an OUT-OF-BAND flag

Also solves, per joint, the offset that WOULD land a given sim pose on the
sample state (assuming pure-degree scale 57.3), so we can see which joints need
correcting and by how much.

Run:
    .venv/bin/python scripts/debug_joints.py
    .venv/bin/python scripts/debug_joints.py --pose 0,1.4,-1.4,0.9,0,0.2
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from molmoact2 import adapter as A

NAMES = A.JOINT_NAMES
# The model card's sample state — a real calibrated SO101 in the reference pose.
SAMPLE = np.array([-0.527, 189.14, 181.41, 60.64, -3.604, 1.097])
DEG = float(np.degrees(1.0))


def load_bands():
    ss = json.load(open(REPO / "molmoact2" / "norm_stats.json"))[
        "metadata_by_tag"]["so100_so101_molmoact2"]["state_stats"]
    return ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="0,1.4,-1.4,0.9,0,0.2",
                    help="sim joint angles (radians), comma-separated")
    ap.add_argument("--conv", default="RANGE_ALIGNED",
                    choices=["RANGE_ALIGNED", "SO101_DEG", "IDENTITY"])
    args = ap.parse_args()
    pose = np.array([float(x) for x in args.pose.split(",")])
    conv = getattr(A, args.conv)
    ss = load_bands()

    sent = A.state_sim_to_model(pose, conv)        # what we'd send

    print(f"Pose (rad): {pose.tolist()}   conv: {args.conv}\n")
    print(f"{'joint':14} {'sim_rad':>8} {'WE SEND':>9} | {'q01':>7} {'q99':>7} {'sample':>8} | flag")
    print("-" * 78)
    for i, n in enumerate(NAMES):
        q1, q9 = ss["q01"][i], ss["q99"][i]
        lo, hi = min(q1, q9), max(q1, q9)
        flag = "" if lo - 10 <= sent[i] <= hi + 10 else "**OUT OF BAND**"
        print(f"{n:14} {pose[i]:8.2f} {sent[i]:9.1f} | {q1:7.1f} {q9:7.1f} {SAMPLE[i]:8.1f} | {flag}")

    # Per-joint offset solve: if THIS pose is the reference pose, what offset
    # (with pure-degree scale) lands each joint exactly on the sample state?
    print(f"\nIf this pose == the reference pose, per-joint correction to hit sample")
    print(f"(model_deg = sign*57.3*rad + offset):")
    print(f"{'joint':14} {'pureDEG':>9} {'sample':>8} {'offset_needed':>14}")
    for i, n in enumerate(NAMES):
        pure = pose[i] * DEG
        offset = SAMPLE[i] - pure
        print(f"{n:14} {pure:9.1f} {SAMPLE[i]:8.1f} {offset:14.1f}")


if __name__ == "__main__":
    main()
