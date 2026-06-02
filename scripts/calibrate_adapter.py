"""Recompute the RANGE_ALIGNED joint convention from sim limits + norm_stats.

The numbers baked into molmoact2/adapter.RANGE_ALIGNED come from here, so they
are reproducible rather than hand-copied. Maps each sim joint's URDF range
[lo,hi] (rad) onto the model's typical [q01,q99] (deg) from norm_stats.json:

    scale  = (q99 - q01) / (hi - lo)
    offset = q01 - scale * lo

Run:
    .venv/bin/python scripts/calibrate_adapter.py
Prints scale/offset arrays to paste into adapter.RANGE_ALIGNED (and a sanity
check that the endpoints map as intended).
"""

import sys
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CALIB = os.environ.get("SO101_CALIBRATION", "new").strip().lower()
URDFS = {
    "new": REPO / "assets" / "so101" / "so101_new_calib.urdf",
    "old": REPO / "assets" / "so101" / "so101_old_calib.urdf",
}
if CALIB not in URDFS:
    raise ValueError(f"SO101_CALIBRATION must be one of {sorted(URDFS)}, got {CALIB!r}")
URDF = URDFS[CALIB]
NORM = REPO / "molmoact2" / "norm_stats.json"
ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    print(f"SO101_CALIBRATION={CALIB}  URDF={URDF.name}\n")
    root = ET.parse(URDF).getroot()
    lim = {}
    for j in root.findall("joint"):
        l = j.find("limit")
        if l is not None and l.get("lower") is not None:
            lim[j.get("name")] = (float(l.get("lower")), float(l.get("upper")))

    tag = json.load(open(NORM))["metadata_by_tag"]["so100_so101_molmoact2"]
    ss = tag["state_stats"]
    names = ss["names"]
    assert names == ORDER, f"joint order mismatch: {names} vs {ORDER}"

    scales, offsets = [], []
    print(f"{'joint':14} {'scale':>9} {'offset':>9}   sim_deg(lo,hi) -> model(q01,q99)")
    for i, n in enumerate(ORDER):
        lo, hi = lim[n]
        q1, q9 = ss["q01"][i], ss["q99"][i]
        scale = (q9 - q1) / (hi - lo)
        offset = q1 - scale * lo
        scales.append(round(scale, 3))
        offsets.append(round(offset, 3))
        print(f"{n:14} {scale:9.3f} {offset:9.3f}   "
              f"({math.degrees(lo):+.0f},{math.degrees(hi):+.0f}) -> ({q1:+.0f},{q9:+.0f})")

    print("\nPaste into molmoact2/adapter.RANGE_ALIGNED:")
    print(f"    scale=np.array({scales}, dtype=np.float32),")
    print(f"    offset=np.array({offsets}, dtype=np.float32),")

    # sanity: confirm adapter reproduces these
    sys.path.insert(0, str(REPO))
    from molmoact2 import adapter as A
    import numpy as np
    if not np.allclose(A.RANGE_ALIGNED.scale, scales, atol=1e-3):
        print("\nWARNING: adapter.RANGE_ALIGNED.scale is STALE — update it.")
    elif not np.allclose(A.RANGE_ALIGNED.offset, offsets, atol=1e-3):
        print("\nWARNING: adapter.RANGE_ALIGNED.offset is STALE — update it.")
    else:
        print("\nadapter.RANGE_ALIGNED matches. OK")


if __name__ == "__main__":
    main()
