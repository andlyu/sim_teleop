"""Snapshot the exact observation images MolmoAct2 would receive.

Builds the real scene + the canonical cameras.py rig (top + side), settles the
arm at a representative pose, and writes the policy's-eye-view images to disk so
we can confirm the framing is right BEFORE feeding anything to the policy.

Run:
    .venv/bin/python scripts/snap_obs.py
writes /tmp/obs_top.png and /tmp/obs_side.png
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_scene as S
import cameras as C

POSE = np.array([0.0, 0.5, -0.5, 0.5, 0.0, 0.2])


def main():
    gs.init(backend=gs.cpu, logging_level="error")
    scene, robot, cube = S.build_scene(show_viewer=False)
    cams = C.add_cameras(scene)                  # canonical top + side rig
    scene.build()
    dofs = S.setup_control(robot)
    target = POSE[: len(dofs)]
    for _ in range(150):
        robot.control_dofs_position(target, dofs)
        scene.step()

    imgs = C.observation_images(cams)            # [top, side], policy order
    for name, arr in zip(C.CAMERA_NAMES, imgs):
        out = f"/tmp/obs_{name}.png"
        Image.fromarray(arr).save(out)
        print(f"wrote {out}  shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
