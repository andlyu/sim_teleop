"""Headless check: do the top + side cameras see the arm and workspace?

Builds the scene, adds the camera rig, settles the arm at the zero pose, then
renders both views and saves them to outputs/ so the framing can be eyeballed.

Run:
    .venv/bin/python scripts/check_cameras.py
"""

from pathlib import Path

import numpy as np
import genesis as gs
from PIL import Image

import so101_scene as S
import cameras as C

OUT = Path(__file__).resolve().parent.parent / "outputs"


def main() -> None:
    gs.init(backend=gs.cpu, logging_level="error")

    scene, robot, cube = S.build_scene(show_viewer=False)
    cams = C.add_cameras(scene)
    scene.build()
    dofs_idx = S.setup_control(robot)

    # Settle the arm at the zero pose so the render is a steady state.
    target = np.zeros(len(dofs_idx))
    for _ in range(150):
        robot.control_dofs_position(target, dofs_idx)
        scene.step()

    OUT.mkdir(exist_ok=True)
    frames = C.capture(cams)
    for name, rgb in frames.items():
        path = OUT / f"cam_{name}.png"
        Image.fromarray(rgb).save(path)
        nonblack = float((rgb.reshape(-1, 3).sum(axis=1) > 0).mean())
        print(f"{name:5s} {rgb.shape} saved -> {path.name}  (non-black px: {nonblack:.1%})")

    imgs = C.observation_images(cams)
    print(f"observation_images: {len(imgs)} views, order={C.CAMERA_NAMES}")


if __name__ == "__main__":
    main()
