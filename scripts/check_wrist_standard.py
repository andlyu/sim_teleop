"""Verify the standard wrist camera: mount on the gripper, aim at the grasp
point (from the SO-101 URDF), render what it sees across a couple of poses.

Run:
    .venv/bin/python scripts/check_wrist_standard.py
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
    wrist = C.add_wrist_camera(scene, robot)
    scene.build()
    dofs_idx = S.setup_control(robot)
    C.attach_wrist_camera(wrist)

    poses = {
        "zero": np.zeros(len(dofs_idx)),
        "reach": np.array([0.0, 1.0, -1.0, 0.5, 0.0, 0.2])[: len(dofs_idx)],
    }
    OUT.mkdir(exist_ok=True)
    for name, target in poses.items():
        for _ in range(150):
            robot.control_dofs_position(target, dofs_idx)
            scene.step()
        wrist.move_to_attach()
        rgb = np.asarray(wrist.render()[0])[..., :3].astype(np.uint8)
        path = OUT / f"wrist_std_{name}.png"
        Image.fromarray(rgb).save(path)
        print(f"{name:6s} -> {path.name}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
