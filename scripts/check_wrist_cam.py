"""Mount a camera on the robot and view what it sees (wrist cam).

Attaches a camera rigidly to the gripper link via Camera.attach(link, offset_T)
and updates it each step with move_to_attach(), so the view tracks the arm.
Settles the arm at a couple of poses and saves the on-robot image at each, so
we can confirm the mounted camera actually looks out toward the workspace.

Run:
    .venv/bin/python scripts/check_wrist_cam.py
"""

from pathlib import Path

import numpy as np
import genesis as gs
from PIL import Image

import so101_scene as S

OUT = Path(__file__).resolve().parent.parent / "outputs"
RES = (640, 480)
FOV = 60  # wider FoV: wrist cams sit close to the action

# Mount link and offset transform (camera frame relative to the link).
# Genesis cameras look down their local -Z by convention; we offset the camera
# a few cm ahead of the gripper and tilt it to look forward/down at the grasp.
MOUNT_LINK = "gripper_link"


def offset_transform(pos=(-0.05, 0.0, 0.03), look_dir=(-1.0, 0.0, 0.5)):
    """Build a 4x4 offset T placing the cam at `pos` in the link frame, with its
    -Z (view dir) pointing along `look_dir`.

    Note: the SO101 gripper_link frame is rotated oddly vs world (local +X is
    ~world-up, local +Z is ~world-back). Empirically the grasp point / cube
    sits toward the link's -X (with a little +Z), so the default look_dir aims
    there. `up_hint` uses the link's +Z as a stable 'up' since world-up is
    nearly parallel to the link's X axis."""
    z = -np.asarray(look_dir, dtype=float)        # camera looks down -Z
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])                 # link-frame up hint
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:                   # look_dir parallel to up
        up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, y, z
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def main() -> None:
    gs.init(backend=gs.cpu, logging_level="error")

    scene, robot, cube = S.build_scene(show_viewer=False)
    cam = scene.add_camera(res=RES, fov=FOV, GUI=False)
    scene.build()
    dofs_idx = S.setup_control(robot)

    link = robot.get_link(MOUNT_LINK)
    cam.attach(link, offset_transform())

    poses = {
        "zero": np.zeros(len(dofs_idx)),
        "bent": np.array([0.0, 1.0, -1.0, 0.5, 0.0, 0.2])[: len(dofs_idx)],
    }

    OUT.mkdir(exist_ok=True)
    for name, target in poses.items():
        for _ in range(150):
            robot.control_dofs_position(target, dofs_idx)
            scene.step()
        cam.move_to_attach()  # snap camera to the link's current pose
        rgb = np.asarray(cam.render()[0])[..., :3].astype(np.uint8)
        path = OUT / f"wrist_cam_{name}.png"
        Image.fromarray(rgb).save(path)
        nb = float((rgb.reshape(-1, 3).sum(axis=1) > 0).mean())
        print(f"{name:5s} {rgb.shape} -> {path.name}  (non-black px: {nb:.1%})")


if __name__ == "__main__":
    main()
