"""See the robot AND what its wrist camera sees, side by side.

For each arm pose we render two panels:
  left  = external 3rd-person view of the whole robot in the scene
  right = the wrist-mounted camera's view (the "robot-eye" image)

Stacked into one sheet so you can watch the robot-eye view track the gripper as
the arm moves. Output: outputs/robot_eye_view.png

Run:
    .venv/bin/python scripts/viz_robot_eye.py
"""

from pathlib import Path

import numpy as np
import genesis as gs
from PIL import Image, ImageDraw

import so101_scene as S
from check_wrist_cam import offset_transform, MOUNT_LINK

OUT = Path(__file__).resolve().parent.parent / "outputs"
PANEL = (480, 360)
EXT_FOV = 40
WRIST_FOV = 60

# A few poses to show the view tracking the arm.
POSES = {
    "zero": np.zeros(6),
    "reach": np.array([0.0, 0.8, -0.8, 0.4, 0.0, 0.2]),
    "down": np.array([0.0, 1.2, -1.2, 0.6, 0.0, 0.5]),
}


def _label(img_arr, text):
    im = Image.fromarray(img_arr).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 16], fill=(0, 0, 0))
    d.text((3, 3), text, fill=(255, 255, 255))
    return im


def main() -> None:
    gs.init(backend=gs.cpu, logging_level="error")

    scene, robot, cube = S.build_scene(show_viewer=False)
    # External camera: fixed 3rd-person look at the workspace.
    # Arm is small (~0.3m, centered ~(0.12,0,0.88)) on a big table — frame it tightly.
    ext = scene.add_camera(res=PANEL, pos=(0.55, -0.55, 1.05), lookat=(0.12, 0.0, 0.86), fov=EXT_FOV, GUI=False)
    # Wrist camera: mounted on the gripper.
    wrist = scene.add_camera(res=PANEL, fov=WRIST_FOV, GUI=False)
    scene.build()
    dofs_idx = S.setup_control(robot)
    wrist.attach(robot.get_link(MOUNT_LINK), offset_transform())

    cube_z = S.TABLE_TOP_Z + S.CUBE_SIZE / 2.0
    rows = []
    for name, target in POSES.items():
        cube.set_pos(np.array([S.CUBE_XY[0], S.CUBE_XY[1], cube_z]))
        for _ in range(150):
            robot.control_dofs_position(target[: len(dofs_idx)], dofs_idx)
            scene.step()
        wrist.move_to_attach()
        ext_rgb = np.asarray(ext.render()[0])[..., :3].astype(np.uint8)
        wr_rgb = np.asarray(wrist.render()[0])[..., :3].astype(np.uint8)
        left = _label(ext_rgb, f"{name} | scene (robot)")
        right = _label(wr_rgb, f"{name} | robot-eye (wrist cam)")
        row = Image.new("RGB", (PANEL[0] * 2, PANEL[1]))
        row.paste(left, (0, 0))
        row.paste(right, (PANEL[0], 0))
        rows.append(row)

    sheet = Image.new("RGB", (PANEL[0] * 2, PANEL[1] * len(rows)))
    for i, row in enumerate(rows):
        sheet.paste(row, (0, i * PANEL[1]))

    OUT.mkdir(exist_ok=True)
    path = OUT / "robot_eye_view.png"
    sheet.save(path)
    print(f"robot-eye sheet: {len(rows)} poses -> {path}")


if __name__ == "__main__":
    main()
