"""Take one snapshot from every camera in the scene and combine into one image.

Builds the scene, adds the external policy rig (top + side, base-relative) and
the on-robot wrist cam, settles the arm, then renders one frame per camera and
tiles them side by side with labels. Opens the result.

Run:
    .venv/bin/python scripts/snapshot_cameras.py
"""

from pathlib import Path

import numpy as np
import genesis as gs
from PIL import Image, ImageDraw

import so101_scene as S
import cameras as C
from check_wrist_cam import offset_transform, MOUNT_LINK

OUT = Path(__file__).resolve().parent.parent / "outputs"
PANEL = (480, 480)


def _label(arr, text):
    im = Image.fromarray(arr).convert("RGB").resize(PANEL)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
    d.text((4, 4), text, fill=(255, 255, 255))
    return im


def main() -> None:
    gs.init(backend=gs.cpu, logging_level="error")

    scene, robot, cube = S.build_scene(show_viewer=False)

    # External policy rig, anchored to the real arm base height (table top).
    ext = C.add_cameras(scene, base_pos=(0.0, 0.0, S.TABLE_TOP_Z))
    # On-robot wrist cam.
    wrist = scene.add_camera(res=C.RES, fov=60, GUI=False)

    scene.build()
    dofs_idx = S.setup_control(robot)
    wrist.attach(robot.get_link(MOUNT_LINK), offset_transform())

    # Settle into a pose where the gripper is over the cube.
    target = np.array([0.0, 1.0, -1.0, 0.5, 0.0, 0.2])[: len(dofs_idx)]
    for _ in range(180):
        robot.control_dofs_position(target, dofs_idx)
        scene.step()
    wrist.move_to_attach()

    panels = []
    for name, cam in ext.items():
        rgb = np.asarray(cam.render()[0])[..., :3].astype(np.uint8)
        panels.append(_label(rgb, f"external: {name}"))
    wr = np.asarray(wrist.render()[0])[..., :3].astype(np.uint8)
    panels.append(_label(wr, "on-robot: wrist"))

    sheet = Image.new("RGB", (PANEL[0] * len(panels), PANEL[1]), (30, 30, 30))
    for i, p in enumerate(panels):
        sheet.paste(p, (i * PANEL[0], 0))

    OUT.mkdir(exist_ok=True)
    path = OUT / "all_cameras_snapshot.png"
    sheet.save(path)
    print(f"snapshot: {len(panels)} cameras -> {path}")


if __name__ == "__main__":
    main()
