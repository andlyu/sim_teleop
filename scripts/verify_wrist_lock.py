"""Verify the wrist camera is rigidly locked to the gripper.

The wrist cam offset is locked in the gripper_link frame via cam.attach(), so
when the arm moves the camera must move with it. This proves it: we attach the
wrist cam, drive the arm through several distinct poses, and at each pose draw
the camera's *actual view frustum* (straight from the Camera object) and shoot
it with a fixed third-person camera. If the cone stays glued to the gripper and
keeps aiming at the grasp across all poses, the lock is correct.

Output: outputs/wrist_lock_check.png (a row of poses) — open it and look.

Run:
    .venv/bin/python scripts/verify_wrist_lock.py
"""

from pathlib import Path

import numpy as np
import genesis as gs
from PIL import Image, ImageDraw

import so101_scene as S
import cameras as C

OUT = Path(__file__).resolve().parent.parent / "outputs"
PANEL = (480, 480)

# A few clearly-different arm poses to sweep through.
POSES = {
    "pose_a": np.array([0.0, 1.0, -1.0, 0.5, 0.0, 0.2]),
    "pose_b": np.array([0.6, 0.8, -0.6, 0.2, 0.5, 0.2]),
    "pose_c": np.array([-0.6, 1.3, -1.3, 0.8, -0.4, 0.2]),
}


def _label(arr, text):
    im = Image.fromarray(arr).convert("RGB").resize(PANEL)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
    d.text((4, 4), text, fill=(255, 255, 255))
    return im


def main() -> None:
    gs.init(backend=gs.cpu, logging_level="error")

    scene, robot, cube = S.build_scene(show_viewer=False)
    wrist = C.add_wrist_camera(scene, robot)
    # Fixed third-person observer to watch the gripper + frustum.
    obs = scene.add_camera(res=PANEL, pos=(0.55, -0.55, 1.05),
                           lookat=(0.12, 0.0, 0.86), fov=50, GUI=False)
    scene.build()
    dofs_idx = S.setup_control(robot)
    C.attach_wrist_camera(wrist)

    panels = []
    for name, full_target in POSES.items():
        target = full_target[: len(dofs_idx)]
        for _ in range(150):
            robot.control_dofs_position(target, dofs_idx)
            scene.step()
        wrist.move_to_attach()  # snap wrist cam to the gripper's new pose

        # Draw the wrist cam's real frustum, then shoot it from the observer.
        scene.clear_debug_objects()
        scene.draw_debug_frustum(wrist, color=(1.0, 0.2, 0.2, 0.6))
        rgb = np.asarray(obs.render()[0])[..., :3].astype(np.uint8)
        panels.append(_label(rgb, f"{name}: wrist frustum (locked to gripper)"))

    sheet = Image.new("RGB", (PANEL[0] * len(panels), PANEL[1]), (30, 30, 30))
    for i, p in enumerate(panels):
        sheet.paste(p, (i * PANEL[0], 0))
    OUT.mkdir(exist_ok=True)
    path = OUT / "wrist_lock_check.png"
    sheet.save(path)
    print(f"wrist lock check: {len(panels)} poses -> {path}")


if __name__ == "__main__":
    main()
