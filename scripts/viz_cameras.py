"""Camera/pose decision aid: render a contact sheet of candidate views.

Subtask-2 framing depends on subtask-1's arm pose (a sprawled zero-pose hides
the workspace from an overhead cam). Rather than guess, this renders a grid:

    rows  = arm poses        (zero pose vs. an upright "ready" pose)
    cols  = candidate cameras (several top/side placements)

Each tile is labeled with its (pose, camera) so we can instantly see which
combos keep the workspace and the graspable cube visible.
Output: outputs/camera_contact_sheet.png — open it and pick.

Run:
    .venv/bin/python scripts/viz_cameras.py
"""

from pathlib import Path

import numpy as np
import genesis as gs
from PIL import Image, ImageDraw

import so101_scene as S

OUT = Path(__file__).resolve().parent.parent / "outputs"
RES = (480, 360)
FOV = 45
_LOOKAT = (0.12, 0.0, 0.12)

# Candidate camera placements to compare (name -> pos/lookat).
CANDIDATE_CAMS = {
    "overhead":   dict(pos=(0.12, 0.0, 0.75), lookat=(0.12, 0.0, 0.05)),
    "top_front":  dict(pos=(0.45, 0.0, 0.55), lookat=_LOOKAT),
    "side_3q":    dict(pos=(0.20, 0.50, 0.40), lookat=_LOOKAT),
    "side_low":   dict(pos=(0.12, 0.55, 0.18), lookat=_LOOKAT),
}

# Candidate arm poses (name -> 6-D joint target in radians).
# "ready" lifts the shoulder/elbow so the arm stands up over the table instead
# of lying flat. Values are mid-ish within the URDF limits; tune from the sheet.
CANDIDATE_POSES = {
    "zero":  np.zeros(6),
    "ready": np.array([0.0, 1.4, -1.4, 0.4, 0.0, 0.2]),
}


def _settle(robot, dofs_idx, target, steps=150):
    for _ in range(steps):
        robot.control_dofs_position(target, dofs_idx)
        S_step()


def _label(img_arr, text):
    """Return a PIL image with a text banner drawn top-left."""
    im = Image.fromarray(img_arr).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 16], fill=(0, 0, 0))
    d.text((3, 3), text, fill=(255, 255, 255))
    return im


# scene.step is bound at runtime in main(); kept module-level for _settle.
def S_step():
    _STATE["scene"].step()


_STATE = {}


def main() -> None:
    gs.init(backend=gs.cpu, logging_level="error")

    scene, robot, cube = S.build_scene(show_viewer=False)
    cams = {}
    for name, c in CANDIDATE_CAMS.items():
        cams[name] = scene.add_camera(res=RES, pos=c["pos"], lookat=c["lookat"], fov=FOV, GUI=False)
    scene.build()
    _STATE["scene"] = scene
    dofs_idx = S.setup_control(robot)

    cube_z = S.TABLE_TOP_Z + S.CUBE_SIZE / 2.0

    tiles = {}  # (pose_name, cam_name) -> PIL image
    for pose_name, target in CANDIDATE_POSES.items():
        # reset cube to its start spot and settle the arm into this pose
        cube.set_pos(np.array([S.CUBE_XY[0], S.CUBE_XY[1], cube_z]))
        _settle(robot, dofs_idx, target)
        for cam_name, cam in cams.items():
            rgb = np.asarray(cam.render()[0])[..., :3].astype(np.uint8)
            tiles[(pose_name, cam_name)] = _label(rgb, f"{pose_name} | {cam_name}")

    # Assemble grid: rows=poses, cols=cameras
    poses = list(CANDIDATE_POSES)
    camnames = list(CANDIDATE_CAMS)
    tw, th = RES
    sheet = Image.new("RGB", (tw * len(camnames), th * len(poses)), (30, 30, 30))
    for r, p in enumerate(poses):
        for c, cn in enumerate(camnames):
            sheet.paste(tiles[(p, cn)], (c * tw, r * th))

    OUT.mkdir(exist_ok=True)
    path = OUT / "camera_contact_sheet.png"
    sheet.save(path)
    print(f"contact sheet: {len(poses)} poses x {len(camnames)} cams -> {path}")
    print(f"poses:   {poses}")
    print(f"cameras: {camnames}")


if __name__ == "__main__":
    main()
