"""Show the arm + a CONE/FRUSTUM marking where the wrist camera is and aims.

Instead of inferring the wrist-cam placement from its renders, we draw it
directly in the scene: a debug frustum (the camera's view cone), an arrow along
its view direction, and a sphere at its position. Then we shoot the whole thing
from a third-person camera so you can SEE where the camera sits and where it
points relative to the gripper and the cube.

Output: outputs/camera_cone.png

Run:
    .venv/bin/python scripts/viz_camera_cone.py
"""

from pathlib import Path

import numpy as np
import genesis as gs
from PIL import Image

import so101_scene as S
import cameras as C

OUT = Path(__file__).resolve().parent.parent / "outputs"


def _quat_to_R(q):
    """Genesis quat (w,x,y,z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])


def main() -> None:
    gs.init(backend=gs.cpu, logging_level="error")
    scene, robot, cube = S.build_scene(show_viewer=False)

    # Wrist cam (the thing we're visualizing) + a 3rd-person viewer cam.
    wrist = C.add_wrist_camera(scene, robot)
    third = scene.add_camera(res=(900, 700), pos=(0.6, -0.5, 1.15),
                             lookat=(0.15, 0.0, 0.85), fov=45, GUI=False)
    scene.build()
    dofs_idx = S.setup_control(robot)
    C.attach_wrist_camera(wrist)

    # Settle into a reach pose.
    target = np.array([0.0, 1.0, -1.0, 0.5, 0.0, 0.2])[: len(dofs_idx)]
    for _ in range(150):
        robot.control_dofs_position(target, dofs_idx)
        scene.step()
    wrist.move_to_attach()

    # Draw the wrist camera's ACTUAL view frustum (the green cone), straight
    # from the Camera object — no hand math to get wrong. This is the cone of
    # what the wrist cam sees, drawn in the world for the 3rd-person cam.
    scene.draw_debug_frustum(wrist, color=(0.2, 1.0, 0.4, 0.5))

    rgb = np.asarray(third.render()[0])[..., :3].astype(np.uint8)
    OUT.mkdir(exist_ok=True)
    path = OUT / "camera_cone.png"
    Image.fromarray(rgb).save(path)
    print(f"cam world pos = {tuple(round(float(v),3) for v in cam_p)}")
    print(f"view dir      = {tuple(round(float(v),3) for v in view_dir)}")
    print(f"saved -> {path}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
