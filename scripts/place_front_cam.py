"""Interactively place the FRONT camera by flying the viewer with the mouse.

Workflow (OnShape-like: move around, then capture):
    1. The interactive viewer opens on the scene (arm + table + cube).
    2. Fly with the mouse — orbit (drag), pan (shift/right drag), zoom (scroll) —
       until the view frames the workspace the way you want the front camera.
    3. Press  C  to capture: the front sensor-camera snaps to the current
       viewer POSITION but always AIMS at the arm base (frame center on the
       base). Its frustum is drawn (green cone) and pos/lookat/fov are printed
       to the terminal (paste into cameras.py CAMERA_OFFSETS).
    4. Press  S  to also save what that camera sees -> outputs/front_cam.png.
    5. Repeat C as you refine; close the window to exit.

Run:
    .venv/bin/python scripts/place_front_cam.py
"""

from pathlib import Path

import numpy as np
import genesis as gs
from genesis.vis.keybindings import Key
from genesis.vis.viewer_plugins import ViewerPlugin
from PIL import Image

import so101_scene as S

OUT = Path(__file__).resolve().parent.parent / "outputs"


class FrontCamPlacer(ViewerPlugin):
    """Capture the viewer's current pose onto a sensor camera on keypress."""

    def __init__(self, scene, front_cam, base_pos):
        super().__init__()
        self.scene = scene
        self.cam = front_cam
        self.base = np.asarray(base_pos, dtype=float)

    def _draw_marker(self, pos):
        """Make the front cam VISIBLE: a red ball at its position, a green cone
        for its frustum, and a line to the arm base it aims at."""
        self.scene.clear_debug_objects()
        self.scene.draw_debug_sphere(pos=tuple(pos), radius=0.03, color=(1.0, 0.2, 0.2, 1.0))
        self.scene.draw_debug_frustum(self.cam, color=(0.2, 1.0, 0.4, 0.6))
        self.scene.draw_debug_line(start=tuple(pos), end=tuple(self.base), color=(1.0, 1.0, 0.0, 0.8))

    def show_initial(self):
        """Draw the marker once at startup so the camera is visible immediately."""
        self._draw_marker(np.asarray(self.cam.get_pos(), dtype=float))

    def on_key_press(self, symbol, modifiers):
        v = self.viewer
        if symbol == int(Key.C):
            pos = np.asarray(v.camera_pos, dtype=float)
            look = self.base  # always aim frame center at the arm base
            fov = float(v.camera_fov)
            self.cam.set_pose(pos=tuple(pos), lookat=tuple(look))
            self._draw_marker(pos)
            # Report both absolute and base-relative (for CAMERA_OFFSETS).
            rel_p, rel_l = pos - self.base, look - self.base
            v.set_message_text(f"captured fov={fov:.0f}")
            print("\n--- FRONT CAM CAPTURED ---")
            print(f"abs   pos={tuple(round(x,3) for x in pos)} "
                  f"lookat={tuple(round(x,3) for x in look)} fov={fov:.0f}")
            print(f'offset  pos={tuple(round(x,3) for x in rel_p)} '
                  f'lookat={tuple(round(x,3) for x in rel_l)}  '
                  f'(base={tuple(round(x,3) for x in self.base)})')
        elif symbol == int(Key.S):
            rgb = np.asarray(self.cam.render()[0])[..., :3].astype(np.uint8)
            OUT.mkdir(exist_ok=True)
            Image.fromarray(rgb).save(OUT / "front_cam.png")
            self.viewer.set_message_text("saved front_cam.png")
            print("saved -> outputs/front_cam.png")


def main() -> None:
    gs.init(backend=gs.cpu)
    scene, robot, cube = S.build_scene(show_viewer=True)
    front = scene.add_camera(res=(640, 480), pos=(0.6, 0.0, 1.1),
                             lookat=(0.15, 0.0, 0.85), fov=45, GUI=False)
    scene.build()
    dofs_idx = S.setup_control(robot)

    target = np.zeros(len(dofs_idx))
    for _ in range(80):
        robot.control_dofs_position(target, dofs_idx)
        scene.step()

    base_pos = np.asarray(robot.get_link("base_link").get_pos())
    placer = FrontCamPlacer(scene, front, base_pos)
    scene.viewer.add_plugin(placer)
    placer.show_initial()  # red ball + green cone visible from the start

    print(__doc__)
    print("Fly with the mouse, press C to capture the front cam, S to save its view.")
    while scene.viewer.is_alive():
        scene.step()


if __name__ == "__main__":
    main()
