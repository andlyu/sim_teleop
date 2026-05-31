"""Open the interactive Genesis viewer with the SO101 arm on its desk.

Uses the shared so101_scene (desk + arm + cube) so what you see here matches
every other script. PD gains hold the arm at its zero pose so it doesn't sag.

The viewer stays open until you close the window (no fixed step count — a fixed
count would exit the process and close the window immediately).

Run:
    .venv/bin/python scripts/view_so101.py
"""

import numpy as np
import genesis as gs

import so101_scene as S


def main() -> None:
    gs.init(backend=gs.cpu)

    scene, robot, _cube = S.build_scene(show_viewer=True)
    scene.build()
    dofs_idx = S.setup_control(robot)

    # Hold the zero pose.
    target = np.zeros(len(dofs_idx))

    print("Viewer open — close the window to exit.")
    while scene.viewer.is_alive():
        robot.control_dofs_position(target, dofs_idx)
        scene.step()
    print("Phase B OK: viewer closed cleanly.")


if __name__ == "__main__":
    main()
