"""Open the Genesis viewer holding the arm at the start pose (no policy).

Just shows the scene + arm at HOME_POSE so you can look at the starting state.
Close the window to exit.

Run:
    .venv/bin/python scripts/show_start.py
"""
import sys
from pathlib import Path

import numpy as np
import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_scene as S
import sim_env as E


def main():
    gs.init(backend=gs.cpu)
    scene, robot, _cube = S.build_scene(show_viewer=True)
    scene.build()
    dofs = S.setup_control(robot)
    target = np.clip(
        E.HOME_POSE,
        [S.joint_limits()[n][0] for n in S.JOINT_NAMES],
        [S.joint_limits()[n][1] for n in S.JOINT_NAMES],
    )
    print("Start pose (rad):", E.HOME_POSE.tolist())
    print("Viewer open — arm held at start pose. Close the window to exit.")
    while scene.viewer.is_alive():
        robot.control_dofs_position(target, dofs)
        scene.step()
    print("Viewer closed.")


if __name__ == "__main__":
    main()
