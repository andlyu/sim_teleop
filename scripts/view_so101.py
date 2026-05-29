"""Phase B: open the interactive Genesis viewer with the SO101 arm.

This is the macOS GUI test — headless stepping already works (Phase A); here we
confirm the interactive viewer actually opens a window on Apple Silicon.

PD gains hold the arm at its zero pose so it doesn't sag under gravity.

The viewer stays open until you close the window (no fixed step count — a fixed
count would exit the process and close the window immediately).

Run:
    .venv/bin/python scripts/view_so101.py
"""

from pathlib import Path

import numpy as np
import genesis as gs

REPO = Path(__file__).resolve().parent.parent
URDF = REPO / "assets" / "so101" / "so101_new_calib.urdf"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Per-DOF PD gains. SO101 motors are small (STS3215); modest gains hold pose.
KP = np.array([30.0, 30.0, 30.0, 20.0, 15.0, 10.0])
KV = np.array([2.0, 2.0, 2.0, 1.5, 1.0, 0.8])


def main() -> None:
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, 0.6, 0.4),
            camera_lookat=(0.0, 0.0, 0.1),
            camera_fov=40,
        ),
        show_viewer=True,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file=str(URDF), fixed=True))
    scene.build()

    dofs_idx = [robot.get_joint(n).dofs_idx_local[0] for n in JOINT_NAMES]
    robot.set_dofs_kp(KP, dofs_idx)
    robot.set_dofs_kv(KV, dofs_idx)

    # Hold the zero pose.
    target = np.zeros(len(dofs_idx))

    print("Viewer open — close the window to exit.")
    while scene.viewer.is_alive():
        robot.control_dofs_position(target, dofs_idx)
        scene.step()
    print("Phase B OK: viewer closed cleanly.")


if __name__ == "__main__":
    main()
