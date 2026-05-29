"""Phase A: load the SO101 arm into Genesis and step it headless.

Proves the URDF parses, the vendored meshes resolve, and the arm simulates on
this machine before we touch the interactive viewer or teleop.

Run:
    .venv/bin/python scripts/load_so101.py
"""

from pathlib import Path

import genesis as gs

REPO = Path(__file__).resolve().parent.parent
URDF = REPO / "assets" / "so101" / "so101_new_calib.urdf"

# SO101 actuated joints, base -> gripper (see assets/so101/ATTRIBUTION.md).
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def main() -> None:
    gs.init(backend=gs.cpu)

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(file=str(URDF), fixed=True),
    )
    scene.build()

    # Map joint names -> local DOF indices so we can address joints by name.
    dofs_idx = [robot.get_joint(name).dof_idx_local for name in JOINT_NAMES]
    print(f"\nLoaded {robot.n_links} links, {robot.n_dofs} DOFs")
    print("Joint -> dof_idx_local:")
    for name, idx in zip(JOINT_NAMES, dofs_idx):
        print(f"  {name:14s} {idx}")

    # Step headless; just prove physics advances without error.
    for _ in range(100):
        scene.step()

    q = robot.get_dofs_position(dofs_idx)
    print("\nJoint positions after 100 steps (rad):")
    for name, val in zip(JOINT_NAMES, q):
        print(f"  {name:14s} {float(val):+.4f}")
    print("\nPhase A OK: SO101 loaded, meshes resolved, simulation stepped.")


if __name__ == "__main__":
    main()
