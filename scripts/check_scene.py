"""Headless check: does the cube rest on the table, and is it within reach?

Settles the scene, then reports the cube's resting height and the gripper
position at the zero pose so we can confirm the layout empirically.
"""

import numpy as np
import genesis as gs

import so101_scene as S


def main() -> None:
    gs.init(backend=gs.cpu, logging_level="error")
    scene, robot, cube = S.build_scene(show_viewer=False)
    scene.build()
    dofs_idx = S.setup_control(robot)

    target = np.zeros(len(dofs_idx))
    for _ in range(200):
        robot.control_dofs_position(target, dofs_idx)
        scene.step()

    cube_pos = np.asarray(cube.get_pos())
    expected_rest = S.TABLE_TOP_Z + S.CUBE_SIZE / 2.0
    print(f"table_top_z      = {S.TABLE_TOP_Z:.3f}")
    print(f"cube_pos         = ({cube_pos[0]:+.3f}, {cube_pos[1]:+.3f}, {cube_pos[2]:+.3f})")
    print(f"cube_expected_z  = {expected_rest:.3f}  (rests on table if z matches)")
    print(f"cube_settled_ok  = {abs(float(cube_pos[2]) - expected_rest) < 0.01}")

    # Gripper link position at zero pose, to gauge reach vs the cube.
    ee = robot.get_link("gripper_link")
    ee_pos = np.asarray(ee.get_pos())
    print(f"gripper_pos      = ({ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f})")
    dist = float(np.linalg.norm(ee_pos - cube_pos))
    print(f"gripper_to_cube  = {dist:.3f} m")


if __name__ == "__main__":
    main()
