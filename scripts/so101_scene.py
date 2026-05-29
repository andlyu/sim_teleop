"""Shared scene construction for SO101 sim_teleop.

Keeps scene layout (arm, table, graspable object, PD gains, joint map) in one
place so the viewer / teleop / future recording scripts stay consistent.
"""

import xml.etree.ElementTree as ET
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

KP = np.array([30.0, 30.0, 30.0, 20.0, 15.0, 10.0])
KV = np.array([2.0, 2.0, 2.0, 1.5, 1.0, 0.8])

# Table: a low fixed slab the arm sits on, with a graspable cube on top.
TABLE_SIZE = (0.5, 0.5, 0.1)   # meters (x, y, z)
TABLE_TOP_Z = TABLE_SIZE[2]    # top surface height (table rests on the ground plane)

# A small, light, high-friction cube — sized for the SO101's little gripper.
CUBE_SIZE = 0.03
CUBE_RHO = 400.0       # light (balsa-ish) so the small servos can hold it
CUBE_FRICTION = 1.5    # grippy, for graspability
CUBE_XY = (0.18, 0.0)  # in front of the arm; z is set by the table top


def joint_limits(urdf_path: Path = URDF) -> dict:
    """(lower, upper) per joint name from the URDF."""
    root = ET.parse(urdf_path).getroot()
    out = {}
    for j in root.findall("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("lower") is not None:
            out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


def build_scene(show_viewer: bool = True):
    """Create the scene with ground, table, arm (mounted on the table) and cube.

    Returns (scene, robot, cube, dofs_idx).
    """
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, 0.6, 0.5),
            camera_lookat=(0.0, 0.0, 0.15),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    # Table: fixed slab sitting on the ground, top at TABLE_TOP_Z.
    scene.add_entity(
        gs.morphs.Box(
            pos=(0.1, 0.0, TABLE_SIZE[2] / 2.0),
            size=TABLE_SIZE,
            fixed=True,
        ),
    )

    # Arm: mounted on top of the table.
    robot = scene.add_entity(
        gs.morphs.URDF(file=str(URDF), pos=(0.0, 0.0, TABLE_TOP_Z), fixed=True),
    )

    # Graspable cube resting on the table top.
    cube = scene.add_entity(
        gs.morphs.Box(
            pos=(CUBE_XY[0], CUBE_XY[1], TABLE_TOP_Z + CUBE_SIZE / 2.0),
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        ),
        material=gs.materials.Rigid(rho=CUBE_RHO, friction=CUBE_FRICTION),
    )

    return scene, robot, cube


def setup_control(robot):
    """Map joint names to DOF indices and apply PD gains. Returns dofs_idx."""
    dofs_idx = [robot.get_joint(n).dofs_idx_local[0] for n in JOINT_NAMES]
    robot.set_dofs_kp(KP, dofs_idx)
    robot.set_dofs_kv(KV, dofs_idx)
    return dofs_idx
