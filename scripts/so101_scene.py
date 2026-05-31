"""Shared scene construction for SO101 sim_teleop.

Keeps scene layout (arm, table, graspable object, PD gains, joint map) in one
place so the viewer / teleop / future recording scripts stay consistent.
"""

import xml.etree.ElementTree as ET
import os
from pathlib import Path

import numpy as np
import genesis as gs

REPO = Path(__file__).resolve().parent.parent
SO101_CALIBRATION = os.environ.get("SO101_CALIBRATION", "new").strip().lower()
URDFS = {
    "new": REPO / "assets" / "so101" / "so101_new_calib.urdf",
    "old": REPO / "assets" / "so101" / "so101_old_calib.urdf",
}
if SO101_CALIBRATION not in URDFS:
    raise ValueError(
        f"SO101_CALIBRATION must be one of {sorted(URDFS)}, got {SO101_CALIBRATION!r}"
    )
URDF = URDFS[SO101_CALIBRATION]
ROBOT_EULER = (0.0, 0.0, 90.0) if SO101_CALIBRATION == "old" else (0.0, 0.0, 0.0)

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# PD gains. The weight-bearing joints (shoulder_lift, elbow_flex, wrist_flex)
# need stiff position gains + enough torque, or gravity sags the arm below its
# commanded target. Raised from the original soft values to hold pose firmly.
KP = np.array([120.0, 200.0, 160.0, 80.0, 40.0, 30.0])
KV = np.array([8.0, 12.0, 10.0, 6.0, 3.0, 2.0])
# Per-joint torque limits (N*m). Generous so the controller can actually resist
# gravity; the URDF lists effort=35, we allow a bit more headroom for holding.
FORCE_RANGE = 50.0

# Table: a real desk (top slab + 4 legs) the arm is mounted on top of.
# Standard office desk, rotated 90deg: 60 cm deep (x) x 120 cm wide (y), 74 cm
# tall. The arm is mounted at the back (-x) edge and reaches forward across the
# desk, like a real SO101 clamped to a table edge.
TABLE_TOP_SIZE = (0.6, 1.2, 0.04)   # meters (x depth, y width, slab thickness)
TABLE_HEIGHT = 0.74                 # top surface height above the ground
# Arm base sits at world x=0; center the desk so its back (-x) edge is there.
TABLE_CENTER_XY = (0.6 / 2.0, 0.0)  # (depth/2, 0) -> back edge at x=0
LEG_SIZE = 0.05                     # square leg cross-section (m)
LEG_INSET = 0.06                    # legs inset from the top's edges (m)

# Privacy partition: a 3-sided booth (left + right + front) standing on the desk
# around the workspace in front of the robot, open toward the robot (-x), like a
# testing divider / photo backdrop. Gives the cameras a clean, consistent
# background. The left/right side panels run the full desk depth (back to front).
BACKDROP_ENABLED = True
BACKDROP_HEIGHT = 0.40              # how tall the panels stand above the desk (m)
BACKDROP_THICKNESS = 0.01          # panel thickness (m)
BACKDROP_INSET = 0.02              # inset from the desk edges (m)
BACKDROP_LEN_FRAC = 1.0            # fraction of the desk depth (x) the side panels span

# Single source of truth for the work surface height. The arm base sits here,
# the cube rests here, and (eventually) base-relative cameras anchor to it.
TABLE_TOP_Z = TABLE_HEIGHT
# In the old-calibration URDF the base plate sits 3.008 cm above the root
# frame, so lower the root by that amount to put the plate on the desk.
ROBOT_Z = TABLE_TOP_Z - 0.0300817 if SO101_CALIBRATION == "old" else TABLE_TOP_Z
# Backwards-compat alias for callers that referenced the old TABLE_SIZE.
TABLE_SIZE = TABLE_TOP_SIZE

# A small, light, high-friction cube — sized for the SO101's little gripper.
CUBE_SIZE = 0.03
CUBE_RHO = 400.0       # light (balsa-ish) so the small servos can hold it
CUBE_FRICTION = 1.5    # grippy, for graspability
CUBE_XY = (0.28, 0.10)  # 10 cm farther forward (+x) and left (+y) of the old spot

# Appearance (RGB 0..1). Wooden desk, contrasting red-orange block.
WOOD_COLOR = (0.55, 0.36, 0.20)       # warm medium-brown wood
CUBE_COLOR = (0.85, 0.20, 0.15)       # red-orange, stands out against the wood
BACKDROP_COLOR = (0.85, 0.85, 0.85)   # neutral light gray, clean background


def joint_limits(urdf_path: Path = URDF) -> dict:
    """(lower, upper) per joint name from the URDF."""
    root = ET.parse(urdf_path).getroot()
    out = {}
    for j in root.findall("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("lower") is not None:
            out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


def _add_table(scene):
    """Add a fixed desk: a top slab held up by four legs.

    The top's upper surface sits at TABLE_HEIGHT; legs run from the floor up to
    the underside of the slab. Everything is fixed (static collidable geometry).
    """
    cx, cy = TABLE_CENTER_XY
    tw, td, tt = TABLE_TOP_SIZE  # top width (x), depth (y), thickness (z)

    # Matte wood surface shared by the top and legs.
    wood = gs.surfaces.Rough(color=WOOD_COLOR)

    # Top slab: centered so its upper face is exactly at TABLE_HEIGHT.
    scene.add_entity(
        gs.morphs.Box(
            pos=(cx, cy, TABLE_HEIGHT - tt / 2.0),
            size=(tw, td, tt),
            fixed=True,
        ),
        surface=wood,
    )

    # Four legs, inset from the slab edges, spanning floor -> underside of slab.
    leg_h = TABLE_HEIGHT - tt
    dx = tw / 2.0 - LEG_INSET - LEG_SIZE / 2.0
    dy = td / 2.0 - LEG_INSET - LEG_SIZE / 2.0
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            scene.add_entity(
                gs.morphs.Box(
                    pos=(cx + sx * dx, cy + sy * dy, leg_h / 2.0),
                    size=(LEG_SIZE, LEG_SIZE, leg_h),
                    fixed=True,
                ),
                surface=wood,
            )


def _add_backdrop(scene):
    """Add side privacy panels on the desk.

    Left/right panels run front-to-back along the desk LENGTH (x), spanning half
    of it on the front (+x) side. The front panel is intentionally omitted so it
    does not occlude the MolmoAct camera views.
    """
    cx, cy = TABLE_CENTER_XY
    tw, td, _tt = TABLE_TOP_SIZE
    th, thk, ins = BACKDROP_HEIGHT, BACKDROP_THICKNESS, BACKDROP_INSET

    gray = gs.surfaces.Rough(color=BACKDROP_COLOR)
    z_center = TABLE_HEIGHT + th / 2.0   # panels stand up from the desk surface

    x_far = cx + tw / 2.0 - ins                      # front (+x) desk edge, inset
    # Side panels span a fraction of the (inset) desk depth, anchored at the front.
    panel_len = (tw - 2.0 * ins) * BACKDROP_LEN_FRAC
    x_center = x_far - panel_len / 2.0               # grows backward from the front edge
    y_edge = td / 2.0 - ins - thk / 2.0              # side panels near +/- y edges

    # Left (+y) and right (-y) side panels: run along x, thin along y.
    for sy in (-1.0, 1.0):
        scene.add_entity(
            gs.morphs.Box(
                pos=(x_center, cy + sy * y_edge, z_center),
                size=(panel_len, thk, th),
                fixed=True,
            ),
            surface=gray,
        )


def build_scene(show_viewer: bool = True):
    """Create the scene with ground, table, arm (mounted on the table) and cube.

    Returns (scene, robot, cube, dofs_idx).
    """
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            # Framed on the work surface, which now sits at TABLE_HEIGHT.
            camera_pos=(0.8, 0.8, TABLE_HEIGHT + 0.45),
            camera_lookat=(TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], TABLE_HEIGHT),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(
            # Soft, diffuse overhead lighting: strong ambient fill + a few gentle,
            # near-vertical directional lights and no hard shadows, so the scene
            # is evenly lit from above instead of by one sharp angled light.
            ambient_light=(0.6, 0.6, 0.6),
            shadow=False,
            lights=[
                gs.options.vis.DirectionalLight(
                    dir=(0.0, 0.0, -1.0), color=(1.0, 1.0, 1.0), intensity=3.0),
                gs.options.vis.DirectionalLight(
                    dir=(-0.3, -0.2, -1.0), color=(1.0, 1.0, 1.0), intensity=1.5),
                gs.options.vis.DirectionalLight(
                    dir=(0.3, 0.2, -1.0), color=(1.0, 1.0, 1.0), intensity=1.5),
            ],
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    _add_table(scene)
    if BACKDROP_ENABLED:
        _add_backdrop(scene)

    # Arm: mounted on top of the table.
    # links_to_keep preserves the camera_wrist mount frame; Genesis merges
    # fixed-joint links by default, which would drop it.
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(URDF),
            pos=(0.0, 0.0, ROBOT_Z),
            euler=ROBOT_EULER,
            fixed=True,
            links_to_keep=["camera_wrist"],
        ),
    )

    # Graspable cube resting on the table top.
    cube = scene.add_entity(
        gs.morphs.Box(
            pos=(CUBE_XY[0], CUBE_XY[1], TABLE_TOP_Z + CUBE_SIZE / 2.0),
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        ),
        material=gs.materials.Rigid(rho=CUBE_RHO, friction=CUBE_FRICTION),
        surface=gs.surfaces.Rough(color=CUBE_COLOR),
    )

    return scene, robot, cube


def setup_control(robot):
    """Map joint names to DOF indices and apply PD gains. Returns dofs_idx."""
    dofs_idx = [robot.get_joint(n).dofs_idx_local[0] for n in JOINT_NAMES]
    robot.set_dofs_kp(KP, dofs_idx)
    robot.set_dofs_kv(KV, dofs_idx)
    # Give the motors enough torque headroom to hold the arm against gravity.
    n = len(dofs_idx)
    robot.set_dofs_force_range(
        np.full(n, -FORCE_RANGE), np.full(n, FORCE_RANGE), dofs_idx
    )
    return dofs_idx
