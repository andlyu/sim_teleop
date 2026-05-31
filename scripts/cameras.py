"""Camera rig for sim_teleop observations (subtask 2).

MolmoAct2-SO100_101 consumes two external RGB views ("top" + "side"; the
checkpoint is camera-order agnostic) plus 6-D state and a language instruction.
This module owns the camera placement so the scene / teleop / policy-client
scripts produce observations in one consistent format.

Cameras are static external views (the SO100 mixture uses external cams, not
wrist-mounted), but their placement is defined **relative to the arm base**, not
in absolute world coordinates. That way the framing survives the arm being
remounted (different table height, repositioned base) — move the base and the
cameras move with it. Offsets are resolved to world poses at add-time via a
`base_pos` anchor.

Resolution note: MolmoAct2's processor resizes inputs internally, so the
capture resolution isn't forced by the checkpoint. We render at standard
RealSense RGB (640x480) — close to the real sensors the model was trained on.
"""

import numpy as np
import genesis as gs

# Standard RealSense RGB resolution (W, H). The policy processor resizes anyway.
RES = (640, 480)
FOV = 45

# Arm base anchor in world coords. so101_scene mounts the arm on the table top
# at (0, 0, TABLE_TOP_Z) with TABLE_TOP_Z = 0.74. (The previous 0.1 was a stale
# floor-level anchor from the old thin-table scene, which placed both cameras
# near the floor.) Pass the real base position (e.g.
# robot.get_link("base").get_pos()) to keep cameras attached if the mount moves.
DEFAULT_BASE_POS = (0.0, 0.0, 0.74)

# Camera placements as OFFSETS from the base. `pos` and `lookat` are
# base-relative offsets; `pan`/`tilt`/`roll` (deg) rotate the aim on top of
# "look straight at lookat" (pan=swivel L/R, tilt=up/down, roll=image rotate).
# Order here is the canonical order handed to the policy (order-agnostic, but
# kept stable for reproducibility / overlays / recording). The "side" values
# were tuned interactively via scripts/arm_cam_view.py.
# "top" and "side" tuned via scripts/align_cams.py against the MolmoAct2
# model-card reference images (assets/molmoact_ref/) using the overlay view.
CAMERA_OFFSETS = {
    "top":  dict(pos=(0.17, -0.305, 0.14), lookat=(0.18, 0.0, 0.02),
                 pan=-14.0, tilt=17.0, roll=-2.0, fov=72.0),
    "side": dict(pos=(0.8, 0.015, 0.64), lookat=(0.18, 0.0, 0.02),
                 pan=-2.0, tilt=-7.0, roll=-2.0, fov=28.0),
}

CAMERA_NAMES = list(CAMERA_OFFSETS)  # ["top", "side"]


def _rot(axis, ang, v):
    """Rodrigues rotation of vector v about `axis` by `ang` radians."""
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return v
    axis = axis / n
    return (v * np.cos(ang)
            + np.cross(axis, v) * np.sin(ang)
            + axis * np.dot(axis, v) * (1 - np.cos(ang)))


def _aim(pos, lookat, pan_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
    """(lookat, up) for a cam at `pos` aimed at `lookat`, with pan/tilt/roll.

    (0,0,0) -> look straight at `lookat`, up = world +z. Must match
    scripts/arm_cam_view.py's _aim so the UI preview equals the baked pose.
    """
    pos = np.asarray(pos, float)
    d = np.asarray(lookat, float) - pos
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        d, dist = np.array([1.0, 0.0, 0.0]), 1.0
    fwd = d / dist
    wup = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, wup)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)

    pan, tilt, roll = np.deg2rad([pan_deg, tilt_deg, roll_deg])
    fwd = _rot(wup, pan, fwd)
    right = _rot(wup, pan, right)
    fwd = _rot(right, tilt, fwd)
    up = np.cross(right, fwd)
    up = _rot(fwd, roll, up)
    return tuple(pos + dist * fwd), tuple(up)


def resolve_poses(base_pos=DEFAULT_BASE_POS):
    """World-frame camera poses = base_pos + each offset, with pan/tilt/roll.

    Returns {name: {"pos", "lookat", "up", "fov"}} (up omitted if no rotation).
    """
    base = np.asarray(base_pos, dtype=float)
    poses = {}
    for name, off in CAMERA_OFFSETS.items():
        pos = tuple(base + np.asarray(off["pos"]))
        lookat = tuple(base + np.asarray(off["lookat"]))
        p = dict(pos=pos, lookat=lookat, fov=off["fov"])
        if any(off.get(k) for k in ("pan", "tilt", "roll")):
            p["lookat"], p["up"] = _aim(
                pos, lookat, off.get("pan", 0.0), off.get("tilt", 0.0), off.get("roll", 0.0)
            )
        poses[name] = p
    return poses


def add_cameras(scene, base_pos=DEFAULT_BASE_POS, res=RES, gui=False):
    """Add the top + side cameras to a scene (call before scene.build()).

    Poses are resolved relative to `base_pos` (the arm base). Returns an
    ordered dict {name: Camera} following CAMERA_NAMES.
    """
    poses = resolve_poses(base_pos)
    cams = {}
    for name in CAMERA_NAMES:
        p = poses[name]
        kw = dict(res=res, pos=p["pos"], lookat=p["lookat"], fov=p["fov"], GUI=gui)
        if "up" in p:
            kw["up"] = p["up"]
        cams[name] = scene.add_camera(**kw)
    return cams


# --- On-robot wrist camera ---------------------------------------------------
# Wrist-cam frame adopted from the community SO-101 URDF MSSergeev/so101-lab,
# which adds a calibrated camera to the *same* so101_new_calib model we use
# (identical link names + joint origins, verified). Their fixed joint:
#   parent=gripper_link  child=camera_wrist
#   origin xyz="0.002265 -0.074668 0.009659"  rpy="-2.703668 0 0"
# So the offset is in our gripper_link frame directly — no remapping needed.
WRIST_FOV = 85  # wide: the cam sits close to the action
WRIST_NEAR = 0.01  # near clip; default 0.1 (10cm) clips the close-up gripper
# The camera_wrist link's pose is fully baked into the URDF joint
# (xyz="0.002265 -0.074668 0.009659" rpy="0.437925 0 0"), already oriented so
# its -Z view axis faces outward toward the workspace. So attaching the render
# camera is a pure IDENTITY offset.
WRIST_MOUNT_LINK = "camera_wrist"
WRIST_CAM_XYZ = (0.0, 0.0, 0.0)
WRIST_CAM_RPY = (0.0, 0.0, 0.0)                   # identity: URDF link is correct


def _rpy_to_T(xyz, rpy):
    """4x4 transform from a URDF xyz + rpy (intrinsic X-Y-Z / fixed-axis R=Rz*Ry*Rx)."""
    x, y, z = xyz
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (x, y, z)
    return T


def add_wrist_camera(scene, robot, res=RES, fov=WRIST_FOV, gui=False):
    """Mount a camera on the gripper (call before scene.build()).

    Uses the calibrated wrist-cam offset adopted from MSSergeev/so101-lab.
    Returns the Camera. After scene.build() the caller must call
    `attach_wrist_camera(cam)` once, then `cam.move_to_attach()` each step (or
    after settling) so the view tracks the arm.
    """
    cam = scene.add_camera(res=res, fov=fov, near=WRIST_NEAR, far=10.0, GUI=gui)
    T = _rpy_to_T(WRIST_CAM_XYZ, WRIST_CAM_RPY)
    cam._wrist_attach = (robot.get_link(WRIST_MOUNT_LINK), T)
    return cam


def attach_wrist_camera(cam):
    """Finalize the wrist-cam attach (call once, after scene.build())."""
    link, T = cam._wrist_attach
    cam.attach(link, T)


def capture(cams):
    """Render each camera's RGB. Returns {name: (H, W, 3) uint8 array}.

    Genesis Camera.render() returns (rgb, depth, segmentation, normal); we take
    rgb only. Result is RGB uint8, ready to wrap as a PIL image for the policy.
    """
    out = {}
    for name, cam in cams.items():
        rgb = cam.render()[0]
        out[name] = np.asarray(rgb)[..., :3].astype(np.uint8)
    return out


def observation_images(cams):
    """RGB views as a list in canonical CAMERA_NAMES order, for the policy.

    MolmoAct2 takes images=[...]; this is that list (top, side).
    """
    frames = capture(cams)
    return [frames[name] for name in CAMERA_NAMES]
