"""Camera rig for sim_teleop observations (subtask 2).

MolmoAct2-SO100_101 consumes two external RGB views ("top" + "side"; the
checkpoint is camera-order agnostic) plus 6-D state and a language instruction.
This module owns the camera placement so the scene / teleop / policy-client
scripts produce observations in one consistent format.

Cameras are static, world-frame, external views (the SO100 mixture uses
external cams, not wrist-mounted). Poses are aimed at the workspace centered
roughly on the table + graspable object in so101_scene.

Resolution note: MolmoAct2's processor resizes inputs internally, so the
capture resolution isn't forced by the checkpoint. We render at standard
RealSense RGB (640x480) — close to the real sensors the model was trained on.
"""

from pathlib import Path

import numpy as np
import genesis as gs

# Standard RealSense RGB resolution (W, H). The policy processor resizes anyway.
RES = (640, 480)
FOV = 45

# Workspace center: table sits at x~0.1, arm at origin, cube in front (~0.18, 0).
_LOOKAT = (0.1, 0.0, 0.12)

# Two external views chosen to be complementary so the workspace (and the
# graspable object) stays visible in at least one even when the arm occludes
# the other. "top" = near-overhead bird's-eye; "side" = elevated 3/4 from +y.
# Order here is the canonical order we hand to the policy (it is order-agnostic,
# but we keep it stable for reproducibility / overlays / recording).
CAMERA_POSES = {
    "top": dict(pos=(0.12, 0.0, 0.7), lookat=(0.12, 0.0, 0.1), fov=FOV),
    "side": dict(pos=(0.18, 0.5, 0.4), lookat=_LOOKAT, fov=FOV),
}

CAMERA_NAMES = list(CAMERA_POSES)  # ["top", "side"]


def add_cameras(scene, res=RES, gui=False):
    """Add the top + side cameras to a scene (call before scene.build()).

    Returns an ordered dict {name: Camera} following CAMERA_NAMES.
    """
    cams = {}
    for name in CAMERA_NAMES:
        p = CAMERA_POSES[name]
        cams[name] = scene.add_camera(
            res=res,
            pos=p["pos"],
            lookat=p["lookat"],
            fov=p["fov"],
            GUI=gui,
        )
    return cams


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
