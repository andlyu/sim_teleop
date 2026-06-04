"""State/action adapter: Genesis convention <-> MolmoAct2 SO101 convention.

This is the single highest-risk seam in the whole pipeline. Genesis controls
joints in **radians**; the `MolmoAct2-SO100_101` checkpoint (covers SO100/SO101,
which share joints) was trained on the LeRobot datasets, whose recorded
state/actions are in the robot's **raw scale** (degrees for the revolute joints,
per the model card's sample state):

    robot_state = [-0.527, 189.14, 181.41, 60.64, -3.604, 1.097]   # not radians

So before sending proprioceptive state to the model we convert rad -> deg, and
the actions it returns (robot scale) we convert deg -> rad before feeding them
to `control_dofs_position`.

IMPORTANT — UNITS ARE NOT YET CALIBRATED AGAINST THE LIVE CHECKPOINT.
The degree convention is inferred from the model card example, not verified end
to end. The conversion is therefore expressed as an explicit, per-joint affine
map (scale + offset) so it can be calibrated once we can run the real model,
rather than being a hidden hardcoded guess. Treat `JointConvention.SO101_DEG`
as a starting hypothesis to validate, not ground truth.

Joint order (matches the SO101 URDF and the rest of the repo):
    shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
NUM_JOINTS = len(JOINT_NAMES)


@dataclass(frozen=True)
class JointConvention:
    """Affine map between Genesis radians and the model's raw joint scale.

    For each joint i:   raw_i = scale_i * rad_i + offset_i
                        rad_i = (raw_i - offset_i) / scale_i

    `scale` and `offset` are length-NUM_JOINTS arrays. The default below assumes
    the five revolute arm joints are reported in degrees with no offset, and
    leaves the gripper as a separate hypothesis (its sample value 1.097 is small
    enough to be radians OR a normalized open/close fraction — needs checking).
    """

    scale: np.ndarray
    offset: np.ndarray

    def to_raw(self, rad: np.ndarray) -> np.ndarray:
        rad = np.asarray(rad, dtype=np.float32)
        return (self.scale * rad + self.offset).astype(np.float32)

    def to_rad(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32)
        return ((raw - self.offset) / self.scale).astype(np.float32)


_DEG_PER_RAD = float(np.degrees(1.0))

# Hypothesis A (default): arm joints in degrees, gripper passed through as-is.
# Calibrate against the live checkpoint before trusting (see module docstring).
SO101_DEG = JointConvention(
    scale=np.array([_DEG_PER_RAD] * 5 + [1.0], dtype=np.float32),
    offset=np.zeros(NUM_JOINTS, dtype=np.float32),
)

# Hypothesis B: everything already in radians (identity). Kept so we can A/B
# the two conventions quickly during calibration.
IDENTITY = JointConvention(
    scale=np.ones(NUM_JOINTS, dtype=np.float32),
    offset=np.zeros(NUM_JOINTS, dtype=np.float32),
)

# Convention used by the public SO-101 MolmoAct2 runner linked from the
# official MolmoAct2 README. LeRobot's current SO101 follower reports degrees
# (`use_degrees=True`), while the SO100/101 MolmoAct2 checkpoint was trained on
# the older v2.1 frame. The official backward-compat transform is:
#
#   model_state = signs * arm_degrees + offsets
#   arm_degrees = (model_action - offsets) * signs
#
# with a shoulder_lift sign flip and +90deg offsets on shoulder_lift/elbow_flex.
LEROBOT_V21_COMPAT = JointConvention(
    scale=np.array(
        [_DEG_PER_RAD, -_DEG_PER_RAD, _DEG_PER_RAD, _DEG_PER_RAD, _DEG_PER_RAD, _DEG_PER_RAD],
        dtype=np.float32,
    ),
    offset=np.array([0.0, 90.0, 90.0, 0.0, 0.0, 0.0], dtype=np.float32),
)

# Hypothesis C (RANGE_ALIGNED): map each sim joint's full URDF range [lo,hi] (rad)
# onto the model's typical operating range [q01,q99] (deg) from norm_stats.json.
# Computed from assets/so101 limits + molmoact2/norm_stats.json. This accounts
# for the LeRobot "new calib" convention (zero = middle of range) that gives the
# large shoulder_lift/elbow_flex offsets — pure 57.3deg/rad + offset=0 would put
# those joints far outside the range the model ever saw. Scales (~20-40) differ
# from the physical 57.3 because sim full-range is compressed to q01..q99.
# Built by scripts/calibrate_adapter.py; verify against the live model.
RANGE_ALIGNED_NEW = JointConvention(
    scale=np.array([23.492, 40.563, 39.866, 25.955, 19.049, 22.498], dtype=np.float32),
    offset=np.array([3.192, 114.465, 105.762, 48.746, -11.178, 4.870], dtype=np.float32),
)

RANGE_ALIGNED_OLD = JointConvention(
    scale=np.array([23.492, 40.563, 40.634, 25.955, 19.049, 22.498], dtype=np.float32),
    offset=np.array([3.192, 178.181, 45.480, 48.746, -10.250, 4.870], dtype=np.float32),
)


def _calibration_name() -> str:
    name = os.environ.get("SO101_CALIBRATION", "new").strip().lower()
    if name not in {"new", "old"}:
        raise ValueError(f"SO101_CALIBRATION must be 'new' or 'old', got {name!r}")
    return name


# REAL-HARDWARE convention. The lerobot SO101 *follower* already reports
# DEGREES (use_degrees=True) in the LeRobot v3.0 frame — no sim radians involved.
# The MolmoAct2-SO100_101 checkpoint was trained on the older v2.1 frame, so the
# transform is degree->degree (same shoulder_lift sign flip + 90deg offsets as
# LEROBOT_V21_COMPAT, but with unit scale instead of deg/rad):
#     model = signs * deg + offset           (signs = [1,-1,1,1,1,1])
#     deg   = (model - offset) / signs        (signs are +/-1, so == multiply)
# Use this (not the rad-based maps) for scripts/run_policy_real.py. The gripper
# (scale 1, offset 0) is a pass-through hypothesis — VALIDATE in the dry run, as
# lerobot's gripper unit (degrees vs 0-100) vs the model's is still unconfirmed.
LEROBOT_V21_COMPAT_DEG = JointConvention(
    scale=np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    offset=np.array([0.0, 90.0, 90.0, 0.0, 0.0, 0.0], dtype=np.float32),
)


RANGE_ALIGNED = RANGE_ALIGNED_OLD if _calibration_name() == "old" else RANGE_ALIGNED_NEW

# Default convention used by the client. For the standard/current SO101
# calibration, match the public real-arm MolmoAct2 runner's LeRobot v3.0->v2.1
# frame transform. Keep RANGE_ALIGNED for the old-calibration URDF experiments.
DEFAULT = RANGE_ALIGNED_OLD if _calibration_name() == "old" else LEROBOT_V21_COMPAT


def state_sim_to_model(joint_rad: np.ndarray, conv: JointConvention = DEFAULT) -> np.ndarray:
    """Genesis joint angles (rad) -> 6-D float32 state for MolmoAct2."""
    joint_rad = np.asarray(joint_rad, dtype=np.float32)
    if joint_rad.shape != (NUM_JOINTS,):
        raise ValueError(f"expected ({NUM_JOINTS},) joint vector, got {joint_rad.shape}")
    return conv.to_raw(joint_rad)


def action_model_to_sim(action_raw: np.ndarray, conv: JointConvention = DEFAULT) -> np.ndarray:
    """A MolmoAct2 action (robot scale) -> Genesis joint targets (rad).

    MolmoAct2 returns an action chunk of shape (N, D). This converts a single
    D-vector; callers iterate over the chunk. D is expected to be NUM_JOINTS for
    the SO100/101 checkpoint, but the model pads to width 32 — so we slice the
    first NUM_JOINTS dims defensively.
    """
    action_raw = np.asarray(action_raw, dtype=np.float32).reshape(-1)
    if action_raw.shape[0] < NUM_JOINTS:
        raise ValueError(f"action has {action_raw.shape[0]} dims, need >= {NUM_JOINTS}")
    return conv.to_rad(action_raw[:NUM_JOINTS])
