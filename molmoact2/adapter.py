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


def state_sim_to_model(joint_rad: np.ndarray, conv: JointConvention = SO101_DEG) -> np.ndarray:
    """Genesis joint angles (rad) -> 6-D float32 state for MolmoAct2."""
    joint_rad = np.asarray(joint_rad, dtype=np.float32)
    if joint_rad.shape != (NUM_JOINTS,):
        raise ValueError(f"expected ({NUM_JOINTS},) joint vector, got {joint_rad.shape}")
    return conv.to_raw(joint_rad)


def action_model_to_sim(action_raw: np.ndarray, conv: JointConvention = SO101_DEG) -> np.ndarray:
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
