"""Phase C: keyboard teleoperation of the SO101 arm in Genesis.

A ViewerPlugin captures keystrokes and nudges per-joint position targets, which a
PD controller tracks each sim step. This is the first real "teleop" milestone.

Controls (focus the viewer window):
    1-6      select a joint (shoulder_pan, shoulder_lift, elbow_flex,
             wrist_flex, wrist_roll, gripper)
    = / -    move the selected joint + / -   (hold to keep moving)
    up/down  same as = / -  (alternative)
    0        reset all joints to zero
    close the window to exit

Run:
    .venv/bin/python scripts/teleop_so101.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import genesis as gs
from genesis.vis.keybindings import Key
from genesis.vis.viewer_plugins import ViewerPlugin

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

# Radians moved per sim step while a +/- key is held.
STEP = 0.01

SELECT_KEYS = {
    int(Key._1): 0,
    int(Key._2): 1,
    int(Key._3): 2,
    int(Key._4): 3,
    int(Key._5): 4,
    int(Key._6): 5,
}
INC_KEYS = {int(Key.EQUAL), int(Key.UP)}
DEC_KEYS = {int(Key.MINUS), int(Key.DOWN)}


def joint_limits(urdf_path: Path) -> dict[str, tuple[float, float]]:
    """Read (lower, upper) limits per joint name from the URDF."""
    root = ET.parse(urdf_path).getroot()
    limits = {}
    for j in root.findall("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("lower") is not None:
            limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return limits


class KeyboardTeleop(ViewerPlugin):
    """Nudge SO101 joint targets from the keyboard."""

    def __init__(self, robot, dofs_idx, lowers, uppers):
        super().__init__()
        self.robot = robot
        self.dofs_idx = dofs_idx
        self.lowers = lowers
        self.uppers = uppers
        self.targets = np.zeros(len(dofs_idx))
        self.selected = 0
        self.held = set()

    def on_key_press(self, symbol, modifiers):
        if symbol in SELECT_KEYS:
            self.selected = SELECT_KEYS[symbol]
            self.viewer.set_message_text(f"Joint {self.selected + 1}: {JOINT_NAMES[self.selected]}")
        elif symbol == int(Key._0):
            self.targets[:] = 0.0
            self.viewer.set_message_text("Reset to zero")
        elif symbol in INC_KEYS or symbol in DEC_KEYS:
            self.held.add(symbol)

    def on_key_release(self, symbol, modifiers):
        self.held.discard(symbol)

    def update_on_sim_step(self):
        moved = any(k in self.held for k in INC_KEYS | DEC_KEYS)
        if moved:
            direction = 0.0
            if any(k in self.held for k in INC_KEYS):
                direction += 1.0
            if any(k in self.held for k in DEC_KEYS):
                direction -= 1.0
            i = self.selected
            self.targets[i] = np.clip(
                self.targets[i] + direction * STEP, self.lowers[i], self.uppers[i]
            )
            self.viewer.set_message_text(
                f"{JOINT_NAMES[i]} = {self.targets[i]:+.3f} rad"
            )
        self.robot.control_dofs_position(self.targets, self.dofs_idx)


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

    limits = joint_limits(URDF)
    lowers = np.array([limits[n][0] for n in JOINT_NAMES])
    uppers = np.array([limits[n][1] for n in JOINT_NAMES])

    teleop = KeyboardTeleop(robot, dofs_idx, lowers, uppers)
    scene.viewer.add_plugin(teleop)

    print(__doc__)
    print("Teleop ready — focus the viewer window and press 1-6, then = / -.")
    while scene.viewer.is_alive():
        scene.step()
    print("Teleop closed.")


if __name__ == "__main__":
    main()
