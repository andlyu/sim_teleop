"""Keyboard teleoperation of the SO101 arm in Genesis.

A ViewerPlugin captures keystrokes and nudges per-joint position targets, which a
PD controller tracks each sim step. The scene (table + graspable cube) comes from
so101_scene, so you can drive the arm to pick up the cube.

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

import numpy as np
import genesis as gs
from genesis.vis.keybindings import Key
from genesis.vis.viewer_plugins import ViewerPlugin

import so101_scene as S

JOINT_NAMES = S.JOINT_NAMES

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

    scene, robot, _cube = S.build_scene(show_viewer=True)
    scene.build()
    dofs_idx = S.setup_control(robot)

    limits = S.joint_limits()
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
