"""Keyboard joint teleop of the REAL SO101 follower (lerobot 0.5.1).

Select a joint and nudge it; position targets go straight to the calibrated
follower via send_action. No IK — direct joint control (simple + robust, no
URDF/frame-convention risk). Run in a terminal (needs raw keyboard + the arm).

    .venv-lerobot/bin/python scripts/keyboard_teleop_real.py

Controls (focus this terminal):
    1-6      select joint: pan, lift, elbow, wrist_flex, wrist_roll, gripper
    k / +    increase selected joint   (hold to keep moving)
    j / -    decrease selected joint
    [  ]     gripper close / open  (shortcut, any selection)
    h        go to home pose
    s        print current state
    q        quit (torque stays on, arm holds its pose)
"""
import sys
import time
import select
import termios
import tty

import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/tty.usbmodem58FD0169761"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HOME = np.array([-0.5, -99.0, 91.0, 60.0, -3.6, 1.1], dtype=np.float32)
STEP = 2.0          # deg per keypress for arm joints
GRIP_STEP = 6.0     # per keypress for the gripper
# soft limits (deg) to stay inside the follower's range (avoids slamming a stop)
LO = np.array([-110, -110, -110, -110, -180,   0], dtype=np.float32)
HI = np.array([ 110,  110,  110,  110,  180, 100], dtype=np.float32)


def main():
    robot = SO101Follower(SO101FollowerConfig(
        port=PORT, id="blupe_follower",
        max_relative_target=None, disable_torque_on_disconnect=False))
    robot.connect(calibrate=False)

    def read_state():
        o = robot.get_observation()
        return np.array([o[f"{j}.pos"] for j in JOINTS], dtype=np.float32)

    def send(target):
        try:
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, target)})
        except Exception:
            pass  # transient bus glitch — skip this tick rather than crash

    target = np.clip(read_state(), LO, HI)
    sel = 0
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print("keyboard teleop — 1-6 select | k/j move | [ ] gripper | h home | q quit")
    try:
        tty.setcbreak(fd)
        while True:
            while select.select([sys.stdin], [], [], 0)[0]:
                c = sys.stdin.read(1)
                if c == "q":
                    raise KeyboardInterrupt
                elif c in "123456":
                    sel = int(c) - 1
                elif c in ("k", "+", "="):
                    target[sel] += GRIP_STEP if sel == 5 else STEP
                elif c in ("j", "-", "_"):
                    target[sel] -= GRIP_STEP if sel == 5 else STEP
                elif c == "]":
                    target[5] += GRIP_STEP
                elif c == "[":
                    target[5] -= GRIP_STEP
                elif c == "h":
                    target = HOME.copy()
                elif c == "s":
                    sys.stdout.write(f"\nstate={np.round(read_state(),1).tolist()}\n")
            target = np.clip(target, LO, HI)
            send(target)
            sys.stdout.write(f"\r[{JOINTS[sel]:13s}] target={np.round(target,1).tolist()}      ")
            sys.stdout.flush()
            time.sleep(0.03)  # ~30 Hz
    except KeyboardInterrupt:
        print("\nquit")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        robot.disconnect()
        print("[disconnected — torque held, arm holding pose]")


if __name__ == "__main__":
    main()
