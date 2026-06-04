"""Cartesian (end-effector) keyboard teleop of the REAL SO101 follower.

Mirrors scripts/sim_teleop_ee.py — same anti-flail IK (placo DLS regularization +
velocity & joint limits), fixed ~45deg-down orientation, yaw-follows-azimuth, and
workspace clamp — but reads keys from the terminal and drives the real follower
via send_action (degrees). The follower loads the same so101 URDF placo uses, so
FK/IK are consistent with the arm.

SAFETY: the EE target starts at FK(current pose) → NO motion until you press a
key; every commanded joint step is hard-limited to --max-step-deg; torque holds
on quit. Run it in a terminal (raw keyboard + the arm connected).

    .venv-lerobot/bin/python scripts/real_teleop_ee.py

Controls (focus this terminal):
    arrows   EE +x/-x (up/down) and +y/-y (left/right)
    r / f    EE +z / -z  (up / down)
    e / d    rotate gripper (wrist roll)
    space    gripper open / close
    0        return EE to home pose
    q        quit (torque stays on, arm holds)
"""
import os
import sys
import time
import select
import termios
import tty

import numpy as np
import placo
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(REPO, "assets", "so101", "so101_new_calib.urdf")
PORT = "/dev/tty.usbmodem58FD0169761"
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINTS = ARM + ["gripper"]

POS_STEP = 0.004          # m per key event
ROLL_STEP = np.deg2rad(3) # rad per key event
MAX_STEP_DEG = 2.0        # hard cap on commanded joint change per control tick
POS_W, ORI_W = 1.0, 0.6
WS_LO = np.array([0.08, -0.25, -0.02])   # reachable EE box (m)
WS_HI = np.array([0.34, 0.25, 0.30])
GRIP_OPEN, GRIP_CLOSED = 45.0, 2.0

# ---- placo IK (radians), with the anti-flail config from the sim ----------
_k = placo.RobotWrapper(URDF)
_s = placo.KinematicsSolver(_k)
_s.mask_fbase(True)
_task = _s.add_frame_task("gripper_frame_link", np.eye(4))
_s.dt = 0.05
_s.enable_joint_limits(True)
_s.enable_velocity_limits(True)
for _n in ARM:
    _k.set_velocity_limit(_n, np.deg2rad(120))
_s.add_regularization_task(1e-3)


def fk(q_rad):
    for n, a in zip(ARM, q_rad):
        _k.set_joint(n, float(a))
    _k.update_kinematics()
    return np.array(_k.get_T_world_frame("gripper_frame_link"))


def ik(q0_rad, T_target):
    for n, a in zip(ARM, q0_rad):
        _k.set_joint(n, float(a))
    _task.T_world_frame = T_target
    _task.configure("gripper_frame_link", "soft", POS_W, ORI_W)
    _s.solve(True)
    _k.update_kinematics()
    return np.array([_k.get_joint(n) for n in ARM])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def read_events(fd):
    """Return a list of key events from pending stdin (raw mode)."""
    if not select.select([sys.stdin], [], [], 0)[0]:
        return []
    buf = os.read(fd, 64)
    ev, i = [], 0
    arrows = {0x41: "UP", 0x42: "DOWN", 0x43: "RIGHT", 0x44: "LEFT"}
    while i < len(buf):
        if buf[i] == 0x1B and i + 2 < len(buf) and buf[i + 1] == 0x5B:
            ev.append(arrows.get(buf[i + 2], ""))
            i += 3
        else:
            ev.append(chr(buf[i]))
            i += 1
    return ev


def main():
    robot = SO101Follower(SO101FollowerConfig(
        port=PORT, id="blupe_follower",
        max_relative_target=None, disable_torque_on_disconnect=False))
    robot.connect(calibrate=False)

    def read_arm_rad():
        o = robot.get_observation()
        return np.deg2rad([o[f"{j}.pos"] for j in ARM])

    q_now = read_arm_rad()
    q_cmd_deg = np.rad2deg(q_now)   # internal commanded joint state (NOT re-read from
                                    # the noisy/lagging sensor each tick -> smooth)
    T0 = fk(q_now)
    R_home = T0[:3, :3]
    pos = T0[:3, 3].copy()
    home_pos = pos.copy()
    roll = 0.0
    grip = GRIP_OPEN
    print(f"start EE pos {np.round(pos,3)} — no motion until you press a key.")

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            for e in read_events(fd):
                if e == "q":
                    raise KeyboardInterrupt
                elif e == "UP":    pos[0] += POS_STEP
                elif e == "DOWN":  pos[0] -= POS_STEP
                elif e == "LEFT":  pos[1] += POS_STEP
                elif e == "RIGHT": pos[1] -= POS_STEP
                elif e == "r":     pos[2] += POS_STEP
                elif e == "f":     pos[2] -= POS_STEP
                elif e == "e":     roll += ROLL_STEP
                elif e == "d":     roll -= ROLL_STEP
                elif e == " ":     grip = GRIP_CLOSED if grip == GRIP_OPEN else GRIP_OPEN
                elif e == "0":     pos[:] = home_pos; roll = 0.0; grip = GRIP_OPEN
            pos[:] = np.clip(pos, WS_LO, WS_HI)

            T = np.eye(4)
            psi = np.arctan2(pos[1], pos[0])
            T[:3, :3] = Rz(psi) @ R_home @ Rz(roll)
            T[:3, 3] = pos
            # seed IK from the COMMANDED state (smooth), not the measured sensor
            q_ik_deg = np.rad2deg(ik(np.deg2rad(q_cmd_deg), T))
            # hard rate limit -> never command more than MAX_STEP_DEG/joint/tick
            q_cmd_deg = q_cmd_deg + np.clip(q_ik_deg - q_cmd_deg, -MAX_STEP_DEG, MAX_STEP_DEG)
            action = {f"{j}.pos": float(v) for j, v in zip(ARM, q_cmd_deg)}
            action["gripper.pos"] = float(grip)
            try:
                robot.send_action(action)
            except Exception:
                pass
            sys.stdout.write(f"\rEE{np.round(pos,3)} roll{roll:+.2f} grip{grip:.0f}   ")
            sys.stdout.flush()
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\nquit")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        robot.disconnect()
        print("[disconnected — torque held]")


if __name__ == "__main__":
    main()
