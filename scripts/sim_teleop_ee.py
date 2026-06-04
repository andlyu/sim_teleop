"""Cartesian (end-effector) keyboard teleop of the SO101 in Genesis sim.

The EE orientation is held FIXED at the home pose's ~50deg-down tilt (the SO101
is 5-DOF, so orientation isn't freely controllable — we just hold this one). You
translate the EE in Cartesian space and spin/clench the gripper; placo IK turns
the target EE pose into joint targets each step.

Controls (focus the 3D viewer window):
    up / down     EE +x / -x   (forward / back)
    left / right  EE +y / -y   (left / right)
    r / f         EE +z / -z   (up / down)
    e / d         rotate gripper  (wrist roll)
    space         toggle gripper open / close
    close window  exit

Run:
    .venv/bin/python scripts/sim_teleop_ee.py
"""
import sys
from pathlib import Path

import numpy as np
import placo
import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_scene as S
from genesis.vis.viewer_plugins import ViewerPlugin
from genesis.vis.keybindings import Key

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
URDF = str(S.URDF)
# Reference pose whose EE orientation we hold (the "~45deg down" the user picked).
REF_DEG = np.array([-0.5, -99.0, 91.0, 60.0, -3.6])
POS_STEP = 0.0025   # m per sim-step while a key is held
ROLL_STEP = 0.02    # rad per step for wrist roll
POS_W, ORI_W = 1.0, 0.6
# Reachable EE box (m): clamp the target so it can't be driven into the
# unreachable/near-singular zone where IK has no stable solution.
WS_LO = np.array([0.08, -0.25, -0.02])
WS_HI = np.array([0.34, 0.25, 0.30])


# ---- placo IK (radians) --------------------------------------------------
_k = placo.RobotWrapper(URDF)
_solver = placo.KinematicsSolver(_k)
_solver.mask_fbase(True)
_task = _solver.add_frame_task("gripper_frame_link", np.eye(4))
# Anti-flail: damped-least-squares regularization + velocity & joint limits, so
# the solver returns a STABLE, small-step solution near singularities/unreachable
# targets instead of flipping between IK branches (validated: converges & holds
# steady where the un-regularized solver swung ~60deg/step).
_solver.dt = 0.05
_solver.enable_joint_limits(True)
_solver.enable_velocity_limits(True)
for _n in ARM:
    _k.set_velocity_limit(_n, np.deg2rad(120))   # 120 deg/s -> <=6 deg per solve
_solver.add_regularization_task(1e-3)


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
    _solver.solve(True)
    _k.update_kinematics()
    return np.array([_k.get_joint(n) for n in ARM])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _patch_cpuinfo():
    try:
        import cpuinfo
    except Exception:
        return
    orig = cpuinfo.get_cpu_info
    def patched():
        info = orig()
        if not any(info.get(k) for k in ("brand_raw", "hardware_raw", "vendor_id_raw")):
            info["brand_raw"] = "Apple ARM CPU"
        return info
    cpuinfo.get_cpu_info = patched


def main():
    _patch_cpuinfo()
    gs.init(backend=gs.cpu, logging_level="error")
    scene, robot, cube = S.build_scene(show_viewer=True)
    scene.build()
    dofs = S.setup_control(robot)

    # fixed orientation = the reference pose's EE rotation
    R_home = fk(np.deg2rad(REF_DEG))[:3, :3]
    T0 = fk(np.deg2rad(REF_DEG))
    start_pos = T0[:3, 3].copy()

    grip_lim = S.joint_limits().get("gripper", (0.0, 1.0))
    grip_open, grip_closed = max(grip_lim), min(grip_lim)

    # drive to the reference pose first
    start = np.concatenate([np.deg2rad(REF_DEG), [grip_open]])
    for _ in range(150):
        robot.control_dofs_position(start, dofs)
        scene.step()

    state = {"pos": start_pos.copy(), "roll": 0.0, "grip": grip_open}
    logf = open("/tmp/ee_teleop_log.csv", "w")
    logf.write("i,ee_x,ee_y,ee_z,roll,grip,j1,j2,j3,j4,j5,ee_x_actual,ee_y_actual,ee_z_actual\n")
    HELD_POS = {  # key -> (axis, sign)
        int(Key.UP): (0, +1), int(Key.DOWN): (0, -1),
        int(Key.LEFT): (1, +1), int(Key.RIGHT): (1, -1),
        int(Key.R): (2, +1), int(Key.F): (2, -1),
    }
    HELD_ROLL = {int(Key.E): +1, int(Key.D): -1}

    class _EE(ViewerPlugin):
        def __init__(self):
            super().__init__()
            self.held = set()
            self.i = 0

        def on_key_press(self, symbol, modifiers):
            if symbol == int(Key._0):
                state["pos"] = start_pos.copy()
                state["roll"] = 0.0
                state["grip"] = grip_open
                return True  # reset to home pose
            elif symbol == int(Key.SPACE):
                state["grip"] = grip_closed if state["grip"] == grip_open else grip_open
                return True  # EVENT_HANDLED — don't let it reach viewer toggles
            elif symbol in HELD_POS or symbol in HELD_ROLL:
                self.held.add(symbol)
                return True
            return None  # let other keys (help text, etc.) through

        def on_key_release(self, symbol, modifiers):
            if symbol in HELD_POS or symbol in HELD_ROLL or symbol == int(Key.SPACE):
                self.held.discard(symbol)
                return True
            return None

        def update_on_sim_step(self):
            for k in self.held:
                if k in HELD_POS:
                    ax, sg = HELD_POS[k]
                    state["pos"][ax] += sg * POS_STEP
                elif k in HELD_ROLL:
                    state["roll"] += HELD_ROLL[k] * ROLL_STEP
            state["pos"][:] = np.clip(state["pos"], WS_LO, WS_HI)  # stay reachable
            # Target EE pose: gripper faces the reach direction (yaw = azimuth)
            # at the fixed ~45deg-down pitch, spun about its approach axis by the
            # wrist roll. Letting yaw follow azimuth is what makes lateral (y)
            # motion reachable on this 5-DOF arm (pure fixed-world orientation
            # fights the base pan).
            T = np.eye(4)
            psi = np.arctan2(state["pos"][1], state["pos"][0])  # world-z azimuth
            T[:3, :3] = Rz(psi) @ R_home @ Rz(state["roll"])
            T[:3, 3] = state["pos"]
            q_now = np.asarray(robot.get_dofs_position(dofs)).reshape(-1)[:5]
            q_arm = ik(q_now, T)
            target = np.concatenate([q_arm, [state["grip"]]])
            robot.control_dofs_position(target, dofs)
            self.viewer.set_message_text(
                f"EE {np.round(state['pos'],3)}  roll {state['roll']:+.2f}  "
                f"grip {'open' if state['grip']==grip_open else 'closed'}")
            # track positions: log target EE + IK joints + ACTUAL achieved EE
            self.i += 1
            if self.i % 3 == 0:
                ee_act = fk(q_now)[:3, 3]
                row = [self.i, *np.round(state["pos"], 4), round(state["roll"], 3),
                       round(state["grip"], 3), *np.round(np.rad2deg(q_arm), 2),
                       *np.round(ee_act, 4)]
                logf.write(",".join(map(str, row)) + "\n")
                logf.flush()
                if self.i % 30 == 0:
                    err = np.linalg.norm(state["pos"] - ee_act) * 1000
                    print(f"i{self.i:4d} EE_tgt{np.round(state['pos'],3)} "
                          f"j(deg){np.round(np.rad2deg(q_arm),1)} "
                          f"grip{state['grip']:.2f}  track_err {err:.0f}mm", flush=True)

    scene.viewer.add_plugin(_EE())
    print("Cartesian teleop: arrows=xy, r/f=z, e/d=roll, space=gripper, 0=reset. Close window to exit.")
    while scene.viewer.is_alive():
        scene.step()
    print("closed.")


if __name__ == "__main__":
    main()
