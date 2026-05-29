"""Headless SO101 sim environment with the tuned camera rig (no web UI).

A small gym-style wrapper around so101_scene + cameras.py that produces exactly
the observation MolmoAct2 consumes: top + side external RGB, the wrist (gripper)
RGB, and the 6-D joint state. Cameras are placed at the values tuned in
arm_cam_view.py and baked into cameras.py.

This is the reusable env the policy/teleop loops plug into (plan subtask 1+2).

Usage:
    from sim_env import SimEnv
    env = SimEnv()
    obs = env.reset()
    obs = env.step(target_rad)          # 6 joint targets (radians)
    imgs = env.observation_images()     # [top, side] for the policy
    # obs = {"state": (6,), "images": {"top","side","wrist"}}

CLI (smoke test, writes snapshots to /tmp):
    .venv/bin/python scripts/sim_env.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_scene as S
import cameras as C

# Wrist-cam offset tuned via the web UI (roll 20, pitch 0, yaw 180 deg).
WRIST_RPY_DEG = (20.0, 0.0, 180.0)

# A neutral "ready" pose (radians), clipped to limits at reset.
HOME_POSE = np.array([0.0, 0.5, -0.5, 0.5, 0.0, 0.2])


def _rpy_to_T(xyz, rpy):
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = xyz
    return T


class SimEnv:
    """Headless SO101 + table + cube, with top/side/wrist cameras."""

    def __init__(self, settle_steps=150, sub_steps=20, backend=gs.cpu,
                 viewer=False, gui_cams=False):
        gs.init(backend=backend, logging_level="error")
        scene, robot, cube = S.build_scene(show_viewer=viewer)

        # External top + side cameras (tuned placements from cameras.py).
        # gui_cams -> each camera opens a live OpenCV window (Genesis GUI=True).
        ext = C.add_cameras(scene, gui=gui_cams)   # {"top","side"} ordered
        # On-robot wrist (gripper) camera.
        wrist = scene.add_camera(
            res=C.RES, fov=C.WRIST_FOV, near=C.WRIST_NEAR, far=10.0, GUI=gui_cams
        )
        self.gui_cams = gui_cams

        scene.build()
        dofs = S.setup_control(robot)
        link = robot.get_link(C.WRIST_MOUNT_LINK)
        wrist.attach(link, _rpy_to_T((0.0, 0.0, 0.0), np.deg2rad(WRIST_RPY_DEG)))

        limits = S.joint_limits()
        self.lowers = np.array([limits[n][0] for n in S.JOINT_NAMES])
        self.uppers = np.array([limits[n][1] for n in S.JOINT_NAMES])

        self.scene, self.robot, self.cube = scene, robot, cube
        self.dofs, self.wrist = dofs, wrist
        self.cams = {**ext, "wrist": wrist}
        self.settle_steps, self.sub_steps = settle_steps, sub_steps
        self.target = np.clip(HOME_POSE[: len(dofs)], self.lowers, self.uppers)

    def reset(self, pose=None):
        """Drive to a start pose, settle, and return the observation."""
        p = HOME_POSE if pose is None else np.asarray(pose, float)
        self.target = np.clip(p[: len(self.dofs)], self.lowers, self.uppers)
        for _ in range(self.settle_steps):
            self.robot.control_dofs_position(self.target, self.dofs)
            self.scene.step()
        self.wrist.move_to_attach()
        return self.observation()

    def step(self, target_rad):
        """Set joint targets (radians), let the PD controller track, observe."""
        self.target = np.clip(np.asarray(target_rad, float), self.lowers, self.uppers)
        for _ in range(self.sub_steps):
            self.robot.control_dofs_position(self.target, self.dofs)
            self.scene.step()
        self.wrist.move_to_attach()
        return self.observation()

    def state(self):
        """Current 6-D joint angles (radians)."""
        return np.asarray(self.robot.get_dofs_position(self.dofs)).reshape(-1)

    def observation_images(self):
        """[top, side] RGB list, canonical policy order (from cameras.py)."""
        return [np.asarray(self.cams[n].render()[0])[..., :3].astype(np.uint8)
                for n in C.CAMERA_NAMES]

    def observation(self):
        """{"state": (6,), "images": {name: HxWx3 uint8}} for top/side/wrist."""
        imgs = {n: np.asarray(c.render()[0])[..., :3].astype(np.uint8)
                for n, c in self.cams.items()}
        return {"state": self.state(), "images": imgs}

    def run_viewer(self):
        """Hold the current target and step until the viewer window is closed.

        Requires viewer=True at construction (must run on the main thread).
        """
        if not getattr(self.scene, "viewer", None):
            raise RuntimeError("SimEnv was built with viewer=False")
        print("Viewer open — close the window to exit.")
        while self.scene.viewer.is_alive():
            self.robot.control_dofs_position(self.target, self.dofs)
            self.scene.step()
            self.wrist.move_to_attach()
        print("Viewer closed.")

    def run_teleop(self, render_every=16):
        """3D viewer + keyboard arm control + live camera windows, one loop.

        Requires viewer=True AND gui_cams=True. Keyboard (focus the 3D window):
          1-6      select a joint
          = / up   increase selected joint;  - / down  decrease
          0        reset all joints to zero
        The top/side/wrist OpenCV windows refresh every step so you can verify
        the camera inputs while you move the arm.

        NOTE: this composes the Genesis viewer + GUI cameras + a ViewerPlugin in
        a single process. Each piece is a documented Genesis pattern; combining
        all three is our composition (the docs show them separately).
        """
        import cv2
        from genesis.vis.viewer_plugins import ViewerPlugin
        from genesis.vis.keybindings import Key

        if not getattr(self.scene, "viewer", None):
            raise RuntimeError("run_teleop needs viewer=True")
        if not self.gui_cams:
            raise RuntimeError("run_teleop needs gui_cams=True")

        sel = {int(getattr(Key, f"_{i+1}")): i for i in range(len(self.dofs))}
        inc = {int(Key.EQUAL), int(Key.UP)}
        dec = {int(Key.MINUS), int(Key.DOWN)}
        env = self
        STEP = 0.01

        class _Teleop(ViewerPlugin):
            def __init__(self):
                super().__init__()
                self.selected = 0
                self.held = set()

            def on_key_press(self, symbol, modifiers):
                if symbol in sel:
                    self.selected = sel[symbol]
                    self.viewer.set_message_text(
                        f"Joint {self.selected+1}: {S.JOINT_NAMES[self.selected]}")
                elif symbol == int(Key._0):
                    env.target[:] = 0.0
                    self.viewer.set_message_text("Reset to zero")
                elif symbol in inc or symbol in dec:
                    self.held.add(symbol)

            def on_key_release(self, symbol, modifiers):
                self.held.discard(symbol)

            def update_on_sim_step(self):
                d = (1.0 if self.held & inc else 0.0) - (1.0 if self.held & dec else 0.0)
                if d:
                    i = self.selected
                    env.target[i] = float(np.clip(
                        env.target[i] + d * STEP, env.lowers[i], env.uppers[i]))
                    self.viewer.set_message_text(
                        f"{S.JOINT_NAMES[i]} = {env.target[i]:+.3f} rad")
                env.robot.control_dofs_position(env.target, env.dofs)

        self.scene.viewer.add_plugin(_Teleop())
        print("Teleop: focus the 3D window. 1-6 pick a joint, =/- (or arrows) move it, 0 reset.")
        print(f"Top/side/wrist camera windows refresh every {render_every} steps. "
              "Close 3D window to exit.")
        # Camera rendering (3x offscreen rasterize) is the per-step bottleneck on
        # CPU; the 3D viewer renders itself, so we throttle only the cameras —
        # they update every `render_every` steps while the sim/viewer stay smooth.
        t0, frames = time.perf_counter(), 0
        try:
            i = 0
            while self.scene.viewer.is_alive():
                self.scene.step()
                self.wrist.move_to_attach()
                if i % render_every == 0:
                    if not self.scene.viewer.is_alive():
                        break
                    for cam in self.cams.values():
                        cam.render()
                    cv2.waitKey(1)
                    frames += 1
                i += 1
        except gs.GenesisException:
            pass  # viewer closed during render — clean shutdown
        finally:
            dt = time.perf_counter() - t0
            if frames:
                print(f"avg camera-frame render: {dt / frames * 1000:.0f} ms "
                      f"({frames} frames over {i} steps)")
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("Teleop closed.")

    def stream_cams(self, steps=None):
        """Stream live camera windows (Genesis GUI=True) for the current state.

        Each camera (top/side/wrist) shows in its own OpenCV window, refreshed
        every step. Requires gui_cams=True at construction. Runs `steps`
        iterations, or forever (until Ctrl-C) if steps is None.
        """
        import cv2

        if not self.gui_cams:
            raise RuntimeError("SimEnv was built with gui_cams=False")
        print("Streaming camera windows — Ctrl-C (or 'q' in a window) to stop.")
        i = 0
        try:
            while steps is None or i < steps:
                self.robot.control_dofs_position(self.target, self.dofs)
                self.scene.step()
                self.wrist.move_to_attach()
                for cam in self.cams.values():
                    cam.render()          # refreshes that camera's GUI window
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                i += 1
        except KeyboardInterrupt:
            print("\nstopped.")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true",
                    help="open the interactive Genesis 3D window")
    ap.add_argument("--cams", action="store_true",
                    help="stream live OpenCV windows of the top/side/wrist cameras")
    ap.add_argument("--teleop", action="store_true",
                    help="3D viewer + keyboard arm control + live camera windows")
    ap.add_argument("--render-every", type=int, default=16,
                    help="refresh camera windows every N sim steps (higher = faster sim)")
    args = ap.parse_args()

    if args.teleop:
        print("Building SimEnv (viewer + keyboard + camera windows)...")
        env = SimEnv(viewer=True, gui_cams=True)
        env.reset()
        env.run_teleop(render_every=args.render_every)
        return

    if args.viewer:
        print("Building SimEnv (with 3D viewer)...")
        env = SimEnv(viewer=True)
        env.reset()
        env.run_viewer()
        return

    if args.cams:
        print("Building SimEnv (live camera windows)...")
        env = SimEnv(gui_cams=True)
        env.reset()
        env.stream_cams()
        return

    from PIL import Image
    print("Building SimEnv (headless)...")
    env = SimEnv()
    obs = env.reset()
    print("state (rad):", np.round(obs["state"], 3))
    for name, img in obs["images"].items():
        out = f"/tmp/env_{name}.png"
        Image.fromarray(img).save(out)
        print(f"  wrote {out}  {img.shape} {img.dtype}")
    print("OK")


if __name__ == "__main__":
    main()
