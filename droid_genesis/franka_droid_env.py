"""Genesis port of the arhanjain/sim-evals DROID environment.

Loads the REAL Franka + Robotiq 2F-85 robot (the sim-evals USD) directly —
Genesis reads USD via gs.morphs.USD (needs `usd-core`). Because the actual robot
is used, the Robotiq `base_link` exists and the wrist camera attaches there with
the exact sim-evals CameraCfg offset (no frame guessing), and the gripper geometry
matches DROID.

Produces the DROID policy observation/action contract so the same openpi
pi0-FAST/pi0.5 server + rollout client work unchanged (only the sim swaps).

Contract:
  observation: arm_joint_pos (7, rad), gripper_pos (1, 0=open..1=closed),
               external_rgb (HxWx3 u8), wrist_rgb (HxWx3 u8)
  action (8-D): [0:7] absolute joint-pos targets (rad), [7] binary gripper (>0.5 close)

Timing: 15 Hz control, decimation 8, sim dt 1/120 (from sim-evals EnvCfg).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import genesis as gs

ASSET_DIR = Path(__file__).resolve().parent / "assets"

# --- Robot: the real Franka + Robotiq 2F-85 from sim-evals (loaded as USD) ----
ROBOT_USD = str(ASSET_DIR / "franka_robotiq_2f_85_flattened.usd")
ARM_JOINTS = [
    "/panda/panda_link0/panda_joint1",
    "/panda/panda_link1/panda_joint2",
    "/panda/panda_link2/panda_joint3",
    "/panda/panda_link3/panda_joint4",
    "/panda/panda_link4/panda_joint5",
    "/panda/panda_link5/panda_joint6",
    "/panda/panda_link6/panda_joint7",
]
FINGER_JOINT = "/panda/Gripper/Robotiq_2F_85/Joints/finger_joint"   # drive joint
GRIPPER_LINKAGE = [   # Robotiq passive linkage joints; held at 0 (open) for now
    "/panda/Gripper/Robotiq_2F_85/Joints/right_outer_knuckle_joint",
    "/panda/Gripper/Robotiq_2F_85/Joints/left_inner_finger_joint",
    "/panda/Gripper/Robotiq_2F_85/Joints/right_inner_finger_joint",
    "/panda/Gripper/Robotiq_2F_85/Joints/left_inner_finger_knuckle_joint",
    "/panda/Gripper/Robotiq_2F_85/Joints/right_inner_finger_knuckle_joint",
]
BASE_LINK = "/panda/Gripper/Robotiq_2F_85/base_link"   # real wrist-cam parent frame

# Franka init pose (rad) — sim-evals nvidia_droid.py NVIDIA_DROID.init_state.
HOME_POSE = np.array(
    [0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0], dtype=np.float32
)

# Robotiq binary gripper: finger_joint 0=open, pi/4=closed (sim-evals ActionCfg).
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = float(np.pi / 4)

# Wrist-cam offset relative to base_link — VERBATIM from sim-evals CameraCfg.OffsetCfg
# (opengl convention; Genesis cameras also view -Z, so it maps directly).
WRIST_CAM_POS = (0.011, -0.031, -0.074)
WRIST_CAM_QUAT = (-0.420, 0.570, 0.576, -0.409)   # (w, x, y, z)

KP = np.array([400.0] * 7, dtype=np.float32)
KV = np.array([80.0] * 7, dtype=np.float32)
GRIP_KP = 100.0
GRIP_KV = 10.0

CONTROL_HZ = 15
DECIMATION = 8
SIM_DT = 1.0 / (CONTROL_HZ * DECIMATION)
MAX_EPISODE_STEPS = CONTROL_HZ * 30   # 450
CAM_RES = (1280, 720)


def _quat_to_R(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def _patch_cpuinfo_for_genesis():
    """Genesis CPU backend wants a CPU brand string; py-cpuinfo can omit it (Mac)."""
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


class FrankaDroidEnv:
    """Genesis Franka + Robotiq 2F-85 env producing the DROID policy observation."""

    def __init__(self, scene_id: int = 1, viewer: bool = False, backend=gs.cpu,
                 settle_steps: int = 60):
        if backend == gs.cpu:
            _patch_cpuinfo_for_genesis()
        gs.init(backend=backend, logging_level="error")

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=SIM_DT),
            # Disable self-collision on articulations (matches sim-evals'
            # enabled_self_collisions=False). The USD's convexified gripper/arm
            # hulls otherwise self-collide and shove the arm off its commanded pose.
            rigid_options=gs.options.RigidOptions(enable_self_collision=False),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, 0.0, 1.2), camera_lookat=(0.0, 0.0, 0.2), camera_fov=40,
            ),
            # The scene1 HDR room is a RayTracer (LuisaRender) feature, unavailable
            # here, so we rasterize with approximate floor/walls (see _add_scene_objects).
            vis_options=gs.options.VisOptions(ambient_light=(0.6, 0.6, 0.6), shadow=False),
            show_viewer=viewer,
        )

        self.scene.add_entity(
            gs.morphs.Plane(),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ImageTexture(
                image_path=str(ASSET_DIR / "floor.png"))),
        )
        # Real Franka + Robotiq robot, fixed base. gravity_compensation=1.0 makes
        # the arm hold its commanded joint pose without sagging (matches sim-evals'
        # disable_gravity on the robot); the cube/bowl still fall under gravity.
        self.robot = self.scene.add_entity(
            # collision=False: the USD's gripper base_link convexifies into a huge
            # bad hull that penetrates the floor and shoves the arm off pose. Disable
            # robot collision so it tracks commanded joints cleanly. TODO: re-enable
            # with proper per-link convex decomposition when we need grasp contact.
            gs.morphs.USD(file=ROBOT_USD, pos=(0.0, 0.0, 0.0), fixed=True, collision=False),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
        self._add_scene_objects(scene_id)
        self._add_cameras()

        self.scene.build()

        self.arm_dofs = [self.robot.get_joint(n).dofs_idx_local[0] for n in ARM_JOINTS]
        self.finger_dof = self.robot.get_joint(FINGER_JOINT).dofs_idx_local[0]
        self.linkage_dofs = [self.robot.get_joint(n).dofs_idx_local[0] for n in GRIPPER_LINKAGE]
        self.grip_dofs = [self.finger_dof] + self.linkage_dofs
        self.robot.set_dofs_kp(KP, self.arm_dofs)
        self.robot.set_dofs_kv(KV, self.arm_dofs)
        self.robot.set_dofs_kp(np.full(len(self.grip_dofs), GRIP_KP), self.grip_dofs)
        self.robot.set_dofs_kv(np.full(len(self.grip_dofs), GRIP_KV), self.grip_dofs)

        # Attach the wrist camera to the REAL Robotiq base_link with the exact
        # sim-evals offset — no frame guessing.
        link = self.robot.get_link(BASE_LINK)
        T = np.eye(4)
        T[:3, :3] = _quat_to_R(*WRIST_CAM_QUAT)
        T[:3, 3] = WRIST_CAM_POS
        self.wrist_cam.attach(link, T)

        self.arm_target = HOME_POSE.copy()
        self.grip_target = GRIPPER_OPEN
        self.settle_steps = settle_steps
        self._step_count = 0

    def _add_scene_objects(self, scene_id: int):
        """Real cube + bowl meshes (world-baked from scene1.usd) + approximate
        table + walls (HDR room needs the ray tracer; unavailable here)."""
        self.objects = {}
        if scene_id != 1:
            raise NotImplementedError("Only scene 1 ported so far (cube+bowl meshes).")
        wall = gs.surfaces.Rough(color=(0.82, 0.80, 0.76))
        H, EXT, THK = 2.5, 1.6, 0.1
        for pos, size in [
            ((EXT, 0.0, H / 2), (THK, 2 * EXT, H)),
            ((-EXT, 0.0, H / 2), (THK, 2 * EXT, H)),
            ((0.0, EXT, H / 2), (2 * EXT, THK, H)),
            ((0.0, -EXT, H / 2), (2 * EXT, THK, H)),
        ]:
            self.scene.add_entity(gs.morphs.Box(pos=pos, size=size, fixed=True), surface=wall)
        self.scene.add_entity(
            gs.morphs.Box(pos=(0.4, 0.05, 0.025), size=(0.6, 0.8, 0.05), fixed=True),
            surface=gs.surfaces.Rough(color=(0.55, 0.42, 0.30)),
        )
        # The real cube's color is an Omniverse MDL material (RTX-only) — not
        # renderable by Genesis's rasterizer (like the HDR room). The cube is
        # geometrically ~a 5.8cm box, so approximate it with a rubik's-cube
        # textured box at its true scene position. (Geometry-only OBJ had no
        # UVs/material, hence the gray look.)
        # UV-mapped cube mesh (Genesis primitive Box has no UVs, so a texture can't
        # map to it). Rubik texture; real MDL material isn't renderable in Genesis.
        self.objects["rubiks_cube"] = self.scene.add_entity(
            gs.morphs.Mesh(file=str(ASSET_DIR / "cube_uv.obj"),
                           pos=(0.36, -0.08, 0.08), fixed=False),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ImageTexture(
                image_path=str(ASSET_DIR / "rubiks.png"))),
        )
        # Bowl: real MDL material not renderable -> solid color (the OBJ has no
        # material, hence the white look).
        self.objects["bowl"] = self.scene.add_entity(
            gs.morphs.Mesh(file=str(ASSET_DIR / "scene1__24_bowl.obj"),
                           pos=(0.0, 0.0, 0.0), fixed=True),
            surface=gs.surfaces.Rough(color=(0.30, 0.45, 0.80)),
        )

    def _add_cameras(self):
        # External camera — approximate sim-evals external_cam framing. TODO: tune.
        self.external_cam = self.scene.add_camera(
            res=CAM_RES, pos=(0.05, 0.57, 0.66), lookat=(0.45, 0.0, 0.1), fov=55,
        )
        self.wrist_cam = self.scene.add_camera(res=CAM_RES, fov=70, near=0.01, far=10.0)

    # --- control helper ----------------------------------------------------
    def _apply_control(self):
        self.robot.control_dofs_position(self.arm_target, self.arm_dofs)
        self.robot.control_dofs_position([self.grip_target], [self.finger_dof])
        self.robot.control_dofs_position(np.zeros(len(self.linkage_dofs)), self.linkage_dofs)

    # --- gym-style API -----------------------------------------------------
    def reset(self):
        self.robot.set_dofs_position(
            np.concatenate([HOME_POSE, np.zeros(len(self.grip_dofs))]),
            self.arm_dofs + self.grip_dofs,
        )
        self.arm_target = HOME_POSE.copy()
        self.grip_target = GRIPPER_OPEN
        for _ in range(self.settle_steps):
            self._apply_control()
            self.scene.step()
        self.wrist_cam.move_to_attach()
        self._step_count = 0
        return self.observation()

    def step(self, action):
        """action: (8,) = 7 joint-position targets (rad) + 1 binary gripper."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self.arm_target = action[:7]
        self.grip_target = GRIPPER_CLOSE if action[7] > 0.5 else GRIPPER_OPEN
        for _ in range(DECIMATION):
            self._apply_control()
            self.scene.step()
        self.wrist_cam.move_to_attach()
        self._step_count += 1
        done = self._step_count >= MAX_EPISODE_STEPS
        return self.observation(), done

    # --- observation -------------------------------------------------------
    def _arm_joint_pos(self) -> np.ndarray:
        return np.asarray(self.robot.get_dofs_position(self.arm_dofs)).reshape(-1)

    def _gripper_pos(self) -> float:
        """0 = open, 1 = closed (sim-evals gripper_pos = finger_joint / (pi/4))."""
        f = float(np.asarray(self.robot.get_dofs_position([self.finger_dof])).reshape(-1)[0])
        return float(np.clip(f / GRIPPER_CLOSE, 0.0, 1.0))

    def observation(self) -> dict:
        ext = np.asarray(self.external_cam.render()[0])[..., :3].astype(np.uint8)
        wrist = np.asarray(self.wrist_cam.render()[0])[..., :3].astype(np.uint8)
        return {
            "arm_joint_pos": self._arm_joint_pos().astype(np.float32),
            "gripper_pos": np.array([self._gripper_pos()], dtype=np.float32),
            "external_rgb": ext,
            "wrist_rgb": wrist,
        }
