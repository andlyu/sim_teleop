"""Open the live MuJoCo viewer on the Franka Kitchen scene.

On macOS the interactive viewer must be launched with `mjpython` (not `python`),
and from your own terminal so the window attaches to your display:

    .venv/bin/mjpython scripts/franka_viewer.py

Drag to orbit, scroll to zoom. Close the window to exit. The arm holds still
(zero control) so you can just look around the scene.
"""
import gymnasium as gym
import gymnasium_robotics
import mujoco
import mujoco.viewer

gym.register_envs(gymnasium_robotics)

env = gym.make("FrankaKitchen-v1", tasks_to_complete=["microwave"])
env.reset(seed=0)

# Reach into the underlying MuJoCo model/data to drive the native viewer.
unwrapped = env.unwrapped
model = unwrapped.model if hasattr(unwrapped, "model") else unwrapped.robot_env.model
data = unwrapped.data if hasattr(unwrapped, "data") else unwrapped.robot_env.data

print("Opening MuJoCo viewer — drag to orbit, scroll to zoom, close window to exit.")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
