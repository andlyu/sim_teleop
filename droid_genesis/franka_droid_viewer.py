"""Open the Genesis FrankaDroidEnv viewer and hold the DROID home pose.

macOS: run from a terminal (Genesis viewer needs the main thread):
    .venv/bin/python droid_genesis/franka_droid_viewer.py

Holds the arm at the home pose so you can visually verify the port step.
Close the window to exit.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from franka_droid_env import FrankaDroidEnv


def main():
    env = FrankaDroidEnv(scene_id=1, viewer=True)
    env.reset()
    print("Viewer open — Franka at the DROID home pose. Close the window to exit.")
    while env.scene.viewer.is_alive():
        env._apply_control()
        env.scene.step()
        env.wrist_cam.move_to_attach()
    print("Viewer closed.")


if __name__ == "__main__":
    main()
