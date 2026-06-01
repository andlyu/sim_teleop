"""Run an openpi DROID policy on the Genesis Franka env (sim-evals port).

This is the Genesis-side equivalent of sim-evals/run_eval.py: it drives
FrankaDroidEnv with the SAME openpi policy server (pi0-FAST/pi0.5 DROID,
joint-position), using the SAME request format the sim-evals client uses.

Prereqs:
  - An openpi policy server reachable at --host:--port (default localhost:8000),
    e.g. on the GPU box:
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py \
        policy:checkpoint --policy.config=pi0_fast_droid_jointpos \
        --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos
    (tunnel it to localhost if running remotely:  ssh -N -L 8000:127.0.0.1:8000 ...)
  - `pip install openpi-client`  (lightweight websocket client; from the openpi repo
    packages/openpi-client). Falls back with a clear message if missing.

Run:
  .venv/bin/python droid_genesis/run_droid_rollout.py --host 127.0.0.1 --port 8000 \
      --scene 1 --steps 450 --video droid_genesis/runs/episode_0.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from franka_droid_env import FrankaDroidEnv

SCENE_INSTRUCTIONS = {
    1: "put the cube in the bowl",
    2: "put the can in the mug",
    3: "put banana in the bin",
}


def resize_with_pad(img: np.ndarray, h: int = 224, w: int = 224) -> np.ndarray:
    """Resize preserving aspect ratio, zero-pad to (h, w) — matches openpi image_tools."""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    resized = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    out = np.zeros((h, w, 3), dtype=np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized[..., :3]
    return out


def build_request(obs: dict, instruction: str) -> dict:
    """DROID policy request — matches sim-evals droid_jointpos.Client.infer."""
    return {
        "observation/exterior_image_1_left": resize_with_pad(obs["external_rgb"], 224, 224),
        "observation/wrist_image_left": resize_with_pad(obs["wrist_rgb"], 224, 224),
        "observation/joint_position": obs["arm_joint_pos"],
        "observation/gripper_position": obs["gripper_pos"],
        "prompt": instruction,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--scene", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--open-loop-horizon", type=int, default=8,
                    help="execute this many actions from a chunk before re-querying")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--video", type=Path, default=Path("droid_genesis/runs/episode_0.mp4"))
    args = ap.parse_args()

    try:
        from openpi_client import websocket_client_policy
    except ImportError:
        raise SystemExit(
            "openpi-client not installed. Install it from the openpi repo:\n"
            "  pip install -e <openpi>/packages/openpi-client   (or `pip install openpi-client`)"
        )

    instruction = SCENE_INSTRUCTIONS[args.scene]
    print(f"Scene {args.scene}: {instruction!r}")
    env = FrankaDroidEnv(scene_id=args.scene, viewer=args.viewer)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f"Connected to policy server {args.host}:{args.port}")

    obs = env.reset()
    frames = []
    pred_chunk = None
    done_in_chunk = 0

    for t in range(args.steps):
        if pred_chunk is None or done_in_chunk >= args.open_loop_horizon:
            done_in_chunk = 0
            pred_chunk = client.infer(build_request(obs, instruction))["actions"]

        action = np.asarray(pred_chunk[done_in_chunk], dtype=np.float32)
        done_in_chunk += 1
        # binarize gripper (last dim), matching sim-evals
        action = np.concatenate([action[:-1], [1.0 if action[-1] > 0.5 else 0.0]])

        # viz: the two views the policy sees, side by side
        viz = np.concatenate(
            [resize_with_pad(obs["external_rgb"]), resize_with_pad(obs["wrist_rgb"])], axis=1)
        frames.append(viz)

        obs, done = env.step(action)
        if done:
            break

    args.video.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio
        imageio.mimsave(args.video, frames, fps=15)
        print(f"Wrote {args.video} ({len(frames)} frames)")
    except Exception as e:
        print(f"(video save skipped: {e})")
    print(f"Rollout done — {len(frames)} steps, final joints "
          f"{np.round(obs['arm_joint_pos'], 3).tolist()}")


if __name__ == "__main__":
    main()
