"""Run a policy in the sim: SimEnv observation -> policy.act() -> step.

The rollout loop that connects everything. Policy-agnostic: the same loop runs
the StubPolicy (free, Mac-side) or the real MolmoActClient (remote GPU) behind
the identical `observation -> action` seam. Build/debug with the stub here, then
swap to --url to point at a MolmoAct2 server with zero loop changes.

Observation handed to the policy each step (MolmoAct2-SO100_101 contract):
    images       = camera RGB list (default [top_rgb, side_rgb])
    state_rad    = 6 joint angles, radians (adapter converts to model degrees)
    instruction  = the task string

Run (Mac, free, stub policy):
    .venv/bin/python scripts/run_policy.py --steps 20 --stub wiggle
Run (against a live MolmoAct2 server on the GPU):
    .venv/bin/python scripts/run_policy.py --url http://<vast-ip>:8202 \
        --instruction "pick up the red cube"
Run with the Genesis viewer open:
    .venv/bin/python scripts/run_policy.py --viewer --url http://<vast-ip>:8202
"""

import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import sim_env as E
from molmoact2.client import Observation, StubPolicy
from molmoact2.client.policy import Policy  # noqa: F401  (type ref)


def build_policy(args) -> "Policy":
    if args.url:
        from molmoact2.client import MolmoActClient
        print(f"Policy: MolmoActClient -> {args.url}")
        return MolmoActClient(args.url)
    print(f"Policy: StubPolicy(mode={args.stub})")
    return StubPolicy(mode=args.stub)


def parse_camera_names(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    valid = {"top", "side", "wrist"}
    bad = [name for name in names if name not in valid]
    if bad:
        raise ValueError(f"unknown policy camera(s) {bad}; valid: {sorted(valid)}")
    if len(names) != 2:
        raise ValueError("MolmoAct2 expects exactly two images; pass two cameras, e.g. side,wrist")
    return names


def save_policy_images(images, names: list[str], camera_dir: Path, chunk_idx: int) -> None:
    camera_dir.mkdir(parents=True, exist_ok=True)
    for name, image in zip(names, images):
        im = Image.fromarray(image)
        im.save(camera_dir / f"{chunk_idx:03d}_{name}.png")
        im.save(camera_dir / f"latest_{name}.png")


def limit_step(target_rad, current_rad, max_step_deg: float):
    """Scale a joint delta so no joint moves more than max_step_deg."""
    target = np.asarray(target_rad, dtype=np.float32)
    current = np.asarray(current_rad, dtype=np.float32)
    max_step = float(np.deg2rad(max_step_deg))
    delta = target - current
    biggest = float(np.max(np.abs(delta)))
    if biggest <= max_step or biggest == 0.0:
        return target
    return (current + delta * (max_step / biggest)).astype(np.float32)


def rollout(env, policy, instruction, steps, camera_names, camera_dir: Path | None = None,
            max_step_deg: float = 15.0):
    """Run `steps` policy actions. Returns list of per-chunk infos."""
    policy.reset()
    obs = env.reset()
    history = []
    executed = 0
    while executed < steps:
        images = env.observation_images(camera_names)
        if camera_dir is not None:
            save_policy_images(images, camera_names, camera_dir, len(history))
        o = Observation(images=images, state_rad=obs["state"], instruction=instruction)
        chunk = policy.act(o)                       # (N, 6) radians (adapter applied)
        history.append({"chunk_shape": tuple(np.shape(chunk)),
                        "state": np.round(obs["state"], 3).tolist()})
        for a in chunk:
            a = limit_step(a, obs["state"], max_step_deg)
            obs = env.step(a, render=False)  # rollout only reads obs["state"]
            executed += 1
            if executed >= steps:
                break
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="MolmoAct2 server URL (omit to use the stub)")
    ap.add_argument("--stub", default="wiggle", choices=["hold", "wiggle", "random"],
                    help="stub policy mode when --url is not given")
    ap.add_argument("--instruction", default="pick up the red cube")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--viewer", action="store_true", default=True,
                    help="open the Genesis 3D viewer (default on); use --no-viewer to disable")
    ap.add_argument("--no-viewer", dest="viewer", action="store_false",
                    help="run headless (no 3D window)")
    ap.add_argument("--camera-dir", type=Path, default=None,
                    help="write the exact policy input images for each policy call")
    ap.add_argument("--policy-cameras", default="top,side",
                    help="comma-separated pair sent to MolmoAct2; valid cameras: top,side,wrist")
    ap.add_argument("--max-step-deg", type=float, default=15.0,
                    help="cap each executed joint delta, matching the real SO101 runner default")
    args = ap.parse_args()
    camera_names = parse_camera_names(args.policy_cameras)

    print(f"Building SimEnv ({'viewer' if args.viewer else 'headless'})...")
    env = E.SimEnv(viewer=args.viewer)
    policy = build_policy(args)
    print(f"Instruction: {args.instruction!r}")
    print(f"Policy cameras: {camera_names}")
    print(f"Rolling out {args.steps} steps "
          f"({'watch the 3D window' if args.viewer else 'headless'})...\n")

    try:
        hist = rollout(
            env, policy, args.instruction, args.steps, camera_names,
            camera_dir=args.camera_dir, max_step_deg=args.max_step_deg,
        )
    except Exception as e:
        # macOS: the Genesis window can close mid-rollout. Don't crash the run.
        if "Viewer closed" in str(e):
            print("\nViewer window was closed — stopping rollout early.")
            return
        raise
    for i, h in enumerate(hist):
        print(f"  chunk {i}: shape {h['chunk_shape']}  state {h['state']}")
    print(f"\nOK — {args.steps} steps executed, final state "
          f"{np.round(env.state(), 3).tolist()}")
    if args.camera_dir is not None:
        print(f"Camera inputs written to {args.camera_dir}")
    if args.viewer:
        env.run_viewer()


if __name__ == "__main__":
    main()
