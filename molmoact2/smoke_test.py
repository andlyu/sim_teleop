"""Mac-side smoke test for the MolmoAct2 integration — NO GPU, NO network.

Verifies the parts we can verify for free:
  1. the adapter round-trips (rad -> model scale -> rad) within tolerance,
  2. the action chunk shapes are correct,
  3. the StubPolicy produces well-formed chunks for the loop,
  4. (optional) a live server responds, if --url is given.

Run:
    .venv/bin/python molmoact2/smoke_test.py
    .venv/bin/python molmoact2/smoke_test.py --url http://<VAST_IP>:8202
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from molmoact2.adapter import (
    LEROBOT_V21_COMPAT,
    NUM_JOINTS,
    SO101_DEG,
    action_model_to_sim,
    state_sim_to_model,
)
from molmoact2.client import Observation, StubPolicy


def test_adapter_roundtrip():
    rng = np.random.default_rng(0)
    rad = rng.uniform(-2.0, 2.0, size=NUM_JOINTS).astype(np.float32)
    for conv in (SO101_DEG, LEROBOT_V21_COMPAT):
        raw = state_sim_to_model(rad, conv)
        back = action_model_to_sim(raw, conv)
        assert np.allclose(rad, back, atol=1e-4), (conv, rad, back)
    # LeRobot compat sanity: shoulder_lift sign flips around a +90deg offset.
    raw = state_sim_to_model(np.zeros(NUM_JOINTS, np.float32), LEROBOT_V21_COMPAT)
    assert np.allclose(raw[:3], [0.0, 90.0, 90.0]), raw
    print(f"  adapter round-trip OK (max err {np.abs(rad - back).max():.2e})")


def test_stub_shapes():
    obs = Observation(
        images=[np.zeros((224, 224, 3), np.uint8), np.zeros((224, 224, 3), np.uint8)],
        state_rad=np.zeros(NUM_JOINTS, np.float32),
        instruction="pick up the block",
    )
    for mode in ("hold", "wiggle", "random"):
        pol = StubPolicy(mode=mode, chunk_len=8)
        pol.reset()
        chunk = pol.act(obs)
        assert chunk.shape == (8, NUM_JOINTS), (mode, chunk.shape)
        assert chunk.dtype == np.float32
    print("  stub policy chunk shapes OK (hold/wiggle/random)")


def test_live_server(url: str):
    from molmoact2.client import MolmoActClient

    client = MolmoActClient(url)
    print("  health:", client.health())
    obs = Observation(
        images=[np.zeros((224, 224, 3), np.uint8), np.zeros((224, 224, 3), np.uint8)],
        state_rad=np.zeros(NUM_JOINTS, np.float32),
        instruction="pick up the block",
    )
    chunk = client.act(obs)
    assert chunk.ndim == 2 and chunk.shape[1] == NUM_JOINTS, chunk.shape
    print(f"  live server returned chunk {chunk.shape} OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=None, help="live MolmoAct2 server to ping")
    args = p.parse_args()

    print("adapter:")
    test_adapter_roundtrip()
    print("stub:")
    test_stub_shapes()
    if args.url:
        print("live server:")
        test_live_server(args.url)
    else:
        print("live server: skipped (no --url)")
    print("\nALL OK")


if __name__ == "__main__":
    main()
