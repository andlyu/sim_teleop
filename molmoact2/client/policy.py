"""Policy interface + a stub, so the sim loop is policy-agnostic.

The whole harness talks to one seam:

    action_chunk = policy.act(Observation(...))

Teleop, the stub, and the remote MolmoAct2 client all implement `Policy`, so
swapping "human" / "fake" / "real net" is a one-line change. This file has the
interface and the StubPolicy (free, Mac-side). The remote client lives in
`molmoact_client.py` (imported lazily so this module needs no GPU deps).

Convention: `act()` returns an action **chunk** of shape (N, NUM_JOINTS) in
Genesis radians (already adapted from model scale). The caller executes the
chunk one step at a time, then asks for the next observation/chunk. A single
action is just N=1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..adapter import NUM_JOINTS


@dataclass
class Observation:
    """What every policy consumes — mirrors MolmoAct2's required inputs."""

    images: list[np.ndarray]  # RGB uint8 (H, W, 3); for SO101: [top, side]
    state_rad: np.ndarray     # (NUM_JOINTS,) current joint angles, radians
    instruction: str          # natural-language task


class Policy(Protocol):
    def act(self, obs: Observation) -> np.ndarray:
        """Return an action chunk (N, NUM_JOINTS) in radians."""
        ...

    def reset(self) -> None:
        """Clear any per-episode internal state (action queue, etc.)."""
        ...


class StubPolicy:
    """A fake policy for wiring up the loop with zero GPU cost.

    Modes:
      - "hold":   return the current pose (arm stays put) — safest smoke test.
      - "wiggle": small sinusoidal offsets so you can SEE motion in the viewer.
      - "random": uniform noise within a small radius (stress the plumbing).

    Returns chunks shaped (chunk_len, NUM_JOINTS) so it exercises the same
    chunk-execution path the real model uses.
    """

    def __init__(self, mode: str = "hold", chunk_len: int = 8, amp_rad: float = 0.1, seed: int = 0):
        if mode not in ("hold", "wiggle", "random"):
            raise ValueError(f"unknown stub mode {mode!r}")
        self.mode = mode
        self.chunk_len = chunk_len
        self.amp_rad = amp_rad
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def reset(self) -> None:
        self._t = 0

    def act(self, obs: Observation) -> np.ndarray:
        base = np.asarray(obs.state_rad, dtype=np.float32).reshape(NUM_JOINTS)
        out = np.tile(base, (self.chunk_len, 1))
        if self.mode == "wiggle":
            for k in range(self.chunk_len):
                phase = (self._t + k) * 0.1
                out[k] += self.amp_rad * np.sin(phase + np.arange(NUM_JOINTS))
        elif self.mode == "random":
            out += self._rng.uniform(-self.amp_rad, self.amp_rad, size=out.shape).astype(np.float32)
        self._t += self.chunk_len
        return out.astype(np.float32)
