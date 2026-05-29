"""Mac-side client for the remote MolmoAct2-SO100_101 server (drives our SO101).

Implements the same `Policy` seam as StubPolicy, so the sim loop can't tell the
difference. All it does is:

  1. adapt sim state (rad) -> model scale,
  2. POST {images, state, instruction} to the Vast GPU server,
  3. adapt the returned action chunk (model scale) -> sim (rad).

The server (molmoact2/server/host_server_so100.py) owns the actual model. This
keeps every heavy dependency (torch+cuda, transformers, the 5B checkpoint) off
the Mac. The only deps here are numpy + requests.

Wire protocol (JSON, numpy arrays base64-encoded via the shared codec):
  request:  {"images": [HWC uint8, ...], "state": [6] float32, "instruction": str}
  response: {"actions": [N, D] float32, "dt_ms": float}
"""
from __future__ import annotations

import numpy as np

from ..adapter import SO101_DEG, JointConvention, action_model_to_sim, state_sim_to_model
from .codec import decode_array, encode_array
from .policy import Observation


class MolmoActClient:
    def __init__(
        self,
        url: str,
        conv: JointConvention = SO101_DEG,
        timeout_s: float = 30.0,
    ):
        # Lazy import so the rest of the package needs no `requests`.
        import requests  # noqa: F401

        self._requests = requests
        self.url = url.rstrip("/")
        self.conv = conv
        self.timeout_s = timeout_s

    def reset(self) -> None:
        # The model's per-episode state (action queue / KV cache) lives server
        # side; tell it to clear. Best-effort — ignored if the server is stateless.
        try:
            self._requests.post(f"{self.url}/reset", timeout=self.timeout_s)
        except Exception:
            pass

    def act(self, obs: Observation) -> np.ndarray:
        payload = {
            "images": [encode_array(np.asarray(im, dtype=np.uint8)) for im in obs.images],
            "state": encode_array(state_sim_to_model(obs.state_rad, self.conv)),
            "instruction": obs.instruction,
        }
        resp = self._requests.post(f"{self.url}/predict_action", json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        body = resp.json()
        actions_raw = decode_array(body["actions"])  # (N, D) model scale
        chunk = np.stack([action_model_to_sim(a, self.conv) for a in actions_raw])
        return chunk.astype(np.float32)

    def health(self) -> dict:
        resp = self._requests.get(f"{self.url}/health", timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()
