"""Tiny numpy<->JSON codec shared by the client and the server.

Arrays are sent as {"__ndarray__": <base64 raw bytes>, "dtype": str, "shape": [..]}.
Base64 of the raw buffer keeps images compact-ish and avoids float repr drift.
This same file is copied into the server bundle so both ends agree byte-for-byte.
"""
from __future__ import annotations

import base64

import numpy as np


def encode_array(arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    return {
        "__ndarray__": base64.b64encode(arr.tobytes()).decode("ascii"),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def decode_array(obj: dict) -> np.ndarray:
    raw = base64.b64decode(obj["__ndarray__"])
    return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
