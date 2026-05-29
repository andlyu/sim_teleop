"""GPU-side FastAPI server hosting allenai/MolmoAct2-SO100_101.

Runs on the rented Vast box (NOT the Mac). Loads the 5B checkpoint once into
GPU memory and serves action predictions over HTTP. The Mac-side
`MolmoActClient` is the only intended caller.

The model call follows the official model card exactly:

    out = model.predict_action(
        processor=processor,
        images=[top_rgb, side_rgb],
        task=task,
        state=robot_state,                 # 6-D float32, raw robot scale
        norm_tag="so100_so101_molmoact2",
        inference_action_mode="continuous",
        enable_depth_reasoning=False,      # required False for this checkpoint
        num_steps=10,
    )
    actions = out.actions                  # (N, D) float32, robot scale

Launch (on the GPU box):
    python host_server_so101.py --host 0.0.0.0 --port 8202 --dtype bfloat16

VRAM: ~26GB float32, <16GB bfloat16 (per the model card). bfloat16 recommended
for a cheap single-GPU Vast box (RTX 3090/4090).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from PIL import Image

# codec.py is copied next to this file in the server bundle (see provision_vast.sh)
from codec import decode_array, encode_array

REPO_ID = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"


def load_model(dtype_str: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_str]
    processor = AutoProcessor.from_pretrained(REPO_ID, trust_remote_code=True)
    model = (
        AutoModelForImageTextToText.from_pretrained(
            REPO_ID, trust_remote_code=True, dtype=dtype
        )
        .to("cuda")
        .eval()
    )
    return processor, model, dtype


def build_app(processor, model, dtype, num_steps: int):
    import torch
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="MolmoAct2-SO101 server")

    class PredictRequest(BaseModel):
        images: list[dict]
        state: dict
        instruction: str

    @app.get("/health")
    def health():
        return {"status": "ok", "repo": REPO_ID, "dtype": str(dtype), "num_steps": num_steps}

    @app.post("/reset")
    def reset():
        # MolmoAct2 predict_action is stateless per call (no server-side queue to
        # clear), so reset is a no-op. Endpoint exists for client symmetry.
        return {"status": "ok"}

    @app.post("/predict_action")
    def predict_action(req: PredictRequest):
        images = [Image.fromarray(decode_array(im)).convert("RGB") for im in req.images]
        state = decode_array(req.state).astype(np.float32).reshape(-1)

        t0 = time.perf_counter()
        if dtype == torch.bfloat16:
            ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            ctx = torch.inference_mode()
        with torch.inference_mode(), ctx:
            out = model.predict_action(
                processor=processor,
                images=images,
                task=req.instruction,
                state=state,
                norm_tag=NORM_TAG,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=num_steps,
                normalize_language=True,
            )
        actions = np.asarray(out.actions, dtype=np.float32)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return {"actions": encode_array(actions), "dt_ms": dt_ms}

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8202)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--num-steps", type=int, default=10)
    args = p.parse_args()

    import uvicorn

    print(f"Loading {REPO_ID} ({args.dtype}) — this takes a minute...")
    processor, model, dtype = load_model(args.dtype)
    app = build_app(processor, model, dtype, args.num_steps)
    print(f"Ready. Serving on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
