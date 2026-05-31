# Running MolmoAct2 on the SO101 sim — status

What we're doing right now: driving our Genesis SO101 simulation with the
**MolmoAct2-SO100_101** policy running on a rented GPU.

## The setup

```
   MAC (sim, free)                         VAST GPU (policy, paid)
   SimEnv (SO101 + top/side/wrist cams)    Quadro RTX 8000, 46GB
        │  obs: 2 imgs + 6-D state + task        │
        │ ────────── HTTP ──────────────────────▶│  host_server_so101.py
        │ ◀───────── action chunk ───────────────│  MolmoAct2-SO100_101 (5B)
   control_dofs_position → scene.step()          (bf16, ~16GB)
```

- **Sim side (Mac):** `scripts/sim_env.py` produces the observation MolmoAct2
  wants — top + side RGB, wrist RGB, 6-D joint state. `scripts/run_policy.py`
  is the rollout loop (`obs → policy.act() → step`), policy-agnostic.
- **Policy seam:** `molmoact2/client/` — `StubPolicy` (free Mac testing) and
  `MolmoActClient` (HTTP to the GPU) implement the same interface, so swapping
  is one flag: `run_policy.py --url http://<gpu>:8202`.
- **GPU side (Vast):** `molmoact2/server/host_server_so101.py` loads the 5B
  checkpoint once and serves `/predict_action`.

## The Vast instance

- Contract **38499662**, Quadro RTX 8000 (46GB), **~$0.29/hr**, launched ~18:33 PDT 2026-05-29.
- SSH: `ssh -p 19662 root@ssh2.vast.ai` (key generated on this Mac, registered).
- Checkpoint `allenai/MolmoAct2-SO100_101` (~21GB) downloaded to the box.
- **Manage cost:** `vastai show instances` to see spend; **`vastai destroy
  instance 38499662`** the moment we're done.

## The calibration problem (the main risk)

Genesis uses **radians**; MolmoAct2 expects **LeRobot-calibrated degrees** (5 arm
joints in degrees, gripper 0-100). Their zeros differ. We can now run either
official SO101 URDF:

- default/new calibration: zero is the middle of each joint's range
- old calibration: zero is the fully extended horizontal arm

Set `SO101_CALIBRATION=old` to use the old URDF and the matching old-calibration
state/action adapter. `molmoact2/adapter.py` encodes per-joint affine maps built
from the selected URDF limits + the model's `norm_stats.json` ranges;
`scripts/calibrate_adapter.py` regenerates them. **Unverified against the live
model** — if the arm flails, A/B old vs new calibration before trying
`SO101_DEG` or `IDENTITY`.

## Inputs to the model (per step)

- `images=[top_rgb, side_rgb]` — PIL RGB, order-agnostic for this checkpoint.
- `state` — 6-D float32, calibrated degrees (adapter converts from sim radians).
- `task` — language instruction, e.g. "pick up the red cube".
- → returns `(N, 6)` absolute joint-pose actions; adapter converts back to radians.

## Setup gotchas hit (and fixed)

- transformers version: model needs **>=4.57,<4.58** (the molmoact2 repo pin),
  NOT the 4.52 the model card loosely states. 4.52 lacks
  `transformers.video_utils.make_batched_metadata`. Installed 4.57.6.
- `timm` was missing on the box — installed.
- `protobuf` was missing on the box; without it tokenizer fallback reports a
  protobuf import error.
- The checkpoint's `tokenizer_config.json` stores `extra_special_tokens` as a
  list, but the Transformers 4.57 Qwen2 fast tokenizer path expects dict-like
  metadata. `host_server_so101.py` passes `extra_special_tokens={}` when loading
  the processor; the tokens are already present in `tokenizer.json`.
- FastAPI rejected `/predict_action` as a missing query arg because the request
  model was defined inside `build_app()` while annotations are deferred. The
  `PredictRequest` model now lives at module scope.
- MolmoAct returns CUDA tensors for actions; the server now detaches/moves them
  to CPU before NumPy encoding.

## Where we are

- [x] Mac sim + cameras + rollout loop, stub-tested (free).
- [x] Adapter calibrated against norm_stats (range-aligned hypothesis).
- [x] GPU rented, checkpoint downloaded, deps fixed.
- [x] Server loads the 5B model and serves `/health`.
- [x] First Mac -> Vast live smoke test:
      `molmoact2/smoke_test.py --url http://127.0.0.1:8202` through an SSH
      tunnel returned action chunk `(1, 6)`.
- [ ] First live rollout via `run_policy.py --url ...`.
- [ ] Live-calibrate the adapter by watching the arm.
- [ ] Destroy the instance.
