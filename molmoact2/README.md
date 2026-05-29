# MolmoAct2 policy integration

Subtask 3 (policy side) of [the plan](../PLAN.md). Runs `allenai/MolmoAct2-SO100_101`
(the checkpoint covers both SO100/SO101; our sim arm is the **SO101**) as a
remote policy behind the sim's `observation -> action` seam.

## Why a client/server split

MolmoAct2 is a 5B VLA — it does **not** run on the Mac (needs ~16GB VRAM in
bf16). So the Mac runs the sim and a thin **client**; a rented **Vast GPU box**
runs the heavy **server**. The sim can't tell whether it's talking to the real
model, a stub, or a human — they all implement the same `Policy`.

```
  MAC (free)                         VAST GPU (paid, ~$0.30/hr)
  sim ── Observation ──> MolmoActClient ──HTTP──> host_server_so101.py
      <── action chunk ──            <──────────  MolmoAct2-SO100_101 (5B, SO101)
```

## Layout

| Path | Where it runs | What it is |
|------|---------------|-----------|
| `adapter.py` | Mac + GPU | Genesis radians <-> model raw scale (`SO101_DEG`). **Calibrate before trusting.** |
| `client/policy.py` | Mac | `Policy` seam + `StubPolicy` (no GPU) |
| `client/molmoact_client.py` | Mac | HTTP client (deps: numpy, requests) |
| `client/codec.py` | both | numpy<->JSON wire format (copied to server) |
| `server/host_server_so101.py` | GPU | FastAPI host, loads the 5B checkpoint once |
| `server/requirements.txt` | GPU | server deps |
| `server/provision_vast.sh` | GPU | one-shot Vast setup |
| `smoke_test.py` | Mac | free verification (adapter + stub), optional live ping |

## Status / what's verified

- [x] Code authored; importable on the Mac with no GPU deps.
- [x] Adapter + stub verified by `smoke_test.py` (free).
- [ ] **Live model not yet run** — needs a Vast GPU (paid; your credentials).
- [ ] **Units NOT calibrated.** `adapter.SO101_DEG` assumes arm joints are in
      degrees and the gripper passes through unchanged — inferred from the model
      card sample, not verified end to end. This is the #1 silent-failure risk;
      validate against the live checkpoint before trusting any rollout.

## Run the free checks (Mac)

```bash
.venv/bin/python molmoact2/smoke_test.py
```

## Bring up the real model (Vast — paid)

1. Rent a single RTX 3090/4090 (24GB), >=60GB disk, PyTorch+CUDA 12.1 image.
2. On the box: `bash molmoact2/server/provision_vast.sh --serve` (launches `host_server_so101.py`)
3. From the Mac, point the client at it and ping:
   ```bash
   .venv/bin/python molmoact2/smoke_test.py --url http://<VAST_PUBLIC_IP>:8202
   ```
4. **Calibrate the adapter**: confirm a known sim pose maps to sensible model
   state, and that returned actions move the arm coherently (not to limits).
   A/B `adapter.SO101_DEG` vs `adapter.IDENTITY` if motion looks wrong.

> Do everything on the Mac with `StubPolicy` first. Rent the GPU only once the
> full loop works against the stub — that's the difference between a $2 and a
> $30 session.
