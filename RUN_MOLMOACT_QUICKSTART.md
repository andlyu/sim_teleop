# Run MolmoAct Live Policy On Vast

Quick commands for running the Genesis SO101 sim on the Mac with the MolmoAct2
policy hosted on a Vast GPU box.

## Best box (standing default): RTX 4090 24 GB

The workload is **batch-1, real-time closed-loop inference** of a 5B VLA, so the
bottleneck is single-stream latency, not VRAM (bf16 needs ~16 GB). Pick high
clocks + native bf16, not a big datacenter card:

- **RTX 4090 (24 GB)** — best latency-per-dollar; native bf16; default choice.
- **RTX 3090 (24 GB)** — works, native bf16, cheaper, ~30–50% slower per action.
- **V100 32 GB** — avoid: no good bf16, forces `--dtype float32` (slower + more mem).
- **A100 / H100** — overkill; the SSH-tunnel network RTT + per-step image upload
  rivals/exceeds GPU time, so a faster card buys little at 3–6× the cost.

Because the model runs **remotely with a tunnel to the Mac**, network latency
matters as much as the GPU: **choose an instance geographically close to you**,
with good bandwidth and a solid reliability score, ≥60 GB disk, PyTorch+CUDA 12.1.

Replace these placeholders with the current Vast SSH target:

```bash
VAST_HOST=ssh3.vast.ai
VAST_PORT=17688
```

Current known-good setup from May 30:

- Vast target: `ssh3.vast.ai:17688`
- GPU: V100 32GB
- Server dtype: `float32`
- Checkpoint cache: successfully downloaded after installing `hf_transfer`
- Server health response: `{"status":"ok","repo":"allenai/MolmoAct2-SO100_101","dtype":"torch.float32","num_steps":10}`

## 1. Copy Server Files

Run from the Mac repo:

```bash
cd /Users/andrewlyubovsky/Projects/Blupe/sim_teleop

scp -P "$VAST_PORT" \
  molmoact2/server/host_server_so101.py \
  molmoact2/client/codec.py \
  molmoact2/server/requirements.txt \
  root@"$VAST_HOST":/root/
```

## 2. Install Vast Dependencies

Run on the Mac:

```bash
ssh -p "$VAST_PORT" root@"$VAST_HOST" '
  python -m pip install --no-cache-dir -r /root/requirements.txt timm hf_transfer
  python - <<'"'"'PY'"'"'
import torch, transformers, hf_transfer
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("hf_transfer ok")
PY
'
```

Do not set `HF_HUB_ENABLE_HF_TRANSFER=1` unless `hf_transfer` is installed. If
that package is missing, Hugging Face fails immediately and downloads nothing.
This was the failure mode on the V100 box: the cache stayed at `4.0K` and
`dl.log` showed `ModuleNotFoundError: No module named 'hf_transfer'`.

## 3. Download The Checkpoint

Run on the Mac:

```bash
ssh -p "$VAST_PORT" root@"$VAST_HOST" '
  cd /root
  export HF_HUB_ENABLE_HF_TRANSFER=1
  python -c "from huggingface_hub import snapshot_download; snapshot_download(\"allenai/MolmoAct2-SO100_101\"); print(\"checkpoint cached\")"
'
```

Expected cache size after download is many GB, not KB:

```bash
ssh -p "$VAST_PORT" root@"$VAST_HOST" 'du -sh /root/.cache/huggingface'
```

Known-good cache size for this checkpoint is about `21G`.

If the download fails, inspect:

```bash
ssh -p "$VAST_PORT" root@"$VAST_HOST" 'tail -120 /root/dl.log'
```

## 4. Start The Server

Use `float32` on V100. Use `bfloat16` on 3090/4090/A-series unless testing says
otherwise.

V100:

```bash
ssh -p "$VAST_PORT" root@"$VAST_HOST" '
  cd /root
  python host_server_so101.py --host 0.0.0.0 --port 8202 --dtype float32
'
```

Background V100 launch:

```bash
ssh -p "$VAST_PORT" root@"$VAST_HOST" '
  cd /root
  rm -f server.log
  nohup python host_server_so101.py --host 0.0.0.0 --port 8202 --dtype float32 > server.log 2>&1 < /dev/null &
'
```

3090/4090/A-series:

```bash
ssh -p "$VAST_PORT" root@"$VAST_HOST" '
  cd /root
  python host_server_so101.py --host 0.0.0.0 --port 8202 --dtype bfloat16
'
```

Health check from another terminal:

```bash
ssh -p "$VAST_PORT" root@"$VAST_HOST" 'curl http://127.0.0.1:8202/health'
```

Expected response includes `"status":"ok"`.

If health is empty, check process/log/GPU memory:

```bash
ssh -p "$VAST_PORT" root@"$VAST_HOST" '
  ps -ef | grep -E "host_server_so101|uvicorn" | grep -v grep || true
  tail -120 /root/server.log 2>/dev/null || true
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
'
```

## 5. Open A Local Tunnel

On the Mac:

```bash
cd /Users/andrewlyubovsky/Projects/Blupe/sim_teleop
ssh -N -L 8202:127.0.0.1:8202 -p "$VAST_PORT" root@"$VAST_HOST"
```

Keep this terminal open while running policy commands.

If the tunnel is stale or broken, close it and reopen:

```bash
pkill -f "8202:127.0.0.1:8202"
ssh -N -L 8202:127.0.0.1:8202 -p "$VAST_PORT" root@"$VAST_HOST"
```

## 6. Run The Policy

Headless rollout:

```bash
SO101_CALIBRATION=old .venv/bin/python scripts/run_policy.py \
  --url http://127.0.0.1:8202 \
  --steps 20 \
  --instruction "pick up the red cube" \
  --policy-cameras top,side \
  --camera-dir outputs/policy_cam_top_side_live
```

Visible Genesis viewer rollout:

```bash
SO101_CALIBRATION=old .venv/bin/python -u scripts/run_policy.py \
  --viewer \
  --url http://127.0.0.1:8202 \
  --steps 20 \
  --instruction "pick up the red cube" \
  --policy-cameras top,side \
  --camera-dir outputs/policy_cam_top_side_visible
```

If the viewer does not appear when launched by automation, run the visible command
directly in a normal macOS Terminal window.

To launch the visible rollout from automation into a normal Terminal window:

```bash
osascript -e 'tell application "Terminal" to do script "cd /Users/andrewlyubovsky/Projects/Blupe/sim_teleop && SO101_CALIBRATION=old .venv/bin/python -u scripts/run_policy.py --viewer --url http://127.0.0.1:8202 --steps 20 --instruction \"pick up the red cube\" --policy-cameras top,side --camera-dir outputs/policy_cam_top_side_visible"' \
  -e 'tell application "Terminal" to activate'
```

## Camera Inputs

Use `--policy-cameras top,side` for the MolmoAct2 SO100/SO101 checkpoint. This
matches the model-card reference images better than `side,wrist`.

Saved camera frames go to the directory passed with `--camera-dir`, for example:

```text
outputs/policy_cam_top_side_live/000_top.png
outputs/policy_cam_top_side_live/000_side.png
```

## Notes

- The cube start position is in `scripts/so101_scene.py` as `CUBE_XY`.
- Camera poses are in `scripts/cameras.py`.
- The server passes `norm_tag="so100_so101_molmoact2"` at inference time.
- The Mac client converts sim radians to the model scale through `molmoact2/adapter.py`.
- The front backdrop panel is intentionally removed; it occluded the camera.
- Use `SO101_CALIBRATION=old` to run the official old-calibration URDF and the
  matching old-calibration state/action mapping. Omit it to use the new
  calibration URDF.

## Stop Everything

Stop local tunnel:

```bash
pkill -f "8202:127.0.0.1:8202"
```

Stop the Vast server:

```bash
ssh -p "$VAST_PORT" root@"$VAST_HOST" 'pkill -f host_server_so101.py'
```

Destroy the Vast instance when done billing:

```bash
vastai destroy instance <INSTANCE_ID>
```
