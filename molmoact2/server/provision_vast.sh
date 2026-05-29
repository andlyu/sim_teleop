#!/usr/bin/env bash
# Provision a Vast.ai GPU box to serve MolmoAct2-SO100_101.
#
# Run this ON THE VAST INSTANCE (after SSHing in), not on the Mac. It assumes a
# PyTorch + CUDA 12.1 base image (Vast's "pytorch/pytorch" templates work).
#
# Recommended instance: single RTX 3090/4090 (24GB), >=60GB disk. bfloat16 fits
# in <16GB, so 24GB gives comfortable headroom.
#
# Usage:
#   bash provision_vast.sh                 # install deps + download checkpoint
#   bash provision_vast.sh --serve         # ...then launch the server
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ID="allenai/MolmoAct2-SO100_101"
PORT="${PORT:-8202}"
DTYPE="${DTYPE:-bfloat16}"

echo "==> Installing server dependencies"
pip install --no-cache-dir -r "$HERE/requirements.txt"

# torch: only install if the base image didn't ship a CUDA build already.
python - <<'PY' || pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
import torch, sys
assert torch.cuda.is_available(), "no CUDA torch"
print("torch CUDA OK:", torch.__version__)
PY

echo "==> Pre-downloading checkpoint $REPO_ID (cached for reuse on a persistent volume)"
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$REPO_ID")
print("checkpoint cached")
PY

# The codec must match the Mac client byte-for-byte; copy it next to the server.
cp "$HERE/../client/codec.py" "$HERE/codec.py"
echo "==> codec.py synced into server dir"

if [[ "${1:-}" == "--serve" ]]; then
    echo "==> Launching server on 0.0.0.0:$PORT ($DTYPE)"
    exec python "$HERE/host_server_so101.py" --host 0.0.0.0 --port "$PORT" --dtype "$DTYPE"
fi

echo "==> Done. Start the server with:"
echo "    python $HERE/host_server_so101.py --host 0.0.0.0 --port $PORT --dtype $DTYPE"
echo "Then from the Mac, point MolmoActClient at  http://<VAST_PUBLIC_IP>:$PORT"
