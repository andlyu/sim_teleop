#!/bin/bash
cd /root/openpi
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
exec uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid_jointpos --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos
