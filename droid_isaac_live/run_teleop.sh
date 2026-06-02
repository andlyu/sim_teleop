#!/bin/bash
cd /root/sim-evals
export PATH="$HOME/.local/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
exec uv run python /root/teleop_cam.py
