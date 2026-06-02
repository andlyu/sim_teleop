#!/bin/bash
# Set up the openpi policy server (joint-position DROID configs on karl/droid_policies).
# Run as root on the same box after setup_sim.sh. Then launch with run_serve.sh.
set -e
export PATH="$HOME/.local/bin:$PATH"
git config --global url."https://github.com/".insteadOf "git@github.com:"
cd /root
[ -d openpi ] || git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi && git checkout karl/droid_policies
GIT_LFS_SKIP_SMUDGE=1 uv sync
echo OPENPI_SETUP_DONE
