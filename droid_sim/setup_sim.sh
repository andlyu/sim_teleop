#!/bin/bash
# Provision a fresh GPU box for the DROID Isaac Sim eval (arhanjain/sim-evals).
# Run as root on an RTX-class box with a CUDA-12.x driver (Isaac's RTX renderer
# crashes on CUDA-13 drivers). Pairs with setup_openpi.sh for the policy server.
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends libgl1 libglib2.0-0 libxrender1 libxext6 libsm6 libice6 libegl1 libgomp1 libgl1-mesa-glx libvulkan1 vulkan-tools libglu1-mesa libnss3 libasound2 libxt6 libxmu6 libxi6 libxrandr2 libxcursor1 libxinerama1 ffmpeg git curl > /root/apt.log 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh > /root/uv.log 2>&1
export PATH="$HOME/.local/bin:$PATH"; export HF_HUB_ENABLE_HF_TRANSFER=1
git config --global url."https://github.com/".insteadOf "git@github.com:"
cd /root; [ -d sim-evals ] || git clone --recurse-submodules https://github.com/arhanjain/sim-evals.git
cd sim-evals
# flatdict (transitive Isaac Lab dep) needs pkg_resources under build isolation; pin old setuptools
grep -q "extra-build-dependencies" pyproject.toml || printf "\n[tool.uv.extra-build-dependencies]\nflatdict = [\"setuptools<80\"]\n" >> pyproject.toml
uv sync
uvx --from huggingface_hub hf download owhan/DROID-sim-environments --repo-type dataset --local-dir assets
echo NEWBOX_SETUP_DONE
