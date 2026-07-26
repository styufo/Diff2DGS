#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda create -y -n diff2dgs -c conda-forge python=3.9 pip ninja ffmpeg
conda run -n diff2dgs python -m pip install --upgrade pip "setuptools<70" wheel
conda run -n diff2dgs python -m pip install \
  torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
conda run -n diff2dgs python -m pip install -r "${repo_root}/requirements.txt"
conda run -n diff2dgs python -m pip install -e "${repo_root}"
conda run -n diff2dgs python -m pip install --no-build-isolation \
  "${repo_root}/reconstruction/submodules/simple-knn"
conda run -n diff2dgs python -m pip install --no-build-isolation \
  "${repo_root}/reconstruction/submodules/diff-surfel-rasterization"

echo "Environment ready. Run: conda activate diff2dgs"
