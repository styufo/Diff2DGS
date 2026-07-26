#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
weights="${repo_root}/weights"
mkdir -p "${weights}/propainter"

python -m pip install --quiet "huggingface_hub[cli]>=0.23,<1"

huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --local-dir "${weights}/stable-diffusion-v1-5" \
  --include model_index.json 'feature_extractor/*' 'safety_checker/*' 'scheduler/*' 'text_encoder/*' 'tokenizer/*'
huggingface-cli download stabilityai/sd-vae-ft-mse \
  --local-dir "${weights}/sd-vae-ft-mse"
huggingface-cli download wangfuyun/PCM_Weights \
  --local-dir "${weights}/PCM_Weights" --include 'sd15/*'

base_url="https://github.com/sczhou/ProPainter/releases/download/v0.1.0"
for file in ProPainter.pth raft-things.pth recurrent_flow_completion.pth; do
  curl -L --fail --continue-at - "${base_url}/${file}" -o "${weights}/propainter/${file}"
done

cat <<'EOF'
Standard weights downloaded. Download the Diff2DGS inpainting checkpoint from:
https://drive.google.com/drive/folders/1TZPRpgjMtV274dyqo3XBy_0PB93upHSy?usp=sharing
and place brushnet/ and unet_main/ under weights/diffinpaint/.
EOF
