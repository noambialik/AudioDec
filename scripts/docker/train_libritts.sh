#!/usr/bin/env bash
set -euo pipefail

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"; assert torch.cuda.device_count() == 1, "expected exactly one mapped GPU"; print(f"Training on {torch.cuda.get_device_name(0)} as cuda:0")'

exec python /workspace/AudioDec/codecTrain.py \
    --config /workspace/AudioDec/config/autoencoder/symAD_libritts_24000_hop300.yaml \
    --tag autoencoder/symAD_libritts_24000_hop300 \
    --exp_root /workspace/data/audiodec/exp \
    "$@"
