#!/bin/bash
set -euxo pipefail

cd external/fairseq && pip install . && cd ../..

cd external/apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" . && cd ../..



