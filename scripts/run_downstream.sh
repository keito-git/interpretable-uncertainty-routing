#!/bin/bash
set -e
cd "$(cd "$(dirname "$0")/.." && pwd)"
echo "=== [1/3] extract fine-tuned features ==="
python3 -m python.data.extract_features --checkpoint data/ckpt/vit_ft_best.pt \
  --output-dir data/features_ft --save-patch-splits test --batch-size 256 --device cuda:0 2>&1 | tail -25
echo "=== [2/3] train 4 methods on fine-tuned features ==="
for M in deterministic mcdropout ensemble proposed; do
  echo "----- $M -----"
  python3 -m python.train.train_csgmcmc --features-dir data/features_ft --method $M \
    --label soft --seed 42 --out data/outputs_ft/${M}_soft_s42 --device cuda:0 2>&1 | grep -A20 "\"metrics\""
done
echo "=== [3/3] spatial maps ==="
python3 -m python.eval.gen_spatial_maps --features-dir data/features_ft --seed 42 \
  --n-images 64 --out data/outputs_ft/spatial_maps_s42 --device cuda:0 2>&1 | tail -8
echo "=== DOWNSTREAM DONE ==="
