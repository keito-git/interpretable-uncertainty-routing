#!/bin/bash
set -e
cd "$(cd "$(dirname "$0")/.." && pwd)"
CK5="data/ckpt_dino_s42/vit_ft_best.pt,data/ckpt_dino_s43/vit_ft_best.pt,data/ckpt_dino_s44/vit_ft_best.pt,data/ckpt_dino_s45/vit_ft_best.pt,data/ckpt_dino_s46/vit_ft_best.pt"
CK1="data/ckpt_dino_s42/vit_ft_best.pt"
BB="vit_base_patch14_dinov2.lvd142m"
conds="none:0 gnoise:2 gnoise:4 blur:2 blur:4 occlude:2 occlude:4 bright:2 bright:4 pixelate:2 pixelate:4"
echo "=== ENSEMBLE (K=5) inference ==="
for cs in $conds; do C=${cs%%:*}; S=${cs##*:}; tag=${C}_s${S};
  python3 -m python.eval.ensemble_infer --ckpts "$CK5" --backbone $BB --corruption $C --severity $S --out data/shift_ens/$tag 2>&1 | tail -1
done
echo "=== SINGLE (K=1) inference ==="
for cs in $conds; do C=${cs%%:*}; S=${cs##*:}; tag=${C}_s${S};
  python3 -m python.eval.ensemble_infer --ckpts "$CK1" --backbone $BB --corruption $C --severity $S --out data/shift_single/$tag 2>&1 | tail -1
done
echo "=== shift_eval ==="
SH=$(for cs in $conds; do C=${cs%%:*}; S=${cs##*:}; if [ "$C" != "none" ]; then echo -n "data/shift_ens/${C}_s${S},"; fi; done)
SS=$(for cs in $conds; do C=${cs%%:*}; S=${cs##*:}; if [ "$C" != "none" ]; then echo -n "data/shift_single/${C}_s${S},"; fi; done)
python3 -m python.eval.shift_eval --clean data/shift_ens/none_s0 --shifted "${SH%,}" --single-clean data/shift_single/none_s0 --single-shifted "${SS%,}" --out data/analysis_shift 2>&1 | tail -40
echo SHIFT_ALLDONE
