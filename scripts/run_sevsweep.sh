#!/bin/bash
cd "$(cd "$(dirname "$0")/.." && pwd)"
CK5="data/ckpt_dino_s42/vit_ft_best.pt,data/ckpt_dino_s43/vit_ft_best.pt,data/ckpt_dino_s44/vit_ft_best.pt,data/ckpt_dino_s45/vit_ft_best.pt,data/ckpt_dino_s46/vit_ft_best.pt"
CK1="data/ckpt_dino_s42/vit_ft_best.pt"
BB="vit_base_patch14_dinov2.lvd142m"
for C in gnoise blur occlude bright pixelate; do
 for S in 1 3 5; do
  python3 -m python.eval.ensemble_infer --ckpts "$CK5" --backbone $BB --corruption $C --severity $S --out data/shift_ens/${C}_s${S} >/dev/null 2>&1
  python3 -m python.eval.ensemble_infer --ckpts "$CK1" --backbone $BB --corruption $C --severity $S --out data/shift_single/${C}_s${S} >/dev/null 2>&1
 done
 echo "$C s1,3,5 done"
done
echo SEVSWEEP_DONE
