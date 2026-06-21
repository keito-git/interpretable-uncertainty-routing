#!/bin/bash
cd "$(cd "$(dirname "$0")/.." && pwd)"
echo "a0,wd,acc,ece,nll,jsd,mis_auroc,c3,mean_ale,mean_epi"
for A0 in 5e-4 1e-5 1e-6 1e-7; do
 for WD in 1e-4 1e-3; do
  OUT=data/sweep/a${A0}_wd${WD}
  python3 -m python.train.train_csgmcmc --features-dir data/features --method proposed \
    --label soft --seed 42 --out $OUT --device cuda:0 \
    --alpha0 $A0 --prior-weight-decay $WD >/dev/null 2>&1
  python3 - "$A0" "$WD" "$OUT/metrics.json" <<PY
import json,sys
a0,wd,p=sys.argv[1],sys.argv[2],sys.argv[3]
m=json.load(open(p))["metrics"]
print(f"{a0},{wd},{m[\"accuracy\"]:.3f},{m[\"ece\"]:.3f},{m[\"nll\"]:.3f},{m[\"jsd_votes\"]:.3f},{m[\"mis_auroc\"]:.3f},{m[\"c3_spearman_cat_ale_vs_disagreement\"]:.3f},{m[\"mean_aleatoric\"]:.3f},{m[\"mean_epistemic\"]:.3f}")
PY
 done
done
echo "SWEEP DONE"
