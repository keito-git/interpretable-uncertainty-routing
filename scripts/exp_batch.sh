#!/bin/bash
cd "$(cd "$(dirname "$0")/.." && pwd)"
echo "########## [A] AGGREGATE + per-sample C3 + risk-coverage ##########"
python3 -m python.eval.analyze_results --final-dir data/final --features-dir data/features \
  --methods deterministic,mcdropout,ensemble,proposed --seeds 42,43,44 --out data/analysis 2>&1 | tail -40
echo "########## [B] DATA-FRACTION (epistemic vs data size, proposed) ##########"
for FR in 0.01 0.02 0.05 0.1 0.25 0.5 1.0; do
  python3 -m python.train.train_csgmcmc --features-dir data/features --method proposed \
    --label soft --seed 42 --train-frac $FR --out data/datafrac/f${FR} --device cuda:0 >/dev/null 2>&1
done
python3 - <<PY
import json,glob,os
print("frac,acc,mean_ale,mean_epi,nll,jsd")
for p in sorted(glob.glob("data/datafrac/*/metrics.json"), key=lambda x: float(x.split("/f")[1].split("/")[0])):
    m=json.load(open(p))["metrics"]; fr=p.split("/f")[1].split("/")[0]
    print("{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(fr,m["accuracy"],m["mean_aleatoric"],m["mean_epistemic"],m["nll"],m["jsd_votes"]))
PY
echo "########## [C] M (samples) ABLATION ##########"
for SPC in 1 3 5 10; do
  python3 -m python.train.train_csgmcmc --features-dir data/features --method proposed \
    --label soft --seed 42 --samples-per-cycle $SPC --out data/Mabl/spc${SPC} --device cuda:0 >/dev/null 2>&1
done
python3 - <<PY
import json,glob,os
print("samples_per_cycle,num_members,acc,nll,jsd,mean_ale,mean_epi")
for p in sorted(glob.glob("data/Mabl/*/metrics.json")):
    m=json.load(open(p))["metrics"]; spc=os.path.basename(os.path.dirname(p))
    print("{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(spc,m["num_members"],m["accuracy"],m["nll"],m["jsd_votes"],m["mean_aleatoric"],m["mean_epistemic"]))
PY
echo "########## [D] OCCLUSION causal test (spatial aleatoric) ##########"
python3 -m python.eval.occlusion_test --features-dir data/features --processed-dir data/processed \
  --pixels data/raw/fer2013_pixels.npy --seed 42 --n-images 200 --top-k 3 --out data/analysis/occlusion 2>&1 | tail -20
echo "########## EXP BATCH DONE ##########"
