#!/bin/bash
set -e
cd "$(cd "$(dirname "$0")/.." && pwd)"
echo "=== extract DINOv2 features ==="
python3 -m python.data.extract_features --backbone vit_base_patch14_dinov2.lvd142m \
  --output-dir data/features_dino --save-patch-splits test --batch-size 256 --device cuda:0 2>&1 | tail -8
echo "=== linear probe (DINOv2 ceiling) ==="
python3 - <<PY
import numpy as np
from sklearn.linear_model import LogisticRegression
tr=np.load("data/features_dino/feat_train.npz",allow_pickle=True); te=np.load("data/features_dino/feat_test.npz",allow_pickle=True)
clf=LogisticRegression(max_iter=3000).fit(tr["cls"].astype(np.float32),tr["hard"])
print("DINOv2 linear-probe test acc:", round(float((clf.predict(te["cls"].astype(np.float32))==te["hard"]).mean()),4))
PY
echo "=== 4 methods (seed42) on DINOv2 ==="
for M in deterministic ensemble proposed; do
  python3 -m python.train.train_csgmcmc --features-dir data/features_dino --method $M --label soft --seed 42 --out data/dino_out/${M}_s42 --device cuda:0 >/dev/null 2>&1
done
python3 - <<PY
import json,glob,os
print("method acc ece nll jsd misAUROC c3 mean_ale mean_epi")
for p in sorted(glob.glob("data/dino_out/*/metrics.json")):
    m=json.load(open(p))["metrics"]; c=os.path.basename(os.path.dirname(p))
    print("{} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(c,m["accuracy"],m["ece"],m["nll"],m["jsd_votes"],m["mis_auroc"],m["c3_spearman_cat_ale_vs_disagreement"],m["mean_aleatoric"],m["mean_epistemic"]))
PY
echo "DINO RUN DONE"
