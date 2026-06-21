#!/bin/bash
cd "$(cd "$(dirname "$0")/.." && pwd)"
CK=data/ckpt_mcdo_s42/vit_ft_best.pt; BB=vit_base_patch14_dinov2.lvd142m
for cs in none:0 gnoise:4 occlude:4 bright:4 pixelate:4 blur:4; do C=${cs%%:*}; S=${cs##*:};
  python3 -m python.eval.ensemble_infer --ckpts "$CK" --backbone $BB --mc-dropout-samples 20 --corruption $C --severity $S --out data/mcdo_shift/${C}_s${S} >/dev/null 2>&1
done
python3 - <<PY
import numpy as np
from python.eval.metrics import _auroc, spearman_rho, expected_calibration_error
EPS=1e-12
def ent(p): pc=np.clip(p,EPS,1); return -np.sum(pc*np.log(pc),-1)
meta=np.load("data/processed/ferplus_test.npz",allow_pickle=True); ve=ent(meta["soft"].astype(float)); dis=meta["disagreement"].astype(float)
mc=np.load("data/mcdo_shift/none_s0/predictions.npz"); nc=len(mc["epistemic"]); ve=ve[:nc]; dis=dis[:nc]
en=np.load("data/shift_ens/none_s0/predictions.npz")
print("=== MC-Dropout(full ft, M=20) vs Deep Ensemble(K5) [clean] ===")
print("acc: mcdo=%.3f ens=%.3f | ECE: mcdo=%.3f ens=%.3f"%((mc["mean_prob"].argmax(1)==mc["hard"]).mean(),(en["mean_prob"].argmax(1)==en["hard"]).mean(),expected_calibration_error(mc["mean_prob"],mc["hard"]),expected_calibration_error(en["mean_prob"],en["hard"])))
print("rho(ale,dis): mcdo=%.3f ens=%.3f | mean_epi: mcdo=%.3f ens=%.3f"%(spearman_rho(mc["aleatoric"],dis),spearman_rho(en["aleatoric"],dis),mc["epistemic"].mean(),en["epistemic"].mean()))
def ood(tag,P):
    c=np.load("data/%s/%s/predictions.npz"%(P,tag)); base=np.load("data/%s/none_s0/predictions.npz"%P)
    lab=np.concatenate([np.zeros(len(base["epistemic"])),np.ones(len(c["epistemic"]))])
    return _auroc(np.concatenate([base["epistemic"],c["epistemic"]]),lab)
print("OOD-detection AUROC(epistemic) by corruption_s4: mcdo vs ensemble")
for c in ["gnoise","occlude","bright","pixelate","blur"]:
    print("  %-9s mcdo=%.3f  ens=%.3f"%(c,ood(c+"_s4","mcdo_shift"),ood(c+"_s4","shift_ens")))
PY
echo MCDO_EVAL_DONE
