#!/bin/bash
cd "$(cd "$(dirname "$0")/.." && pwd)"
CK=data/ckpt_dgt_s42/vit_ft_best.pt,data/ckpt_dgt_s43/vit_ft_best.pt,data/ckpt_dgt_s44/vit_ft_best.pt
BB=vit_base_patch14_dinov2.lvd142m
for cs in none:0 bright:2 bright:4 pixelate:2 pixelate:4 gnoise:4; do C=${cs%%:*}; S=${cs##*:};
  python3 -m python.eval.ensemble_infer --ckpts "$CK" --backbone $BB --corruption $C --severity $S --out data/dgt_shift/${C}_s${S} >/dev/null 2>&1
done
python3 - <<PY
import numpy as np
from python.eval.metrics import _auroc, spearman_rho
EPS=1e-12
def ent(p): pc=np.clip(p,EPS,1); return -np.sum(pc*np.log(pc),-1)
def decompK(member,K):
    m=member[:K]; mean=m.mean(0); tot=ent(mean); ale=np.stack([ent(m[k]) for k in range(K)]).mean(0); return mean,tot,ale,np.maximum(tot-ale,0)
meta=np.load("data/processed/ferplus_test.npz",allow_pickle=True); ve=ent(meta["soft"].astype(float)); dis=meta["disagreement"].astype(float)
# DGT clean
dg=np.load("data/dgt_shift/none_s0/predictions.npz"); nc=len(dg["epistemic"]); ve=ve[:nc]; dis=dis[:nc]
# plain K3 from saved 5-member ensemble
pe=np.load("data/shift_ens/none_s0/predictions.npz"); _,_,pale,pepi=decompK(pe["member"],3)
print("=== B: DGT(K3, align+OE) vs Plain(K3) ===")
print("clean: rho(ale,dis)  DGT=%.3f  Plain=%.3f"%(spearman_rho(dg["aleatoric"],dis), spearman_rho(pale,dis)))
hi=np.quantile(ve,2/3); lo=np.quantile(ve,1/3); m=(ve>=hi)|(ve<=lo); lab=(ve[m]>=hi).astype(int)
print("clean: ADD-AUROC      DGT=%.3f  Plain=%.3f"%(_auroc(dg["aleatoric"][m],lab), _auroc(pale[m],lab)))
print("clean acc: DGT=%.3f"%((dg["mean_prob"].argmax(1)==dg["hard"]).mean()))
# held-out OOD detection (bright,pixelate) DGT epi vs plain-K3 epi
def ood(dgtag):
    e=np.load("data/dgt_shift/%s/predictions.npz"%dgtag); lab=np.concatenate([np.zeros(nc),np.ones(len(e["epistemic"]))])
    dgt_auc=_auroc(np.concatenate([dg["epistemic"],e["epistemic"]]),lab)
    pc=np.load("data/shift_ens/%s/predictions.npz"%dgtag); _,_,_,pe2=decompK(pc["member"],3)
    plain_auc=_auroc(np.concatenate([pepi,pe2]),lab)
    return dgt_auc,plain_auc
print("HELD-OUT OOD-detection AUROC (trained OE on gnoise/blur/occlude):")
for t in ["bright_s2","bright_s4","pixelate_s2","pixelate_s4"]:
    d,p=ood(t); print("  %-12s DGT-epi=%.3f  Plain-epi=%.3f  %s"%(t,d,p,"(DGT better)" if d>p else "(plain better)"))
print("TRAINED corruption (sanity): gnoise_s4")
d,p=ood("gnoise_s4"); print("  gnoise_s4    DGT-epi=%.3f  Plain-epi=%.3f"%(d,p))
PY
echo DGTEVAL_DONE
