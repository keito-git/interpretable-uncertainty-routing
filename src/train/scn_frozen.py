"""
scn_frozen.py - SCN uncertainty mechanism on frozen DINOv2 features (stable).

Faithfully reproduces SCN's (Wang et al., CVPR 2020) self-cure uncertainty
mechanism -- importance weighting, Rank Regularization, and relabeling -- but on
top of the (frozen) fine-tuned DINOv2 features, so the backbone does not
collapse. SCN's per-sample uncertainty is u = 1 - alpha (low importance =
uncertain). We evaluate u against human annotator disagreement; SCN provides a
single uncertainty with no aleatoric/epistemic split, no shift detector, and no
routing -- the conceptual point of the comparison.

CLI
---
    python -m python.train.scn_frozen --features-dir data/features_dino --out data/scn_frozen
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from python.eval.metrics import _auroc, spearman_rho


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=Path("data/features_dino"))
    ap.add_argument("--votes", type=Path, default=Path("data/processed/ferplus_test.npz"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--margin-rr", type=float, default=0.15)
    ap.add_argument("--relabel-margin", type=float, default=0.5)
    ap.add_argument("--relabel-start", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    tr = np.load(args.features_dir / "feat_train.npz", allow_pickle=True)
    te = np.load(args.features_dir / "feat_test.npz", allow_pickle=True)
    Xtr = torch.from_numpy(tr["cls"].astype(np.float32)).to(device)
    ytr = torch.from_numpy(tr["hard"].astype(np.int64)).to(device)
    Xte = torch.from_numpy(te["cls"].astype(np.float32)).to(device)
    # standardise
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    D = Xtr.shape[1]; labels = ytr.clone(); N = len(Xtr); bs = 512

    fc = nn.Linear(D, 8).to(device); att = nn.Linear(D, 1).to(device)
    opt = torch.optim.Adam(list(fc.parameters()) + list(att.parameters()), lr=1e-3, weight_decay=1e-4)

    for ep in range(args.epochs):
        perm = torch.randperm(N, device=device)
        for s in range(0, N, bs):
            bidx = perm[s:s + bs]
            f = Xtr[bidx]
            logits = fc(f); alpha = torch.sigmoid(att(f)).squeeze(-1)
            ce = F.cross_entropy(logits, labels[bidx], reduction="none")
            loss_cls = (alpha * ce).mean()
            k = max(int(args.beta * len(alpha)), 1)
            a_sorted, _ = torch.sort(alpha, descending=True)
            hi = a_sorted[:k].mean(); lo = a_sorted[k:].mean() if k < len(a_sorted) else a_sorted[-1]
            loss = loss_cls + F.relu(args.margin_rr - (hi - lo))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep >= args.relabel_start:
            with torch.no_grad():
                logits = fc(Xtr); alpha = torch.sigmoid(att(Xtr)).squeeze(-1)
                p = F.softmax(logits, -1); maxp, arg = p.max(1)
                givenp = p[torch.arange(N, device=device), labels]
                thr = torch.quantile(alpha, 0.2)
                flip = (alpha < thr) & (maxp > 0.6) & ((maxp - givenp) > args.relabel_margin)
                labels[flip] = arg[flip]

    with torch.no_grad():
        acc = (fc(Xtr).argmax(1) == ytr).float().mean().item()
        alpha_te = torch.sigmoid(att(Xte)).squeeze(-1).cpu().numpy()
        ent_te = (-(F.softmax(fc(Xte), -1) * F.log_softmax(fc(Xte), -1)).sum(-1)).cpu().numpy()
    u = 1 - alpha_te

    meta = np.load(args.votes, allow_pickle=True)
    soft = meta["soft"].astype(np.float64); dis = meta["disagreement"].astype(np.float64)
    ve = -(np.clip(soft, 1e-12, 1) * np.log(np.clip(soft, 1e-12, 1))).sum(1)
    n = min(len(u), len(dis)); u = u[:n]; ent_te = ent_te[:n]; dis = dis[:n]; ve = ve[:n]
    hi = np.quantile(ve, 2/3); lo = np.quantile(ve, 1/3); m = (ve >= hi) | (ve <= lo); lab = (ve[m] >= hi).astype(int)
    res = {"train_acc_on_frozen": acc,
           "scn_importance_uncertainty": {"rho_vs_disagreement": spearman_rho(u, dis), "ADD_AUROC": _auroc(u[m], lab)},
           "scn_predictive_entropy": {"rho_vs_disagreement": spearman_rho(ent_te, dis), "ADD_AUROC": _auroc(ent_te[m], lab)},
           "note": "SCN gives a single uncertainty (importance/entropy): no aleatoric/epistemic split, no shift detector, no routing."}
    json.dump(res, open(args.out / "scn_frozen_eval.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
