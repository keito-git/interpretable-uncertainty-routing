"""
scn_train.py - Faithful Self-Cure Network (SCN, Wang et al. CVPR 2020) on DINOv2.

Implements SCN's core uncertainty mechanism for FER as a SOTA FER-uncertainty
baseline (R1-C3 / reviewer request to compare against SCN/EAC):
- self-attention importance weight alpha_i = sigmoid(FC_att(feature))
- logit-weighted cross-entropy (Rank-Regularized importance)
- Rank Regularization: with the top-beta fraction as the high-importance group,
  L_RR = max(0, margin - (mean_high_alpha - mean_low_alpha))
- Relabeling: after warmup, low-importance samples whose max-class prob exceeds
  the given-label prob by a margin are relabeled to the argmax.

SCN's per-sample uncertainty is u = 1 - alpha (low importance = uncertain).
We evaluate u against human annotator disagreement (Spearman, ADD-AUROC) and as
an OOD score under corruption, and contrast with our ensemble's aleatoric and
epistemic. SCN, being a single model with a single uncertainty, cannot
decompose, detect shift, or route.

CLI
---
    python -m python.train.scn_train --backbone vit_base_patch14_dinov2.lvd142m \
        --epochs 10 --out data/ckpt_scn_s42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from python.train.finetune_backbone import (
    load_split_tensors, gpu_resize_flip, normalize, _IMAGENET_MEAN, _IMAGENET_STD)
from python.eval.metrics import _auroc, spearman_rho


class SCNHead(nn.Module):
    def __init__(self, dim, ncls=8):
        super().__init__()
        self.fc = nn.Linear(dim, ncls)
        self.att = nn.Linear(dim, 1)

    def forward(self, feat):
        alpha = torch.sigmoid(self.att(feat)).squeeze(-1)  # (B,)
        logits = self.fc(feat)
        return logits, alpha


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--pixels", type=Path, default=Path("data/raw/fer2013_pixels.npy"))
    ap.add_argument("--proc", type=Path, default=Path("data/processed"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--beta", type=float, default=0.7, help="high-importance fraction")
    ap.add_argument("--margin-rr", type=float, default=0.15)
    ap.add_argument("--relabel-margin", type=float, default=0.5)
    ap.add_argument("--relabel-start", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    pixels = np.load(args.pixels)

    import timm
    try:
        bk = timm.create_model(args.backbone, pretrained=True, num_classes=0, img_size=224)
    except TypeError:
        bk = timm.create_model(args.backbone, pretrained=True, num_classes=0)
    bk = bk.to(device)
    dim = bk.num_features
    head = SCNHead(dim).to(device)
    mean = _IMAGENET_MEAN.to(device); std = _IMAGENET_STD.to(device)

    tr_raw, tr_soft, tr_hard = load_split_tensors(pixels, args.proc / "ferplus_train.npz", device)
    va_raw, _, va_hard = load_split_tensors(pixels, args.proc / "ferplus_valid.npz", device)
    labels = tr_hard.clone()
    N = len(tr_raw); bs = args.bs

    opt = torch.optim.AdamW(list(bk.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    def feat_of(raw, train):
        x = normalize(gpu_resize_flip(raw, device, train), device)
        return bk.forward_features(x)[:, 0, :]

    for ep in range(args.epochs):
        bk.train(); head.train()
        perm = torch.randperm(N, device=device)
        for s in range(0, N - bs + 1, bs):
            bidx = perm[s:s + bs]
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                f = feat_of(tr_raw[bidx], True)
                logits, alpha = head(f)
                # logit-weighted CE (SCN): weight the loss by importance
                ce = F.cross_entropy(logits, labels[bidx], reduction="none")
                loss_cls = (alpha * ce).mean()
                # rank regularization
                k = max(int(args.beta * len(alpha)), 1)
                a_sorted, _ = torch.sort(alpha, descending=True)
                hi = a_sorted[:k].mean(); lo = a_sorted[k:].mean() if k < len(a_sorted) else a_sorted[-1]
                loss_rr = F.relu(args.margin_rr - (hi - lo))
                loss = loss_cls + loss_rr
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        # relabeling after warmup
        if ep >= args.relabel_start:
            bk.eval(); head.eval()
            with torch.no_grad():
                for s in range(0, N, bs):
                    idx = torch.arange(s, min(s + bs, N), device=device)
                    f = feat_of(tr_raw[idx], False)
                    logits, alpha = head(f)
                    p = F.softmax(logits, -1)
                    maxp, arg = p.max(1)
                    givenp = p[torch.arange(len(idx)), labels[idx]]
                    # conservative SCN relabel: only confidently-wrong, low-importance samples
                    thr_lo = torch.quantile(alpha, 0.2)
                    flip = (alpha < thr_lo) & (maxp > 0.6) & ((maxp - givenp) > args.relabel_margin)
                    labels[idx[flip]] = arg[flip]
        # val
        bk.eval(); head.eval(); cor = tot = 0
        with torch.no_grad():
            for s in range(0, len(va_raw), bs):
                f = feat_of(va_raw[s:s + bs], False)
                pred = head(f)[0].argmax(1)
                cor += (pred == va_hard[s:s + bs]).sum().item(); tot += len(pred)
        print(f"epoch {ep} val_acc {cor/tot:.4f} relabeled {int((labels!=tr_hard).sum())}", flush=True)

    torch.save({"backbone": bk.state_dict(), "head": head.state_dict(), "bb": args.backbone}, args.out / "scn.pt")

    # ---- evaluate SCN uncertainty (u = 1 - alpha) ----
    from python.eval.ensemble_infer import corrupt
    proc_te = np.load(args.proc / "ferplus_test.npz", allow_pickle=True)
    te_ri = proc_te["row_index"].astype(np.int64); te_hard = proc_te["hard"].astype(np.int64)
    soft = proc_te["soft"].astype(np.float64); ve = -(np.clip(soft, 1e-12, 1) * np.log(np.clip(soft, 1e-12, 1))).sum(1)
    dis = proc_te["disagreement"].astype(np.float64)
    gen = torch.Generator(device=device).manual_seed(0)

    def scn_scores(ctype, sev):
        us = []; ents = []
        bk.eval(); head.eval()
        with torch.no_grad():
            for s in range(0, len(te_ri), bs):
                idx = te_ri[s:s + bs]
                raw = torch.from_numpy(pixels[idx].astype(np.float32) / 255.).to(device)
                x = F.interpolate(raw.unsqueeze(1), 224, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
                x = corrupt(x, ctype, sev, device, gen); x = (x - mean) / std
                logits, alpha = head(bk.forward_features(x)[:, 0, :])
                us.append((1 - alpha).cpu().numpy())
                p = F.softmax(logits, -1).cpu().numpy(); ents.append(-(np.clip(p, 1e-12, 1) * np.log(np.clip(p, 1e-12, 1))).sum(1))
        return np.concatenate(us), np.concatenate(ents)

    u_clean, ent_clean = scn_scores("none", 0)
    hi = np.quantile(ve, 2/3); lo = np.quantile(ve, 1/3); m = (ve >= hi) | (ve <= lo); lab = (ve[m] >= hi).astype(int)
    res = {"scn_importance_uncertainty": {"rho_vs_disagreement": spearman_rho(u_clean, dis),
                                          "ADD_AUROC": _auroc(u_clean[m], lab)},
           "scn_predictive_entropy": {"rho_vs_disagreement": spearman_rho(ent_clean, dis),
                                      "ADD_AUROC": _auroc(ent_clean[m], lab)},
           "ood_detection_AUROC_by_corruption_s4": {}}
    for c in ["gnoise", "occlude", "bright", "pixelate", "blur"]:
        u_c, _ = scn_scores(c, 4)
        labo = np.concatenate([np.zeros(len(u_clean)), np.ones(len(u_c))])
        res["ood_detection_AUROC_by_corruption_s4"][c] = _auroc(np.concatenate([u_clean, u_c]), labo)
    res["note"] = "SCN yields a single importance-based uncertainty: it has no aleatoric/epistemic split, no shift detector, and cannot route."
    json.dump(res, open(args.out / "scn_eval.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
