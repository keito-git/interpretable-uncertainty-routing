"""
eac_train.py - EAC-style training (Erasing Attention Consistency, Zhang et al.
ECCV 2022) on DINOv2, as a second SOTA FER-uncertainty/robustness baseline.

EAC's core ideas are (i) random erasing augmentation and (ii) flip attention
consistency: the spatial response of an image and its horizontal flip should be
consistent. We adapt this to a ViT by enforcing consistency between the patch-
token feature map of an image and the (un-flipped) patch map of its flipped
input. The resulting model is a single classifier whose uncertainty is the
predictive entropy. Like SCN and LDL, EAC provides a single uncertainty with no
aleatoric/epistemic split, no shift detector, and no routing.

CLI
---
    python -m python.train.eac_train --epochs 10 --out data/ckpt_eac_s42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from python.train.finetune_backbone import load_split_tensors, _IMAGENET_MEAN, _IMAGENET_STD
from python.eval.metrics import _auroc, spearman_rho


def random_erase(x, gen, p=0.5, area=(0.02, 0.2)):
    B, C, H, W = x.shape
    out = x.clone()
    for i in range(B):
        if torch.rand(1, generator=gen, device=x.device).item() > p:
            continue
        a = (area[0] + (area[1] - area[0]) * torch.rand(1, generator=gen, device=x.device).item()) * H * W
        h = int(a ** 0.5); w = h
        if h < 1 or h >= H:
            continue
        top = int(torch.randint(0, H - h, (1,), generator=gen, device=x.device).item())
        left = int(torch.randint(0, W - w, (1,), generator=gen, device=x.device).item())
        out[i, :, top:top + h, left:left + w] = 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--pixels", type=Path, default=Path("data/raw/fer2013_pixels.npy"))
    ap.add_argument("--proc", type=Path, default=Path("data/processed"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--consistency-weight", type=float, default=1.0)
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
        model = timm.create_model(args.backbone, pretrained=True, num_classes=8, img_size=224)
    except TypeError:
        model = timm.create_model(args.backbone, pretrained=True, num_classes=8)
    model = model.to(device)
    mean = _IMAGENET_MEAN.to(device); std = _IMAGENET_STD.to(device)
    gen = torch.Generator(device=device).manual_seed(args.seed + 5)

    tr_raw, _, tr_hard = load_split_tensors(pixels, args.proc / "ferplus_train.npz", device)
    va_raw, _, va_hard = load_split_tensors(pixels, args.proc / "ferplus_valid.npz", device)
    N = len(tr_raw); bs = args.bs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    def to224(raw):
        return F.interpolate(raw.unsqueeze(1), 224, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        for s in range(0, N - bs + 1, bs):
            bidx = perm[s:s + bs]
            x0 = to224(tr_raw[bidx])
            x = random_erase(x0, gen)                       # random erasing
            xn = (x - mean) / std
            xf = (torch.flip(x, dims=[3]) - mean) / std     # flipped view
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                feat = model.forward_features(xn)           # (B,1+P,D)
                logits = model.forward_head(feat)
                loss_cls = F.cross_entropy(logits, tr_hard[bidx])
                # flip attention/feature consistency on patch tokens
                pf = model.forward_features(xf)[:, 1:, :]
                p0 = feat[:, 1:, :]
                gw = int(p0.shape[1] ** 0.5)
                if gw * gw == p0.shape[1]:
                    p0g = p0.transpose(1, 2).reshape(p0.shape[0], -1, gw, gw)
                    pfg = pf.transpose(1, 2).reshape(pf.shape[0], -1, gw, gw)
                    cons = F.mse_loss(p0g, torch.flip(pfg, dims=[3]))
                else:
                    cons = F.mse_loss(p0, pf)
                loss = loss_cls + args.consistency_weight * cons
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        model.eval(); cor = tot = 0
        with torch.no_grad():
            for s in range(0, len(va_raw), bs):
                pred = model((to224(va_raw[s:s + bs]) - mean) / std).argmax(1)
                cor += (pred == va_hard[s:s + bs]).sum().item(); tot += len(pred)
        print(f"epoch {ep} val_acc {cor/tot:.4f}", flush=True)

    torch.save({"state_dict": model.state_dict(), "backbone": args.backbone}, args.out / "vit_ft_best.pt")

    # evaluate EAC predictive-entropy uncertainty
    from python.eval.ensemble_infer import corrupt
    proc_te = np.load(args.proc / "ferplus_test.npz", allow_pickle=True)
    te_ri = proc_te["row_index"].astype(np.int64)
    soft = proc_te["soft"].astype(np.float64); ve = -(np.clip(soft, 1e-12, 1) * np.log(np.clip(soft, 1e-12, 1))).sum(1)
    dis = proc_te["disagreement"].astype(np.float64); te_hard = proc_te["hard"].astype(np.int64)
    cgen = torch.Generator(device=device).manual_seed(0)

    def ent_scores(ct, sev):
        es = []; accs = []
        model.eval()
        with torch.no_grad():
            for s in range(0, len(te_ri), bs):
                idx = te_ri[s:s + bs]
                raw = torch.from_numpy(pixels[idx].astype(np.float32) / 255.).to(device)
                x = to224(raw); x = corrupt(x, ct, sev, device, cgen); x = (x - mean) / std
                p = F.softmax(model(x), -1).cpu().numpy()
                es.append(-(np.clip(p, 1e-12, 1) * np.log(np.clip(p, 1e-12, 1))).sum(1))
                accs.append(p.argmax(1))
        return np.concatenate(es), np.concatenate(accs)

    e_clean, pred_clean = ent_scores("none", 0)
    hi = np.quantile(ve, 2/3); lo = np.quantile(ve, 1/3); m = (ve >= hi) | (ve <= lo); lab = (ve[m] >= hi).astype(int)
    res = {"eac_predictive_entropy": {"rho_vs_disagreement": spearman_rho(e_clean, dis),
                                      "ADD_AUROC": _auroc(e_clean[m], lab),
                                      "test_acc": float((pred_clean == te_hard).mean())},
           "ood_detection_AUROC_by_corruption_s4": {}}
    for c in ["gnoise", "occlude", "bright", "pixelate", "blur"]:
        e_c, _ = ent_scores(c, 4)
        labo = np.concatenate([np.zeros(len(e_clean)), np.ones(len(e_c))])
        res["ood_detection_AUROC_by_corruption_s4"][c] = _auroc(np.concatenate([e_clean, e_c]), labo)
    res["note"] = "EAC is a single robust classifier; its predictive entropy is one uncertainty with no a/e split, shift detector, or routing."
    json.dump(res, open(args.out / "eac_eval.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
