"""
ensemble_infer.py - Deep Ensemble inference with on-the-fly distribution shift.

Genuine epistemic uncertainty for FER: a Deep Ensemble of K FULLY fine-tuned
DINOv2 models (each end-to-end). Each model k yields p_k(y|x); we decompose:
    total      = H[ mean_k p_k ]
    aleatoric  = mean_k H[ p_k ]
    epistemic  = total - aleatoric   (= model disagreement / mutual information)

Unlike the frozen-head cSG-MCMC (whose epistemic was inert), a Deep Ensemble of
full models produces epistemic that RESPONDS to distribution shift -> the regime
where 1-D confidence is known to fail. We evaluate clean and shifted inputs:
corruptions (applied on GPU) and cross-dataset pixels.

Outputs predictions.npz: mean_prob, total, aleatoric, epistemic, maxprob, hard,
soft (if available), plus per-model probs for analysis.

CLI
---
    python -m python.eval.ensemble_infer \
        --ckpts data/ckpt_dino_s42/vit_ft_best.pt,data/ckpt_dino_s43/vit_ft_best.pt,... \
        --backbone vit_base_patch14_dinov2.lvd142m \
        --pixels data/raw/fer2013_pixels.npy --row-index-from data/processed/ferplus_test.npz \
        --corruption none --severity 0 --out data/shift/clean
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_kernel(sigma, device):
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2)); k = k / k.sum()
    return k.view(1, 1, -1), r


def corrupt(x, ctype, severity, device, gen):
    """x: (B,3,224,224) normalised-or-not? We corrupt the [0,1] image BEFORE
    normalisation. Caller passes un-normalised [0,1] images."""
    if ctype == "none" or severity == 0:
        return x
    if ctype == "gnoise":
        std = [0, 0.04, 0.08, 0.12, 0.18, 0.26][severity]
        return (x + torch.randn(x.shape, generator=gen, device=device) * std).clamp(0, 1)
    if ctype == "blur":
        sigma = [0, 0.5, 1.0, 1.5, 2.5, 4.0][severity]
        k1, r = _gaussian_kernel(sigma, device)  # (1,1,L)
        B, C, H, W = x.shape
        L = k1.shape[-1]
        w_h = k1.view(1, 1, 1, L).repeat(C, 1, 1, 1)  # (C,1,1,L)
        w_v = k1.view(1, 1, L, 1).repeat(C, 1, 1, 1)  # (C,1,L,1)
        xx = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), w_h, groups=C)
        xx = F.conv2d(F.pad(xx, (0, 0, r, r), mode="reflect"), w_v, groups=C)
        return xx.clamp(0, 1)
    if ctype == "occlude":
        frac = [0, 0.1, 0.2, 0.3, 0.45, 0.6][severity]
        s = int(224 * frac)
        if s > 0:
            top = (224 - s) // 2
            x = x.clone(); x[:, :, top:top + s, top:top + s] = 0.0
        return x
    if ctype == "bright":
        d = [0, 0.1, 0.2, 0.35, 0.5, 0.7][severity]
        return (x + d).clamp(0, 1)
    if ctype == "pixelate":
        f = [1, 0.75, 0.6, 0.45, 0.3, 0.2][severity]
        s = max(8, int(224 * f))
        xx = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        return F.interpolate(xx, size=(224, 224), mode="nearest")
    raise ValueError(ctype)


def _entropy(p, eps=1e-12):
    pc = np.clip(p, eps, 1.0)
    return -np.sum(pc * np.log(pc), axis=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=str, required=True, help="comma-separated checkpoint paths")
    ap.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--pixels", type=Path, default=Path("data/raw/fer2013_pixels.npy"))
    ap.add_argument("--row-index-from", type=Path, default=Path("data/processed/ferplus_test.npz"),
                    help="npz providing row_index (+ soft/hard) for the eval images")
    ap.add_argument("--corruption", type=str, default="none")
    ap.add_argument("--severity", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--mc-dropout-samples", type=int, default=0,
                    help="if >0, treat the single ckpt as an MC-Dropout model and draw this many stochastic forwards as members")
    args = ap.parse_args()

    import timm
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    pixels = np.load(args.pixels)
    meta = np.load(args.row_index_from, allow_pickle=True)
    row_index = meta["row_index"].astype(np.int64)
    soft = meta["soft"].astype(np.float32) if "soft" in meta.files else None
    hard = meta["hard"].astype(np.int64) if "hard" in meta.files else None
    n = len(row_index)

    imnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    gen = torch.Generator(device=device).manual_seed(0)

    ckpts = [c for c in args.ckpts.split(",") if c]
    mc = int(args.mc_dropout_samples)
    K = mc if mc > 0 else len(ckpts)
    member = np.zeros((K, n, 8), dtype=np.float64)

    def _build(cp):
        try:
            m = timm.create_model(args.backbone, pretrained=False, num_classes=8, img_size=224)
        except TypeError:
            m = timm.create_model(args.backbone, pretrained=False, num_classes=8)
        sd = torch.load(cp, map_location="cpu"); sd = sd.get("state_dict", sd)
        m.load_state_dict(sd, strict=True)
        return m.to(device)

    def _input(idx):
        raw = torch.from_numpy(pixels[idx].astype(np.float32) / 255.0).to(device)
        x = raw.unsqueeze(1)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        x = corrupt(x, args.corruption, args.severity, device, gen)
        return (x - imnet_mean) / imnet_std

    if mc > 0:
        # MC-Dropout: one model, dropout active at inference, mc stochastic passes.
        model = _build(ckpts[0]); model.eval()
        for mod in model.modules():
            if mod.__class__.__name__.startswith("Dropout"):
                mod.train()
        torch.manual_seed(0)
        with torch.no_grad():
            for j in range(mc):
                for s in range(0, n, args.batch_size):
                    idx = row_index[s:s + args.batch_size]
                    member[j, s:s + args.batch_size] = torch.softmax(model(_input(idx)), -1).cpu().numpy()
        del model; torch.cuda.empty_cache()
    else:
        for k, cp in enumerate(ckpts):
            model = _build(cp); model.eval()
            with torch.no_grad():
                for s in range(0, n, args.batch_size):
                    idx = row_index[s:s + args.batch_size]
                    member[k, s:s + args.batch_size] = torch.softmax(model(_input(idx)), -1).cpu().numpy()
            del model; torch.cuda.empty_cache()

    mean_p = member.mean(0)
    total = _entropy(mean_p)
    aleatoric = np.stack([_entropy(member[k]) for k in range(K)]).mean(0)
    epistemic = np.maximum(total - aleatoric, 0.0)
    out = {"mean_prob": mean_p, "total": total, "aleatoric": aleatoric,
           "epistemic": epistemic, "maxprob": mean_p.max(1), "member": member}
    if hard is not None:
        out["hard"] = hard
    if soft is not None:
        out["soft"] = soft
    np.savez_compressed(args.out / "predictions.npz", **out)

    rec = {"corruption": args.corruption, "severity": args.severity, "K": K, "n": int(n),
           "mean_total": float(total.mean()), "mean_aleatoric": float(aleatoric.mean()),
           "mean_epistemic": float(epistemic.mean()), "mean_maxprob": float(mean_p.max(1).mean())}
    if hard is not None:
        rec["accuracy"] = float((mean_p.argmax(1) == hard).mean())
    json.dump(rec, open(args.out / "summary.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(rec, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
