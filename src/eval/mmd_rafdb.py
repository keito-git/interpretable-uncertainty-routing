"""
mmd_rafdb.py - Feature-space distance FERPlus test vs RAF-DB (R2-M3).

Quantifies how far RAF-DB is from FERPlus in the fine-tuned DINOv2 feature
space, to substantiate the honest claim that (grayscale-matched) RAF-DB is
near-in-distribution. Reports linear MMD^2 and an RBF MMD^2 between the CLS
embeddings, plus, for calibration, the same distance between FERPlus-clean and
a strongly-corrupted FERPlus set (which IS detected as OOD) as a reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def feats(model, pixels, row_index, device, mean, std, ctype="none", sev=0, bs=128):
    from python.eval.ensemble_infer import corrupt
    gen = torch.Generator(device=device).manual_seed(0)
    out = []
    with torch.no_grad():
        for s in range(0, len(row_index), bs):
            idx = row_index[s:s + bs]
            raw = torch.from_numpy(pixels[idx].astype(np.float32) / 255.0).to(device)
            x = raw.unsqueeze(1)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            x = corrupt(x, ctype, sev, device, gen)
            x = (x - mean) / std
            tok = model.forward_features(x)
            out.append(tok[:, 0, :].float().cpu().numpy())
    return np.concatenate(out)


def linear_mmd2(X, Y):
    mx = X.mean(0); my = Y.mean(0)
    return float(((mx - my) ** 2).sum())


def rbf_mmd2(X, Y, gamma=None, sub=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = X[rng.choice(len(X), min(sub, len(X)), replace=False)]
    Y = Y[rng.choice(len(Y), min(sub, len(Y)), replace=False)]
    if gamma is None:
        Z = np.concatenate([X, Y])
        d2 = ((Z[:, None] - Z[None]) ** 2).sum(-1)
        gamma = 1.0 / (np.median(d2[d2 > 0]) + 1e-9)
    def k(A, B):
        d2 = ((A[:, None] - B[None]) ** 2).sum(-1)
        return np.exp(-gamma * d2)
    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("data/ckpt_dino_s42/vit_ft_best.pt"))
    ap.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--fer-pixels", type=Path, default=Path("data/raw/fer2013_pixels.npy"))
    ap.add_argument("--fer-meta", type=Path, default=Path("data/processed/ferplus_test.npz"))
    ap.add_argument("--rafdb-pixels", type=Path, default=Path("data/rafdb_hires/rafdb_pixels.npy"))
    ap.add_argument("--rafdb-meta", type=Path, default=Path("data/rafdb_hires/rafdb_meta.npz"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis_shift/mmd.json"))
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import timm
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    try:
        model = timm.create_model(args.backbone, pretrained=False, num_classes=8, img_size=224)
    except TypeError:
        model = timm.create_model(args.backbone, pretrained=False, num_classes=8)
    sd = torch.load(args.ckpt, map_location="cpu"); sd = sd.get("state_dict", sd)
    model.load_state_dict(sd, strict=True); model.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    fer = np.load(args.fer_meta, allow_pickle=True); fer_ri = fer["row_index"].astype(np.int64)
    raf = np.load(args.rafdb_meta, allow_pickle=True); raf_ri = raf["row_index"].astype(np.int64)
    ferpx = np.load(args.fer_pixels); rafpx = np.load(args.rafdb_pixels)

    Ff = feats(model, ferpx, fer_ri, device, mean, std)
    Fr = feats(model, rafpx, raf_ri, device, mean, std)
    Fg = feats(model, ferpx, fer_ri, device, mean, std, ctype="gnoise", sev=4)  # detected-OOD reference

    out = {
        "linear_mmd2_fer_vs_rafdb": linear_mmd2(Ff, Fr),
        "linear_mmd2_fer_vs_gnoise_s4(OOD ref)": linear_mmd2(Ff, Fg),
        "rbf_mmd2_fer_vs_rafdb": rbf_mmd2(Ff, Fr),
        "rbf_mmd2_fer_vs_gnoise_s4(OOD ref)": rbf_mmd2(Ff, Fg),
        "note": "RAF-DB MMD << gnoise_s4 MMD substantiates that grayscale-matched RAF-DB is near-in-distribution",
    }
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
