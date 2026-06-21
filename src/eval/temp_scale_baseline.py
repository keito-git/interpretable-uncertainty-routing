"""
temp_scale_baseline.py - Temperature-scaled single model (THE simple baseline).

The whole project's reference baseline is "single model + temperature
calibration". Temperature scaling (Guo et al. 2017) fits ONE scalar T on a
held-out CLEAN set; it makes the clean predictions well-calibrated but, being a
constant, it CANNOT adapt to distribution shift. This script demonstrates that
under corruption the temperature-scaled single model stays overconfident
(ECE rises) while the Deep Ensemble does not -> the ensemble's calibration
benefit is not reproducible by simple post-hoc calibration.

Fits T on the FERPlus VALID (clean) logits, then evaluates ECE/acc of the
T-scaled single model on clean test + corruptions (sev 0..5) + RAF-DB.

CLI
---
    python -m python.eval.temp_scale_baseline --ckpt data/ckpt_dino_s42/vit_ft_best.pt \
        --out data/analysis_shift/tempscale.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from python.eval.ensemble_infer import corrupt
from python.eval.metrics import expected_calibration_error

CORR = ["gnoise", "blur", "occlude", "bright", "pixelate"]
SEV = [1, 2, 3, 4, 5]


def _logits(model, pixels, row_index, device, imnet_mean, imnet_std, ctype, sev, gen, bs=128):
    out = []
    with torch.no_grad():
        for s in range(0, len(row_index), bs):
            idx = row_index[s:s + bs]
            raw = torch.from_numpy(pixels[idx].astype(np.float32) / 255.0).to(device)
            x = raw.unsqueeze(1)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            x = corrupt(x, ctype, sev, device, gen)
            x = (x - imnet_mean) / imnet_std
            out.append(model(x).cpu().numpy())
    return np.concatenate(out, 0)


def _ece_from_logits(logits, T, hard):
    p = torch.softmax(torch.from_numpy(logits) / T, -1).numpy()
    return expected_calibration_error(p, hard), float((p.argmax(1) == hard).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("data/ckpt_dino_s42/vit_ft_best.pt"))
    ap.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--pixels", type=Path, default=Path("data/raw/fer2013_pixels.npy"))
    ap.add_argument("--proc-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--rafdb-pixels", type=Path, default=Path("data/rafdb/rafdb_pixels.npy"))
    ap.add_argument("--rafdb-meta", type=Path, default=Path("data/rafdb/rafdb_meta.npz"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis_shift/tempscale.json"))
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
    imnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    gen = torch.Generator(device=device).manual_seed(0)
    pixels = np.load(args.pixels)

    # fit T on VALID (clean)
    va = np.load(args.proc_dir / "ferplus_valid.npz", allow_pickle=True)
    va_ri = va["row_index"].astype(np.int64); va_hard = va["hard"].astype(np.int64)
    val_logits = _logits(model, pixels, va_ri, device, imnet_mean, imnet_std, "none", 0, gen)
    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=80)
    vl = torch.from_numpy(val_logits); vy = torch.from_numpy(va_hard)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(vl / T, vy); loss.backward(); return loss
    opt.step(closure)
    T_opt = float(T.detach().item())

    te = np.load(args.proc_dir / "ferplus_test.npz", allow_pickle=True)
    te_ri = te["row_index"].astype(np.int64); te_hard = te["hard"].astype(np.int64)

    result = {"T": T_opt, "per_corruption": {}, "avg_curve": {}}
    # clean (sev 0)
    clean_logits = _logits(model, pixels, te_ri, device, imnet_mean, imnet_std, "none", 0, gen)
    ece0, acc0 = _ece_from_logits(clean_logits, T_opt, te_hard)
    result["clean"] = {"ece": ece0, "acc": acc0}
    sev_ece = {0: [ece0]}; sev_acc = {0: [acc0]}
    for c in CORR:
        rows = []
        for s in SEV:
            lg = _logits(model, pixels, te_ri, device, imnet_mean, imnet_std, c, s, gen)
            ece, acc = _ece_from_logits(lg, T_opt, te_hard)
            rows.append({"severity": s, "ece": ece, "acc": acc})
            sev_ece.setdefault(s, []).append(ece); sev_acc.setdefault(s, []).append(acc)
        result["per_corruption"][c] = rows
    for s in sorted(sev_ece):
        result["avg_curve"][s] = {"ece": float(np.mean(sev_ece[s])), "acc": float(np.mean(sev_acc[s]))}

    # RAF-DB
    rp = np.load(args.rafdb_pixels); rm = np.load(args.rafdb_meta, allow_pickle=True)
    rlg = _logits(model, rp, rm["row_index"].astype(np.int64), device, imnet_mean, imnet_std, "none", 0, gen)
    rece, racc = _ece_from_logits(rlg, T_opt, rm["hard"].astype(np.int64))
    result["rafdb"] = {"ece": rece, "acc": racc}

    json.dump(result, open(args.out, "w"), ensure_ascii=False, indent=2)
    print("Temperature T =", round(T_opt, 3))
    print("Temp-scaled SINGLE  ECE/acc by severity (avg over corruptions):")
    for s in sorted(result["avg_curve"]):
        a = result["avg_curve"][s]
        print(f"  s{s}: ece={a['ece']:.3f} acc={a['acc']:.3f}")
    print(f"RAF-DB: ece={rece:.3f} acc={racc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
