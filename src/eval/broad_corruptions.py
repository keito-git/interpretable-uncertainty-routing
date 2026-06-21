"""broad_corruptions.py - extend the shift suite with 6 ImageNet-C-style
corruptions (shot_noise, impulse_noise, motion_blur, contrast, jpeg, elastic),
distinct from the original 5. Runs the K=5 Deep Ensemble once per condition
(corruption applied once, all members share it) and saves per-sample
predictions.npz to data/shift_ens_ext/{corr}_s{sev}/ (member[0] doubles as the
single-model baseline, matching the original pipeline's CK1=s42).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import timm
from torchvision.io import encode_jpeg, decode_jpeg

EPS = 1e-12
DEV = torch.device("cuda:0")
CK = [f"data/ckpt_dino_s{s}/vit_ft_best.pt" for s in [42, 43, 44, 45, 46]]
BB = "vit_base_patch14_dinov2.lvd142m"
NEW_CORR = ["shot_noise", "impulse_noise", "motion_blur", "contrast", "jpeg", "elastic"]
SEV = [1, 2, 3, 4, 5]


def ent(p):
    pc = np.clip(p, EPS, 1.0)
    return -np.sum(pc * np.log(pc), axis=-1)


def build(cp):
    try:
        m = timm.create_model(BB, pretrained=False, num_classes=8, img_size=224)
    except TypeError:
        m = timm.create_model(BB, pretrained=False, num_classes=8)
    sd = torch.load(cp, map_location="cpu"); sd = sd.get("state_dict", sd)
    m.load_state_dict(sd, strict=True)
    return m.to(DEV).eval()


def _gauss_kernel(sigma, device):
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2)); k = k / k.sum()
    return k, r


def _gblur(x, sigma):
    k, r = _gauss_kernel(sigma, x.device)
    C = x.shape[1]; L = k.numel()
    wh = k.view(1, 1, 1, L).repeat(C, 1, 1, 1)
    wv = k.view(1, 1, L, 1).repeat(C, 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), wh, groups=C)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), wv, groups=C)
    return x


def corrupt(x, ctype, sev, gen):
    """x: (B,3,224,224) in [0,1]. Returns corrupted [0,1]."""
    if ctype == "shot_noise":
        c = [60.0, 25.0, 12.0, 5.0, 3.0][sev - 1]
        return (torch.poisson(x * c, generator=gen) / c).clamp(0, 1)
    if ctype == "impulse_noise":
        amt = [0.03, 0.06, 0.09, 0.17, 0.27][sev - 1]
        u = torch.rand(x.shape, generator=gen, device=x.device)
        out = x.clone()
        out[u < amt / 2] = 0.0
        out[u > 1 - amt / 2] = 1.0
        return out.clamp(0, 1)
    if ctype == "motion_blur":
        L = [3, 5, 9, 13, 17][sev - 1]
        C = x.shape[1]
        ker = torch.ones(C, 1, 1, L, device=x.device) / L
        return F.conv2d(F.pad(x, (L // 2, L // 2, 0, 0), mode="reflect"), ker, groups=C).clamp(0, 1)
    if ctype == "contrast":
        c = [0.7, 0.55, 0.4, 0.25, 0.12][sev - 1]
        m = x.mean(dim=(2, 3), keepdim=True)
        return ((x - m) * c + m).clamp(0, 1)
    if ctype == "jpeg":
        q = [80, 65, 45, 30, 18][sev - 1]
        xb = (x * 255).round().to(torch.uint8).cpu()
        out = torch.empty_like(xb)
        for i in range(xb.shape[0]):
            out[i] = decode_jpeg(encode_jpeg(xb[i], quality=q))
        return (out.float() / 255.0).to(x.device).clamp(0, 1)
    if ctype == "elastic":
        alpha = [1.0, 2.0, 4.0, 6.0, 8.0][sev - 1]; sigma = 4.0
        B, C, H, W = x.shape
        dx = _gblur(torch.randn(B, 1, H, W, generator=gen, device=x.device), sigma) * alpha
        dy = _gblur(torch.randn(B, 1, H, W, generator=gen, device=x.device), sigma) * alpha
        ys, xs = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij")
        gx = (xs.unsqueeze(0) + dx.squeeze(1)) / (W - 1) * 2 - 1
        gy = (ys.unsqueeze(0) + dy.squeeze(1)) / (H - 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=True).clamp(0, 1)
    raise ValueError(ctype)


def main():
    pixels = np.load("data/raw/fer2013_pixels.npy")
    meta = np.load("data/processed/ferplus_test.npz", allow_pickle=True)
    row_index = meta["row_index"].astype(np.int64)
    hard = meta["hard"].astype(np.int64)
    n = len(row_index)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEV)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEV)
    bs = 128

    models = [build(cp) for cp in CK]  # load all 5 once

    summary = {}
    for ctype in NEW_CORR:
        for sev in SEV:
            gen = torch.Generator(device=DEV).manual_seed(1000 + sev)
            member = np.zeros((5, n, 8), dtype=np.float64)
            with torch.no_grad():
                for s in range(0, n, bs):
                    idx = row_index[s:s + bs]
                    raw = torch.from_numpy(pixels[idx].astype(np.float32) / 255.0).to(DEV)
                    x = raw.unsqueeze(1)
                    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
                    x = corrupt(x, ctype, sev, gen)
                    xn = (x - mean) / std
                    for k, mdl in enumerate(models):
                        member[k, s:s + bs] = torch.softmax(mdl(xn), -1).cpu().numpy()
            mp = member.mean(0)
            tot = ent(mp); ale = np.stack([ent(member[kk]) for kk in range(5)]).mean(0)
            epi = np.maximum(tot - ale, 0.0)
            out = {"mean_prob": mp, "total": tot, "aleatoric": ale, "epistemic": epi,
                   "maxprob": mp.max(1), "member": member, "hard": hard}
            d = Path(f"data/shift_ens_ext/{ctype}_s{sev}"); d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / "predictions.npz", **out)
            acc = float((mp.argmax(1) == hard).mean())
            summary[f"{ctype}_s{sev}"] = {"acc": round(acc, 4), "mean_epi": round(float(epi.mean()), 4),
                                          "mean_maxprob": round(float(mp.max(1).mean()), 4)}
            print(f"{ctype}_s{sev}: acc={acc:.3f} mean_epi={epi.mean():.4f} maxp={mp.max(1).mean():.3f}", flush=True)
    json.dump(summary, open("data/analysis_shift/broad_corruptions_summary.json", "w"), indent=2)
    print("BROAD_DONE")


if __name__ == "__main__":
    raise SystemExit(main())
