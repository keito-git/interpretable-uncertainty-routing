"""p5_gnoise_confound.py - reviewer W4: is gnoise epistemic confounded by
per-member noise realizations?

ensemble_infer.py shares one RNG across members, so member k and k+1 see
DIFFERENT gaussian-noise draws of the same image. That injects input-noise
variance into the member disagreement (epistemic) on gnoise. Here we recompute
gnoise_s4 epistemic with the SAME noise tensor applied to all 5 members
(reseed before each member) and compare epistemic magnitude + OOD-detection
AUROC against the shared-RNG version.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import timm

from python.eval.metrics import _auroc

EPS = 1e-12
DEV = torch.device("cuda:0")
CK = [f"data/ckpt_dino_s{s}/vit_ft_best.pt" for s in [42, 43, 44, 45, 46]]
BB = "vit_base_patch14_dinov2.lvd142m"
STD_S4 = 0.18  # gnoise severity-4 std (matches ensemble_infer corrupt table)


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


def main():
    pixels = np.load("data/raw/fer2013_pixels.npy")
    meta = np.load("data/processed/ferplus_test.npz", allow_pickle=True)
    row_index = meta["row_index"].astype(np.int64)
    ve = ent(meta["soft"].astype(np.float64))
    n = len(row_index)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEV)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEV)
    bs = 128

    # Pre-generate ONE fixed noise tensor per image (shared across members).
    g = torch.Generator(device=DEV).manual_seed(12345)

    def run(shared_noise: bool):
        member = np.zeros((5, n, 8), dtype=np.float64)
        for k, cp in enumerate(CK):
            model = build(cp)
            gen = torch.Generator(device=DEV).manual_seed(12345 if shared_noise else 1000 + k)
            with torch.no_grad():
                for s in range(0, n, bs):
                    idx = row_index[s:s + bs]
                    raw = torch.from_numpy(pixels[idx].astype(np.float32) / 255.0).to(DEV)
                    x = raw.unsqueeze(1)
                    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
                    noise = torch.randn(x.shape, generator=gen, device=DEV) * STD_S4
                    x = (x + noise).clamp(0, 1)
                    xn = (x - mean) / std
                    member[k, s:s + bs] = torch.softmax(model(xn), -1).cpu().numpy()
            del model; torch.cuda.empty_cache()
        mp = member.mean(0)
        tot = ent(mp)
        ale = np.stack([ent(member[kk]) for kk in range(5)]).mean(0)
        epi = np.maximum(tot - ale, 0.0)
        return epi

    # clean epistemic (no noise) for OOD-AUROC reference
    clean = np.load("data/shift_ens/none_s0/predictions.npz")["epistemic"]

    epi_shared = run(shared_noise=True)    # SAME noise all members (no input-noise confound)
    epi_indep = run(shared_noise=False)    # DIFFERENT noise per member (current pipeline)

    def ood_auroc(epi_ood):
        lab = np.concatenate([np.zeros(n), np.ones(n)])
        return _auroc(np.concatenate([clean, epi_ood]), lab)

    out = {
        "severity": 4, "corruption": "gnoise",
        "shared_noise_all_members (confound-free)": {
            "mean_epistemic": round(float(epi_shared.mean()), 4),
            "ood_auroc_vs_clean": round(float(ood_auroc(epi_shared)), 4),
        },
        "independent_noise_per_member (current pipeline)": {
            "mean_epistemic": round(float(epi_indep.mean()), 4),
            "ood_auroc_vs_clean": round(float(ood_auroc(epi_indep)), 4),
        },
        "note": ("If shared-noise epistemic stays well above clean and its OOD-AUROC "
                 "remains high, the gnoise shift signal is NOT merely an artifact of "
                 "per-member noise realizations."),
    }
    json.dump(out, open("data/analysis_shift/p5_gnoise_confound.json", "w"),
              ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
