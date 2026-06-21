"""natural_ood.py - NATURAL (non-synthetic) distribution shift from FERPlus.

FERPlus crowd labels include NF (not-a-face) and unknown (unclassifiable). The
standard pipeline DROPS samples with unknown+NF>=5. Those dropped samples are a
NATURAL OOD source needing no acquisition:
  - NF-dominant  = non-face images  -> genuine distribution shift.
  - unknown-dominant = ambiguous/unclassifiable faces -> high aleatoric, in-dist-ish.

We run the same 5-member Deep Ensemble on these and test:
  (1) does epistemic detect NF (non-face) natural OOD above chance / above the
      single-model max-prob baseline?  (-> epistemic generalizes to REAL shift)
  (2) are unknown (ambiguous) samples high in ALEATORIC but NOT epistemic?
      (-> a/e distinction holds on natural data)
Clean reference = FERPlus test faces (shift_ens/none_s0).
"""
from __future__ import annotations
import csv, json
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
EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]


def ent(p):
    pc = np.clip(p, EPS, 1.0)
    return -np.sum(pc * np.log(pc), axis=-1)


def build_model(cp):
    try:
        m = timm.create_model(BB, pretrained=False, num_classes=8, img_size=224)
    except TypeError:
        m = timm.create_model(BB, pretrained=False, num_classes=8)
    sd = torch.load(cp, map_location="cpu"); sd = sd.get("state_dict", sd)
    m.load_state_dict(sd, strict=True)
    return m.to(DEV).eval()


def infer(idx, pixels):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEV)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEV)
    n = len(idx); member = np.zeros((5, n, 8), dtype=np.float64); bs = 128
    for k, cp in enumerate(CK):
        model = build_model(cp)
        with torch.no_grad():
            for s in range(0, n, bs):
                bi = idx[s:s + bs]
                raw = torch.from_numpy(pixels[bi].astype(np.float32) / 255.0).to(DEV)
                x = raw.unsqueeze(1)
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
                xn = (x - mean) / std
                member[k, s:s + bs] = torch.softmax(model(xn), -1).cpu().numpy()
        del model; torch.cuda.empty_cache()
    mp = member.mean(0)
    tot = ent(mp)
    ale = np.stack([ent(member[kk]) for kk in range(5)]).mean(0)
    epi = np.maximum(tot - ale, 0.0)
    return {"epistemic": epi, "aleatoric": ale, "total": tot, "maxprob": mp.max(1)}


def main():
    rows = list(csv.DictReader(open("data/raw/fer2013new.csv")))
    nf_idx, unk_idx = [], []
    for i, r in enumerate(rows):
        try:
            nf = float(r["NF"]); unk = float(r["unknown"])
        except (ValueError, KeyError):
            continue
        if nf >= 5: nf_idx.append(i)
        elif unk >= 5: unk_idx.append(i)
    nf_idx = np.array(nf_idx); unk_idx = np.array(unk_idx)

    pixels = np.load("data/raw/fer2013_pixels.npy")
    nf = infer(nf_idx, pixels)
    unk = infer(unk_idx, pixels)
    clean = np.load("data/shift_ens/none_s0/predictions.npz")
    csingle = np.load("data/shift_single/none_s0/predictions.npz")
    epi_c = clean["epistemic"]; ale_c = clean["aleatoric"]; smx_c = 1 - csingle["maxprob"]

    def auroc(ood_score, clean_score):
        lab = np.concatenate([np.zeros(len(clean_score)), np.ones(len(ood_score))])
        return float(_auroc(np.concatenate([clean_score, ood_score]), lab))

    out = {
        "n_clean_test_faces": int(len(epi_c)),
        "n_NF_nonface": int(len(nf_idx)),
        "n_unknown_ambiguous": int(len(unk_idx)),
        "NF_natural_OOD_detection": {
            "epistemic_AUROC": round(auroc(nf["epistemic"], epi_c), 4),
            "single_maxprob_AUROC": round(auroc(1 - nf["maxprob"], smx_c), 4),
            "aleatoric_AUROC": round(auroc(nf["aleatoric"], ale_c), 4),
            "mean_epistemic_NF": round(float(nf["epistemic"].mean()), 4),
            "mean_epistemic_clean": round(float(epi_c.mean()), 4),
        },
        "unknown_ambiguous_check": {
            "epistemic_AUROC": round(auroc(unk["epistemic"], epi_c), 4),
            "aleatoric_AUROC": round(auroc(unk["aleatoric"], ale_c), 4),
            "mean_aleatoric_unknown": round(float(unk["aleatoric"].mean()), 4),
            "mean_aleatoric_clean": round(float(ale_c.mean()), 4),
        },
        "note": ("NF = non-face natural OOD (should raise epistemic); unknown = ambiguous "
                 "faces (should raise aleatoric more than epistemic). Tests whether the a/e "
                 "distinction holds on NATURAL, non-synthetic data."),
    }
    Path("data/analysis_shift").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("data/analysis_shift/natural_ood.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
