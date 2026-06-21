"""
occlusion_test.py - Causal validation of the spatial aleatoric map.

Hypothesis: masking the image regions our spatial-aleatoric map flags as
high-ambiguity should disturb the model's agreement with the ANNOTATOR vote
distribution more than masking random regions. If so, those regions causally
drive the ambiguity the annotators disagreed over (not just class saliency).

Procedure (frozen ViT-B/16, tuned cSG-MCMC head):
1. Re-derive the proposed posterior head + train-fitted scaler.
2. For each selected test image, compute per-patch aleatoric (head over patch
   tokens), pick the top-k high-aleatoric 14x14 cells.
3. Re-extract the CLS feature for the image under three conditions: no mask,
   mask high-aleatoric cells (16x16 px blocks at 224), mask random cells.
4. Posterior-mean predict; measure JSD(pred, votes) for each condition.
Report JSD shift (high-aleatoric vs random). high > random => validated.

CLI
---
    python -m python.eval.occlusion_test --features-dir data/features \
        --processed-dir data/processed --pixels data/raw/fer2013_pixels.npy \
        --backbone vit_base_patch16_224 --seed 42 --n-images 200 --top-k 3 \
        --out data/analysis/occlusion
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from python.train.train_csgmcmc import _load_split, train_proposed, fit_scaler, apply_scaler
from python.eval.metrics import jensen_shannon

GRID = 14
PATCH_PX = 16  # 224 / 14


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=Path("data/features"))
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--pixels", type=Path, default=Path("data/raw/fer2013_pixels.npy"))
    ap.add_argument("--backbone", type=str, default="vit_base_patch16_224")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    import timm
    import torch.nn.functional as F
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1. proposed head + scaler
    train = _load_split(args.features_dir, "train")
    valid = _load_split(args.features_dir, "valid")
    test = _load_split(args.features_dir, "test")
    num_classes = int(train["soft"].shape[1])
    mu, sd = fit_scaler(train["cls"])
    train["cls"] = apply_scaler(train["cls"], mu, sd)
    cfg = dict(num_cycles=8, cycle_length=50, exploitation_fraction=0.25,
               samples_per_cycle=5, burn_in_cycles=1, alpha_0=1e-5,
               temperature=1.0, prior_weight_decay=1e-3, batch_size=512)
    head, states = train_proposed(train, valid, num_classes, "soft", device, cfg, args.seed)
    mu_d, sd_d = mu.to(device), sd.to(device)

    def posterior_mean(cls_feat):  # cls_feat (B,D) raw -> (B,C)
        x = apply_scaler(cls_feat, mu_d, sd_d)
        probs = np.zeros((len(states), x.shape[0], num_classes))
        with torch.no_grad():
            for m, st in enumerate(states):
                head.load_state_dict(st)
                probs[m] = torch.softmax(head(x), -1).cpu().numpy()
        return probs.mean(0)

    # 2. ViT + test pixels + per-image patch aleatoric
    model = timm.create_model(args.backbone, pretrained=True, num_classes=0).eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    pixels = np.load(args.pixels)
    proc = np.load(args.processed_dir / "ferplus_test.npz", allow_pickle=True)
    row_index = proc["row_index"].astype(np.int64)  # aligned to feat_test order
    votes = test["soft"].numpy()
    patch = torch.from_numpy(test["patch"].astype(np.float32))  # (N,196,D)

    # per-patch aleatoric for all selected images
    order = np.argsort(-test["disagreement"])  # most ambiguous first
    sel = order[:args.n_images]

    imnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def make_input(idx_list):  # row indices -> (B,3,224,224)
        raw = torch.from_numpy(pixels[idx_list].astype(np.float32) / 255.0).to(device)
        x = raw.unsqueeze(1)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        return x  # un-normalised; we mask then normalise

    def cls_from_imgs(x_unnorm):
        x = (x_unnorm - imnet_mean) / imnet_std
        with torch.no_grad():
            tok = model.forward_features(x)
        return tok[:, 0, :].float()

    def mask_cells(x_unnorm, cells_per_img):
        x = x_unnorm.clone()
        for bi, cells in enumerate(cells_per_img):
            for c in cells:
                r, col = divmod(int(c), GRID)
                x[bi, :, r*PATCH_PX:(r+1)*PATCH_PX, col*PATCH_PX:(col+1)*PATCH_PX] = 0.0
        return x

    # per-patch aleatoric (apply states to standardized patch tokens)
    sel_patch = patch[sel].to(device)  # (n,196,D)
    n, P, D = sel_patch.shape
    member = np.zeros((len(states), n, P, num_classes))
    xp = apply_scaler(sel_patch.reshape(-1, D), mu_d, sd_d)
    with torch.no_grad():
        for m, st in enumerate(states):
            head.load_state_dict(st)
            member[m] = torch.softmax(head(xp), -1).reshape(n, P, num_classes).cpu().numpy()
    eps = 1e-12
    ale_patch = np.mean([-(np.clip(member[m], eps, 1) * np.log(np.clip(member[m], eps, 1))).sum(-1)
                         for m in range(len(states))], axis=0)  # (n,P)

    # 3-4. occlusion conditions
    idx_rows = row_index[sel]
    base_in = make_input(idx_rows)
    base_pred = posterior_mean(cls_from_imgs(base_in))
    high_cells = [list(np.argsort(-ale_patch[i])[:args.top_k]) for i in range(n)]
    rand_cells = [list(rng.choice(P, size=args.top_k, replace=False)) for _ in range(n)]
    high_pred = posterior_mean(cls_from_imgs(mask_cells(base_in, high_cells)))
    rand_pred = posterior_mean(cls_from_imgs(mask_cells(base_in, rand_cells)))

    v = votes[sel]
    base_jsd = jensen_shannon(base_pred, v, axis=1)
    high_jsd = jensen_shannon(high_pred, v, axis=1)
    rand_jsd = jensen_shannon(rand_pred, v, axis=1)

    res = {
        "n_images": int(n), "top_k": int(args.top_k),
        "base_jsd": float(base_jsd.mean()),
        "high_aleatoric_masked_jsd": float(high_jsd.mean()),
        "random_masked_jsd": float(rand_jsd.mean()),
        "high_minus_base": float((high_jsd - base_jsd).mean()),
        "random_minus_base": float((rand_jsd - base_jsd).mean()),
        "high_minus_random": float((high_jsd - rand_jsd).mean()),
        "paired_high_gt_random_frac": float((high_jsd > rand_jsd).mean()),
        "note": "high_minus_random > 0 => masking high-aleatoric regions disturbs "
                "agreement with annotator votes more than random => spatial aleatoric "
                "marks the ambiguity-driving regions",
    }
    json.dump(res, open(args.out / "occlusion_result.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
