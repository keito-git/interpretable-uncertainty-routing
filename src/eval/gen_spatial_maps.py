"""
gen_spatial_maps.py - Spatial aleatoric/epistemic maps for test faces.

Reproduces the proposed cSG-MCMC head (same seed/cfg as train_csgmcmc), then
applies the posterior samples to the per-patch ViT tokens to obtain, for each
selected test image, per-patch predictive distributions -> a/e decomposition
-> 14x14 spatial maps. Saves the maps (and indices) for figure generation.

Also computes the reviewer-defence quantities where possible without the raw
image: saliency-style baseline = per-patch top-class probability mass (a crude
class-saliency proxy) and its dissimilarity to the aleatoric map.

CLI
---
    python -m python.eval.gen_spatial_maps \
        --features-dir data/features --seed 42 \
        --n-images 64 --out data/outputs/spatial_maps_s42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from python.models.csgmcmc_head import LinearHead, decompose_uncertainty
from python.models.spatial_aleatoric import (
    patch_decomposition, map_grid, normalize_map, saliency_dissimilarity,
)
from python.train.train_csgmcmc import _load_split, train_proposed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=Path("data/features"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-images", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    # mirror train_csgmcmc proposed defaults
    ap.add_argument("--num-cycles", type=int, default=8)
    ap.add_argument("--cycle-length", type=int, default=50)
    ap.add_argument("--exploitation-fraction", type=float, default=0.25)
    ap.add_argument("--samples-per-cycle", type=int, default=5)
    ap.add_argument("--burn-in-cycles", type=int, default=1)
    ap.add_argument("--alpha0", type=float, default=5e-4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--prior-weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    train = _load_split(args.features_dir, "train")
    valid = _load_split(args.features_dir, "valid")
    test = _load_split(args.features_dir, "test")
    num_classes = int(train["soft"].shape[1])
    if test["patch"] is None:
        raise RuntimeError("test split has no patch features; re-run extract_features "
                           "with --save-patch-splits test")

    cfg = dict(num_cycles=args.num_cycles, cycle_length=args.cycle_length,
               exploitation_fraction=args.exploitation_fraction,
               samples_per_cycle=args.samples_per_cycle,
               burn_in_cycles=args.burn_in_cycles, alpha_0=args.alpha0,
               temperature=args.temperature, prior_weight_decay=args.prior_weight_decay,
               batch_size=args.batch_size)
    head, states = train_proposed(train, valid, num_classes, "soft", device, cfg, args.seed)

    patch = torch.from_numpy(test["patch"].astype(np.float32))  # (N,P,D)
    grid_h = grid_w = 14
    N, P, D = patch.shape

    # Select most-ambiguous images by image-level disagreement for compelling figs.
    order = np.argsort(-test["disagreement"])
    sel = order[:args.n_images]

    # member probs per patch for selected images: (M, n_sel, P, C)
    head.eval()
    M = len(states)
    sel_patch = patch[sel].to(device)  # (n,P,D)
    member = np.zeros((M, len(sel), P, num_classes), dtype=np.float32)
    with torch.no_grad():
        for m, st in enumerate(states):
            head.load_state_dict(st)
            logits = head(sel_patch.reshape(-1, D))            # (n*P, C)
            probs = torch.softmax(logits, -1).reshape(len(sel), P, num_classes)
            member[m] = probs.cpu().numpy()

    dec = patch_decomposition(member)  # total/aleatoric/epistemic each (n,P)

    # saliency proxy = per-patch max class prob (class-discriminative mass)
    saliency = dec["mean_prob"].max(axis=-1)  # (n,P)

    maps = {"index_in_test": sel.astype(np.int64),
            "disagreement": test["disagreement"][sel].astype(np.float32),
            "hard": test["hard"].numpy()[sel].astype(np.int64),
            "aleatoric_grid": np.stack([map_grid(dec["aleatoric"][i], grid_h, grid_w) for i in range(len(sel))]),
            "epistemic_grid": np.stack([map_grid(dec["epistemic"][i], grid_h, grid_w) for i in range(len(sel))]),
            "saliency_grid": np.stack([map_grid(saliency[i], grid_h, grid_w) for i in range(len(sel))])}

    # aggregate reviewer-defence stat: mean dissimilarity(aleatoric, saliency)
    dissims = [saliency_dissimilarity(normalize_map(maps["aleatoric_grid"][i]),
                                      normalize_map(maps["saliency_grid"][i]))
               for i in range(len(sel))]
    summary = {"n_images": int(len(sel)), "num_members": int(M),
               "mean_aleatoric_vs_saliency_dissimilarity": float(np.nanmean(dissims)),
               "note": "high dissimilarity => spatial aleatoric != class saliency"}

    np.savez_compressed(args.out / "spatial_maps.npz", **maps)
    with open(args.out / "spatial_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
