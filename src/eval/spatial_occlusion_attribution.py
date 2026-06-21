"""
spatial_occlusion_attribution.py - Causal spatial aleatoric attribution.

New spatial mechanism (replaces the CLS-head-on-patch-tokens map, which failed
the causal occlusion test). We DEFINE the spatial aleatoric map causally:

    S_i[p] = H(base_pred_i) - H(pred_i | patch p occluded)

i.e. how much removing region p REDUCES the model's predictive entropy. High
S[p] => region p was driving the ambiguity (its removal resolves the emotion).

Non-circular validation (the map is defined by entropy-drop, so we must NOT
validate it with entropy-drop). We test whether S-guided occlusion RESOLVES the
face toward the MAJORITY-perceived emotion better than random- or naive-map-
guided occlusion:
  - decisiveness: max predictive probability after occluding top-k regions
  - agreement:    JSD(pred, majority one-hot) and argmax==majority accuracy
S-guided should make the model both more confident AND more aligned with the
human majority than random/naive guidance -> the attributed regions are the
ambiguity sources, not arbitrary or merely class-salient.

Comparators: random patches; "naive" = the old per-patch aleatoric map
(CLS head applied to patch tokens).

CLI
---
    python -m python.eval.spatial_occlusion_attribution \
        --features-dir data/features --processed-dir data/processed \
        --pixels data/raw/fer2013_pixels.npy --seed 42 --n-images 400 \
        --out data/analysis/occ_attr
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
PX = 16


def _entropy(p, eps=1e-12):
    pc = np.clip(p, eps, 1.0)
    return -np.sum(pc * np.log(pc), axis=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=Path("data/features"))
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--pixels", type=Path, default=Path("data/raw/fer2013_pixels.npy"))
    ap.add_argument("--backbone", type=str, default="vit_base_patch16_224")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-images", type=int, default=400)
    ap.add_argument("--topks", type=str, default="1,3,5,10")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    import timm
    import torch.nn.functional as F
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    topks = [int(k) for k in args.topks.split(",")]

    # head + scaler
    train = _load_split(args.features_dir, "train")
    valid = _load_split(args.features_dir, "valid")
    test = _load_split(args.features_dir, "test")
    C = int(train["soft"].shape[1])
    mu, sd = fit_scaler(train["cls"]); mu_d, sd_d = mu.to(device), sd.to(device)
    train["cls"] = apply_scaler(train["cls"], mu, sd)
    cfg = dict(num_cycles=8, cycle_length=50, exploitation_fraction=0.25,
               samples_per_cycle=5, burn_in_cycles=1, alpha_0=1e-5,
               temperature=1.0, prior_weight_decay=1e-3, batch_size=512)
    head, states = train_proposed(train, valid, C, "soft", device, cfg, args.seed)
    state_mats = [(st["classifier.weight"].to(device), st["classifier.bias"].to(device)) for st in states]

    def post_mean_from_cls(cls):  # cls (B,D) raw -> (B,C) posterior mean
        x = apply_scaler(cls, mu_d, sd_d)
        acc = torch.zeros(x.shape[0], C, device=device)
        for W, b in state_mats:
            acc += torch.softmax(x @ W.t() + b, -1)
        return (acc / len(state_mats)).cpu().numpy()

    # ViT (img_size=224; DINOv2 defaults to 518)
    try:
        model = timm.create_model(args.backbone, pretrained=True, num_classes=0, img_size=224)
    except TypeError:
        model = timm.create_model(args.backbone, pretrained=True, num_classes=0)
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    imnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    pixels = np.load(args.pixels)
    proc = np.load(args.processed_dir / "ferplus_test.npz", allow_pickle=True)
    row_index = proc["row_index"].astype(np.int64)
    votes = test["soft"].numpy(); hard = test["hard"].numpy()
    patch = test["patch"]  # (N,196,D) for the naive map

    # select most-ambiguous test images
    order = np.argsort(-test["disagreement"])
    sel = order[:args.n_images]

    def base_img(idx_rows):  # -> (B,3,224,224) un-normalised
        raw = torch.from_numpy(pixels[idx_rows].astype(np.float32) / 255.0).to(device)
        x = raw.unsqueeze(1)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x.repeat(1, 3, 1, 1)

    def cls_of(x_unnorm):
        with torch.no_grad():
            tok = model.forward_features((x_unnorm - imnet_mean) / imnet_std)
        return tok[:, 0, :].float()

    P = GRID * GRID
    # ---- compute S map per image (196 occlusions each) + naive map ----
    S = np.zeros((len(sel), P)); base_pred = np.zeros((len(sel), C))
    naive = np.zeros((len(sel), P))
    for i, ti in enumerate(sel):
        img = base_img(row_index[ti:ti+1])  # (1,3,224,224)
        bp = post_mean_from_cls(cls_of(img))[0]
        base_pred[i] = bp; base_H = _entropy(bp)
        # build 196 occluded variants
        occ = img.repeat(P, 1, 1, 1)
        for p in range(P):
            r, c = divmod(p, GRID)
            occ[p, :, r*PX:(r+1)*PX, c*PX:(c+1)*PX] = 0.0
        preds = post_mean_from_cls(cls_of(occ))  # (P,C)
        S[i] = base_H - _entropy(preds)
        # naive map = per-patch aleatoric from CLS head on patch tokens.
        # Only meaningful when the token grid matches the occlusion grid.
        if patch.shape[1] == P:
            xp = apply_scaler(torch.from_numpy(patch[ti].astype(np.float32)).to(device), mu_d, sd_d)
            accH = np.zeros(P)
            for W, b in state_mats:
                pr = torch.softmax(xp @ W.t() + b, -1).cpu().numpy()
                accH += _entropy(pr)
            naive[i] = accH / len(state_mats)

    # ---- validation: occlude top-k by {S, naive, random}; measure resolve ----
    maj = hard[sel]
    def occlude_and_eval(rank_scores):
        res = {}
        for k in topks:
            occ = base_img(row_index[sel])  # (n,3,224,224)
            for i in range(len(sel)):
                cells = np.argsort(-rank_scores[i])[:k]
                for cc in cells:
                    r, c = divmod(int(cc), GRID)
                    occ[i, :, r*PX:(r+1)*PX, c*PX:(c+1)*PX] = 0.0
            pred = post_mean_from_cls(cls_of(occ))
            onehot = np.zeros_like(pred); onehot[np.arange(len(pred)), maj] = 1.0
            res[k] = {
                "mean_entropy": float(_entropy(pred).mean()),
                "mean_maxprob": float(pred.max(1).mean()),
                "jsd_to_majority": float(jensen_shannon(pred, onehot, axis=1).mean()),
                "acc_vs_majority": float((pred.argmax(1) == maj).mean()),
            }
        return res

    base_onehot = np.zeros_like(base_pred); base_onehot[np.arange(len(base_pred)), maj] = 1.0
    out = {
        "n_images": int(len(sel)),
        "base": {"mean_entropy": float(_entropy(base_pred).mean()),
                 "mean_maxprob": float(base_pred.max(1).mean()),
                 "jsd_to_majority": float(jensen_shannon(base_pred, base_onehot, axis=1).mean()),
                 "acc_vs_majority": float((base_pred.argmax(1) == maj).mean())},
        "S_guided": occlude_and_eval(S),
        "naive_guided": (occlude_and_eval(naive) if patch.shape[1] == P else "skipped(grid!=tokens)"),
        "random_guided": occlude_and_eval(rng.random((len(sel), P))),
        "note": "S_guided should LOWER entropy & jsd_to_majority and RAISE maxprob & "
                "acc_vs_majority more than random/naive -> attributed regions are the "
                "ambiguity sources that, once removed, resolve the human-majority emotion.",
    }
    np.savez_compressed(args.out / "S_maps.npz", S=S, naive=naive,
                        index_in_test=sel.astype(np.int64), base_pred=base_pred)
    json.dump(out, open(args.out / "occ_attr_result.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "S_vs_naive_map_spearman_meanperimg"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
