"""
train_csgmcmc.py - Train the head over frozen ViT features and evaluate
the uncertainty decomposition (image-level), porting the NLP recipe.

Methods
-------
- proposed   : cSG-MCMC posterior over a linear head + soft-label loss.
- ensemble   : K independently-initialised heads (Deep Ensemble baseline).
- mcdropout  : single head, dropout sampled at eval (MC Dropout baseline).
- deterministic : single head, no posterior (epistemic == 0 reference).

All methods share the same frozen ViT CLS features and the same soft-label
objective (KL to the annotator vote distribution) for a fair comparison,
matching the NLP study's protocol.

Evaluation (on test): C1 ECE/Brier/NLL, C2 JSD/KL to votes, C5 AURC/AUGRC,
misclassification AUROC, plus the total/aleatoric/epistemic decomposition
and the C3 per-category (aleatoric vs disagreement) Spearman rho.

CLI
---
    python -m python.train.train_csgmcmc \
        --features-dir data/features --method proposed --label soft \
        --seed 42 --out data/outputs/proposed_soft_s42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

from python.models.csgmcmc_head import (
    LinearHead, sgld_step, soft_label_loss, cyclical_lr, is_sampling_phase,
    compute_sample_steps, decompose_uncertainty,
)
from python.eval.metrics import compute_all_metrics, spearman_rho


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _load_split(features_dir: Path, split: str):
    d = np.load(features_dir / f"feat_{split}.npz", allow_pickle=True)
    return {
        "cls": torch.from_numpy(d["cls"].astype(np.float32)),
        "soft": torch.from_numpy(d["soft"].astype(np.float32)),
        "hard": torch.from_numpy(d["hard"].astype(np.int64)),
        "disagreement": d["disagreement"].astype(np.float64),
        "patch": (d["patch"] if "patch" in d.files else None),
    }


def fit_scaler(feats: torch.Tensor):
    """Per-dim mean/std from TRAIN cls features. ViT features have large,
    uneven scale (std~1.6, |max|~23) which destabilises the SGLD head;
    standardising fixes the gradient/step magnitude."""
    mu = feats.mean(0, keepdim=True)
    sd = feats.std(0, keepdim=True).clamp_min(1e-6)
    return mu, sd


def apply_scaler(feats: torch.Tensor, mu, sd):
    return (feats - mu) / sd


def _targets(batch_soft, batch_hard, label_mode):
    if label_mode == "soft":
        return batch_soft
    onehot = torch.zeros_like(batch_soft)
    onehot[torch.arange(len(batch_hard)), batch_hard] = 1.0
    return onehot


# --------------------------------------------------------------------------
# Member predictions per method -> (M, N, C) softmax
# --------------------------------------------------------------------------

def _member_probs_from_states(head, states, feats, device, bs=4096):
    head.eval()
    outs = []
    with torch.no_grad():
        for st in states:
            head.load_state_dict(st)
            probs = []
            for s in range(0, len(feats), bs):
                x = feats[s:s + bs].to(device)
                probs.append(torch.softmax(head(x), -1).cpu().numpy())
            outs.append(np.concatenate(probs, 0))
    return np.stack(outs, 0)  # (M,N,C)


def train_proposed(train, valid, num_classes, label_mode, device, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    feats = train["cls"]; N = len(feats)
    head = LinearHead(feats.shape[1], num_classes, dropout_p=0.0).to(device)
    params = list(head.parameters())
    targets_all = _targets(train["soft"], train["hard"], label_mode).to(device)

    L = cfg["cycle_length"]; ncyc = cfg["num_cycles"]
    ef = cfg["exploitation_fraction"]; spc = cfg["samples_per_cycle"]
    burn = cfg["burn_in_cycles"]; a0 = cfg["alpha_0"]; T = cfg["temperature"]
    wd = cfg["prior_weight_decay"]; bs = cfg["batch_size"]
    sample_steps = set(compute_sample_steps(L, spc, ef))
    gen = torch.Generator(device=device); gen.manual_seed(seed + 9999)

    states = []
    feats_dev = feats.to(device)
    step = 0
    for cyc in range(ncyc):
        in_burn = cyc < burn
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed + cyc))
        ptr = 0
        for u in range(L):
            if ptr + bs > N:
                perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed + cyc + step))
                ptr = 0
            bidx = perm[ptr:ptr + bs]; ptr += bs
            x = feats_dev[bidx]; tgt = targets_all[bidx]
            logits = head(x)
            loss = soft_label_loss(logits, tgt)
            for p in params:
                if p.grad is not None: p.grad.detach_().zero_()
            loss.backward()
            lr = cyclical_lr(u, cycle_length=L, alpha_0=a0)
            in_samp = is_sampling_phase(u, cycle_length=L, exploitation_fraction=ef)
            sgld_step(params, lr=lr, temperature=(T if in_samp else 0.0),
                      weight_decay=wd, posterior_scale=float(N), generator=gen)
            if (not in_burn) and (u in sample_steps):
                states.append({k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
            step += 1
    return head, states


def train_pointwise(train, num_classes, label_mode, device, seed, epochs=20,
                    lr=1e-3, dropout_p=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    feats = train["cls"].to(device); N = len(feats)
    head = LinearHead(feats.shape[1], num_classes, dropout_p=dropout_p).to(device)
    targets_all = _targets(train["soft"], train["hard"], label_mode).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    bs = 4096
    for ep in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N, bs):
            bidx = perm[s:s + bs]
            logits = head(feats[bidx])
            loss = soft_label_loss(logits, targets_all[bidx])
            opt.zero_grad(); loss.backward(); opt.step()
    return head


def evaluate(member_probs, test, num_classes):
    dec = decompose_uncertainty(member_probs)
    soft = test["soft"].numpy(); hard = test["hard"].numpy()
    metrics = compute_all_metrics(dec["mean_prob"], hard, targets_soft=soft,
                                  uncertainty_total=dec["total"])
    # C3: per-category aleatoric vs per-category disagreement
    dis = test["disagreement"]
    cat_ale = np.array([dec["aleatoric"][hard == c].mean() if (hard == c).any() else np.nan
                        for c in range(num_classes)])
    cat_dis = np.array([dis[hard == c].mean() if (hard == c).any() else np.nan
                        for c in range(num_classes)])
    ok = ~(np.isnan(cat_ale) | np.isnan(cat_dis))
    metrics["c3_spearman_cat_ale_vs_disagreement"] = spearman_rho(cat_ale[ok], cat_dis[ok])
    metrics["mean_total"] = float(dec["total"].mean())
    metrics["mean_aleatoric"] = float(dec["aleatoric"].mean())
    metrics["mean_epistemic"] = float(dec["epistemic"].mean())
    metrics["num_members"] = int(member_probs.shape[0])
    return metrics, dec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=Path("data/features"))
    ap.add_argument("--method", choices=["proposed", "ensemble", "mcdropout", "deterministic"],
                    default="proposed")
    ap.add_argument("--label", choices=["soft", "hard"], default="soft")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ensemble-k", type=int, default=10)
    ap.add_argument("--mc-samples", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda:0")
    # cSG-MCMC hyperparameters (review-robust defaults from the NLP study)
    ap.add_argument("--num-cycles", type=int, default=8)
    ap.add_argument("--cycle-length", type=int, default=50)
    ap.add_argument("--exploitation-fraction", type=float, default=0.25)
    ap.add_argument("--samples-per-cycle", type=int, default=5)
    ap.add_argument("--burn-in-cycles", type=int, default=1)
    # Tuned for frozen ViT features (2026-06-17 sweep): effective drift
    # alpha0*N must be a sane LR; alpha0=1e-5 fixes the earlier overconfidence
    # (alpha0=5e-4 gave NLL 2.82, aleatoric~0). T=1 = standard Bayesian.
    ap.add_argument("--alpha0", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--prior-weight-decay", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--no-standardize", action="store_true",
                    help="disable train-fitted feature standardization")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="subsample this fraction of TRAIN (for data-size study)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    train = _load_split(args.features_dir, "train")
    valid = _load_split(args.features_dir, "valid")
    test = _load_split(args.features_dir, "test")
    num_classes = int(train["soft"].shape[1])

    # Data-size study: subsample TRAIN (seeded). Epistemic should fall as the
    # training set grows; aleatoric (irreducible emotion ambiguity) should not.
    if args.train_frac < 1.0:
        g = torch.Generator().manual_seed(args.seed)
        n_all = len(train["cls"]); n_keep = max(int(n_all * args.train_frac), num_classes * 4)
        keep = torch.randperm(n_all, generator=g)[:n_keep]
        for k in ("cls", "soft", "hard"):
            train[k] = train[k][keep]
        train["disagreement"] = train["disagreement"][keep.numpy()]

    # Standardize cls features (mean/std fitted on TRAIN). Save scaler so the
    # spatial-map step applies the identical transform to patch tokens.
    if not args.no_standardize:
        mu, sd = fit_scaler(train["cls"])
        for split in (train, valid, test):
            split["cls"] = apply_scaler(split["cls"], mu, sd)
        np.savez(args.out / "scaler.npz", mu=mu.numpy(), sd=sd.numpy())

    cfg = dict(num_cycles=args.num_cycles, cycle_length=args.cycle_length,
               exploitation_fraction=args.exploitation_fraction,
               samples_per_cycle=args.samples_per_cycle,
               burn_in_cycles=args.burn_in_cycles, alpha_0=args.alpha0,
               temperature=args.temperature, prior_weight_decay=args.prior_weight_decay,
               batch_size=args.batch_size)

    if args.method == "proposed":
        head, states = train_proposed(train, valid, num_classes, args.label, device, cfg, args.seed)
        member = _member_probs_from_states(head, states, test["cls"], device)
    elif args.method == "ensemble":
        states = []
        for k in range(args.ensemble_k):
            h = train_pointwise(train, num_classes, args.label, device, args.seed + k)
            states.append({kk: v.detach().cpu().clone() for kk, v in h.state_dict().items()})
        head = LinearHead(train["cls"].shape[1], num_classes).to(device)
        member = _member_probs_from_states(head, states, test["cls"], device)
    elif args.method == "mcdropout":
        head = train_pointwise(train, num_classes, args.label, device, args.seed, dropout_p=0.2)
        head.train()  # keep dropout active
        feats = test["cls"]; outs = []
        with torch.no_grad():
            for _ in range(args.mc_samples):
                pr = []
                for s in range(0, len(feats), 4096):
                    pr.append(torch.softmax(head(feats[s:s+4096].to(device)), -1).cpu().numpy())
                outs.append(np.concatenate(pr, 0))
        member = np.stack(outs, 0)
    else:  # deterministic
        head = train_pointwise(train, num_classes, args.label, device, args.seed)
        feats = test["cls"]; pr = []
        head.eval()
        with torch.no_grad():
            for s in range(0, len(feats), 4096):
                pr.append(torch.softmax(head(feats[s:s+4096].to(device)), -1).cpu().numpy())
        member = np.concatenate(pr, 0)[None, ...]  # (1,N,C) -> epistemic 0

    metrics, dec = evaluate(member, test, num_classes)
    np.savez_compressed(args.out / "predictions.npz",
                        mean_prob=dec["mean_prob"], total=dec["total"],
                        aleatoric=dec["aleatoric"], epistemic=dec["epistemic"],
                        hard=test["hard"].numpy(), soft=test["soft"].numpy())
    record = {"method": args.method, "label": args.label, "seed": args.seed,
              "cfg": cfg, "metrics": metrics}
    with open(args.out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
