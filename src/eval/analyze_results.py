"""
analyze_results.py - Aggregate multi-seed results and run the reviewer-robust
analyses that strengthen the paper beyond the headline table.

Produces:
- aggregate.json : per-method mean+/-std over seeds for all metrics.
- per_sample_c3.json : per-SAMPLE Spearman(aleatoric, annotator-disagreement)
  with bootstrap CI, plus a 10-bin monotonicity table -> escapes the n=8
  per-category critique.
- risk_coverage.json : risk-coverage curve points (selective prediction fig).

CLI
---
    python -m python.eval.analyze_results --final-dir data/final \
        --features-dir data/features --methods deterministic,mcdropout,ensemble,proposed \
        --seeds 42,43,44 --out data/analysis
"""
from __future__ import annotations

import argparse
import json
import glob
import os
from pathlib import Path

import numpy as np

from python.eval.metrics import spearman_rho, jensen_shannon


KEYS = ["accuracy", "ece", "brier", "nll", "aurc", "augrc", "mis_auroc",
        "jsd_votes", "kl_votes", "c3_spearman_cat_ale_vs_disagreement",
        "mean_total", "mean_aleatoric", "mean_epistemic"]


def aggregate(final_dir: Path, methods, seeds) -> dict:
    out = {}
    for m in methods:
        vals = {k: [] for k in KEYS}
        for s in seeds:
            p = final_dir / f"{m}_s{s}" / "metrics.json"
            if not p.exists():
                continue
            mm = json.load(open(p))["metrics"]
            for k in KEYS:
                if k in mm and mm[k] is not None:
                    vals[k].append(float(mm[k]))
        out[m] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                  for k, v in vals.items() if v}
    return out


def per_sample_c3(final_dir: Path, features_dir: Path, method: str, seed: int,
                  n_boot: int = 1000) -> dict:
    pred = np.load(final_dir / f"{method}_s{seed}" / "predictions.npz")
    feat = np.load(features_dir / "feat_test.npz", allow_pickle=True)
    ale = pred["aleatoric"].astype(np.float64)
    dis = feat["disagreement"].astype(np.float64)
    n = min(len(ale), len(dis)); ale, dis = ale[:n], dis[:n]

    rho = spearman_rho(ale, dis)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(spearman_rho(ale[idx], dis[idx]))
    boots = np.array([b for b in boots if not np.isnan(b)])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    # 10-bin monotonicity: mean disagreement per aleatoric decile
    order = np.argsort(ale)
    bins = np.array_split(order, 10)
    bin_tbl = [{"bin": i, "mean_aleatoric": float(ale[b].mean()),
                "mean_disagreement": float(dis[b].mean()), "n": int(len(b))}
               for i, b in enumerate(bins)]
    return {"method": method, "seed": seed, "n_samples": int(n),
            "per_sample_spearman": float(rho),
            "bootstrap_ci95": [float(lo), float(hi)],
            "decile_table": bin_tbl}


def risk_coverage(final_dir: Path, method: str, seed: int) -> dict:
    pred = np.load(final_dir / f"{method}_s{seed}" / "predictions.npz")
    total = pred["total"].astype(np.float64)
    mean_p = pred["mean_prob"]; hard = pred["hard"]
    correct = (mean_p.argmax(1) == hard).astype(np.float64)
    order = np.argsort(total)  # low uncertainty first = keep
    c = correct[order]
    k = np.arange(1, len(c) + 1)
    coverage = (k / len(c)).tolist()
    risk = (np.cumsum(1.0 - c) / k).tolist()
    # subsample to 100 points for a compact figure
    idx = np.linspace(0, len(c) - 1, 100).astype(int)
    return {"method": method, "seed": seed,
            "coverage": [coverage[i] for i in idx],
            "risk": [risk[i] for i in idx]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-dir", type=Path, default=Path("data/final"))
    ap.add_argument("--features-dir", type=Path, default=Path("data/features"))
    ap.add_argument("--methods", type=str, default="deterministic,mcdropout,ensemble,proposed")
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--out", type=Path, default=Path("data/analysis"))
    args = ap.parse_args()
    methods = args.methods.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)

    agg = aggregate(args.final_dir, methods, seeds)
    json.dump(agg, open(args.out / "aggregate.json", "w"), ensure_ascii=False, indent=2)

    c3 = {m: per_sample_c3(args.final_dir, args.features_dir, m, seeds[0])
          for m in methods}
    json.dump(c3, open(args.out / "per_sample_c3.json", "w"), ensure_ascii=False, indent=2)

    rc = {m: risk_coverage(args.final_dir, m, seeds[0]) for m in methods}
    json.dump(rc, open(args.out / "risk_coverage.json", "w"), ensure_ascii=False, indent=2)

    # compact console summary
    print("=== aggregate (mean+/-std over seeds) ===")
    hdr = ["method", "acc", "ece", "nll", "jsd", "misAUROC", "c3", "ale", "epi"]
    print("  ".join(h.rjust(9) for h in hdr))
    short = {"acc": "accuracy", "ece": "ece", "nll": "nll", "jsd": "jsd_votes",
             "misAUROC": "mis_auroc", "c3": "c3_spearman_cat_ale_vs_disagreement",
             "ale": "mean_aleatoric", "epi": "mean_epistemic"}
    for m in methods:
        row = [m.rjust(9)]
        for h in hdr[1:]:
            k = short[h]; d = agg.get(m, {}).get(k)
            row.append((f"{d['mean']:.3f}" if d else "-").rjust(9))
        print("  ".join(row))
    print("\n=== per-sample C3 (proposed) ===")
    pc = c3["proposed"]
    print(f"per-sample Spearman={pc['per_sample_spearman']:.4f} "
          f"CI95={pc['bootstrap_ci95']}  n={pc['n_samples']}")
    print("decile mean_aleatoric -> mean_disagreement:")
    for b in pc["decile_table"]:
        print(f"  bin{b['bin']}: ale={b['mean_aleatoric']:.3f} dis={b['mean_disagreement']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
