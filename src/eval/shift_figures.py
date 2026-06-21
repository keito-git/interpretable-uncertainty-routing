"""
shift_figures.py - Paper figures for the distribution-shift / dual-validation story.

Reads data/analysis_shift/curves.json (severity curves) and the ensemble clean
predictions + FERPlus votes (for the aleatoric<->disagreement decile panel).

Figures:
- fig_ece_vs_severity.png    : ensemble vs single ECE across severity (avg over corruptions)
- fig_ood_vs_severity.png    : epistemic vs single-maxprob OOD-AUROC across severity
- fig_unc_vs_severity.png    : mean aleatoric vs epistemic across severity
- fig_dual_validation.png    : (left) aleatoric decile vs human disagreement (in-dist);
                               (right) epistemic OOD-AUROC vs severity (shift)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CORR = ["gnoise", "blur", "occlude", "bright", "pixelate"]
SEV = [0, 1, 2, 3, 4, 5]


def avg_curve(curves, key):
    out = []
    for sev in SEV:
        vals = [r[key] for c in CORR for r in curves.get(c, []) if r["severity"] == sev]
        out.append(float(np.mean(vals)) if vals else np.nan)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", type=Path, default=Path("data/analysis_shift/curves.json"))
    ap.add_argument("--ens-clean", type=Path, default=Path("data/shift_ens/none_s0/predictions.npz"))
    ap.add_argument("--votes", type=Path, default=Path("data/processed/ferplus_test.npz"))
    ap.add_argument("--out", type=Path, default=Path("data/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    curves = json.load(open(args.curves))

    # 1. ECE vs severity
    plt.figure(figsize=(5, 4))
    plt.plot(SEV, avg_curve(curves, "ens_ece"), "o-", label="Deep Ensemble")
    plt.plot(SEV, avg_curve(curves, "single_ece"), "s--", label="Single model")
    plt.xlabel("Corruption severity"); plt.ylabel("ECE (lower better)")
    plt.title("Calibration under shift"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(args.out / "fig_ece_vs_severity.png", dpi=200); plt.close()

    # 2. OOD-AUROC vs severity
    plt.figure(figsize=(5, 4))
    plt.plot(SEV, avg_curve(curves, "ood_auroc_epistemic"), "o-", label="Ensemble epistemic")
    plt.plot(SEV, avg_curve(curves, "ood_auroc_single_maxprob"), "s--", label="Single 1-maxprob")
    plt.axhline(0.5, color="gray", ls=":", lw=1)
    plt.xlabel("Corruption severity"); plt.ylabel("Shift-detection AUROC")
    plt.title("Distribution-shift detection"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(args.out / "fig_ood_vs_severity.png", dpi=200); plt.close()

    # 3. uncertainty vs severity
    plt.figure(figsize=(5, 4))
    plt.plot(SEV, avg_curve(curves, "mean_epistemic"), "o-", label="epistemic")
    plt.plot(SEV, avg_curve(curves, "mean_aleatoric"), "s-", label="aleatoric")
    plt.xlabel("Corruption severity"); plt.ylabel("Mean uncertainty (nats)")
    plt.title("Uncertainty response to shift"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(args.out / "fig_unc_vs_severity.png", dpi=200); plt.close()

    # 4. dual validation
    e = np.load(args.ens_clean); meta = np.load(args.votes, allow_pickle=True)
    dis = meta["disagreement"].astype(float); ale = e["aleatoric"].astype(float)
    n = min(len(dis), len(ale)); ale, dis = ale[:n], dis[:n]
    order = np.argsort(ale); bins = np.array_split(order, 10)
    bx = [ale[b].mean() for b in bins]; by = [dis[b].mean() for b in bins]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].plot(bx, by, "o-", color="C2")
    ax[0].set_xlabel("Predicted aleatoric (decile mean)")
    ax[0].set_ylabel("Human annotator disagreement")
    ax[0].set_title("aleatoric -> human disagreement (in-dist)"); ax[0].grid(alpha=0.3)
    ax[1].plot(SEV, avg_curve(curves, "ood_auroc_epistemic"), "o-", color="C3")
    ax[1].axhline(0.5, color="gray", ls=":", lw=1)
    ax[1].set_xlabel("Corruption severity"); ax[1].set_ylabel("Epistemic shift-detection AUROC")
    ax[1].set_title("epistemic -> distribution shift"); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(args.out / "fig_dual_validation.png", dpi=200); plt.close()
    print("figures written to", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
