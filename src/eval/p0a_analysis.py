"""
p0a_analysis.py - UAR ablation + K-sensitivity (from SAVED member predictions).

Addresses meta-review P0-A / P1-A:
- UAR routing AUC (ambiguous in-dist retention vs OOD rejection) for epistemic
  vs single-maxprob, with paired bootstrap 95% CI on the difference, plus the
  full threshold-sweep curve (robustness).
- K-sensitivity: recompute the ensemble decomposition for K=2..5 from the saved
  per-member softmax (no re-inference) -> epistemic shift-detection AUROC and
  UAR AUC vs K (for the computational-cost discussion).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from python.eval.metrics import _auroc

CORR = ["gnoise", "blur", "occlude", "bright", "pixelate"]
SEV = [1, 2, 3, 4, 5]
_EPS = 1e-12


def _ent(p):
    pc = np.clip(p, _EPS, 1.0)
    return -np.sum(pc * np.log(pc), axis=-1)


def _load(d):
    p = Path(d) / "predictions.npz"
    return np.load(p) if p.exists() else None


def decomp_K(member, K):
    """member (Kfull,N,C) -> total/ale/epi/maxprob using first K members."""
    m = member[:K]
    mean_p = m.mean(0)
    tot = _ent(mean_p)
    ale = np.stack([_ent(m[k]) for k in range(K)]).mean(0)
    epi = np.maximum(tot - ale, 0.0)
    return mean_p, tot, ale, epi


def routing_auc(score_ind, score_ood_pool, sel, amb, n_clean):
    score = np.concatenate([score_ind, score_ood_pool[sel]])
    is_ood = np.concatenate([np.zeros(n_clean), np.ones(n_clean)])
    ambm = np.concatenate([amb, np.zeros(n_clean, bool)])
    order = np.argsort(-score)
    rej = np.zeros(len(score), bool)
    rows = []
    na = max(int(ambm.sum()), 1)
    step = max(len(order) // 300, 1)
    for k in range(0, len(order) + 1, step):
        rej[:] = False; rej[order[:k]] = True
        rows.append(((rej & (is_ood == 1)).sum() / n_clean, (~rej & ambm).sum() / na))
    rows = np.array(rows)
    idx = np.argsort(rows[:, 0])
    return float(np.trapz(rows[idx, 1], rows[idx, 0]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens-dir", type=Path, default=Path("data/shift_ens"))
    ap.add_argument("--single-dir", type=Path, default=Path("data/shift_single"))
    ap.add_argument("--votes", type=Path, default=Path("data/processed/ferplus_test.npz"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis_shift/p0a.json"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    meta = np.load(args.votes, allow_pickle=True)
    soft = meta["soft"].astype(np.float64)
    ce = _load(args.ens_dir / "none_s0"); cs = _load(args.single_dir / "none_s0")
    n_clean = len(ce["epistemic"])
    ve = _ent(soft)[:n_clean]
    hi = np.quantile(ve, 2/3); amb = ve >= hi
    rng = np.random.default_rng(0)

    out = {}

    # ---- K-sensitivity from saved members ----
    memb_clean = ce["member"]  # (5,N,C)
    Kfull = memb_clean.shape[0]
    # pre-load corrupted members + single maxprob
    ood_members = [(_load(args.ens_dir / f"{c}_s{s}"), _load(args.single_dir / f"{c}_s{s}"))
                   for c in CORR for s in SEV]
    out["K_sensitivity"] = {}
    for K in range(2, Kfull + 1):
        _, _, ale_c, epi_c = decomp_K(memb_clean, K)
        # shift detection AUROC avg
        aucs = []
        epi_ood_all = []
        for e, _sg in ood_members:
            _, _, _, epi_o = decomp_K(e["member"], K)
            lab = np.concatenate([np.zeros(n_clean), np.ones(len(epi_o))])
            aucs.append(_auroc(np.concatenate([epi_c, epi_o]), lab))
            epi_ood_all.append(epi_o)
        epi_ood_pool = np.concatenate(epi_ood_all)
        sel = rng.choice(len(epi_ood_pool), size=n_clean, replace=False)
        uar = routing_auc(epi_c, epi_ood_pool, sel, amb, n_clean)
        out["K_sensitivity"][f"K{K}"] = {"shift_auroc_avg": float(np.mean(aucs)),
                                         "uar_amb_auc": uar}

    # ---- UAR AUC + paired bootstrap CI (K=5 epistemic vs single maxprob) ----
    _, _, _, epi5 = decomp_K(memb_clean, Kfull)
    epi_ood = np.concatenate([decomp_K(e["member"], Kfull)[3] for e, _ in ood_members])
    smx_ood = np.concatenate([1 - sg["maxprob"] for _, sg in ood_members])
    smx_clean = 1 - cs["maxprob"]
    sel = rng.choice(len(epi_ood), size=n_clean, replace=False)
    base_epi = routing_auc(epi5, epi_ood, sel, amb, n_clean)
    base_smx = routing_auc(smx_clean, smx_ood, sel, amb, n_clean)
    # paired bootstrap over samples (resample clean indices + OOD selection)
    diffs = []
    for _ in range(1000):
        ci = rng.integers(0, n_clean, n_clean)
        sel_b = rng.choice(len(epi_ood), size=n_clean, replace=False)
        de = routing_auc(epi5[ci], epi_ood, sel_b, amb[ci], n_clean)
        ds = routing_auc(smx_clean[ci], smx_ood, sel_b, amb[ci], n_clean)
        diffs.append(de - ds)
    diffs = np.array(diffs)
    out["uar"] = {"auc_epistemic": base_epi, "auc_single_maxprob": base_smx,
                  "diff": base_epi - base_smx,
                  "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                  "significant": bool(np.percentile(diffs, 2.5) > 0)}

    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print("K-sensitivity (shift-detection AUROC avg, UAR ambiguous-retention AUC):")
    for k, v in out["K_sensitivity"].items():
        print(f"  {k}: shift_auroc={v['shift_auroc_avg']:.3f}  uar_auc={v['uar_amb_auc']:.3f}")
    u = out["uar"]
    print(f"UAR AUC: epistemic={u['auc_epistemic']:.3f} vs single_maxprob={u['auc_single_maxprob']:.3f}")
    print(f"  diff={u['diff']:.3f} CI95={[round(x,3) for x in u['diff_ci95']]} significant={u['significant']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
