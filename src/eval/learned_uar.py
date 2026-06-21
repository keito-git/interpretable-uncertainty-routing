"""
learned_uar.py - Learned Uncertainty-Aware Routing (L-UAR) + held-out shift.

Upgrades UAR from a two-threshold rule to a LEARNED router on the ensemble's
uncertainty features, and evaluates GENERALIZATION to UNSEEN corruption types.
This defeats the "UAR is just an if-else" critique and adds a held-out-shift
robustness result, all from saved member predictions (no re-inference).

Feature per sample (from the K-member ensemble):
  [H_tot, H_ale, H_epi, mean_maxprob, mean_top2gap, std_member_maxprob,
   std_member_entropy]  -- richer than (H_ale,H_epi) so a learned boundary can
   beat axis-aligned thresholds.

Protocol (honest, no leakage of test corruption type):
  TRAIN reject-classifier on: clean (in-dist, label 0) + {gnoise,blur,occlude}
     all severities (OOD, label 1).
  TEST on: clean + {bright,pixelate} (UNSEEN corruption types).
  Compare reject scores: L-UAR P(OOD)  vs  epistemic (threshold-UAR)  vs
     single-model 1-maxprob, by the ambiguous-in-dist-retention vs OOD-rejection
     AUC on the held-out test pool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression

_EPS = 1e-12
TRAIN_CORR = ["gnoise", "blur", "occlude"]
TEST_CORR = ["bright", "pixelate"]
SEV = [1, 2, 3, 4, 5]


def _ent(p):
    pc = np.clip(p, _EPS, 1.0)
    return -np.sum(pc * np.log(pc), axis=-1)


def _load(d):
    p = Path(d) / "predictions.npz"
    return np.load(p) if p.exists() else None


def feats(pred):
    """per-sample uncertainty feature matrix from ensemble member probs."""
    m = pred["member"]  # (K,N,C)
    mean_p = pred["mean_prob"]
    tot = _ent(mean_p)
    ale = pred["aleatoric"]; epi = pred["epistemic"]
    maxp = mean_p.max(1)
    srt = np.sort(mean_p, axis=1)
    top2gap = srt[:, -1] - srt[:, -2]
    mem_maxp = m.max(2)               # (K,N)
    std_mem_maxp = mem_maxp.std(0)
    mem_ent = np.stack([_ent(m[k]) for k in range(m.shape[0])])  # (K,N)
    std_mem_ent = mem_ent.std(0)
    return np.stack([tot, ale, epi, maxp, top2gap, std_mem_maxp, std_mem_ent], axis=1)


def routing_auc(reject_score, is_ood, amb, n_clean):
    order = np.argsort(-reject_score)
    rej = np.zeros(len(reject_score), bool)
    rows = []
    na = max(int(amb.sum()), 1)
    step = max(len(order) // 300, 1)
    for k in range(0, len(order) + 1, step):
        rej[:] = False; rej[order[:k]] = True
        rows.append(((rej & (is_ood == 1)).sum() / max((is_ood == 1).sum(), 1),
                     (~rej & amb).sum() / na))
    rows = np.array(rows); idx = np.argsort(rows[:, 0])
    return float(np.trapz(rows[idx, 1], rows[idx, 0]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens-dir", type=Path, default=Path("data/shift_ens"))
    ap.add_argument("--single-dir", type=Path, default=Path("data/shift_single"))
    ap.add_argument("--votes", type=Path, default=Path("data/processed/ferplus_test.npz"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis_shift/learned_uar.json"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    meta = np.load(args.votes, allow_pickle=True)
    ce = _load(args.ens_dir / "none_s0"); cs = _load(args.single_dir / "none_s0")
    n_clean = len(ce["epistemic"])
    ve = _ent(meta["soft"].astype(np.float64))[:n_clean]
    hi = np.quantile(ve, 2/3); amb_clean = ve >= hi

    Fc = feats(ce)
    # ---- TRAIN reject classifier: clean(0) + train-corruptions(1) ----
    Xtr = [Fc]; ytr = [np.zeros(n_clean)]
    for c in TRAIN_CORR:
        for s in SEV:
            e = _load(args.ens_dir / f"{c}_s{s}")
            Xtr.append(feats(e)); ytr.append(np.ones(len(e["epistemic"])))
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit((Xtr - mu) / sd, ytr)

    # ---- TEST pool: clean + held-out corruptions ----
    def pred_reject_scores(conds):
        # returns per-condition (luar P(ood), epistemic, single 1-maxprob, n, is the clean block)
        blocks = []
        # clean block first
        blocks.append(("clean", clf.predict_proba((Fc - mu) / sd)[:, 1], ce["epistemic"],
                       1 - cs["maxprob"], n_clean))
        for c in conds:
            for s in SEV:
                e = _load(args.ens_dir / f"{c}_s{s}"); sg = _load(args.single_dir / f"{c}_s{s}")
                blocks.append((f"{c}_s{s}", clf.predict_proba((feats(e) - mu) / sd)[:, 1],
                               e["epistemic"], 1 - sg["maxprob"], len(e["epistemic"])))
        return blocks

    blocks = pred_reject_scores(TEST_CORR)
    luar = np.concatenate([b[1] for b in blocks])
    epi = np.concatenate([b[2] for b in blocks])
    smx = np.concatenate([b[3] for b in blocks])
    is_ood = np.concatenate([np.zeros(blocks[0][4])] + [np.ones(b[4]) for b in blocks[1:]])
    amb = np.concatenate([amb_clean] + [np.zeros(b[4], bool) for b in blocks[1:]])

    auc_luar = routing_auc(luar, is_ood, amb, n_clean)
    auc_epi = routing_auc(epi, is_ood, amb, n_clean)
    auc_smx = routing_auc(smx, is_ood, amb, n_clean)
    # paired bootstrap CI on (luar - epi) and (luar - smx) over clean indices
    def boot(a, b):
        ds = []
        for _ in range(800):
            ci = rng.integers(0, n_clean, n_clean)
            # resample only clean block (in-dist + ambiguous) for the retention part
            la = np.concatenate([a[:n_clean][ci], a[n_clean:]])
            lb = np.concatenate([b[:n_clean][ci], b[n_clean:]])
            io = np.concatenate([np.zeros(n_clean), is_ood[n_clean:]])
            am = np.concatenate([amb[:n_clean][ci], amb[n_clean:]])
            ds.append(routing_auc(la, io, am, n_clean) - routing_auc(lb, io, am, n_clean))
        return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))
    lo1, hi1 = boot(luar, epi); lo2, hi2 = boot(luar, smx)

    out = {
        "protocol": {"train_corruptions": TRAIN_CORR, "test_corruptions(unseen)": TEST_CORR},
        "held_out_routing_auc": {"L-UAR(learned)": auc_luar, "threshold-UAR(epistemic)": auc_epi,
                                 "single-maxprob": auc_smx},
        "L-UAR_minus_threshold_UAR_ci95": [lo1, hi1],
        "L-UAR_minus_single_maxprob_ci95": [lo2, hi2],
        "feature_weights": dict(zip(
            ["H_tot", "H_ale", "H_epi", "mean_maxprob", "top2gap", "std_mem_maxp", "std_mem_ent"],
            [float(w) for w in clf.coef_[0]])),
    }
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"Held-out unseen corruptions {TEST_CORR} (trained on {TRAIN_CORR}):")
    print(f"  ambiguous-retention-vs-OOD-rejection AUC:")
    print(f"    L-UAR (learned)        = {auc_luar:.3f}")
    print(f"    threshold-UAR (epi)    = {auc_epi:.3f}")
    print(f"    single-maxprob         = {auc_smx:.3f}")
    print(f"  L-UAR - threshold-UAR CI95 = [{lo1:.3f}, {hi1:.3f}]")
    print(f"  L-UAR - single-maxprob CI95 = [{lo2:.3f}, {hi2:.3f}]")
    print("  feature weights:", {k: round(v,2) for k,v in out["feature_weights"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
