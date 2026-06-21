"""
p0_analysis.py - Critical (P0) revision analyses for the reviewer round.

All computed from existing ensemble/single predictions (no retraining).

P0-3 Statistical significance:
  - Spearman(aleatoric, human disagreement) and (epistemic, .) with bootstrap 95% CI.
  - OOD-detection AUROC per severity for {epistemic, aleatoric, 1-maxprob(ens),
    1-maxprob(single)} with bootstrap CI on the paired difference (epi - single maxprob).

P0-2 Conditional separation (answers the "both rise" self-contradiction):
  - which component WINS at which task: aleatoric should win disagreement detection
    (in-dist), epistemic should win shift detection. Report the 2x2.

P0-1 Uncertainty-Aware Routing (UAR) -- the methodological contribution:
  Task: serve in-distribution faces (INCLUDING genuinely ambiguous ones) while
  rejecting out-of-distribution inputs. A 1-D confidence rule rejects ambiguous
  in-dist faces (low maxprob) together with OOD; an epistemic rule rejects OOD
  while keeping ambiguous in-dist (high aleatoric, low epistemic). We measure, at
  a matched OOD-rejection rate, the retention of in-dist and especially of
  AMBIGUOUS in-dist faces.

CLI
---
    python -m python.eval.p0_analysis --ens-dir data/shift_ens \
        --single-dir data/shift_single --votes data/processed/ferplus_test.npz \
        --out data/analysis_shift/p0.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from python.eval.metrics import _auroc, spearman_rho

CORR = ["gnoise", "blur", "occlude", "bright", "pixelate"]
SEV = [1, 2, 3, 4, 5]
_EPS = 1e-12


def _ent(p):
    pc = np.clip(p, _EPS, 1.0)
    return -np.sum(pc * np.log(pc), axis=-1)


def _load(d):
    p = Path(d) / "predictions.npz"
    return np.load(p) if p.exists() else None


def boot_ci_spearman(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    base = spearman_rho(x, y)
    vals = []
    m = len(x)
    for _ in range(n):
        idx = rng.integers(0, m, m)
        vals.append(spearman_rho(x[idx], y[idx]))
    vals = np.array([v for v in vals if not np.isnan(v)])
    return base, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_auroc_diff(scoreA, scoreB, labels, n=2000, seed=0):
    """paired bootstrap CI of AUROC(scoreA)-AUROC(scoreB) on same labels."""
    rng = np.random.default_rng(seed)
    base = _auroc(scoreA, labels) - _auroc(scoreB, labels)
    m = len(labels)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        la = labels[idx]
        if la.min() == la.max():
            continue
        diffs.append(_auroc(scoreA[idx], la) - _auroc(scoreB[idx], la))
    diffs = np.array(diffs)
    return base, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens-dir", type=Path, default=Path("data/shift_ens"))
    ap.add_argument("--single-dir", type=Path, default=Path("data/shift_single"))
    ap.add_argument("--votes", type=Path, default=Path("data/processed/ferplus_test.npz"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis_shift/p0.json"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    meta = np.load(args.votes, allow_pickle=True)
    soft = meta["soft"].astype(np.float64); dis = meta["disagreement"].astype(np.float64)
    ve = _ent(soft)  # vote entropy
    ce = _load(args.ens_dir / "none_s0"); cs = _load(args.single_dir / "none_s0")
    n_clean = len(ce["epistemic"])
    dis = dis[:n_clean]; ve = ve[:n_clean]; soft = soft[:n_clean]

    out = {}

    # ---- P0-3a: Spearman with bootstrap CI (DINOv2 ensemble, clean) ----
    ale_c = ce["aleatoric"].astype(np.float64); epi_c = ce["epistemic"].astype(np.float64)
    r_ale, lo_ale, hi_ale = boot_ci_spearman(ale_c, dis)
    r_epi, lo_epi, hi_epi = boot_ci_spearman(epi_c, dis)
    out["spearman_ale_vs_disagreement"] = {"rho": r_ale, "ci95": [lo_ale, hi_ale]}
    out["spearman_epi_vs_disagreement"] = {"rho": r_epi, "ci95": [lo_epi, hi_epi]}

    # ---- P0-2: conditional separation 2x2 ----
    # task1: disagreement detection (in-dist), positive = top tertile vote entropy
    hi_t = np.quantile(ve, 2/3); lo_t = np.quantile(ve, 1/3)
    mask = (ve >= hi_t) | (ve <= lo_t); lab_dis = (ve[mask] >= hi_t).astype(int)
    add_ale = _auroc(ale_c[mask], lab_dis); add_epi = _auroc(epi_c[mask], lab_dis)
    out["disagreement_detection_AUROC"] = {"aleatoric": add_ale, "epistemic": add_epi}

    # task2: shift detection (severity-averaged), positive = corrupted
    def shift_auroc(score_clean_key):
        aucs = {"epistemic": [], "aleatoric": [], "ens_1mmaxprob": [], "single_1mmaxprob": []}
        for c in CORR:
            for s in SEV:
                e = _load(args.ens_dir / f"{c}_s{s}"); sg = _load(args.single_dir / f"{c}_s{s}")
                if e is None or sg is None:
                    continue
                lab = np.concatenate([np.zeros(n_clean), np.ones(len(e["epistemic"]))])
                aucs["epistemic"].append(_auroc(np.concatenate([epi_c, e["epistemic"]]), lab))
                aucs["aleatoric"].append(_auroc(np.concatenate([ale_c, e["aleatoric"]]), lab))
                aucs["ens_1mmaxprob"].append(_auroc(np.concatenate([1-ce["maxprob"], 1-e["maxprob"]]), lab))
                aucs["single_1mmaxprob"].append(_auroc(np.concatenate([1-cs["maxprob"], 1-sg["maxprob"]]), lab))
        return {k: float(np.mean(v)) for k, v in aucs.items()}
    out["shift_detection_AUROC_avg"] = shift_auroc(None)

    # ---- P0-3b: significance of epi vs single-maxprob shift detection at sev5 ----
    sig = {}
    for c in CORR:
        e = _load(args.ens_dir / f"{c}_s5"); sg = _load(args.single_dir / f"{c}_s5")
        if e is None or sg is None:
            continue
        lab = np.concatenate([np.zeros(n_clean), np.ones(len(e["epistemic"]))])
        sA = np.concatenate([epi_c, e["epistemic"]])
        sB = np.concatenate([1-cs["maxprob"], 1-sg["maxprob"]])
        d, lo, hi = boot_auroc_diff(sA, sB, lab)
        sig[f"{c}_s5"] = {"auroc_diff_epi_minus_singlemaxprob": d, "ci95": [lo, hi],
                          "significant": bool(lo > 0)}
    out["shift_detection_significance_sev5"] = sig

    # ---- P0-1: Uncertainty-Aware Routing ----
    # pool = clean (in-dist) + all corrupted (OOD). in-dist label 0, OOD label 1.
    ens_pool = [ce] + [_load(args.ens_dir / f"{c}_s{s}") for c in CORR for s in SEV]
    sng_pool = [cs] + [_load(args.single_dir / f"{c}_s{s}") for c in CORR for s in SEV]
    is_ood = np.concatenate([np.zeros(n_clean)] +
                            [np.ones(len(_load(args.ens_dir / f"{c}_s{s}")["epistemic"]))
                             for c in CORR for s in SEV])
    epi_pool = np.concatenate([p["epistemic"] for p in ens_pool])
    ens_mxp_pool = np.concatenate([p["maxprob"] for p in ens_pool])
    sng_mxp_pool = np.concatenate([p["maxprob"] for p in sng_pool])
    # ambiguous in-dist: clean samples with high human disagreement (top tertile)
    amb_indist = np.zeros(len(is_ood), dtype=bool)
    amb_indist[:n_clean] = ve >= hi_t
    indist = is_ood == 0

    def routing_curve(reject_score):
        # reject highest reject_score first; sweep threshold to match OOD-rejection
        order = np.argsort(-reject_score)
        rejected = np.zeros(len(reject_score), dtype=bool)
        rows = []
        n_ood = int(is_ood.sum()); n_amb = int(amb_indist.sum()); n_ind = int(indist.sum())
        step = max(len(order)//200, 1)
        for k in range(0, len(order)+1, step):
            rejected[:] = False; rejected[order[:k]] = True
            ood_rej = (rejected & (is_ood == 1)).sum()/max(n_ood,1)
            amb_ret = (~rejected & amb_indist).sum()/max(n_amb,1)
            ind_ret = (~rejected & indist).sum()/max(n_ind,1)
            rows.append((ood_rej, ind_ret, amb_ret))
        return np.array(rows)
    rc_epi = routing_curve(epi_pool)
    rc_ensmxp = routing_curve(1-ens_mxp_pool)
    rc_sngmxp = routing_curve(1-sng_mxp_pool)

    def retention_at(curve, target_ood=0.8):
        # interpolate in-dist & ambiguous-in-dist retention at target OOD rejection
        o = curve[:,0]; ind = curve[:,1]; amb = curve[:,2]
        i = np.argmin(np.abs(o-target_ood))
        return float(ind[i]), float(amb[i]), float(o[i])
    out["routing"] = {}
    for tgt in [0.7, 0.8, 0.9]:
        ie, ae, oe = retention_at(rc_epi, tgt)
        im, am, om = retention_at(rc_ensmxp, tgt)
        isg, asg, osg = retention_at(rc_sngmxp, tgt)
        out["routing"][f"ood_reject_{tgt}"] = {
            "epistemic": {"indist_retention": ie, "ambiguous_indist_retention": ae, "actual_ood_rej": oe},
            "ens_maxprob": {"indist_retention": im, "ambiguous_indist_retention": am, "actual_ood_rej": om},
            "single_maxprob": {"indist_retention": isg, "ambiguous_indist_retention": asg, "actual_ood_rej": osg},
        }
    out["routing"]["pool"] = {"n_clean_indist": int(n_clean),
                              "n_ambiguous_indist": int(amb_indist.sum()),
                              "n_ood": int(is_ood.sum())}

    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    # print
    print("P0-3 Spearman (DINOv2 ensemble, clean):")
    print(f"  ale vs disagreement: rho={r_ale:.3f} CI{[round(lo_ale,3),round(hi_ale,3)]}")
    print(f"  epi vs disagreement: rho={r_epi:.3f} CI{[round(lo_epi,3),round(hi_epi,3)]}")
    print("P0-2 conditional separation:")
    print(f"  disagreement-detection AUROC: ale={add_ale:.3f}  epi={add_epi:.3f}  (ale wins)")
    sd = out["shift_detection_AUROC_avg"]
    print(f"  shift-detection AUROC(avg): epi={sd['epistemic']:.3f} ale={sd['aleatoric']:.3f} "
          f"ens_maxprob={sd['ens_1mmaxprob']:.3f} single_maxprob={sd['single_1mmaxprob']:.3f}")
    print("P0-3 significance (epi - single maxprob, sev5):")
    for k, v in sig.items():
        print(f"  {k}: diff={v['auroc_diff_epi_minus_singlemaxprob']:.3f} CI{[round(v['ci95'][0],3),round(v['ci95'][1],3)]} sig={v['significant']}")
    print("P0-1 UAR routing (retention at OOD-rejection target):")
    for tgt, r in out["routing"].items():
        if tgt == "pool":
            continue
        print(f"  {tgt}: epi amb_ret={r['epistemic']['ambiguous_indist_retention']:.3f} "
              f"vs single_maxprob amb_ret={r['single_maxprob']['ambiguous_indist_retention']:.3f} "
              f"(indist_ret epi={r['epistemic']['indist_retention']:.3f} vs {r['single_maxprob']['indist_retention']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
