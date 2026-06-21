"""loco_uar.py - Leave-One-Corruption-Out generalization for L-UAR.

Reviewer-robustness upgrade: the fixed split (train {gnoise,blur,occlude} ->
test {bright,pixelate}) can be called cherry-picked. Here we hold out EACH
corruption in turn, train the learned router on clean + the other four (all
severities), and evaluate routing on clean + the held-out corruption. Reports
L-UAR vs threshold-UAR(epistemic) vs single-maxprob routing AUC for every
held-out corruption. All from saved predictions (no re-inference).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-12
ALL = ["gnoise", "blur", "occlude", "bright", "pixelate"]
SEV = [1, 2, 3, 4, 5]
ENS = Path("data/shift_ens")
SGL = Path("data/shift_single")


def ent(p):
    pc = np.clip(p, EPS, 1.0)
    return -np.sum(pc * np.log(pc), axis=-1)


def load(d):
    p = d / "predictions.npz"
    return np.load(p) if p.exists() else None


def feats(pred):
    m = pred["member"]
    mean_p = pred["mean_prob"]
    tot = ent(mean_p)
    ale = pred["aleatoric"]; epi = pred["epistemic"]
    maxp = mean_p.max(1)
    srt = np.sort(mean_p, axis=1)
    top2gap = srt[:, -1] - srt[:, -2]
    mem_maxp = m.max(2)
    std_mem_maxp = mem_maxp.std(0)
    mem_ent = np.stack([ent(m[k]) for k in range(m.shape[0])])
    std_mem_ent = mem_ent.std(0)
    return np.stack([tot, ale, epi, maxp, top2gap, std_mem_maxp, std_mem_ent], axis=1)


def routing_auc(reject_score, is_ood, amb, n_clean):
    order = np.argsort(-reject_score)
    rej = np.zeros(len(reject_score), bool)
    rows = []
    na = max(int(amb.sum()), 1)
    no = max(int((is_ood == 1).sum()), 1)
    step = max(len(order) // 300, 1)
    for k in range(0, len(order) + 1, step):
        rej[:] = False; rej[order[:k]] = True
        rows.append(((rej & (is_ood == 1)).sum() / no, (~rej & amb).sum() / na))
    rows = np.array(rows); idx = np.argsort(rows[:, 0])
    return float(np.trapz(rows[idx, 1], rows[idx, 0]))


def main():
    meta = np.load("data/processed/ferplus_test.npz", allow_pickle=True)
    ce = load(ENS / "none_s0"); cs = load(SGL / "none_s0")
    n_clean = len(ce["epistemic"])
    ve = ent(meta["soft"].astype(np.float64))[:n_clean]
    amb_clean = ve >= np.quantile(ve, 2 / 3)
    Fc = feats(ce)

    out = {"protocol": "leave-one-corruption-out; train on clean + 4 corruptions (all sev), "
                       "test on clean + held-out corruption", "per_heldout": {}}
    luar_list, epi_list, smx_list = [], [], []
    for held in ALL:
        train_corr = [c for c in ALL if c != held]
        Xtr = [Fc]; ytr = [np.zeros(n_clean)]
        for c in train_corr:
            for s in SEV:
                e = load(ENS / f"{c}_s{s}")
                Xtr.append(feats(e)); ytr.append(np.ones(len(e["epistemic"])))
        Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit((Xtr - mu) / sd, ytr)

        # test pool: clean + held-out corruption (all severities)
        luar = [clf.predict_proba((Fc - mu) / sd)[:, 1]]
        epi = [ce["epistemic"]]; smx = [1 - cs["maxprob"]]
        is_ood = [np.zeros(n_clean)]; amb = [amb_clean]
        for s in SEV:
            e = load(ENS / f"{held}_s{s}"); sg = load(SGL / f"{held}_s{s}")
            luar.append(clf.predict_proba((feats(e) - mu) / sd)[:, 1])
            epi.append(e["epistemic"]); smx.append(1 - sg["maxprob"])
            is_ood.append(np.ones(len(e["epistemic"]))); amb.append(np.zeros(len(e["epistemic"]), bool))
        luar = np.concatenate(luar); epi = np.concatenate(epi); smx = np.concatenate(smx)
        is_ood = np.concatenate(is_ood); amb = np.concatenate(amb)
        a_l = routing_auc(luar, is_ood, amb, n_clean)
        a_e = routing_auc(epi, is_ood, amb, n_clean)
        a_s = routing_auc(smx, is_ood, amb, n_clean)
        out["per_heldout"][held] = {"L-UAR": round(a_l, 4), "threshold-UAR(epistemic)": round(a_e, 4),
                                    "single-maxprob": round(a_s, 4)}
        luar_list.append(a_l); epi_list.append(a_e); smx_list.append(a_s)

    out["mean_over_heldout"] = {
        "L-UAR": round(float(np.mean(luar_list)), 4),
        "threshold-UAR(epistemic)": round(float(np.mean(epi_list)), 4),
        "single-maxprob": round(float(np.mean(smx_list)), 4),
    }
    out["L-UAR_wins_vs_single_count"] = int(np.sum(np.array(luar_list) > np.array(smx_list)))
    out["L-UAR_wins_vs_threshold_count"] = int(np.sum(np.array(luar_list) > np.array(epi_list)))
    Path("data/analysis_shift").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("data/analysis_shift/loco_uar.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
