"""broad_curves.py - OOD-detection AUROC (epistemic vs single max-prob) for the
6 new corruptions, merged with the original 5 into an 11-corruption table.
single baseline = ensemble member[0] (= ckpt s42, the original single model).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from python.eval.metrics import _auroc

EPS = 1e-12
NEW = ["shot_noise", "impulse_noise", "motion_blur", "contrast", "jpeg", "elastic"]
OLD = ["gnoise", "blur", "occlude", "bright", "pixelate"]
SEV = [1, 2, 3, 4, 5]


def ent(p):
    pc = np.clip(p, EPS, 1.0)
    return -np.sum(pc * np.log(pc), axis=-1)


def main():
    clean = np.load("data/shift_ens/none_s0/predictions.npz")
    csingle = np.load("data/shift_single/none_s0/predictions.npz")
    epi_c = clean["epistemic"]
    smx_c = 1 - csingle["maxprob"]
    n = len(epi_c)

    def auroc(ood, cl):
        lab = np.concatenate([np.zeros(len(cl)), np.ones(len(ood))])
        return float(_auroc(np.concatenate([cl, ood]), lab))

    curves = {"epistemic": {}, "single_maxprob": {}}
    # NEW corruptions from shift_ens_ext (single = member[0])
    for c in NEW:
        curves["epistemic"][c] = {}; curves["single_maxprob"][c] = {}
        for s in SEV:
            d = np.load(f"data/shift_ens_ext/{c}_s{s}/predictions.npz")
            epi_o = d["epistemic"]; smx_o = 1 - d["member"][0].max(1)
            curves["epistemic"][c][f"s{s}"] = round(auroc(epi_o, epi_c), 4)
            curves["single_maxprob"][c][f"s{s}"] = round(auroc(smx_o, smx_c), 4)
    # OLD corruptions from shift_ens / shift_single
    for c in OLD:
        curves["epistemic"][c] = {}; curves["single_maxprob"][c] = {}
        for s in SEV:
            e = np.load(f"data/shift_ens/{c}_s{s}/predictions.npz")
            sg = np.load(f"data/shift_single/{c}_s{s}/predictions.npz")
            curves["epistemic"][c][f"s{s}"] = round(auroc(e["epistemic"], epi_c), 4)
            curves["single_maxprob"][c][f"s{s}"] = round(auroc(1 - sg["maxprob"], smx_c), 4)

    # aggregate over all 11 corruptions x 5 severities
    all_corr = NEW + OLD
    epi_all = [curves["epistemic"][c][f"s{s}"] for c in all_corr for s in SEV]
    smx_all = [curves["single_maxprob"][c][f"s{s}"] for c in all_corr for s in SEV]
    # per-corruption mean (epistemic)
    epi_percorr = {c: round(float(np.mean([curves["epistemic"][c][f"s{s}"] for s in SEV])), 4)
                   for c in all_corr}
    smx_percorr = {c: round(float(np.mean([curves["single_maxprob"][c][f"s{s}"] for s in SEV])), 4)
                   for c in all_corr}
    win = sum(1 for c in all_corr if epi_percorr[c] > smx_percorr[c])

    out = {
        "n_corruptions": len(all_corr),
        "curves": curves,
        "epistemic_mean_per_corruption": epi_percorr,
        "single_maxprob_mean_per_corruption": smx_percorr,
        "epistemic_beats_single_count": f"{win}/{len(all_corr)}",
        "aggregate_epistemic_mean_AUROC": round(float(np.mean(epi_all)), 4),
        "aggregate_single_maxprob_mean_AUROC": round(float(np.mean(smx_all)), 4),
    }
    json.dump(out, open("data/analysis_shift/broad_curves.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps({"epistemic_mean_per_corruption": epi_percorr,
                      "single_maxprob_mean_per_corruption": smx_percorr,
                      "epistemic_beats_single_count": out["epistemic_beats_single_count"],
                      "aggregate_epistemic": out["aggregate_epistemic_mean_AUROC"],
                      "aggregate_single": out["aggregate_single_maxprob_mean_AUROC"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
