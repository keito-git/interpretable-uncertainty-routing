"""
shift_eval.py - Does a/e disentanglement beat 1-D confidence under shift?

Consumes a set of predictions.npz produced by ensemble_infer for the CLEAN
test and for shifted inputs (corruptions at severities, and/or cross-dataset),
then evaluates the regime where 1-D confidence is known to fail.

Analyses
--------
1. Shift/OOD detection: clean (label 0) vs shifted (label 1). AUROC for each
   scorer: epistemic, aleatoric, total, 1-maxprob. Claim: epistemic (ensemble
   disagreement) detects shift better than 1-maxprob.
2. Calibration under shift: ECE vs severity, ensemble vs single model.
3. Selective prediction under shift: on clean+shifted pool, risk-coverage AURC
   using epistemic-guided vs maxprob-guided abstention.
4. Accuracy degradation vs severity.

CLI
---
    python -m python.eval.shift_eval --clean data/shift_ens/clean \
        --shifted data/shift_ens/blur_s3,data/shift_ens/gnoise_s3,... \
        --single data/shift_single/clean,data/shift_single/blur_s3,... \
        --out data/analysis_shift
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from python.eval.metrics import _auroc, expected_calibration_error, aurc


def _entropy(p, eps=1e-12):
    pc = np.clip(p, eps, 1.0)
    return -np.sum(pc * np.log(pc), axis=-1)


def _load(d):
    p = np.load(Path(d) / "predictions.npz")
    s = json.load(open(Path(d) / "summary.json"))
    return p, s


def detection_auroc(clean_score, shifted_score):
    scores = np.concatenate([clean_score, shifted_score])
    labels = np.concatenate([np.zeros(len(clean_score)), np.ones(len(shifted_score))])
    return _auroc(scores, labels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=Path, required=True, help="ensemble clean dir")
    ap.add_argument("--shifted", type=str, required=True, help="comma-separated ensemble shifted dirs")
    ap.add_argument("--single-clean", type=Path, default=None, help="single-model clean dir")
    ap.add_argument("--single-shifted", type=str, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    clean, cmeta = _load(args.clean)
    shifted_dirs = [d for d in args.shifted.split(",") if d]

    result = {"clean": cmeta, "detection": {}, "calibration": {}, "selective": {}, "accuracy": {}}

    # scorers on clean
    clean_scorers = {
        "epistemic": clean["epistemic"], "aleatoric": clean["aleatoric"],
        "total": clean["total"], "one_minus_maxprob": 1.0 - clean["maxprob"],
    }
    single_clean = None
    if args.single_clean is not None:
        sc, _ = _load(args.single_clean)
        single_clean = sc
        clean_scorers["single_maxprob_baseline"] = 1.0 - sc["maxprob"]

    single_shifted_map = {}
    if args.single_shifted:
        for d in args.single_shifted.split(","):
            if d:
                ss, sm = _load(d)
                single_shifted_map[f"{sm['corruption']}_s{sm['severity']}"] = (ss, sm)

    for d in shifted_dirs:
        sh, sm = _load(d)
        tag = f"{sm['corruption']}_s{sm['severity']}"
        # 1. detection AUROC (clean vs this shifted)
        det = {}
        for name, cscore in clean_scorers.items():
            if name == "single_maxprob_baseline":
                if tag in single_shifted_map:
                    sscore = 1.0 - single_shifted_map[tag][0]["maxprob"]
                else:
                    continue
            else:
                sscore = {"epistemic": sh["epistemic"], "aleatoric": sh["aleatoric"],
                          "total": sh["total"], "one_minus_maxprob": 1.0 - sh["maxprob"]}[name]
            det[name] = detection_auroc(cscore, sscore)
        result["detection"][tag] = det
        # 4. accuracy + 2. ECE under shift (ensemble vs single)
        if "hard" in sh.files:
            hard = sh["hard"]
            result["accuracy"][tag] = {
                "ensemble_acc": float((sh["mean_prob"].argmax(1) == hard).mean()),
                "ensemble_ece": expected_calibration_error(sh["mean_prob"], hard),
            }
            if tag in single_shifted_map:
                ss = single_shifted_map[tag][0]
                result["accuracy"][tag]["single_acc"] = float((ss["mean_prob"].argmax(1) == hard).mean())
                result["accuracy"][tag]["single_ece"] = expected_calibration_error(ss["mean_prob"], hard)

    # 3. selective prediction on clean+all-shifted pool (ensemble): epistemic vs maxprob abstention
    pools = [clean]
    for d in shifted_dirs:
        pools.append(_load(d)[0])
    if all("hard" in p.files for p in pools):
        mp = np.concatenate([p["mean_prob"] for p in pools], 0)
        hard = np.concatenate([p["hard"] for p in pools], 0)
        correct = (mp.argmax(1) == hard).astype(np.float64)
        epi = np.concatenate([p["epistemic"] for p in pools], 0)
        mxp = np.concatenate([p["maxprob"] for p in pools], 0)
        tot = np.concatenate([p["total"] for p in pools], 0)
        result["selective"]["pool_n"] = int(len(hard))
        result["selective"]["aurc_epistemic"] = aurc(-epi, correct)
        result["selective"]["aurc_maxprob"] = aurc(mxp, correct)
        result["selective"]["aurc_total"] = aurc(-tot, correct)
        if single_clean is not None and single_shifted_map:
            # single model pool maxprob
            try:
                smp = [single_clean["maxprob"]]
                for d in shifted_dirs:
                    sh, sm = _load(d); tag = f"{sm['corruption']}_s{sm['severity']}"
                    smp.append(single_shifted_map[tag][0]["maxprob"])
                smp = np.concatenate(smp, 0)
                scorr = np.concatenate([(single_clean["mean_prob"].argmax(1) ==
                                         np.concatenate([p["hard"] for p in [clean]],0))] +
                                       [(single_shifted_map[f"{_load(d)[1]['corruption']}_s{_load(d)[1]['severity']}"][0]["mean_prob"].argmax(1)
                                         == _load(d)[0]["hard"]) for d in shifted_dirs], 0).astype(np.float64)
                result["selective"]["aurc_single_maxprob_baseline"] = aurc(smp, scorr)
            except Exception as e:
                result["selective"]["single_baseline_note"] = repr(e)[:150]

    json.dump(result, open(args.out / "shift_result.json", "w"), ensure_ascii=False, indent=2)
    # compact print
    print("=== Shift/OOD detection AUROC (clean vs shifted) ===")
    for tag, det in result["detection"].items():
        line = "  " + tag.ljust(14)
        for k in ["epistemic", "one_minus_maxprob", "single_maxprob_baseline", "aleatoric"]:
            if k in det:
                line += f"  {k[:9]}={det[k]:.3f}"
        print(line)
    if result["selective"]:
        print("=== Selective prediction on clean+shift pool (AURC, lower=better) ===")
        s = result["selective"]
        for k in ["aurc_epistemic", "aurc_maxprob", "aurc_total", "aurc_single_maxprob_baseline"]:
            if k in s:
                print(f"  {k} = {s[k]:.4f}")
    print("=== Accuracy/ECE under shift ===")
    for tag, a in result["accuracy"].items():
        print(f"  {tag}: ens_acc={a.get('ensemble_acc',0):.3f} ens_ece={a.get('ensemble_ece',0):.3f}"
              + (f" single_acc={a['single_acc']:.3f} single_ece={a['single_ece']:.3f}" if 'single_acc' in a else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
