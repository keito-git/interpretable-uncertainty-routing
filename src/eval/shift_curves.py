"""
shift_curves.py - Severity curves for the distribution-shift story.

For each corruption and severity 0..5, computes (ensemble vs single):
- accuracy, ECE
- OOD-detection AUROC (clean vs this severity) for ensemble-epistemic and
  single-model (1-maxprob)
- mean aleatoric and mean epistemic (to show epistemic RISES with severity
  while aleatoric is comparatively flat -> disentanglement evidence)

Severity 0 maps to the shared clean run (none_s0).

CLI
---
    python -m python.eval.shift_curves --ens-dir data/shift_ens \
        --single-dir data/shift_single --out data/analysis_shift/curves.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from python.eval.metrics import _auroc, expected_calibration_error

CORRUPTIONS = ["gnoise", "blur", "occlude", "bright", "pixelate"]
SEVERITIES = [0, 1, 2, 3, 4, 5]


def _load(d):
    p = Path(d) / "predictions.npz"
    return np.load(p) if p.exists() else None


def _det(a, b):
    return _auroc(np.concatenate([a, b]), np.concatenate([np.zeros(len(a)), np.ones(len(b))]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens-dir", type=Path, default=Path("data/shift_ens"))
    ap.add_argument("--single-dir", type=Path, default=Path("data/shift_single"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis_shift/curves.json"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ce = _load(args.ens_dir / "none_s0"); cs = _load(args.single_dir / "none_s0")
    curves = {}
    for corr in CORRUPTIONS:
        rows = []
        for sev in SEVERITIES:
            tag = "none_s0" if sev == 0 else f"{corr}_s{sev}"
            e = _load(args.ens_dir / tag); s = _load(args.single_dir / tag)
            if e is None or s is None:
                continue
            row = {
                "severity": sev,
                "ens_acc": float((e["mean_prob"].argmax(1) == e["hard"]).mean()),
                "ens_ece": expected_calibration_error(e["mean_prob"], e["hard"]),
                "single_acc": float((s["mean_prob"].argmax(1) == s["hard"]).mean()),
                "single_ece": expected_calibration_error(s["mean_prob"], s["hard"]),
                "mean_aleatoric": float(e["aleatoric"].mean()),
                "mean_epistemic": float(e["epistemic"].mean()),
            }
            if sev == 0:
                row["ood_auroc_epistemic"] = 0.5
                row["ood_auroc_single_maxprob"] = 0.5
            else:
                row["ood_auroc_epistemic"] = _det(ce["epistemic"], e["epistemic"])
                row["ood_auroc_single_maxprob"] = _det(1 - cs["maxprob"], 1 - s["maxprob"])
            rows.append(row)
        curves[corr] = rows

    json.dump(curves, open(args.out, "w"), ensure_ascii=False, indent=2)

    # console: averaged-over-corruptions curve
    print("severity | ens_acc single_acc | ens_ece single_ece | OOD_epi OOD_single | mean_ale mean_epi")
    for sev in SEVERITIES:
        vals = [r for c in CORRUPTIONS for r in curves.get(c, []) if r["severity"] == sev]
        if not vals:
            continue
        def m(k):
            return np.mean([v[k] for v in vals])
        print(f"   s{sev}    |  {m('ens_acc'):.3f}   {m('single_acc'):.3f}  | "
              f"{m('ens_ece'):.3f}   {m('single_ece'):.3f}  | "
              f"{m('ood_auroc_epistemic'):.3f}   {m('ood_auroc_single_maxprob'):.3f}  | "
              f"{m('mean_aleatoric'):.3f}    {m('mean_epistemic'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
