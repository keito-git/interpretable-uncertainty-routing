# Interpretable Uncertainty Routing for Facial Expression Recognition

Code for the paper *"Interpretable Uncertainty Routing: Separating Emotion
Ambiguity from Distribution Shift in Facial Expression Recognition"*.

This repository is released for reproducibility. Model weights are **not**
included; all results can be regenerated from the scripts below using the
public datasets.

## Overview

The method decomposes predictive uncertainty into **aleatoric** (emotion
ambiguity / annotator disagreement) and **epistemic** (distribution shift)
components using a Deep Ensemble of fully fine-tuned DINOv2 (ViT-B/14) models,
validates each component against an independent external signal (annotator
disagreement and distribution shift), and turns the separation into an
inference-time **Uncertainty-Aware Routing (UAR)** mechanism with a learned
variant (**L-UAR**). The 2-D `(H_ale, H_epi)` decision space is partitioned into
Accept / Defer / Reject regions and presented in a human-interpretable form.

## Repository structure

```
src/
  data/    dataset acquisition & preprocessing (FERPlus, RAF-DB, feature extraction)
  train/   backbone fine-tuning, deep ensemble, LDL / SCN / EAC baselines
  eval/    uncertainty decomposition, dual validation, shift detection,
           UAR / L-UAR routing, MMD, natural-OOD, broad-corruption sweeps
  models/  uncertainty heads (cSG-MCMC, spatial aleatoric) -- exploratory
  utils/   helpers
scripts/   shell entry points (run from the repository root)
```

## Requirements

```
pip install -r requirements.txt
```
Python 3.10+, a CUDA GPU is recommended (experiments were run on a single
NVIDIA H100). See `requirements.txt` for package versions.

## Data

- **FERPlus** (FER2013 with 10-annotator crowd votes): used for training and for
  aleatoric validation against annotator disagreement.
- **RAF-DB**: used as a cross-dataset shift probe.

Both are publicly available. `src/data/` contains the acquisition and
preprocessing scripts (votes -> soft/hard labels, grayscale matching, feature
extraction). Run them first to populate `data/`.

## Reproduction

All scripts are run from the repository root and read/write under `data/`.

```bash
# 1. Build datasets and fine-tune the K=5 deep ensemble + baselines
bash scripts/run_dino.sh

# 2. Distribution-shift inference (corruptions + cross-dataset) and evaluation
bash scripts/run_shift.sh
bash scripts/run_sevsweep.sh

# 3. Routing / decomposition analyses
python -m src.eval.p0_analysis        # conditional separation + UAR
python -m src.eval.p0a_analysis       # UAR AUC CI + K-sensitivity
python -m src.eval.learned_uar        # L-UAR (held-out corruptions)
python -m src.eval.loco_uar           # leave-one-corruption-out generalization
python -m src.eval.broad_corruptions  # 11-corruption suite (ImageNet-C style)
python -m src.eval.broad_curves       # OOD-detection AUROC over 11 corruptions
python -m src.eval.routing11          # per-corruption routing over 11 corruptions
python -m src.eval.natural_ood        # natural OOD (non-face) detection
python -m src.eval.mmd_rafdb          # feature-space MMD (RAF-DB vs corruption)
```

(Module paths assume the package is importable as `src`; adjust `PYTHONPATH` or
the `python -m` invocation to match your layout.)


```

## License

Released under the MIT License (see `LICENSE`).
