"""
spatial_aleatoric.py - Spatial aleatoric/epistemic maps (the novelty pillar).

Idea
----
Text is a token sequence; a face image has 2-D spatial structure. Instead of
one ambiguity scalar per image, we ask *which facial regions generate the
ambiguity*. With a ViT backbone we apply the SAME cSG-MCMC posterior head to
every patch token, decompose uncertainty per patch, and reshape to the patch
grid -> a spatial aleatoric map and a spatial epistemic map.

This file is backbone-agnostic: it consumes per-patch member predictions
(M posterior samples x N images x P patches x C classes). Producing those
from real ViT features is the GPU step; the map math here is CPU/NumPy and
unit-testable.

Reviewer-defence helpers included:
- ``map_grid``            : reshape (P,) patch scores to (H, W) grid
- ``saliency_dissimilarity`` : 1 - |corr| between an aleatoric map and a
      class-saliency map (shows aleatoric != saliency)
- ``occlusion_delta``     : given a callable that re-predicts under a mask,
      compare JSD shift when masking high-aleatoric vs random patches
      (causal check that the highlighted regions really drive ambiguity)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from python.models.csgmcmc_head import decompose_uncertainty
from python.eval.metrics import jensen_shannon

_EPS = 1e-12


def patch_decomposition(member_probs_patch: np.ndarray) -> dict[str, np.ndarray]:
    """Per-patch a/e decomposition.

    member_probs_patch : (M, N, P, C)
    Returns total/aleatoric/epistemic each shaped (N, P).
    """
    M, N, P, C = member_probs_patch.shape
    flat = member_probs_patch.reshape(M, N * P, C)
    dec = decompose_uncertainty(flat)
    return {
        "total": dec["total"].reshape(N, P),
        "aleatoric": dec["aleatoric"].reshape(N, P),
        "epistemic": dec["epistemic"].reshape(N, P),
        "mean_prob": dec["mean_prob"].reshape(N, P, C),
    }


def map_grid(scores_p: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Reshape a (P,) patch-score vector to an (H, W) grid for heatmaps."""
    if scores_p.shape[-1] != grid_h * grid_w:
        raise ValueError(f"P={scores_p.shape[-1]} != {grid_h}x{grid_w}")
    return scores_p.reshape(grid_h, grid_w)


def normalize_map(m: np.ndarray) -> np.ndarray:
    lo, hi = float(m.min()), float(m.max())
    if hi - lo < _EPS:
        return np.zeros_like(m)
    return (m - lo) / (hi - lo)


def saliency_dissimilarity(aleatoric_map: np.ndarray,
                           saliency_map: np.ndarray) -> float:
    """1 - |Pearson corr| between two flattened maps.

    High value => aleatoric map is NOT the class-saliency map (it highlights
    ambiguity-driving regions, not class-discriminative regions).
    """
    a = aleatoric_map.ravel().astype(np.float64)
    s = saliency_map.ravel().astype(np.float64)
    a -= a.mean(); s -= s.mean()
    denom = np.sqrt((a ** 2).sum() * (s ** 2).sum())
    if denom < _EPS:
        return float("nan")
    return float(1.0 - abs((a * s).sum() / denom))


def occlusion_delta(image_idx: int,
                    aleatoric_patch: np.ndarray,
                    vote_distribution: np.ndarray,
                    repredict_under_mask: Callable[[int, np.ndarray], np.ndarray],
                    top_k: int,
                    rng: np.random.Generator) -> dict[str, float]:
    """Causal check: does masking high-aleatoric patches disturb the
    human-aligned prediction more than masking random patches?

    image_idx            : index of the image under test
    aleatoric_patch      : (P,) per-patch aleatoric for this image
    vote_distribution    : (C,) annotator vote distribution (soft label)
    repredict_under_mask : fn(image_idx, mask_bool_P) -> predicted (C,) dist
                           [this is the GPU-backed model call]
    top_k                : number of patches to mask

    Returns JSD(pred||votes) shift for high-aleatoric masking vs random.
    Larger ``high_minus_random`` => aleatoric regions genuinely drive the
    ambiguity the annotators disagreed over.
    """
    P = aleatoric_patch.shape[0]
    base_pred = repredict_under_mask(image_idx, np.zeros(P, dtype=bool))
    base_jsd = float(jensen_shannon(base_pred, vote_distribution, axis=-1))

    high_idx = np.argsort(-aleatoric_patch)[:top_k]
    high_mask = np.zeros(P, dtype=bool); high_mask[high_idx] = True
    high_pred = repredict_under_mask(image_idx, high_mask)
    high_jsd = float(jensen_shannon(high_pred, vote_distribution, axis=-1))

    rand_idx = rng.choice(P, size=top_k, replace=False)
    rand_mask = np.zeros(P, dtype=bool); rand_mask[rand_idx] = True
    rand_pred = repredict_under_mask(image_idx, rand_mask)
    rand_jsd = float(jensen_shannon(rand_pred, vote_distribution, axis=-1))

    return {
        "base_jsd": base_jsd,
        "high_aleatoric_jsd": high_jsd,
        "random_jsd": rand_jsd,
        "high_minus_base": high_jsd - base_jsd,
        "high_minus_random": high_jsd - rand_jsd,
    }
