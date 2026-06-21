"""
csgmcmc_head.py - Cyclical SG-MCMC head + uncertainty decomposition.

Ported from the NLP/GoEmotions study (train_proposed.py) and made
*backbone-agnostic*: the head consumes pre-extracted features (e.g. a
frozen ViT [CLS] embedding, or per-patch token embeddings) instead of
running a text encoder. This lets us reuse the exact SGLD recipe while
swapping in a vision backbone.

Core pieces reused verbatim (domain-independent):
- cyclical_lr            : cosine cyclical step size (Zhang et al. ICLR 2020)
- is_sampling_phase      : exploration (SGD) vs sampling (SGLD) within a cycle
- compute_sample_steps   : where to snapshot the head inside the exploit phase
- sgld_step              : in-place SGLD update (Welling & Teh 2011)
- decompose_uncertainty  : total = aleatoric + epistemic (Depeweg et al. 2018)

GPU note: this module only DEFINES the recipe; running it over real ViT
features is the GPU step. It is unit-testable on CPU with random features.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:  # torch is only needed for the head/SGLD, not for decompose_uncertainty
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


# --------------------------------------------------------------------------
# Cyclical schedule (verbatim recipe from the NLP study)
# --------------------------------------------------------------------------

def cyclical_lr(step: int, *, cycle_length: int, alpha_0: float) -> float:
    """lr_t = (alpha_0/2) * (cos(pi * (step mod L)/L) + 1)."""
    if cycle_length <= 0:
        raise ValueError("cycle_length must be > 0")
    if alpha_0 <= 0:
        raise ValueError("alpha_0 must be > 0")
    u = step % cycle_length
    return 0.5 * alpha_0 * (math.cos(math.pi * (u / cycle_length)) + 1.0)


def is_sampling_phase(step: int, *, cycle_length: int,
                      exploitation_fraction: float) -> bool:
    if not 0.0 < exploitation_fraction < 1.0:
        raise ValueError("exploitation_fraction must be in (0,1)")
    u = step % cycle_length
    return u >= cycle_length * (1.0 - exploitation_fraction)


def compute_sample_steps(cycle_length: int, samples_per_cycle: int,
                         exploitation_fraction: float) -> list[int]:
    if samples_per_cycle < 1:
        raise ValueError("samples_per_cycle must be >= 1")
    start = int(cycle_length * (1.0 - exploitation_fraction))
    end = cycle_length - 1
    span = max(end - start, 1)
    return [start + int(round(span * (i + 1) / (samples_per_cycle + 1)))
            for i in range(samples_per_cycle)]


# --------------------------------------------------------------------------
# Uncertainty decomposition (pure NumPy; no torch needed)
# --------------------------------------------------------------------------

def decompose_uncertainty(member_probs: np.ndarray) -> dict[str, np.ndarray]:
    """Decompose predictive uncertainty from posterior member predictions.

    member_probs : (M, N, C) softmax outputs of M posterior samples.
    Returns per-example total / aleatoric / epistemic entropy arrays (N,).

    total      = H[ mean_m p_m ]
    aleatoric  = mean_m H[ p_m ]
    epistemic  = total - aleatoric   (= mutual information I[y; theta | x])
    (Depeweg et al. ICML 2018; identical to the NLP study.)
    """
    eps = 1e-12
    M = member_probs.shape[0]
    if M < 1:
        raise ValueError("need at least one posterior sample")
    mean_p = member_probs.mean(axis=0)
    mean_p = mean_p / np.clip(mean_p.sum(axis=-1, keepdims=True), eps, None)

    def _H(p):
        pc = np.clip(p, eps, 1.0)
        return -np.sum(pc * np.log(pc), axis=-1)

    total = _H(mean_p)
    aleatoric = np.stack([_H(member_probs[m]) for m in range(M)], axis=0).mean(axis=0)
    epistemic = np.maximum(total - aleatoric, 0.0)
    return {"mean_prob": mean_p, "total": total,
            "aleatoric": aleatoric, "epistemic": epistemic}


# --------------------------------------------------------------------------
# SGLD step + head (torch)
# --------------------------------------------------------------------------

if _HAS_TORCH:

    def sgld_step(params, *, lr: float, temperature: float = 1.0,
                  weight_decay: float = 0.0, posterior_scale: float = 1.0,
                  generator: Optional["torch.Generator"] = None) -> None:
        """In-place SGLD update (Welling & Teh 2011 / Zhang et al. 2020 §3.1).

        theta <- theta - lr*(posterior_scale*grad + wd*theta)
                       + sqrt(2*lr*temperature) * N(0, I)

        posterior_scale should be the training-set size N (review-robust:
        targets the true posterior, not a tempered/MAP target).
        """
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        noise_scale = (2.0 * lr * temperature) ** 0.5
        for p in params:
            if p.grad is None:
                continue
            update = posterior_scale * p.grad.detach()
            if weight_decay > 0:
                update = update + weight_decay * p.data
            p.data.add_(update, alpha=-lr)
            if temperature > 0:
                if generator is not None:
                    noise = torch.empty_like(p.data).normal_(0.0, 1.0, generator=generator)
                else:
                    noise = torch.randn_like(p.data)
                p.data.add_(noise, alpha=noise_scale)

    class LinearHead(nn.Module):
        """Single linear classifier over pre-extracted features.

        Operates on either a pooled feature (N, D) -> (N, C), or per-patch
        features (N, P, D) -> (N, P, C) when ``per_token=True`` is passed at
        call time. The same weights drive both, so the posterior over this
        head induces both an image-level and a spatial (per-patch) predictive
        distribution -> the basis of the spatial-aleatoric map.
        """

        def __init__(self, in_dim: int, num_classes: int, dropout_p: float = 0.0):
            super().__init__()
            self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
            self.classifier = nn.Linear(in_dim, num_classes)

        def forward(self, feats: "torch.Tensor") -> "torch.Tensor":
            return self.classifier(self.dropout(feats))

    def soft_label_loss(logits: "torch.Tensor", soft_targets: "torch.Tensor") -> "torch.Tensor":
        """KL(votes || pred) up to a constant == cross-entropy with soft
        targets. Matches the NLP study's soft-label objective."""
        logp = F.log_softmax(logits, dim=-1)
        return -(soft_targets * logp).sum(dim=-1).mean()
