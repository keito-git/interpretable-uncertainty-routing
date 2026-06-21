"""
metrics.py - Uncertainty-quality metrics (pure NumPy, no GPU).

Implements the evaluation axes ported from the NLP/GoEmotions study to the
face domain:

- C1 calibration : ECE, Brier, NLL (vs hard-label argmax target)
- C2 distribution fidelity : Jensen-Shannon divergence (bits) and KL to the
      annotator vote distribution (soft label)
- C3 correlation : Spearman rho helper (per-category aleatoric vs delta^(c))
- C5 selective prediction : AURC (Geifman & El-Yaniv 2017) and AUGRC
      (Traub et al. NeurIPS 2024); misclassification detection AUROC
      (Hendrycks & Gimpel 2017)

All functions take NumPy arrays so they can be unit-tested on CPU with
synthetic predictions before any model is trained.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------
# Basic distribution utilities
# --------------------------------------------------------------------------

def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    pc = np.clip(p, _EPS, 1.0)
    return -np.sum(pc * np.log(pc), axis=axis)


def _kl(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    pc = np.clip(p, _EPS, 1.0)
    qc = np.clip(q, _EPS, 1.0)
    return np.sum(pc * (np.log(pc) - np.log(qc)), axis=axis)


def jensen_shannon(p: np.ndarray, q: np.ndarray, axis: int = -1,
                   base2: bool = True) -> np.ndarray:
    """JSD(p||q). Symmetric, bounded. Returns per-row JSD; mean externally.

    base2=True returns bits (matches the NLP study's JSD unit).
    """
    m = 0.5 * (p + q)
    jsd = 0.5 * _kl(p, m, axis=axis) + 0.5 * _kl(q, m, axis=axis)
    if base2:
        jsd = jsd / np.log(2.0)
    return np.maximum(jsd, 0.0)


# --------------------------------------------------------------------------
# C1: calibration
# --------------------------------------------------------------------------

def expected_calibration_error(probs: np.ndarray, targets_hard: np.ndarray,
                               n_bins: int = 15) -> float:
    """Standard top-label ECE (Guo et al. 2017)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == targets_hard).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc - avg_conf)
    return float(ece)


def brier_score(probs: np.ndarray, targets_hard: np.ndarray) -> float:
    """Multi-class Brier score vs one-hot of the hard target."""
    n, c = probs.shape
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), targets_hard] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def negative_log_likelihood(probs: np.ndarray, targets_hard: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(targets_hard)), targets_hard], _EPS, 1.0)
    return float(-np.mean(np.log(p)))


# --------------------------------------------------------------------------
# C2: distribution fidelity to annotator votes
# --------------------------------------------------------------------------

def mean_jsd_to_votes(probs: np.ndarray, soft_targets: np.ndarray) -> float:
    return float(jensen_shannon(probs, soft_targets, axis=1, base2=True).mean())


def mean_kl_to_votes(probs: np.ndarray, soft_targets: np.ndarray) -> float:
    # KL(votes || pred): how well the prediction covers the human distribution.
    return float(_kl(soft_targets, probs, axis=1).mean())


# --------------------------------------------------------------------------
# C3: correlation helper
# --------------------------------------------------------------------------

def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without SciPy (rank then Pearson)."""
    def _rank(a: np.ndarray) -> np.ndarray:
        order = a.argsort()
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a), dtype=np.float64)
        # average ties
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        if (counts > 1).any():
            sums = np.zeros(len(counts))
            np.add.at(sums, inv, ranks)
            means = sums / counts
            ranks = means[inv]
        return ranks
    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom < _EPS:
        return float("nan")
    return float((rx * ry).sum() / denom)


# --------------------------------------------------------------------------
# C5: selective prediction & misclassification detection
# --------------------------------------------------------------------------

def _risk_coverage(confidence: np.ndarray, correct: np.ndarray):
    """Return coverage grid and risk (cumulative error) sorted by descending
    confidence. ``confidence`` high = keep first."""
    order = np.argsort(-confidence)
    correct_sorted = correct[order].astype(np.float64)
    n = len(correct_sorted)
    cum_err = np.cumsum(1.0 - correct_sorted)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = cum_err / k
    return coverage, risk


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area Under the Risk-Coverage curve (lower is better).

    ``confidence`` is the negative uncertainty (higher = more confident).
    """
    coverage, risk = _risk_coverage(confidence, correct)
    return float(np.trapz(risk, coverage))


def augrc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area Under the Generalised Risk-Coverage curve (Traub et al. 2024).

    Uses the *generalised risk* = cumulative error / n (normalised by the
    full set, not by current coverage), which Traub et al. show fixes
    pathologies of AURC at low coverage. Lower is better.
    """
    order = np.argsort(-confidence)
    correct_sorted = correct[order].astype(np.float64)
    n = len(correct_sorted)
    gen_risk = np.cumsum(1.0 - correct_sorted) / n
    coverage = np.arange(1, n + 1) / n
    return float(np.trapz(gen_risk, coverage))


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC where higher score should indicate label==1. Rank-based."""
    pos = labels == 1
    neg = ~pos
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def misclassification_auroc(uncertainty: np.ndarray, correct: np.ndarray) -> float:
    """AUROC for detecting errors: higher uncertainty -> incorrect (label=1)."""
    err = (1 - correct).astype(np.int64)
    return _auroc(uncertainty, err)


# --------------------------------------------------------------------------
# Aggregator
# --------------------------------------------------------------------------

def compute_all_metrics(probs: np.ndarray, targets_hard: np.ndarray,
                        targets_soft: np.ndarray | None = None,
                        uncertainty_total: np.ndarray | None = None) -> dict:
    """Compute the full metric bundle from a mean predictive distribution.

    probs            : (N, C) mean predictive distribution
    targets_hard     : (N,) majority-vote labels
    targets_soft     : (N, C) annotator vote distribution (optional, for C2)
    uncertainty_total: (N,) total predictive uncertainty for C5 (defaults to
                       predictive entropy of ``probs``)
    """
    pred = probs.argmax(axis=1)
    correct = (pred == targets_hard).astype(np.int64)
    if uncertainty_total is None:
        uncertainty_total = entropy(probs, axis=1)
    confidence = -uncertainty_total

    out = {
        "accuracy": float((pred == targets_hard).mean()),
        "ece": expected_calibration_error(probs, targets_hard),
        "brier": brier_score(probs, targets_hard),
        "nll": negative_log_likelihood(probs, targets_hard),
        "aurc": aurc(confidence, correct),
        "augrc": augrc(confidence, correct),
        "mis_auroc": misclassification_auroc(uncertainty_total, correct),
    }
    if targets_soft is not None:
        out["jsd_votes"] = mean_jsd_to_votes(probs, targets_soft)
        out["kl_votes"] = mean_kl_to_votes(probs, targets_soft)
    return out
