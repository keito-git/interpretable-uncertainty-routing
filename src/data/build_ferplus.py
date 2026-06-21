"""
build_ferplus.py - FERPlus annotator-vote preprocessing (LABELS ONLY, no GPU).

Purpose
-------
Turn the public FERPlus crowd-vote file (``fer2013new.csv``, 10 annotators
per image) into the artefacts our uncertainty-decomposition study needs:

- soft labels  q_i : the normalised annotator vote distribution over the
  8 emotion categories (= the *aleatoric ground truth*).
- hard labels  y_i : majority vote (argmax of the vote distribution).
- per-image disagreement  delta_i : 1 - max_c q_i^c  (and Shannon entropy).
- per-category disagreement rate  delta^(c) : mean disagreement over images
  whose majority label is c  (used for the C3 correlation: per-category
  aleatoric uncertainty vs. annotator disagreement).

This mirrors ``per_emotion_disagreement.csv`` from the NLP/GoEmotions study
so the same C3 analysis transfers to faces.

IMPORTANT
---------
This script uses ONLY the vote CSV. The FER2013 pixel data
(``fer2013.csv`` / ``icml_face_data.csv``, from Kaggle) is NOT needed here
and is consumed later by the (GPU) feature-extraction step. Image names are
preserved so labels can be joined to pixels downstream.

FERPlus columns
---------------
    Usage, Image name, neutral, happiness, surprise, sadness, anger,
    disgust, fear, contempt, unknown, NF

- ``Usage`` in {Training, PublicTest, PrivateTest} -> {train, valid, test}.
- The 8 emotion columns are the supervised classes; ``unknown`` and ``NF``
  (not-a-face) are non-emotion and are excluded from the label distribution.

Filtering rule (documented & conservative)
------------------------------------------
An image is DROPPED when either:
  (a) unknown + NF >= majority of the 10 votes (>= 5), i.e. the crowd mostly
      judged it not a clear facial emotion; or
  (b) the 8 emotion votes sum to 0.
Otherwise we keep it and set q_i = emotion_votes / sum(emotion_votes).
This is intentionally simple and is logged so it can be revisited; FERPlus'
original PLD scheme can be swapped in later if a reviewer prefers it.

CLI
---
    python -m python.data.build_ferplus \
        --input  data/raw/fer2013new.csv \
        --output-dir data/processed
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

EMOTION_COLUMNS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
]
NUM_CLASSES = len(EMOTION_COLUMNS)
SPECIAL_COLUMNS = ["unknown", "NF"]
USAGE_TO_SPLIT = {
    "Training": "train",
    "PublicTest": "valid",
    "PrivateTest": "test",
}


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pc = np.clip(p, eps, 1.0)
    return -np.sum(pc * np.log(pc), axis=-1)


def parse_rows(input_path: Path) -> list[dict]:
    """Read fer2013new.csv into a list of raw records."""
    with open(input_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in (["Usage", "Image name"] + EMOTION_COLUMNS + SPECIAL_COLUMNS)
                   if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"FERPlus CSV missing columns: {missing}")
        return [row for row in reader]


def build(input_path: Path, output_dir: Path) -> dict:
    rows = parse_rows(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, dict[str, list]] = {
        s: {"image": [], "row_index": [], "usage": [], "votes": [], "soft": [],
            "hard": [], "disagreement": [], "entropy": [], "n_emotion_votes": []}
        for s in ("train", "valid", "test")
    }

    n_total = len(rows)
    n_dropped_nonface = 0
    n_dropped_zero = 0
    n_dropped_usage = 0

    # NOTE: pairing to FER2013 pixels is by ROW INDEX (position in this file),
    # NOT by the "Image name" number -- the name index drifts from the row
    # index (e.g. row 28709 is named fer0028638). acquire_fer2013 proved the
    # row-order alignment (Usage matches per row); pixels[row_index] is correct.
    for ri, row in enumerate(rows):
        usage = row["Usage"].strip()
        split = USAGE_TO_SPLIT.get(usage)
        if split is None:
            n_dropped_usage += 1
            continue

        emo_votes = np.array([float(row[c]) for c in EMOTION_COLUMNS], dtype=np.float64)
        unknown = float(row["unknown"])
        nf = float(row["NF"])
        emo_sum = emo_votes.sum()

        # Filtering rule (see module docstring).
        if (unknown + nf) >= 5.0:
            n_dropped_nonface += 1
            continue
        if emo_sum <= 0.0:
            n_dropped_zero += 1
            continue

        q = emo_votes / emo_sum
        hard = int(np.argmax(emo_votes))
        disagreement = float(1.0 - q.max())
        ent = float(_entropy(q))

        d = splits[split]
        d["image"].append(row["Image name"].strip())
        d["row_index"].append(ri)
        d["usage"].append(usage)
        d["votes"].append(emo_votes)
        d["soft"].append(q)
        d["hard"].append(hard)
        d["disagreement"].append(disagreement)
        d["entropy"].append(ent)
        d["n_emotion_votes"].append(float(emo_sum))

    # Persist per-split npz + a human-readable csv.
    summary: dict = {
        "input": str(input_path),
        "n_total_rows": n_total,
        "n_dropped_nonface(unknown+NF>=5)": n_dropped_nonface,
        "n_dropped_zero_emotion_votes": n_dropped_zero,
        "n_dropped_unrecognised_usage": n_dropped_usage,
        "num_classes": NUM_CLASSES,
        "emotion_columns": EMOTION_COLUMNS,
        "splits": {},
    }

    # Accumulate per-category disagreement over the full kept set (all splits)
    # AND per-split, so C3 can use whichever the protocol fixes.
    cat_dis_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    cat_dis_cnt = np.zeros(NUM_CLASSES, dtype=np.float64)

    for split, d in splits.items():
        n = len(d["image"])
        soft = np.array(d["soft"], dtype=np.float64) if n else np.zeros((0, NUM_CLASSES))
        votes = np.array(d["votes"], dtype=np.float64) if n else np.zeros((0, NUM_CLASSES))
        hard = np.array(d["hard"], dtype=np.int64) if n else np.zeros((0,), dtype=np.int64)
        dis = np.array(d["disagreement"], dtype=np.float64)
        ent = np.array(d["entropy"], dtype=np.float64)
        nev = np.array(d["n_emotion_votes"], dtype=np.float64)

        np.savez_compressed(
            output_dir / f"ferplus_{split}.npz",
            image=np.array(d["image"]),
            row_index=np.array(d["row_index"], dtype=np.int64),
            usage=np.array(d["usage"]),
            votes=votes,
            soft=soft,
            hard=hard,
            disagreement=dis,
            entropy=ent,
            n_emotion_votes=nev,
            emotion_columns=np.array(EMOTION_COLUMNS),
        )

        for c in range(NUM_CLASSES):
            mask = hard == c
            cat_dis_sum[c] += dis[mask].sum()
            cat_dis_cnt[c] += int(mask.sum())

        summary["splits"][split] = {
            "n_images": int(n),
            "mean_disagreement": float(dis.mean()) if n else None,
            "mean_entropy": float(ent.mean()) if n else None,
            "class_counts": {EMOTION_COLUMNS[c]: int((hard == c).sum()) for c in range(NUM_CLASSES)},
        }

    # Per-category disagreement rate delta^(c) over the kept set.
    safe = np.where(cat_dis_cnt > 0, cat_dis_cnt, 1.0)
    cat_dis = cat_dis_sum / safe
    with open(output_dir / "per_category_disagreement.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "mean_disagreement", "n_images"])
        for c in range(NUM_CLASSES):
            w.writerow([EMOTION_COLUMNS[c], f"{cat_dis[c]:.6f}", int(cat_dis_cnt[c])])

    summary["per_category_disagreement"] = {
        EMOTION_COLUMNS[c]: {"mean_disagreement": float(cat_dis[c]),
                             "n_images": int(cat_dis_cnt[c])}
        for c in range(NUM_CLASSES)
    }

    with open(output_dir / "build_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FERPlus vote -> soft/hard label preprocessing.")
    p.add_argument("--input", type=Path, default=Path("data/raw/fer2013new.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    summary = build(args.input, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
