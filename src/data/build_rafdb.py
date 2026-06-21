"""
build_rafdb.py - RAF-DB test set as a CROSS-DATASET shift probe (no GPU).

Downloads RAF-DB (elipaluma/RAF-DB_Kaggle, HF) test split, decodes images to
GRAYSCALE 48x48 (matching the FERPlus training input format so that any OOD
signal stems from genuine distribution difference -- different faces, demos,
label balance -- not from input format), maps RAF-DB 7-class labels to the
FERPlus 8-class index space, and saves a pixels array + meta npz compatible
with ensemble_infer (row_index, hard, soft, disagreement).

RAF-DB label convention (standard): 1 Surprise, 2 Fear, 3 Disgust,
4 Happiness, 5 Sadness, 6 Anger, 7 Neutral.
FERPlus index order: 0 neutral, 1 happiness, 2 surprise, 3 sadness,
4 anger, 5 disgust, 6 fear, 7 contempt.

CLI
---
    python -m python.data.build_rafdb --out-dir data/rafdb
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

RAF_TO_FERPLUS = {1: 2, 2: 6, 3: 5, 4: 1, 5: 3, 6: 4, 7: 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, default="elipaluma/RAF-DB_Kaggle")
    ap.add_argument("--file", type=str, default="data/test-00000-of-00001.parquet")
    ap.add_argument("--out-dir", type=Path, default=Path("data/rafdb"))
    ap.add_argument("--size", type=int, default=48,
                    help="grayscale resize edge; 48 matches FERPlus, larger keeps native detail (real shift)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from PIL import Image

    f = hf_hub_download(args.repo, args.file, repo_type="dataset")
    rows = pq.read_table(f).to_pylist()
    n = len(rows); S = int(args.size)
    pixels = np.zeros((n, S, S), dtype=np.uint8)
    hard = np.zeros(n, dtype=np.int64)
    for i, r in enumerate(rows):
        img = r["image"]
        b = img["bytes"] if isinstance(img, dict) else None
        im = Image.open(io.BytesIO(b)) if b else img
        im = im.convert("L").resize((S, S))
        pixels[i] = np.array(im, dtype=np.uint8)
        hard[i] = RAF_TO_FERPLUS[int(r["label"])]
    soft = np.zeros((n, 8), dtype=np.float32)
    soft[np.arange(n), hard] = 1.0  # dummy one-hot (RAF-DB has no vote dist)

    np.save(args.out_dir / "rafdb_pixels.npy", pixels)
    np.savez_compressed(args.out_dir / "rafdb_meta.npz",
                        row_index=np.arange(n, dtype=np.int64),
                        hard=hard, soft=soft,
                        disagreement=np.zeros(n, dtype=np.float32),
                        image=np.array([f"raf_{i:05d}" for i in range(n)]))
    summary = {"n": int(n), "repo": args.repo,
               "class_counts_ferplus_index": {int(c): int((hard == c).sum()) for c in range(8)},
               "note": "RAF-DB test, grayscale 48x48, labels mapped to FERPlus index space"}
    json.dump(summary, open(args.out_dir / "rafdb_summary.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
