"""
extract_features.py - Frozen ViT-B/16 feature extraction (GPU).

Produces, per split:
- CLS features (N, D)      -> for the cSG-MCMC head (image-level a/e).
- patch tokens (N, P, D)   -> for the spatial-aleatoric map. Saved for the
  TEST split (and an optional small sample) only, to bound disk usage.

Pairing: FERPlus processed npz (image names + soft/hard votes) is joined to
``fer2013_pixels.npy`` by the global index encoded in each image name
(``fer<idx>.png``). Alignment was proven by acquire_fer2013.py.

Backbone: timm ``vit_base_patch16_224`` (ImageNet pretrained), frozen.
``forward_features`` returns (B, 1+P, D): token 0 = CLS, tokens 1.. = patches
(P = 196 = 14x14). 48x48 grayscale faces are resized to 224 and replicated to
3 channels with ImageNet normalisation.

CLI
---
    python -m python.data.extract_features \
        --pixels data/raw/fer2013_pixels.npy \
        --processed-dir data/processed \
        --output-dir data/features \
        --backbone vit_base_patch16_224 \
        --save-patch-splits test
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

_NAME_RE = re.compile(r"fer0*?(\d+)\.png$")


def _name_to_index(name: str) -> int:
    m = _NAME_RE.search(str(name))
    if not m:
        # fall back: strip non-digits
        digits = "".join(ch for ch in str(name) if ch.isdigit())
        return int(digits)
    return int(m.group(1))


def extract(pixels_path: Path, processed_dir: Path, output_dir: Path,
            backbone: str, save_patch_splits: set[str],
            batch_size: int, device_str: str, checkpoint: Path | None = None) -> dict:
    import torch
    import torch.nn.functional as F
    import timm

    output_dir.mkdir(parents=True, exist_ok=True)
    pixels = np.load(pixels_path)  # (35887, 48, 48) uint8

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    try:
        model = timm.create_model(backbone, pretrained=(checkpoint is None), num_classes=0, img_size=224)
    except TypeError:
        model = timm.create_model(backbone, pretrained=(checkpoint is None), num_classes=0)
    n_prefix = int(getattr(model, "num_prefix_tokens", 1))
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        # fine-tuned model had a classification head (num_classes=8); load
        # backbone weights only (strict=False ignores head.* keys).
        missing, unexpected = model.load_state_dict(sd, strict=False)
        loaded = [k for k in sd if not (k.startswith("head.") or k.startswith("fc."))]
        print(f"loaded checkpoint {checkpoint}: matched~{len(loaded)} keys, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    # GPU-side preprocessing (no DataLoader/PIL; container /dev/shm is tiny).
    imnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def _to_input(idx_array):
        raw = torch.from_numpy(pixels[idx_array].astype(np.float32) / 255.0).to(device)
        x = raw.unsqueeze(1)  # (B,1,48,48)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        return (x - imnet_mean) / imnet_std

    summary = {"backbone": backbone, "device": str(device), "splits": {}}

    for split in ("train", "valid", "test"):
        npz = np.load(processed_dir / f"ferplus_{split}.npz", allow_pickle=True)
        names = [str(x) for x in npz["image"]]
        # Pair to pixels by ROW INDEX (proven alignment), NOT the name number.
        idxs = npz["row_index"].astype(np.int64)
        n = len(idxs)
        want_patch = split in save_patch_splits

        cls_feats = None
        patch_feats = None
        write_ptr = 0

        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch_idx = idxs[start:start + batch_size]
                imgs = _to_input(batch_idx)
                tokens = model.forward_features(imgs)  # (B, n_prefix+P, D)
                if tokens.dim() != 3:
                    raise RuntimeError(
                        f"expected token sequence (B,*,D), got {tuple(tokens.shape)}; "
                        f"backbone {backbone} may not expose patch tokens.")
                cls = tokens[:, 0, :].float().cpu().numpy().astype(np.float16)
                patches = tokens[:, n_prefix:, :].float().cpu().numpy().astype(np.float16)
                B, P, D = patches.shape
                if cls_feats is None:
                    cls_feats = np.zeros((n, D), dtype=np.float16)
                    if want_patch:
                        patch_feats = np.zeros((n, P, D), dtype=np.float16)
                cls_feats[write_ptr:write_ptr + B] = cls
                if want_patch:
                    patch_feats[write_ptr:write_ptr + B] = patches
                write_ptr += B

        P_tokens = int(patch_feats.shape[1]) if want_patch else (cls_feats.shape[1] and 0)
        g = int(round(float(np.sqrt(P_tokens)))) if want_patch and P_tokens > 0 else 14
        out = {
            "cls": cls_feats,
            "soft": npz["soft"].astype(np.float32),
            "hard": npz["hard"].astype(np.int64),
            "disagreement": npz["disagreement"].astype(np.float32),
            "image": np.array(names),
            "row_index": idxs.astype(np.int64),
            "grid_h": np.int64(g), "grid_w": np.int64(g),
        }
        if want_patch:
            out["patch"] = patch_feats
        np.savez_compressed(output_dir / f"feat_{split}.npz", **out)
        summary["splits"][split] = {
            "n": int(n), "dim": int(cls_feats.shape[1]),
            "patch_saved": bool(want_patch),
            "patch_shape": (list(patch_feats.shape) if want_patch else None),
        }

    with open(output_dir / "extract_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen ViT feature extraction.")
    p.add_argument("--pixels", type=Path, default=Path("data/raw/fer2013_pixels.npy"))
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--output-dir", type=Path, default=Path("data/features"))
    p.add_argument("--backbone", type=str, default="vit_base_patch16_224")
    p.add_argument("--save-patch-splits", type=str, default="test",
                   help="comma-separated subset of {train,valid,test}")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="fine-tuned backbone checkpoint (.pt); if set, pretrained=False")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    patch_splits = {s.strip() for s in args.save_patch_splits.split(",") if s.strip()}
    summary = extract(args.pixels, args.processed_dir, args.output_dir,
                      args.backbone, patch_splits, args.batch_size, args.device,
                      checkpoint=args.checkpoint)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
