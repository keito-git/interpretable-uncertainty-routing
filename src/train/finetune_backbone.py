"""
finetune_backbone.py - Fine-tune ViT-B/16 on FERPlus (TRAIN split only).

Why: frozen ImageNet ViT features are useless for FER (linear-probe ceiling
32.9% < majority 35.8%). FER requires backbone adaptation. We fine-tune on
the FERPlus TRAIN split only (no leakage of valid/test), with the soft-label
(annotator vote distribution) objective, then save the best-by-valid-accuracy
checkpoint. Feature extraction (extract_features.py --checkpoint) then uses
this fine-tuned backbone; the cSG-MCMC uncertainty head sits on top of the
(now frozen) fine-tuned features -> the "head-Bayesian" story is preserved as
a two-stage recipe (fine-tune -> freeze -> posterior head).

No test leakage: only train images update weights; valid is for model
selection; test is never seen during fine-tuning.

CLI
---
    python -m python.train.finetune_backbone \
        --pixels data/raw/fer2013_pixels.npy --processed-dir data/processed \
        --backbone vit_base_patch16_224 --epochs 25 --out data/ckpt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_NAME_RE = re.compile(r"fer0*?(\d+)\.png$")


def _name_to_index(name: str) -> int:
    m = _NAME_RE.search(str(name))
    return int(m.group(1)) if m else int("".join(c for c in str(name) if c.isdigit()))


# --- GPU-side data path (no DataLoader workers -> no /dev/shm dependence) ---
# Container /dev/shm is tiny; multiprocessing DataLoader bus-errors. Raw 48x48
# faces are tiny, so we keep them on GPU and resize/augment on GPU per batch.

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_split_tensors(pixels, npz_path, device):
    d = np.load(npz_path, allow_pickle=True)
    # Pair to pixels by ROW INDEX (proven alignment), NOT the name number.
    idx = d["row_index"].astype(np.int64)
    raw = torch.from_numpy(pixels[idx].astype(np.float32) / 255.0).to(device)  # (N,48,48)
    soft = torch.from_numpy(d["soft"].astype(np.float32)).to(device)
    hard = torch.from_numpy(d["hard"].astype(np.int64)).to(device)
    return raw, soft, hard


def gpu_resize_flip(raw_batch, device, train: bool):
    """(B,48,48) in [0,1] -> (B,3,224,224) in [0,1] (un-normalised), GPU-side."""
    import torch.nn.functional as F
    x = raw_batch.unsqueeze(1)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    if train:
        flip = torch.rand(x.shape[0], device=device) < 0.5
        x[flip] = torch.flip(x[flip], dims=[3])
    return x


def normalize(x, device):
    return (x - _IMAGENET_MEAN.to(device)) / _IMAGENET_STD.to(device)


def gpu_transform(raw_batch, device, train: bool):
    """Convenience: resize/flip then ImageNet-normalise."""
    return normalize(gpu_resize_flip(raw_batch, device, train), device)


def evaluate(model, raw, hard, device, bs=256):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for s in range(0, len(raw), bs):
            x = gpu_transform(raw[s:s + bs], device, train=False)
            pred = model(x).argmax(1)
            correct += (pred == hard[s:s + bs]).sum().item(); total += x.shape[0]
    return correct / max(total, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixels", type=Path, default=Path("data/raw/fer2013_pixels.npy"))
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--drop-rate", type=float, default=0.0,
                    help="head dropout rate (>0 enables MC-Dropout baseline)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label", choices=["soft", "hard"], default="soft")
    ap.add_argument("--align-weight", type=float, default=0.0,
                    help="DGT: weight on (H[pred]-H[vote])^2 aleatoric-grounding term")
    ap.add_argument("--oe-weight", type=float, default=0.0,
                    help="DGT: weight on outlier-exposure (max entropy on corrupted views)")
    ap.add_argument("--oe-corruptions", type=str, default="gnoise,blur,occlude",
                    help="corruption types used for OE during training (held-out types reserved for eval)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/ckpt"))
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    pixels = np.load(args.pixels)

    import timm
    try:
        model = timm.create_model(args.backbone, pretrained=True, num_classes=8, img_size=224,
                                  drop_rate=args.drop_rate).to(device)
    except TypeError:
        model = timm.create_model(args.backbone, pretrained=True, num_classes=8,
                                  drop_rate=args.drop_rate).to(device)

    tr_raw, tr_soft, tr_hard = load_split_tensors(pixels, args.processed_dir / "ferplus_train.npz", device)
    va_raw, va_soft, va_hard = load_split_tensors(pixels, args.processed_dir / "ferplus_valid.npz", device)
    N = len(tr_raw)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    bs = args.batch_size

    # DGT extras
    if args.align_weight > 0 or args.oe_weight > 0:
        from python.eval.ensemble_infer import corrupt as _corrupt
    oe_types = [c for c in args.oe_corruptions.split(",") if c]
    oe_gen = torch.Generator(device=device).manual_seed(args.seed + 777)
    eps = 1e-8
    logC = float(np.log(8))

    best_acc = -1.0; history = []
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        for s in range(0, N - bs + 1, bs):
            bidx = perm[s:s + bs]
            x_un = gpu_resize_flip(tr_raw[bidx], device, train=True)
            x = normalize(x_un, device)
            tgt = tr_soft[bidx] if args.label == "soft" else F.one_hot(tr_hard[bidx], 8).float()
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = -(tgt * F.log_softmax(logits, -1)).sum(-1).mean()
                if args.align_weight > 0:
                    p = F.softmax(logits, -1)
                    h_pred = -(p * torch.log(p + eps)).sum(-1)
                    q = tr_soft[bidx]
                    h_vote = -(q * torch.log(q + eps)).sum(-1)
                    loss = loss + args.align_weight * ((h_pred - h_vote) ** 2).mean()
                if args.oe_weight > 0 and oe_types:
                    ct = oe_types[int(torch.randint(0, len(oe_types), (1,), generator=oe_gen, device=device).item())]
                    sev = int(torch.randint(2, 6, (1,), generator=oe_gen, device=device).item())
                    x_oe = normalize(_corrupt(x_un.detach(), ct, sev, device, oe_gen), device)
                    p_oe = F.softmax(model(x_oe), -1)
                    h_oe = -(p_oe * torch.log(p_oe + eps)).sum(-1)
                    loss = loss + args.oe_weight * (logC - h_oe).mean()  # maximise entropy on OOD
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        acc = evaluate(model, va_raw, va_hard, device)
        history.append({"epoch": ep, "valid_acc": acc, "last_loss": float(loss.item())})
        print(f"epoch {ep} valid_acc {acc:.4f} loss {loss.item():.4f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(), "backbone": args.backbone,
                        "valid_acc": acc, "epoch": ep, "label": args.label},
                       args.out / "vit_ft_best.pt")

    with open(args.out / "finetune_history.json", "w", encoding="utf-8") as fh:
        json.dump({"best_valid_acc": best_acc, "history": history,
                   "args": vars(args) | {"pixels": str(args.pixels),
                   "processed_dir": str(args.processed_dir), "out": str(args.out)}},
                  fh, ensure_ascii=False, indent=2, default=str)
    print(json.dumps({"best_valid_acc": best_acc}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
