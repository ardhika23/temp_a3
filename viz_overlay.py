#!/usr/bin/env python3
"""
Make nice 3-panel visualisation:

[ MRI Slice ]  [ Ground Truth overlay ]  [ Model Prediction overlay ]

Works for:
- OASIS 2D      (you have input.png / gt.png / pred.png)
- HipMRI 2D     (preds_hipmri2d_triplet/*.png)
- HipMRI 3D     (preds_hipmri3d_triplet/*_zmid.png)  --> still 2D slice, but 3D volume

Run example (OASIS):
    python viz_overlay.py \
        --input preds_oasis_triplet/input.png \
        --gt    preds_oasis_triplet/gt.png \
        --pred  preds_oasis_triplet/pred.png \
        --out   viz_oasis_overlay.png

Run example (HipMRI 2D):
    python viz_overlay.py \
        --input preds_hipmri2d_triplet/input.png \
        --gt    preds_hipmri2d_triplet/gt.png \
        --pred  preds_hipmri2d_triplet/pred.png \
        --out   viz_hipmri2d_overlay.png

Run example (HipMRI 3D, z-mid slice):
    python viz_overlay.py \
        --input preds_hipmri3d_triplet/input_zmid.png \
        --gt    preds_hipmri3d_triplet/gt_zmid.png \
        --pred  preds_hipmri3d_triplet/pred_zmid.png \
        --out   viz_hipmri3d_overlay.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless on Rangpur
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_gray(path: str) -> np.ndarray:
    """Load image as grayscale array [H, W]."""
    img = Image.open(path).convert("L")
    return np.array(img)


def load_mask(path: str) -> np.ndarray:
    """Load mask (binary or 0/255) -> bool array."""
    m = Image.open(path).convert("L")
    arr = np.array(m)
    # anything > 0 is foreground
    return arr > 0


def overlay(base_gray: np.ndarray,
            mask_bool: np.ndarray,
            color=(200, 80, 80),
            alpha=0.45) -> np.ndarray:
    """
    base_gray: [H, W], 0..255
    mask_bool: [H, W] True=foreground
    color: RGB tuple
    return: [H, W, 3] uint8
    """
    h, w = base_gray.shape
    base_rgb = np.stack([base_gray]*3, axis=-1).astype(np.float32)

    overlay_img = np.zeros_like(base_rgb)
    overlay_img[mask_bool] = np.array(color, dtype=np.float32)

    out = base_rgb.copy()
    out[mask_bool] = (1 - alpha) * base_rgb[mask_bool] + alpha * overlay_img[mask_bool]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input MRI slice (png)")
    ap.add_argument("--gt", required=True, help="ground truth mask (png)")
    ap.add_argument("--pred", required=True, help="predicted mask (png)")
    ap.add_argument("--out", required=True, help="output figure png")
    args = ap.parse_args()

    # 1. load images
    img_gray = load_gray(args.input)
    gt_mask = load_mask(args.gt)
    pred_mask = load_mask(args.pred)

    # 2. make overlays
    gt_overlay = overlay(img_gray, gt_mask, color=(196, 107, 90), alpha=0.55)     # brownish/red
    pred_overlay = overlay(img_gray, pred_mask, color=(40, 60, 110), alpha=0.55)  # bluish

    # 3. plot
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), facecolor="black")

    # MRI slice
    axes[0].imshow(img_gray, cmap="gray")
    axes[0].set_title("MRI Slice", color="white", fontsize=10)
    axes[0].axis("off")
    axes[0].set_facecolor("black")

    # GT overlay
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth", color="white", fontsize=10)
    axes[1].axis("off")
    axes[1].set_facecolor("black")

    # Pred overlay
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Model Prediction", color="white", fontsize=10)
    axes[2].axis("off")
    axes[2].set_facecolor("black")

    plt.tight_layout(pad=1.2)
    fig.patch.set_facecolor("black")

    # 4. save
    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()