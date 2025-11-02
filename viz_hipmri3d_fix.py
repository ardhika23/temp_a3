#!/usr/bin/env python3
"""
viz_hipmri3d_fix.py
Visualise HipMRI 3D (one mid slice) and try to auto-fix orientation of the PNG prediction.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib


def load_nifti(path):
    nii = nib.load(path)
    arr = nii.get_fdata().astype(np.float32)
    return arr  # (Z, Y, X) for this dataset


def normalise(img2d: np.ndarray) -> np.ndarray:
    vmin, vmax = float(img2d.min()), float(img2d.max())
    if vmax > vmin:
        return (img2d - vmin) / (vmax - vmin)
    return np.zeros_like(img2d)


def load_pred_png_smart(path: str, target_hw: tuple[int, int]) -> np.ndarray:
    """
    Try to make the pred.png shape == target_hw (H, W).
    We try: original -> transpose -> 4x rotation -> resize (fallback).
    """
    H, W = target_hw
    png = Image.open(path).convert("L")
    pred = np.array(png, dtype=np.float32) / 255.0  # 0..1
    # 1) exact match
    if pred.shape == (H, W):
        return pred
    # 2) transpose match
    if pred.T.shape == (H, W):
        return pred.T
    # 3) try rotate 0..3
    for k in range(4):
        rp = np.rot90(pred, k)
        if rp.shape == (H, W):
            return rp
    # 4) last resort: resize
    pred_resized = np.array(
        Image.fromarray((pred * 255).astype(np.uint8)).resize((W, H), Image.NEAREST),
        dtype=np.float32,
    ) / 255.0
    return pred_resized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="3D MRI volume (.nii.gz)")
    ap.add_argument("--mask", required=True, help="3D GT mask (.nii.gz)")
    ap.add_argument("--pred", required=True, help="2D prediction png (from your model)")
    ap.add_argument("--out", required=True, help="output png")
    ap.add_argument("--z", type=int, default=None, help="optional slice index, default=mid")
    args = ap.parse_args()

    vol = load_nifti(args.img)        # (Z, Y, X)
    gt3d = load_nifti(args.mask)      # (Z, Y, X)

    # pick slice
    if args.z is None:
        z = vol.shape[0] // 2
    else:
        z = max(0, min(args.z, vol.shape[0] - 1))

    img2d = vol[z]                    # (Y, X)
    gt2d = (gt3d[z] > 0).astype(np.float32)

    # load pred and try to match shape
    pred2d = load_pred_png_smart(args.pred, img2d.shape)

    img_disp = normalise(img2d)

    plt.figure(figsize=(12, 4))

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(img_disp, cmap="gray")
    ax1.set_title("MRI Slice")
    ax1.axis("off")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(img_disp, cmap="gray")
    ax2.imshow(gt2d, cmap="Reds", alpha=0.5)
    ax2.set_title("Ground Truth (mid)")
    ax2.axis("off")

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(img_disp, cmap="gray")
    ax3.imshow(pred2d, cmap="Blues", alpha=0.5)
    ax3.set_title("Model Prediction (from PNG)")
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved to {args.out} (slice z={z})")


if __name__ == "__main__":
    main()