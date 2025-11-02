#!/usr/bin/env python3
"""
viz_hipmri3d_raw.py

Visualise HipMRI 3D from *raw* NIfTI files + an external prediction PNG.
This matches the command you tried:

python viz_hipmri3d_raw.py \
  --img  /home/groups/comp3710/HipMRI_Study_open/semantic_MRs/B006_Week0_LFOV.nii.gz \
  --mask /home/groups/comp3710/HipMRI_Study_open/semantic_labels_only/B006_Week0_SEMANTIC.nii.gz \
  --pred /home/Student/s4906271/seg-oasis-to-hipmri-49062717/preds_hipmri3d_triplet/pred_zmid.png \
  --out  viz_hipmri3d_B006_Week0.png
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
    return arr  # (Z, Y, X) biasanya, tergantung dataset


def load_png_mask(path, target_shape):
    """
    pred PNG kamu sekarang 2D. Kita taruh dia persis di tengah (slice zmid),
    supaya ukuran match pas diplot.
    """
    png = Image.open(path).convert("L")
    png = np.array(png, dtype=np.float32) / 255.0  # 0..1
    # target_shape: (H, W)
    if png.shape != target_shape:
        # simple resize biar gak meledak
        png = np.array(Image.fromarray((png * 255).astype(np.uint8)).resize(
            (target_shape[1], target_shape[0]),
            resample=Image.NEAREST
        ), dtype=np.float32) / 255.0
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="3D MRI volume (.nii.gz)")
    ap.add_argument("--mask", required=True, help="3D GT mask (.nii.gz)")
    ap.add_argument("--pred", required=True, help="2D prediction png (your model output)")
    ap.add_argument("--out", required=True, help="output viz png")
    args = ap.parse_args()

    vol = load_nifti(args.img)     # (Z, Y, X)
    gt3d = load_nifti(args.mask)   # (Z, Y, X)

    # pilih slice tengah
    zmid = vol.shape[0] // 2
    img2d = vol[zmid]    # (Y, X)
    gt2d = (gt3d[zmid] > 0).astype(np.float32)

    # pred 2D dari png → samakan size dengan img2d
    pred2d = load_png_mask(args.pred, img2d.shape)

    # normalisasi image
    vmin, vmax = img2d.min(), img2d.max()
    if vmax > vmin:
        img_disp = (img2d - vmin) / (vmax - vmin)
    else:
        img_disp = img2d * 0.0

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
    print(f"[OK] saved to {args.out}")


if __name__ == "__main__":
    main()