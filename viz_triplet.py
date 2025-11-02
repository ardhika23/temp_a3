import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # biar bisa jalan di rangpur/headless
import matplotlib.pyplot as plt
from PIL import Image

try:
    import nibabel as nib
except ImportError:
    nib = None


def load_img(path):
    # load grayscale (png/jpg)
    img = Image.open(path).convert("L")
    return np.array(img)  # (H, W)


def load_mask(path):
    m = Image.open(path).convert("L")
    m = np.array(m).astype(np.float32)
    # binarisasi kalau perlu
    m = (m > 0).astype(np.float32)
    return m  # (H, W)


def overlay(ax, img, mask, color=(1, 0, 0), alpha=0.35):
    ax.imshow(img, cmap="gray")
    # mask==1 → tampil
    masked = np.zeros((*img.shape, 4), dtype=np.float32)
    masked[..., 0] = color[0]
    masked[..., 1] = color[1]
    masked[..., 2] = color[2]
    masked[..., 3] = mask * alpha
    ax.imshow(masked)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input MRI slice (png/jpg) or nifti")
    parser.add_argument("--gt", required=True, help="GT mask (png/jpg)")
    parser.add_argument("--pred", required=True, help="Pred mask (png/jpg)")
    parser.add_argument("--out", default="viz_triplet.png")
    parser.add_argument("--title", default="HipMRI 2D – slice")
    args = parser.parse_args()

    # --- load image ---
    if args.input.endswith(".nii") or args.input.endswith(".nii.gz"):
        assert nib is not None, "install nibabel to read nifti"
        nii = nib.load(args.input)
        vol = nii.get_fdata()
        z = vol.shape[2] // 2
        img = vol[:, :, z]
    else:
        img = load_img(args.input)

    gt = load_mask(args.gt)
    pred = load_mask(args.pred)

    # pastikan ukuran sama
    h, w = img.shape
    if gt.shape != (h, w):
        gt = np.array(Image.fromarray(gt).resize((w, h), resample=Image.NEAREST))
    if pred.shape != (h, w):
        pred = np.array(Image.fromarray(pred).resize((w, h), resample=Image.NEAREST))

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    # kiri: input saja
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("MRI Slice")
    axes[0].axis("off")

    # tengah: GT (merah)
    overlay(axes[1], img, gt, color=(0.8, 0.2, 0.1), alpha=0.45)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # kanan: Pred (biru)
    overlay(axes[2], img, pred, color=(0.1, 0.2, 0.6), alpha=0.45)
    axes[2].set_title("Model Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()