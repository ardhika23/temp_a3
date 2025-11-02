#!/usr/bin/env python3
"""
viz_hipmri3d_from_dataset.py

Visualise HipMRI 3D *using the same dataset pipeline* as training/inference,
so shapes/orientation match your model outputs.

Usage (on Rangpur):

python viz_hipmri3d_from_dataset.py \
  --checkpoint runs/hipmri3d/best.pt \
  --index 0 \
  --out viz_hipmri3d_ds_0.png
"""

import argparse
import os

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import get_dataset
from modules import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to runs/hipmri3d/best.pt")
    p.add_argument("--index", type=int, default=0, help="which sample from val split")
    p.add_argument("--out", default="viz_hipmri3d_ds.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load val set via your pipeline -> this guarantees SAME preprocess
    ds = get_dataset("hipmri3d", root="", split="val")
    img, gt = ds[args.index]        # img: (1, Z, H, W) or (1, D, H, W) depending on your dataset

    # make batch
    img_b = img.unsqueeze(0).to(device)   # (1, 1, Z, H, W)

    # 2) load model from ckpt
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt["args"]["model"]
    model = build_model(model_name, in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        pred = model(img_b)  # (1,1,Z,H,W)

    # go back to cpu
    img  = img.squeeze(0).cpu().numpy()     # (Z,H,W)
    gt   = gt.squeeze(0).cpu().numpy()      # (Z,H,W)
    pred = pred.squeeze(0).squeeze(0).cpu().numpy()  # (Z,H,W)

    # pick middle slice along Z
    z = img.shape[0] // 2
    img2d  = img[z]           # (H,W)
    gt2d   = (gt[z] > 0).astype(np.float32)
    pred2d = (1 / (1 + np.exp(-pred[z])) > 0.5).astype(np.float32)

    # normalise img
    imin, imax = img2d.min(), img2d.max()
    if imax > imin:
        img_disp = (img2d - imin) / (imax - imin)
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
    ax2.set_title("Ground Truth")
    ax2.axis("off")

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(img_disp, cmap="gray")
    ax3.imshow(pred2d, cmap="Blues", alpha=0.5)
    ax3.set_title("Model Prediction")
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved to {args.out}")


if __name__ == "__main__":
    main()