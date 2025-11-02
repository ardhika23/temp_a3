import os
import argparse

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules import build_model
from dataset import get_dataset


def to_pil_gray(t):
    # t: (1,H,W)
    arr = t.squeeze(0).cpu().numpy()
    # normalize to 0..255
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr = (arr * 255).astype("uint8")
    return Image.fromarray(arr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="oasis2d | hipmri2d | hipmri3d")
    p.add_argument("--root", default=".")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--save-dir", default="preds_triplet")
    p.add_argument("--index", type=int, default=0, help="which sample from val set")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1) load val split
    if args.dataset == "hipmri3d":
        ds = get_dataset("hipmri3d", root="", split="val")
    else:
        ds = get_dataset(args.dataset, root=args.root, split="val")

    img, gt = ds[args.index]     # img: (1,H,W) or (1,Z,H,W)
    is_3d = (img.dim() == 4)     # (1,Z,H,W)

    # 2) load model
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_name = ckpt["args"]["model"]
    model = build_model(model_name, in_channels=1, out_channels=1)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        pred = model(img.unsqueeze(0))   # add batch
    pred = torch.sigmoid(pred).squeeze(0)

    if not is_3d:
        # ---------- 2D case ----------
        inp_pil = to_pil_gray(img)
        gt_pil = Image.fromarray((gt.squeeze(0).numpy() > 0).astype("uint8") * 255)
        pred_bin = (pred.squeeze(0) > 0.5).float()
        pred_pil = Image.fromarray(pred_bin.numpy().astype("uint8") * 255)

        inp_pil.save(os.path.join(args.save_dir, "input.png"))
        gt_pil.save(os.path.join(args.save_dir, "gt.png"))
        pred_pil.save(os.path.join(args.save_dir, "pred.png"))

        # make collage
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow(inp_pil, cmap="gray")
        axs[0].set_title("Input")
        axs[1].imshow(gt_pil, cmap="gray")
        axs[1].set_title("Ground Truth")
        axs[2].imshow(pred_pil, cmap="gray")
        axs[2].set_title("Prediction")
        for ax in axs:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.save_dir, "triplet.png"))
        plt.close(fig)
    else:
        # ---------- 3D case ----------
        # pick middle slice
        _, z, h, w = img.shape
        mid = z // 2

        inp_slice = img[0, mid]          # (H,W)
        gt_slice = gt[0, mid]
        pred_slice = (pred[0, mid] > 0.5).float()

        def to_pil(arr):
            arr = arr.cpu().numpy()
            if arr.max() > 1:
                arr = arr / arr.max()
            arr = (arr * 255).astype("uint8")
            return Image.fromarray(arr)

        inp_pil = to_pil(inp_slice)
        gt_pil = to_pil(gt_slice)
        pred_pil = to_pil(pred_slice)

        inp_pil.save(os.path.join(args.save_dir, "input_zmid.png"))
        gt_pil.save(os.path.join(args.save_dir, "gt_zmid.png"))
        pred_pil.save(os.path.join(args.save_dir, "pred_zmid.png"))

        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow(inp_pil, cmap="gray")
        axs[0].set_title("Input (mid)")
        axs[1].imshow(gt_pil, cmap="gray")
        axs[1].set_title("GT (mid)")
        axs[2].imshow(pred_pil, cmap="gray")
        axs[2].set_title("Pred (mid)")
        for ax in axs:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.save_dir, "triplet_zmid.png"))
        plt.close(fig)

    print("done, saved to", args.save_dir)


if __name__ == "__main__":
    main()