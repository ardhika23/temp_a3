# recognition/seg-oasis-to-hipmri-<uqid>/predict.py
"""
Minimal inference / visualisation script.

Example:
python predict.py --dataset hipmri2d \
    --root /home/groups/comp3710/HipMRI_Study_open/keras_slices_data \
    --checkpoint runs/hipmri2d/best_model.pt --save-dir preds/

This will save 2-3 sample predicted masks as .png or .npy.
You can screenshot ONE of them for README.md
"""

import os
import argparse

import torch
import numpy as np
from PIL import Image

from modules import build_model
from dataset import get_dataset


def save_mask(mask_tensor, out_path):
    # mask_tensor: (1,H,W) or (1,Z,Y,X)
    arr = torch.sigmoid(mask_tensor).cpu().numpy()
    arr = (arr > 0.5).astype(np.uint8) * 255

    if arr.ndim == 3:  # (1,H,W)
        img = Image.fromarray(arr[0])
        img.save(out_path)
    else:
        # 3D case: save middle slice
        _, z, h, w = arr.shape
        mid = z // 2
        img = Image.fromarray(arr[0, mid])
        img.save(out_path.replace(".png", f"_z{mid}.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="oasis2d | hipmri2d | hipmri3d")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default=None,
                        help="if None, will use model name from checkpoint")
    parser.add_argument("--save-dir", type=str, default="preds")
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # load dataset
    if args.dataset == "hipmri3d":
        ds = get_dataset("hipmri3d", root="", split="val")
    else:
        ds = get_dataset(args.dataset, root=args.root, split="val")

    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_name = args.model or ckpt["args"]["model"]
    model = build_model(model_name, in_channels=1, out_channels=1)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # run on a few samples
    for i in range(min(args.num_samples, len(ds))):
        img, _ = ds[i]
        img = img.unsqueeze(0).to(device)  # (1,C,H,W) or (1,C,Z,H,W)
        with torch.no_grad():
            pred = model(img)
        out_path = os.path.join(args.save_dir, f"sample_{i}.png")
        save_mask(pred.squeeze(0), out_path)
        print(f"saved {out_path}")

    print("Done. Add one of these images to README under Results.")


if __name__ == "__main__":
    main()