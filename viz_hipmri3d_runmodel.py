# viz_hipmri3d_runmodel.py
# buat visualisasi 3D HipMRI yang SEBARIS (input, GT, pred) dan TIDAK miring
# jalanin dari:  ~/seg-oasis-to-hipmri-49062717
import os
import argparse

import numpy as np
import torch
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules import build_model  # pakai model kamu sendiri

def load_nifti(path):
    nii = nib.load(path)
    arr = nii.get_fdata().astype(np.float32)
    return arr  # shape: (Z, Y, X)

def normalise_img(vol):
    # sama seperti di dataset.py 3D kamu
    return (vol - vol.mean()) / (vol.std() + 1e-5)

def binarise_mask(vol):
    return (vol > 0).astype(np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img", required=True, help="HipMRI volume, e.g. .../semantic_MRs/B006_Week0_LFOV.nii.gz")
    p.add_argument("--mask", required=True, help="HipMRI GT, e.g. .../semantic_labels_only/B006_Week0_SEMANTIC.nii.gz")
    p.add_argument("--checkpoint", required=True, help="runs/hipmri3d/best.pt")
    p.add_argument("--out", default="viz_hipmri3d.png")
    p.add_argument("--slice-idx", type=int, default=None, help="optional, pick specific z; default=middle")
    args = p.parse_args()

    # 1) load volume + gt
    vol = load_nifti(args.img)          # (Z, Y, X)
    gt  = load_nifti(args.mask)         # (Z, Y, X)
    vol_norm = normalise_img(vol)
    gt_bin   = binarise_mask(gt)

    z, h, w = vol_norm.shape
    if args.slice_idx is None:
        zmid = z // 2
    else:
        zmid = max(0, min(z - 1, args.slice_idx))

    # 2) load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt["args"]["model"]  # karena kamu simpan di train.py
    model = build_model(model_name, in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 3) infer full volume (1,1,Z,H,W)
    with torch.no_grad():
        inp = torch.from_numpy(vol_norm).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,Z,H,W)
        pred_logits = model(inp)  # (1,1,Z,H,W)
        pred = torch.sigmoid(pred_logits).squeeze(0).squeeze(0).cpu().numpy()  # (Z,H,W)
        pred_bin = (pred > 0.5).astype(np.float32)

    # 4) ambil slice yang sama
    img_slice = vol_norm[zmid]          # (H,W)
    gt_slice  = gt_bin[zmid]            # (H,W)
    pr_slice  = pred_bin[zmid]          # (H,W)

    # 5) plot rapi
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_slice, cmap="gray")
    axs[0].set_title("MRI Slice (z={})".format(zmid))
    axs[0].axis("off")

    # GT
    axs[1].imshow(img_slice, cmap="gray")
    axs[1].imshow(gt_slice, alpha=0.45, cmap="Reds")
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    # PRED
    axs[2].imshow(img_slice, cmap="gray")
    axs[2].imshow(pr_slice, alpha=0.45, cmap="Blues")
    axs[2].set_title("Model Prediction")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"saved to {args.out}")

if __name__ == "__main__":
    main()