# recognition/seg-oasis-to-hipmri-<uqid>/train.py
"""
Training script for COMP3710 recognition tasks.

Rangpur examples:

# 1. Easy – OASIS 2D
python train.py --dataset oasis2d --root /home/groups/comp3710/OASIS \
    --model 2d --epochs 20 --outdir runs/oasis2d

# 2. Normal – HipMRI 2D
python train.py --dataset hipmri2d --root /home/groups/comp3710/HipMRI_Study_open/keras_slices_data \
    --model 2d-hip --epochs 15 --outdir runs/hipmri2d

# 3. Hard – HipMRI 3D
python train.py --dataset hipmri3d --model 3d-improved --epochs 80 --batch-size 1 \
    --outdir runs/hipmri3d
"""

import os
import argparse
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# matplotlib di rangpur kadang headless, jadi kita amankan
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules import build_model
from dataset import get_dataset


def dice_coef(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * inter + eps) / (union + eps)


def save_plots(outdir, train_losses, val_losses, val_dices):
    plot_dir = os.path.join(outdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # loss
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.title("Loss curves")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(os.path.join(plot_dir, "loss.png"))
    plt.close()

    # dice
    plt.figure()
    plt.plot(val_dices, label="val_dice")
    plt.legend()
    plt.title("Validation Dice")
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.savefig(os.path.join(plot_dir, "dice.png"))
    plt.close()
    # <-- SCREENSHOT THESE FOR README.md -->


def validate(model, loader, device, criterion):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)
            val_loss += loss.item()
            val_dice += dice_coef(logits, masks).item()
    n = len(loader)
    return val_loss / n, val_dice / n


def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="oasis2d | hipmri2d | hipmri3d")
    parser.add_argument("--root", type=str, default=".",
                        help="dataset root (2D). For 3D we use fixed Rangpur paths.")
    parser.add_argument("--model", type=str, default="2d",
                        help="2d | 2d-hip | 3d | 3d-improved")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--outdir", type=str, default="runs/debug")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset selection
    if args.dataset == "hipmri3d":
        # force rangpur 3D paths
        train_ds = get_dataset("hipmri3d", root="", split="train")
        val_ds = get_dataset("hipmri3d", root="", split="val")
        if args.batch-size > 2:
            args.batch_size = 1
    else:
        train_ds = get_dataset(args.dataset, root=args.root, split="train")
        val_ds = get_dataset(args.dataset, root=args.root, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.model, in_channels=1, out_channels=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_dice = 0.0
    train_losses, val_losses, val_dices = [], [], []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_dice = validate(model, val_loader, device, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        print(
            f"[{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
        )

        # save best checkpoint
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_dice": float(val_dice),
                    "args": vars(args),
                },
                os.path.join(args.outdir, "best.pt"),
            )
            print(f"saved new best model to {os.path.join(args.outdir, 'best.pt')}")

    # save last model-only
    torch.save(model.state_dict(), os.path.join(args.outdir, "last_model_only.pt"))

    # plots
    save_plots(args.outdir, train_losses, val_losses, val_dices)

    # text log (for README + PR)
    with open(os.path.join(args.outdir, "train_log.txt"), "w") as f:
        f.write(f"trained at {datetime.now()}\n")
        f.write(f"best_val_dice={best_val_dice:.4f}\n")
        for i, (tl, vl, vd) in enumerate(zip(train_losses, val_losses, val_dices), start=1):
            f.write(f"epoch {i}: train_loss={tl:.4f} val_loss={vl:.4f} val_dice={vd:.4f}\n")


if __name__ == "__main__":
    main()