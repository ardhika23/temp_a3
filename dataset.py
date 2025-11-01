# recognition/seg-oasis-to-hipmri-<uqid>/dataset.py
"""
Dataset loaders for:
- OASIS 2D segmentation (easy)
- HipMRI 2D slices (normal)
- HipMRI 3D volumes (hard)

IMPORTANT (COMP3710):
- Do NOT commit dataset files, .nii.gz, .npy, or generated cache to the repo.
- In README, tell markers to mount /home/groups/comp3710/... on Rangpur.

This version AUTO-DETECTS the Rangpur layout for OASIS:
    /home/groups/comp3710/OASIS/keras_png_slices_train
    /home/groups/comp3710/OASIS/keras_png_slices_seg_train
    /home/groups/comp3710/OASIS/keras_png_slices_validate
    /home/groups/comp3710/OASIS/keras_png_slices_seg_validate
    /home/groups/comp3710/OASIS/keras_png_slices_test
    /home/groups/comp3710/OASIS/keras_png_slices_seg_test

but still supports the "canonical" layout:
    root/train/images, root/train/masks, ...
"""

import os
from typing import Optional, Dict, Tuple, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np

try:
    import nibabel as nib  # for 3D volumes
except ImportError:
    nib = None

from PIL import Image

# ---------------------------------------------------------------------------
# Utility transforms
# ---------------------------------------------------------------------------
to_tensor_2d = T.Compose(
    [
        T.ToTensor(),  # HWC [0,1]
    ]
)


# ---------------------------------------------------------------------------
# OASIS 2D (auto-detect rangpur layout)
# ---------------------------------------------------------------------------
class OASIS2DDataset(Dataset):
    """
    Supports TWO layouts.

    1) Canonical layout (your own machine):
        root/
          train/images/*.png
          train/masks/*.png
          val/images/...
          test/...

    2) Rangpur layout (what you showed):
        root/
          keras_png_slices_train/
          keras_png_slices_seg_train/
          keras_png_slices_validate/
          keras_png_slices_seg_validate/
          keras_png_slices_test/
          keras_png_slices_seg_test/

    We will detect which one exists.
    """

    # map split -> (img_dirname, mask_dirname) for rangpur-style
    RANGPUR_MAP: Dict[str, Tuple[str, str]] = {
        "train": ("keras_png_slices_train", "keras_png_slices_seg_train"),
        "val": ("keras_png_slices_validate", "keras_png_slices_seg_validate"),
        "validate": ("keras_png_slices_validate", "keras_png_slices_seg_validate"),
        "test": ("keras_png_slices_test", "keras_png_slices_seg_test"),
    }

    def __init__(self, root: str, split: str = "train", transform=None):
        self.root = root
        self.split = "val" if split == "validate" else split  # normalize
        self.transform = transform or to_tensor_2d

        # 1) try canonical first
        canonical_img_dir = os.path.join(root, self.split, "images")
        canonical_mask_dir = os.path.join(root, self.split, "masks")

        if os.path.isdir(canonical_img_dir) and os.path.isdir(canonical_mask_dir):
            img_dir = canonical_img_dir
            mask_dir = canonical_mask_dir
        else:
            # 2) fallback to rangpur-style
            if self.split not in self.RANGPUR_MAP:
                raise ValueError(
                    f"OASIS2DDataset: unknown split {self.split} for rangpur layout"
                )
            img_name, mask_name = self.RANGPUR_MAP[self.split]
            img_dir = os.path.join(root, img_name)
            mask_dir = os.path.join(root, mask_name)

            if not (os.path.isdir(img_dir) and os.path.isdir(mask_dir)):
                raise FileNotFoundError(
                    f"Could not find OASIS dirs for split={self.split}. "
                    f"Tried canonical: {canonical_img_dir} and rangpur: {img_dir}"
                )

        self.img_paths: List[str] = sorted(
            [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        )
        self.mask_paths: List[str] = sorted(
            [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]
        )
        assert len(self.img_paths) == len(
            self.mask_paths
        ), f"Images ({len(self.img_paths)}) and masks ({len(self.mask_paths)}) must match"

    def __len__(self):
        return len(self.img_paths)

    def _load_any(self, path: str) -> np.ndarray:
        # .npy support just in case
        if path.endswith(".npy"):
            return np.load(path)
        # png/jpg
        img = Image.open(path)
        return np.array(img)

    def __getitem__(self, idx):
        img = self._load_any(self.img_paths[idx])
        mask = self._load_any(self.mask_paths[idx])

        # ensure channel for image
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        img = self.transform(img)  # -> (C,H,W)

        # mask: to (1,H,W) and BINARIZE
        if mask.ndim == 3:
            mask = mask[..., 0]
        # OASIS masks are usually 0 / 255
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1,H,W)
        return img, mask


# ---------------------------------------------------------------------------
# HipMRI 2D
# ---------------------------------------------------------------------------
class HipMRI2DDataset(Dataset):
    """
    HipMRI 2D processed slices at (typical):
      /home/groups/comp3710/HipMRI_Study_open/keras_slices_data

    I'll keep the canonical layout like:
      root/train/images/*.npy
      root/train/masks/*.npy
    so you can mirror it in your $HOME if needed.
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform or to_tensor_2d

        img_dir = os.path.join(root, split, "images")
        mask_dir = os.path.join(root, split, "masks")

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"HipMRI2DDataset: cannot find images dir at {img_dir} "
                f"(root={root}, split={split})"
            )
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(
                f"HipMRI2DDataset: cannot find masks dir at {mask_dir} "
                f"(root={root}, split={split})"
            )

        self.img_paths = sorted([os.path.join(img_dir, p) for p in os.listdir(img_dir)])
        self.mask_paths = sorted(
            [os.path.join(mask_dir, p) for p in os.listdir(mask_dir)]
        )
        assert len(self.img_paths) == len(
            self.mask_paths
        ), "HipMRI 2D: images and masks must have same length"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        if img_path.endswith(".npy"):
            img = np.load(img_path)
        else:
            img = np.array(Image.open(img_path))

        if mask_path.endswith(".npy"):
            mask = np.load(mask_path)
        else:
            mask = np.array(Image.open(mask_path))

        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        img = self.transform(img)

        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

# ---------------------------------------------------------------------------
# HipMRI 3D
# ---------------------------------------------------------------------------
class HipMRI3DDataset(Dataset):
    """
    3D volumes (nifti) for hard difficulty.
    Paths (adjust on Rangpur):
      /home/groups/comp3710/HipMRI_Study_open/semantic_MRs
      /home/groups/comp3710/HipMRI_Study_open/semantic_labels_only

    NOTE: nibabel is required.
    """

    def __init__(self, img_root: str, mask_root: str, split: str = "train"):
        assert nib is not None, "Please install nibabel for 3D data"
        self.img_root = img_root
        self.mask_root = mask_root
        self.split = split

        all_imgs = sorted(
            [
                f
                for f in os.listdir(img_root)
                if f.endswith(".nii") or f.endswith(".nii.gz")
            ]
        )
        n = len(all_imgs)
        # simple split
        if split == "train":
            self.files = all_imgs[: int(0.7 * n)]
        elif split in ("val", "validate"):
            self.files = all_imgs[int(0.7 * n) : int(0.85 * n)]
        else:
            self.files = all_imgs[int(0.85 * n) :]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_root, fname)
        mask_path = os.path.join(self.mask_root, fname)

        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)
        img = img_nii.get_fdata().astype(np.float32)
        mask = mask_nii.get_fdata().astype(np.float32)

        # normalise per-volume
        img = (img - img.mean()) / (img.std() + 1e-5)

        img = torch.from_numpy(img).unsqueeze(0)  # (1,Z,Y,X)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_dataset(
    name: str,
    root: str,
    split: str = "train",
    extra: Optional[dict] = None,
):
    name = name.lower()
    extra = extra or {}
    if name == "oasis2d":
        return OASIS2DDataset(root=root, split=split)
    if name == "hipmri2d":
        return HipMRI2DDataset(root=root, split=split)
    if name == "hipmri3d":
        img_root = extra.get(
            "img_root", "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
        )
        mask_root = extra.get(
            "mask_root", "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
        )
        return HipMRI3DDataset(img_root=img_root, mask_root=mask_root, split=split)
    raise ValueError(f"Unknown dataset: {name}")