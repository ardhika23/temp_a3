import os
from typing import Optional, Dict, Tuple, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np

try:
    import nibabel as nib  
except ImportError:
    nib = None

from PIL import Image

# Utility transforms
to_tensor_2d = T.Compose(
    [
        T.ToTensor(),  # HWC [0,1]
    ]
)

# OASIS 2D (auto-detect rangpur layout)
class OASIS2DDataset(Dataset):
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

# HipMRI 2D
class HipMRI2DDataset(Dataset):
    SPLIT_MAP = {
        "train": ("keras_slices_train", "keras_slices_seg_train"),
        "val": ("keras_slices_validate", "keras_slices_seg_validate"),
        "validate": ("keras_slices_validate", "keras_slices_seg_validate"),
        "test": ("keras_slices_test", "keras_slices_seg_test"),
    }

    def __init__(self, root: str, split: str = "train", transform=None, target_size=(256, 256)):
        assert nib is not None, "Please install nibabel to read .nii.gz HipMRI slices"
        self.root = root
        self.split = split
        self.transform = transform or to_tensor_2d
        self.target_size = target_size  # (H, W)

        if split not in self.SPLIT_MAP:
            raise ValueError(f"HipMRI2DDataset: unknown split {split}")

        img_dirname, mask_dirname = self.SPLIT_MAP[split]
        img_dir = os.path.join(root, img_dirname)
        mask_dir = os.path.join(root, mask_dirname)

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"HipMRI2DDataset: images dir not found: {img_dir}")
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"HipMRI2DDataset: masks dir not found: {mask_dir}")

        self.img_paths = sorted(
            [os.path.join(img_dir, f) for f in os.listdir(img_dir)
             if f.endswith(".nii") or f.endswith(".nii.gz")]
        )
        self.mask_paths = sorted(
            [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
             if f.endswith(".nii") or f.endswith(".nii.gz")]
        )
        assert len(self.img_paths) == len(self.mask_paths), "HipMRI 2D: images and masks count mismatch"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)

        img = img_nii.get_fdata().astype(np.float32)   # (H,W) or (H,W,1)
        mask = mask_nii.get_fdata().astype(np.float32)

        # make image (H,W,1)
        if img.ndim == 2:
            img = np.expand_dims(img, -1)

        # to tensor first
        img = self.transform(img)  # (1,H,W)

        # --- resize image & mask to same size ---
        # we use torchvision F.resize
        import torchvision.transforms.functional as F

        img = F.resize(img, self.target_size)  # (1, 256, 256)

        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)      # (1,H,W)
        # mask is still original size → resize too
        mask = F.resize(mask, self.target_size, interpolation=Image.NEAREST, antialias=False)  # (1, 256, 256)

        return img, mask

# HipMRI 3D
class HipMRI3DDataset(Dataset):
    def __init__(self, img_root: str, mask_root: str, split: str = "train"):
        assert nib is not None, "Please install nibabel for 3D data"
        self.img_root = img_root
        self.mask_root = mask_root
        self.split = split

        # collect all image and mask files
        img_files = sorted(
            [f for f in os.listdir(img_root) if f.endswith(".nii") or f.endswith(".nii.gz")]
        )
        mask_files = sorted(
            [f for f in os.listdir(mask_root) if f.endswith(".nii") or f.endswith(".nii.gz")]
        )

        # build a dict for masks so we can match them by prefix
        mask_map = {}
        for m in mask_files:
            # example: B006_Week0_SEMANTIC.nii.gz -> B006_Week0
            key = m.replace("_SEMANTIC", "").replace(".nii.gz", "").replace(".nii", "")
            mask_map[key] = m

        paired = []
        for img in img_files:
            # example: B006_Week0_LFOV.nii.gz -> B006_Week0
            key = img.replace("_LFOV", "").replace(".nii.gz", "").replace(".nii", "")
            if key in mask_map:
                paired.append((img, mask_map[key]))

        if not paired:
            raise RuntimeError(
                f"HipMRI3DDataset: still no paired files after fuzzy matching.\n"
                f"img_root={img_root}\nmask_root={mask_root}"
            )

        # simple 70/15/15 split
        n = len(paired)
        if split == "train":
            self.paired = paired[: int(0.7 * n)]
        elif split in ("val", "validate"):
            self.paired = paired[int(0.7 * n): int(0.85 * n)]
        else:
            self.paired = paired[int(0.85 * n):]

    def __len__(self):
        return len(self.paired)

    def __getitem__(self, idx):
        img_name, mask_name = self.paired[idx]
        img_path = os.path.join(self.img_root, img_name)
        mask_path = os.path.join(self.mask_root, mask_name)

        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)

        img = img_nii.get_fdata().astype(np.float32)
        mask = mask_nii.get_fdata().astype(np.float32)

        # normalize image volume
        img = (img - img.mean()) / (img.std() + 1e-5)

        # binarize mask – every voxel > 0 is foreground
        mask = (mask > 0).astype(np.float32)

        img = torch.from_numpy(img).unsqueeze(0)   # (1, Z, Y, X)
        mask = torch.from_numpy(mask).unsqueeze(0) # (1, Z, Y, X)
        return img, mask
    
# Factory
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