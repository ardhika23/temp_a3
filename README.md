# OASIS → HipMRI Segmentation (2D to 3D) – COMP3710 Pattern Analysis

**Student:** `<your name>`  
**UQ ID:** `<uqid>`  
**Folder:** `recognition/seg-oasis-to-hipmri-<uqid>/`  
**Spec:** COMP3710 Pattern Analysis Report v1.64 Final (segmentation pathway)  

## 1. Problem Description
This project follows the recommended difficulty pathway in Section 1.4 of the assignment PDF: start with 2D segmentation on OASIS (easy), then increase difficulty by either (a) changing the dataset to HipMRI 2D **or** (b) upgrading the model to perform 3D segmentation on HipMRI volumes to obtain normal/hard marks.  [oai_citation:0‡COMP3710_Report_v1.64_Final (2).pdf](sediment://file_000000000f7c7208bdece503ca4e4ad6)

## 2. Dataset
- **OASIS 2D** (easy): path on Rangpur: `/home/groups/comp3710/.../OASIS_2D` (adjust to actual path).
- **HipMRI 2D** (normal): `/home/groups/comp3710/HipMRI_Study_open/keras_slices_data`
- **HipMRI 3D** (hard): 
  - images: `/home/groups/comp3710/HipMRI_Study_open/semantic_MRs`
  - labels: `/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only`
- **IMPORTANT:** Dataset files and trained model checkpoints **must not** be committed to Git.

## 3. Method
### 3.1 Baseline (Easy)
- Model: `UNet2DBaseline` (Ronneberger et al., 2015)
- Dataset: OASIS 2D
- Loss: BCEWithLogitsLoss
- Metric: Dice coefficient

### 3.2 Normal
- Model: `UNet2DHipMRI` (deeper 2D UNet)
- Dataset: HipMRI 2D
- Same loss/metric

### 3.3 Hard
- Model: `UNet3DImproved` (3D UNet with context/residual blocks)
- Dataset: HipMRI 3D volumes
- Target metric: **Dice ≥ 0.7** on validation (as per hard description in PDF)

## 4. How to Run
### 4.1 Train
```bash
# OASIS 2D
python train.py --dataset oasis2d --root /home/groups/comp3710/OASIS_2D \
  --model 2d --epochs 20 --outdir runs/oasis2d

# HipMRI 2D
python train.py --dataset hipmri2d --root /home/groups/comp3710/HipMRI_Study_open/keras_slices_data \
  --model 2d-hip --epochs 40 --outdir runs/hipmri2d

# HipMRI 3D (Rangpur GPU)
python train.py --dataset hipmri3d --model 3d-improved --epochs 80 --batch-size 1 \
  --outdir runs/hipmri3d

---

## Git milestones (do these in order)

1. **`feat: scaffold oasis-to-hipmri project`**  
   - add folder + empty files
2. **`feat: add 2d unet baseline modules`**  
   - commit `modules.py` (2D parts)
3. **`feat: add dataset loaders for oasis and hipmri`**  
   - commit `dataset.py`
4. **`feat: add training script with model/dataset switches`**  
   - commit `train.py`
5. **`feat: add prediction script for sample masks`**  
   - commit `predict.py`
6. **`docs: add initial README for seg oasis to hipmri`**  
   - commit `README.md`
7. **open PR** to course repo on `topic-recognition` with description (we’ll write the text in PHASE 3).

---

Next time (PHASE 3) I’ll write you the **final README content** (with discussion, limitations, AI section properly worded) **and** the **PR text** you can paste. Before that, when you’ve run 1–2 trainings, paste your console log (esp. best val dice) so I can put real numbers.