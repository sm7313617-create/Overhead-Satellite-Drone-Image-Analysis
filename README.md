# 🛰️ SpaceNet1 Building Detection — U-Net + SAM (3-Band Evaluation)

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-5C3EE8?style=flat&logo=opencv&logoColor=white)
![SAM](https://img.shields.io/badge/SAM-ViT--B-FF6B35?style=flat)
![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-2×%20T4-76B900?style=flat&logo=nvidia&logoColor=white)
![IoU](https://img.shields.io/badge/U--Net%20IoU-0.631-brightgreen?style=flat)
![Dataset](https://img.shields.io/badge/Dataset-SpaceNet--1-orange?style=flat)

> A focused evaluation notebook comparing a trained **3-band U-Net** against **SAM zero-shot inference** for building footprint segmentation on the SpaceNet-1 dataset — running entirely on Kaggle with no Google Drive dependency.

🌐 [Live Demo](#) · 📓 [Notebook](notebooks/spacenet1-unet-sam-3band-evaluation.ipynb) · 🗃️ [Dataset](https://spacenet.ai/spacenet-buildings-dataset-v1/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Pipeline](#pipeline)
- [Results](#results)
- [Environment & Setup](#environment--setup)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Output Files](#output-files)

---

## Overview

This notebook trains a custom **3-band U-Net** from scratch on SpaceNet-1 RGB imagery and benchmarks it against **SAM ViT-B zero-shot segmentation**. The entire pipeline runs on Kaggle using a pre-generated mask dataset — no Google Drive mounting required.

Key features:
- 3-band (RGB) U-Net with `DoubleConv` blocks and skip connections
- `BCEDiceLoss` with class-imbalance weighting
- DataParallel training across 2× T4 GPUs
- SAM zero-shot inference with custom building-area and aspect-ratio filters
- Side-by-side U-Net vs SAM metric comparison

---

## Dataset

**SpaceNet-1 Buildings (3-band RGB + Pre-generated Masks)**

| Property | Details |
|---|---|
| Region | Rio de Janeiro, Brazil |
| Sensor | WorldView-3 |
| Bands | 3-band RGB (this notebook) |
| Tiles | ~6,940 GeoTIFF images |
| Resolution | 438 × 406 px (native), resized to 512 × 512 |
| Annotations | Binary building footprint masks (pre-generated) |
| Platform | Kaggle dataset: `spacenet1-3band-masks` |

> Dataset stored externally. See [`data/dataset_links.md`](../data/dataset_links.md) for download links.

---

## Model Architectures

### 🔷 U-Net (3-band)

A from-scratch encoder-decoder with skip connections, built for RGB satellite imagery.

```
Input: (B, 3, 512, 512)
  │
  ├─ Encoder: [64 → 128 → 256 → 512]  (DoubleConv + MaxPool)
  ├─ Bottleneck: 1024
  └─ Decoder: [512 → 256 → 128 → 64]  (ConvTranspose2d + skip concat)
  │
Output: (B, 1, 512, 512)  — logits
```

| Component | Detail |
|---|---|
| Input channels | 3 (R, G, B) |
| Feature maps | [64, 128, 256, 512] |
| Loss | BCEDiceLoss (pos_weight=5.0) |
| Optimizer | Adam (lr=1e-4) |
| Augmentation | H/V flip, 90° rotation, brightness/contrast |
| Multi-GPU | DataParallel (2× T4) |
| Checkpoint | Auto-resume from `unet_last.pth` |

### 🔶 SAM ViT-B (Zero-Shot)

Meta's Segment Anything Model applied out-of-the-box with a custom **building filter** post-processor.

| Component | Detail |
|---|---|
| Model | SAM ViT-B (`sam_vit_b_01ec64.pth`, ~375 MB) |
| Mode | Zero-shot (no fine-tuning) |
| Generator | `SamAutomaticMaskGenerator` (points_per_side=16) |
| Building filter | Area: 200–50,000 px; Aspect ratio ≤ 4.0 |
| Input | 3-band RGB uint8 |

---

## Pipeline

```
Kaggle Dataset (3-band RGB + pre-generated masks)
        │
        ├─► Path Discovery (dynamic glob across split folders)
        │
        ├─► Dataset Exploration & Visualisation (Step 4)
        │
        ├─► SpaceNetDataset + DataLoaders  (train/val/test split)
        │        └─ Albumentations augmentation
        │
        ├─► U-Net Training  (Steps 8–10)
        │        ├─ BCEDiceLoss
        │        ├─ DataParallel (2× T4)
        │        └─ Checkpoint-safe (auto-resume)
        │
        ├─► Full Evaluation  (Step 11)
        │        ├─ Loss & IoU curves
        │        ├─ Pixel-level: IoU / Precision / Recall / F1
        │        ├─ Confusion matrix
        │        ├─ Per-image IoU distribution
        │        └─ Prediction grid + polygon overlay
        │
        └─► SAM Inference & Comparison  (Step 12)
                 ├─ SAM zero-shot on 100 val images
                 ├─ Per-image IoU distribution (SAM)
                 ├─ Confusion matrix (SAM)
                 └─ U-Net vs SAM bar chart
```

---

## Results

### U-Net (3-band) — SpaceNet-1 Validation Set

| Metric | Score |
|---|---|
| **Mean IoU** | **0.631** |
| **F1 Score** | **0.724** |
| Precision | 0.601 |
| Recall | 0.910 |

### U-Net vs SAM — Comparison

| Model | Mean IoU | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **U-Net (trained)** | **0.631** | **0.601** | **0.910** | **0.724** |
| SAM ViT-B (zero-shot) | ~0.10 | — | — | — |

> SAM zero-shot struggles with the spectral properties of satellite imagery and large open-ground regions. The trained U-Net outperforms by a significant margin on this domain.

---

## Environment & Setup

### Requirements

```bash
pip install torch torchvision rasterio geopandas shapely fiona \
            albumentations segment-anything opencv-python \
            matplotlib tqdm
```

### SAM Checkpoint

```python
import urllib.request, os
os.makedirs('checkpoints', exist_ok=True)
urllib.request.urlretrieve(
    'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
    'checkpoints/sam_vit_b_01ec64.pth'
)
```

### Kaggle Dataset

Add the following dataset to your Kaggle notebook:
```
sayanmondal772/spacenet1-3band-masks
```

---

## Notebook Walkthrough

| Step | Description |
|---|---|
| 1 | GPU check (`nvidia-smi`, `torch.cuda.device_count()`) |
| 2 | Install dependencies (`rasterio`, `geopandas`, `segment-anything`, etc.) |
| 3 | Config & dynamic path discovery (handles split zip folders) |
| 4 | Dataset exploration — resolution, CRS, band stats, sample visualisation |
| 5 | Path helpers — `geojson_for_image()`, `mask_path_for_image()` |
| 6 | Quick sanity visualisation (RGB tile + binary mask overlay) |
| 7 | `SpaceNetDataset`, train/val/test split, Albumentations transforms |
| 8 | U-Net architecture (3-band, `DoubleConv`, encoder-decoder with skips) |
| 9 | `BCEDiceLoss`, Adam optimizer setup |
| 10 | Training loop with auto-resume checkpoint logic |
| 11a | Loss & IoU curves |
| 11b | Full val-set evaluation (IoU, Precision, Recall, F1) |
| 11c | Confusion matrix |
| 11d | Per-image IoU distribution histogram |
| 11e | Prediction grid (9 samples: RGB / GT mask / Predicted mask) |
| 11f | Polygon contour overlay (OpenCV) |
| 12 | SAM download, `SamAutomaticMaskGenerator` setup, zero-shot inference |
| 12+ | U-Net vs SAM bar chart & IoU distribution comparison |

---

## Output Files

| File | Description |
|---|---|
| `fig1_sample_tile.png` | Sample RGB tile visualisation |
| `fig2_mask_verify.png` | Mask sanity check |
| `fig6_loss_curves.png` | Train/val loss + IoU curves |
| `confusion_matrix.png` | U-Net pixel-level confusion matrix |
| `iou_distribution.png` | Per-image IoU histogram (U-Net) |
| `fig7_predictions.png` | 9-sample prediction grid |
| `fig8_polygons.png` | Polygon contour overlay |
| `fig_unet_vs_sam.png` | U-Net vs SAM metric comparison bar chart |
| `fig11_sam_iou_distribution.png` | Per-image IoU histogram (SAM) |
| `unet_best.pth` | Best U-Net checkpoint (by val loss) |
| `unet_last.pth` | Last epoch checkpoint (for resumption) |
