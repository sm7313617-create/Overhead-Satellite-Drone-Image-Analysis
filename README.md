# Overhead (Satellite/Drone) Image Analysis

A deep learning pipeline for **building footprint extraction** from satellite and drone imagery, covering the full lifecycle from data ingestion through model training, evaluation, and cross-domain transfer learning.

---

## Project Overview

This project implements and compares multiple deep learning architectures for semantic segmentation of building footprints across two distinct image domains:

- **Phase 1 — Satellite Imagery:** Training on the SpaceNet-1 dataset (WorldView-3 multispectral imagery over Rio de Janeiro)
- **Phase 2 — Transfer Learning:** Adapting trained models to Indian drone imagery via the Svamitva dataset

---

## Repository Structure

```
overhead-satellite-image-analysis/
├── data/
│   └── dataset_links.md          # Links and storage info for all datasets
├── notebooks/
│   ├── oiu-sd.ipynb               # Main pipeline: 8-band U-Net + SAM fine-tuning + YOLO (Phases 1 & 2)
│   ├── spacenet1-unet-sam-3band-evaluation.ipynb  # 3-band U-Net training & U-Net vs SAM evaluation
│   └── spacenet1_full_download.ipynb              # Dataset download utility
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Datasets

### SpaceNet-1
- **Source:** [SpaceNet Buildings Dataset v1](https://spacenet.ai/spacenet-buildings-dataset-v1/)
- **AWS:** `s3://spacenet-dataset/spacenet/SN1_buildings/tarballs/`
- **Content:** ~6,940 WorldView-3 GeoTIFF tiles over Rio de Janeiro
  - 3-band RGB imagery
  - 8-band multispectral imagery (coastal, blue, green, yellow, red, red-edge, NIR1, NIR2)
  - GeoJSON building footprint annotations
- **Storage:** Hosted externally on Google Drive / Kaggle

### Svamitva (Phase 2)
- **Content:** Filtered RGB drone aerial imagery with binary building masks
- **Domain:** Indian rural/peri-urban settlements
- **Purpose:** Cross-domain transfer learning target

---

## Model Architectures

### 1. Custom U-Net (8-channel)
A from-scratch encoder-decoder network with skip connections, adapted for 8-band multispectral input.

- **Input:** 8-channel WorldView-3 GeoTIFF tiles at 400×400 px
- **Architecture:** 4-level encoder (`[64, 128, 256, 512]` feature maps), symmetric decoder with skip connections, `DoubleConv` blocks (Conv → BN → ReLU ×2)
- **Loss:** `BCEDiceLoss` (weighted BCE + Dice)
- **Optimizer:** Adam (lr=1e-4), DataParallel on 2× T4 GPUs
- **Augmentation:** Horizontal/vertical flip, 90° rotation, brightness/contrast jitter

### 2. Custom U-Net (3-channel)
Identical architecture adapted for 3-band RGB input; trained and evaluated on Kaggle with a pre-generated mask dataset.

### 3. SAM ViT-B Fine-Tuner
Meta's Segment Anything Model adapted for satellite imagery segmentation.

- **Base model:** SAM ViT-B (`sam_vit_b_01ec64.pth`)
- **Adaptation:** Frozen image encoder + prompt encoder; learnable `channel_adapter` (Conv2d 8→3) prepended; only the mask decoder is fine-tuned
- **Trainable params:** ~4.1M (mask decoder only)
- **Training:** Micro-batched forward passes to manage GPU memory; AMP with gradient clipping

### 4. YOLOv8-Nano (Segmentation)
Applied in Phase 2 as a lightweight detector-segmenter baseline.

- **Dataset format:** YOLO normalized polygon format, converted from binary masks via OpenCV contour extraction
- **Architecture:** YOLOv8-Nano segmentation head
- **Purpose:** Comparison against U-Net and SAM on drone imagery

---

## Pipeline Summary

### Phase 1 — SpaceNet-1 (Satellite)

```
Raw GeoTIFF (8-band / 3-band)
        │
        ├─► Mask Generation (rasterize GeoJSON → binary TIF)
        │
        ├─► Dataset & DataLoader (Albumentations augmentation)
        │
        ├─► U-Net Training (BCEDiceLoss, Adam, DataParallel)
        │
        ├─► U-Net Evaluation (IoU, Precision, Recall, F1, Confusion Matrix)
        │
        └─► SAM Zero-Shot + Fine-Tuning → Comparative Evaluation
```

### Phase 2 — Svamitva Drone (Transfer Learning)

```
Pre-trained 8-band models (U-Net + SAM)
        │
        ├─► Network Surgery (replace 8→3 channel adapter)
        │
        ├─► Zero-Shot Transfer Test
        │
        ├─► Phase 2 Fine-Tuning (drone dataset, lr=1e-4, 10 epochs)
        │
        ├─► U-Net + OpenCV Polygon Post-processing
        │
        └─► YOLOv8-Nano Training & Evaluation
```

---

## Results

### Phase 1 — SpaceNet-1 Test Set

| Model | Mean IoU | Precision | Recall | F1 Score |
|---|---|---|---|---|
| U-Net (3-band, trained) | 0.631 | 0.601 | 0.910 | 0.724 |
| SAM ViT-B (zero-shot) | ~0.10 | — | — | — |
| SAM ViT-B (fine-tuned) | — | — | — | — |

### Phase 2 — Svamitva Drone Dataset

| Model | Accuracy |
|---|---|
| U-Net (fine-tuned, Phase 2) | ~92% |
| YOLOv8-Nano (Segmentation) | Evaluated (see notebook) |

---

## Environment & Setup

### Compute
- **Training:** Kaggle (2× NVIDIA T4 GPUs, DataParallel)
- **Development:** Google Colab (T4 GPU) / Local Windows + VS Code

### Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
```
torch
torchvision
rasterio
geopandas
shapely
fiona
albumentations
segment-anything
ultralytics
opencv-python
scikit-learn
matplotlib
seaborn
tqdm
```

### SAM Checkpoint
Download the SAM ViT-B checkpoint (~375 MB) before running SAM cells:

```python
import urllib.request
urllib.request.urlretrieve(
    'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
    'checkpoints/sam_vit_b_01ec64.pth'
)
```

---

## Notebooks

### `oiu-sd.ipynb` — Main Full Pipeline
The primary research notebook covering both phases end-to-end:

1. SpaceNet-1 file pairing (8-band + GeoJSON by image ID)
2. Binary mask generation from GeoJSON annotations
3. Dataset and DataLoader setup (8-band, 400×400 px)
4. SAM ViT-B channel adapter definition
5. Loss functions (`DiceLoss`, `BCEDiceLoss`)
6. U-Net (8-channel) training + evaluation on SpaceNet-1 test set
7. SAM fine-tuning on SpaceNet-1 (frozen encoder, trainable decoder)
8. Phase 2: Svamitva drone DataLoader setup
9. Network surgery (8→3 channel adapter replacement)
10. Zero-shot transfer test
11. Phase 2 fine-tuning (U-Net + SAM)
12. YOLOv8-Nano dataset preparation and training
13. Comparative evaluation with confusion matrices

### `spacenet1-unet-sam-3band-evaluation.ipynb` — 3-Band Evaluation
A focused evaluation notebook on Kaggle using pre-generated 3-band RGB masks:

1. GPU check and dependency installation
2. Config and dynamic path discovery
3. Dataset exploration and visualisation
4. 3-band U-Net training (DataParallel, 2× T4)
5. Full evaluation: loss/IoU curves, confusion matrix, per-image IoU distribution
6. SAM zero-shot inference with custom building filter (area, aspect ratio)
7. U-Net vs SAM side-by-side comparison

---

## Contributors

| GitHub | Name |
|---|---|
| [@sm7313617-create](https://github.com/sm7313617-create) | Sayan Mondal |
| [@IshanGain](https://github.com/IshanGain) | Ishan Gain |
| [@Arka007-hustle](https://github.com/Arka007-hustle) | Pranjal Basu |

---

## References

- [SpaceNet Buildings Dataset v1](https://spacenet.ai/spacenet-buildings-dataset-v1/)
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) — Kirillov et al., 2023
- [YOLOv8](https://github.com/ultralytics/ultralytics) — Ultralytics
- [Svamitva Drone Aerial Images](https://www.kaggle.com/datasets/utkarshsaxenadn/svamitva-drone-aerial-images) — Kaggle
