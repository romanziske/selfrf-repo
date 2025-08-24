# selfRF

Library for Self-Supervised Learning for Radio Frequency Signals

## Installation

```bash
# Install torchsig
pip install git+https://github.com/TorchDSP/torchsig.git

# Install detectron2
pip install git+https://github.com/facebookresearch/detectron2.git

# Install rf-detr
pip install rfdetr

# Install ultralytics (YOLO)
pip install ultralytics


# Install this package in development mode
pip install -e .
```

## Training

### Pretraining

DenseCL:

```bash
python tools/pretraining.py --num-epochs 300 --batch-size 128 --backbone resnet50 --ssl-model densecl --spectrogram true --root datasets/50k --num-samples 50000
```

VigRegL:

```bash
python tools/pretraining.py --num-epochs 300 --batch-size 128 --backbone resnet50 --ssl-model VICREGL --spectrogram true --root datasets/50k --num-samples 50000
```

MAE:

```bash
python tools/pretraining.py --num-epochs 300 --batch-size 512 --backbone vit_base --ssl-model mae --spectrogram true --root datasets/50k --num-samples 50000
```

### Finetuning

#### Faster-RCNN + ResNet50

Training:

```bash
python tools/train_detection_faster_rcnn.py \
    --root <path_to_dataset> \
    --dataset-path wideband_impaired \
    --max-iter 1500 \
    --num-classes 10 \
    --ims-per-batch 16 \
    --base-lr 0.0025
```

Inference:

```bash
python tools/train_detection_faster_rcnn.py \
    --root <path_to_dataset> \
    --max-iter 90000 \
    --num-classes 10 \
    --ims-per-batch 16 \
    --base-lr 0.0025 \
    --model-path <path_to_model> \
    --inference
```

## Scripts

### Convert ResNet to Detectron2

```bash
python tools/convert/convert_resnet_to_detectron2.py \
    --input_checkpoint path-to-checkpoint.ckpt \
    --output_path path-to-output.pth
```

### Convert ViT to Detectron2

```bash
python tools/convert/convert_vit_to_detectron2.py --input_checkpoint /home/sigence/repos/selfRF/MAE-vit-vit_base_patch16_224-narrowband-spec-e145-b32-loss0.001_backbone.ckpt  --output_path mae_vit_detectron2_new.pth
```

### Convert TorchSig to COCO Format

```bash
# Detection dataset conversion
python tools/convert/torchsig_to_coco.py --input_dir /home/sigence/repos/selfRF/datasets/100k/wideband_impaired  --mode family_recognition

# Recognition dataset conversion
python tools/convert/torchsig_to_coco.py \
    --input_dir /path/to/torchsig/dataset \
    --mode recognition

# Family recognition dataset conversion
python tools/convert/torchsig_to_coco.py \
    --input_dir /path/to/torchsig/dataset \
    --mode family_recognition \
```

### Convert SigMF to COCO Format

```bash
# Detection conversion for detection
python tools/convert/sigmf_to_coco.py \
    --sigmf_input_dir /path/to/sigmf/files \
    --mode detection

# Recognition mode
python tools/convert/sigmf_to_coco.py \
    --sigmf_input_dir /path/to/sigmf/files \
    --mode recognition
```

## Datasets

RadDet: [https://github.com/abcxyzi/RadDet](https://github.com/abcxyzi/RadDet)

TorchSig: [https://github.com/TorchDSP/torchsig](https://github.com/TorchDSP/torchsig)

## Results

### Signal Detection Data Efficincy

Pretrained on TorchSig Narrowband 50k.
Evaluated and trained on Torchsig WB 100k
Batch size 16.
75k iterations
Random Initialization Results:

| Model        | Backbone | Dataset Size | mAP   | mAP50 | mAP75 |
| ------------ | -------- | ------------ | ----- | ----- | ----- |
| Faster R-CNN | ResNet50 | 1k           | 47.71 | 74.90 | 50.83 |
| Faster R-CNN | ResNet50 | 5k           | 55.17 | 81.55 | 59.54 |
| Faster R-CNN | ResNet50 | 10k          | 58.40 | 84.55 | 63.08 |
| Faster R-CNN | ResNet50 | 25k          | 63.31 | 87.77 | 68.25 |
| Faster R-CNN | ResNet50 | 50k          | 63.73 | 87.88 | 69.59 |
| Faster R-CNN | ResNet50 | 100k         | 64.69 | 88.83 | 70.98 |

Imagenet Pretraining Results:

| Model        | Backbone | Dataset Size | mAP   | mAP50 | mAP75 | Δ mAP | Δ %    |
| ------------ | -------- | ------------ | ----- | ----- | ----- | ----- | ------ |
| Faster R-CNN | ResNet50 | 1k           | 53.48 | 79.00 | 57.55 | 5.77  | 12.09% |
| Faster R-CNN | ResNet50 | 5k           | 62.32 | 86.21 | 68.55 | 7.15  | 12.96% |
| Faster R-CNN | ResNet50 | 10k          | 65.67 | 88.44 | 71.95 | 7.27  | 12.45% |
| Faster R-CNN | ResNet50 | 25k          | 66.33 | 88.86 | 73.59 | 3.02  | 4.77%  |
| Faster R-CNN | ResNet50 | 50k          | 67.04 | 89.59 | 73.74 | 3.31  | 5.19%  |
| Faster R-CNN | ResNet50 | 100k         | 67.21 | 89.61 | 73.93 | 2.52  | 3.90%  |

VICRegL Pretraining Results:

| Model        | Backbone | Dataset Size | mAP   | mAP50 | mAP75 | Δ mAP | Δ %    |
| ------------ | -------- | ------------ | ----- | ----- | ----- | ----- | ------ |
| Faster R-CNN | ResNet50 | 1k           | 53.86 | 80.08 | 58.43 | 6.15  | 12.89% |
| Faster R-CNN | ResNet50 | 5k           | 62.89 | 86.18 | 68.82 | 7.72  | 13.99% |
| Faster R-CNN | ResNet50 | 10k          | 67.38 | 88.45 | 74.07 | 8.98  | 15.38% |
| Faster R-CNN | ResNet50 | 25k          | 70.72 | 90.61 | 77.55 | 7.41  | 11.70% |
| Faster R-CNN | ResNet50 | 50k          | 71.13 | 90.74 | 77.83 | 7.40  | 11.61% |
| Faster R-CNN | ResNet50 | 100k         | 71.39 | 91.22 | 77.88 | 6.70  | 10.36% |

DenseCL Pretraining Results:

| Model        | Backbone | Dataset Size | mAP   | mAP50 | mAP75 | Δ mAP | Δ %   |
| ------------ | -------- | ------------ | ----- | ----- | ----- | ----- | ----- |
| Faster R-CNN | ResNet50 | 1k           | 50.65 | 78.71 | 54.33 | 2.94  | 6.16% |
| Faster R-CNN | ResNet50 | 5k           | 58.85 | 85.18 | 64.41 | 3.68  | 6.67% |
| Faster R-CNN | ResNet50 | 10k          | 62.02 | 87.23 | 69.27 | 3.62  | 6.20% |
| Faster R-CNN | ResNet50 | 25k          | 67.08 | 88.96 | 73.42 | 3.77  | 5.95% |
| Faster R-CNN | ResNet50 | 50k          | 67.39 | 89.15 | 74.21 | 3.66  | 5.74% |
| Faster R-CNN | ResNet50 | 100k         | 67.99 | 89.62 | 74.67 | 3.30  | 5.10% |

### Signal Family Recognition Data Efficincy

Pretrained on TorchSig Narrowband 50k.
Evaluated and trained on Torchsig WB 100k

Random Initialization Results:

| Model        | Backbone | Dataset Size | mAP   | mAP50 | mAP75 |
| ------------ | -------- | ------------ | ----- | ----- | ----- |
| Faster R-CNN | ResNet50 | 1k           | 17.15 | 29.94 | 16.90 |
| Faster R-CNN | ResNet50 | 5k           | 24.19 | 36.97 | 26.56 |
| Faster R-CNN | ResNet50 | 10k          | 27.59 | 40.54 | 29.74 |
| Faster R-CNN | ResNet50 | 25k          | 32.40 | 44.40 | 35.46 |
| Faster R-CNN | ResNet50 | 50k          | 32.61 | 44.90 | 35.49 |
| Faster R-CNN | ResNet50 | 100k         | 32.83 | 45.06 | 35.86 |

ImageNet Pretraining Results:

| Model        | Backbone | Dataset Size | mAP   | mAP50 | mAP75 | Δ mAP | Δ %     |
| ------------ | -------- | ------------ | ----- | ----- | ----- | ----- | ------- |
| Faster R‑CNN | ResNet50 | 1k           | 24.18 | 38.13 | 25.11 | 7.03  | 40.99 % |
| Faster R‑CNN | ResNet50 | 5k           | 30.03 | 43.67 | 32.06 | 5.84  | 24.14 % |
| Faster R‑CNN | ResNet50 | 10k          | 33.25 | 46.76 | 36.72 | 5.66  | 20.51 % |
| Faster R‑CNN | ResNet50 | 25k          | 37.95 | 51.93 | 41.74 | 5.55  | 17.13 % |
| Faster R‑CNN | ResNet50 | 50k          | 40.46 | 54.75 | 44.03 | 7.85  | 24.07 % |
| Faster R‑CNN | ResNet50 | 100k         | 40.69 | 55.03 | 44.18 | 7.86  | 23.94 % |

VICRegL Pretraining Results:

| Model        | Backbone | Dataset Size | mAP   | mAP50 | mAP75 | Δ mAP | Δ %     |
| ------------ | -------- | ------------ | ----- | ----- | ----- | ----- | ------- |
| Faster R‑CNN | ResNet50 | 1k           | 23.07 | 38.58 | 25.71 | 5.92  | 34.52 % |
| Faster R‑CNN | ResNet50 | 5k           | 31.09 | 43.65 | 32.47 | 6.90  | 28.52 % |
| Faster R‑CNN | ResNet50 | 10k          | 33.14 | 44.69 | 36.59 | 5.55  | 20.12 % |
| Faster R‑CNN | ResNet50 | 25k          | 39.83 | 55.56 | 43.94 | 7.43  | 22.93 % |
| Faster R‑CNN | ResNet50 | 50k          | 42.75 | 55.65 | 47.19 | 10.14 | 31.09 % |
| Faster R‑CNN | ResNet50 | 100k         | 43.50 | 56.31 | 48.01 | 10.67 | 32.50 % |

DenseCL Pretraining Results:

| Model        | Backbone | Dataset Size | mAP   | mAP50 | mAP75 | Δ mAP | Δ %     |
| ------------ | -------- | ------------ | ----- | ----- | ----- | ----- | ------- |
| Faster R‑CNN | ResNet50 | 1k           | 21.84 | 35.38 | 23.43 | 4.69  | 27.35 % |
| Faster R‑CNN | ResNet50 | 5k           | 29.19 | 41.71 | 31.67 | 5.00  | 20.67 % |
| Faster R‑CNN | ResNet50 | 10k          | 31.29 | 43.81 | 34.34 | 3.70  | 13.41 % |
| Faster R‑CNN | ResNet50 | 25k          | 35.75 | 48.65 | 39.24 | 3.35  | 10.34 % |
| Faster R‑CNN | ResNet50 | 50k          | 38.97 | 51.81 | 43.13 | 6.36  | 19.50 % |
| Faster R‑CNN | ResNet50 | 100k         | 39.18 | 51.91 | 43.28 | 6.35  | 19.34 % |

MAE ViT Results:

| Model  | Backbone             | mAP   | mAP50 | mAP75 | Δ mAP | Δ %    |
| ------ | -------------------- | ----- | ----- | ----- | ----- | ------ |
| ViTDet | ViT-B-scratch        | 35.18 | 47.05 | 38.79 | 0.00  | 0.00%  |
| ViTDet | ViT-B-mae-narrowband | 37.01 | 49.57 | 40.19 | 1.83  | 5.20%  |
| ViTDet | ViT-B-mae-wideband   | 38.06 | 50.08 | 44.17 | 2.88  | 8.19%  |
| ViTDet | ViT-B-mae-imagenet   | 41.14 | 52.62 | 44.42 | 5.96  | 16.94% |

### Signal Recognition Transfer Learning

Dataset: raddet40k512HW009Tv2

| Model        | Backbone | Pretraining | mAP   | mAP50 | mAP75 | Δ mAP | Δ %     |
| ------------ | -------- | ----------- | ----- | ----- | ----- | ----- | ------- |
| Faster R‑CNN | ResNet50 | random      | 27.80 | 36.08 | 31.24 |       |         |
| Faster R‑CNN | ResNet50 | ImageNet    | 43.31 | 49.70 | 45.98 | 15.51 | 55.83 % |
| Faster R‑CNN | ResNet50 | DenseCL     | 33.69 | 42.49 | 37.74 | 5.89  | 21.19 % |
| Faster R‑CNN | ResNet50 | VICRegL     | 37.42 | 46.17 | 42.18 | 9.62  | 34.64 % |
