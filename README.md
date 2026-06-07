# CIFAR-10 Image Classification with Deep Learning

A complete deep learning pipeline for CIFAR-10 classification using PyTorch.

## Overview

CIFAR-10 consists of 60,000 32×32 color images across 10 classes (airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks). This project implements a CNN that progressively learns features from edges and textures to object parts, achieving ~75-85% validation accuracy.

## Architecture
Input (32×32×3)
↓
[Conv3×3 + ReLU + BatchNorm + MaxPool] × 2
↓
[Conv3×3 + ReLU + BatchNorm] × 2
↓
Global Average Pooling
↓
Dense (512 → 256 → 10) + Dropout
↓
Softmax

text

**Why this works:**
- Convolutions learn spatial hierarchies (edges → textures → objects)
- BatchNorm stabilizes training and enables higher learning rates
- Dropout prevents overfitting (50K training images is modest)
- Global pooling reduces parameters while preserving spatial info

## Training Pipeline

1. **Data augmentation** - Random flips, crops, rotations (+8-10% accuracy)
2. **Normalization** - Standardize to mean=0.5, std=0.5
3. **Loss function** - Cross-entropy (penalizes confident wrong predictions)
4. **Optimizer** - Adam (adaptive) or SGD with momentum
5. **Learning rate scheduling** - Reduce on plateau
6. **Early stopping** - Save best model by validation accuracy

## Performance Benchmarks

| Model | Params | Accuracy | Training (GPU) |
|-------|--------|----------|----------------|
| Simple CNN | ~100K | 70-75% | 5 min |
| Medium CNN (+BN) | ~500K | 78-82% | 15 min |
| ResNet-20 | ~270K | 88-90% | 30 min |

**Human baseline:** ~94% (challenging due to 32×32 resolution)

## Common Challenges & Solutions

| Problem | Fix |
|---------|-----|
| Overfitting (99% train, 70% val) | Add dropout, augment data, reduce model size |
| Underfitting (<60% both) | Increase capacity, train longer |
| Vanishing gradients | BatchNorm, residual connections |
| Class imbalance | Class weights, oversampling |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train.py --epochs 30 --batch-size 128

# Generate results
python generate_presentation.py
