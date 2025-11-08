# Framework Design Document

## Overview

This document explains the design principles and architecture of the Few-Shot Sketch Recognition Framework, detailing how each component works and why specific design decisions were made.

## Problem Statement (FSUGR-2)

**Goal**: Design a few-shot sketch recognition framework that can recognize previously unseen object classes after being shown only a few example sketches.

**Key Challenges**:
1. Limited training examples per class (few-shot constraint)
2. Generalization to completely unseen classes
3. Capturing abstract sketch features despite high variance
4. Efficient learning from minimal data

## Architecture Design

### Two-Stage Learning Pipeline

#### Stage 1: Self-Supervised Pretraining

**Purpose**: Learn robust, generalizable sketch representations without class labels.

**Why Self-Supervised Learning?**
- Creates embedding space optimized for sketch structure
- Learns features that transfer to unseen classes
- Doesn't overfit to specific class labels
- Leverages large amounts of unlabeled sketch data

**Methods Implemented**:

1. **SimCLR (Simple Framework for Contrastive Learning)**
   - **How it works**: Creates two augmented views of each sketch and trains encoder to maximize agreement between views while minimizing agreement with other sketches
   - **Loss function**: NT-Xent (Normalized Temperature-scaled Cross Entropy)
   - **Key parameters**: Temperature (controls softness of similarity)
   - **Advantages**: Simple, effective, proven to work well
   - **Design choice**: We use temperature=0.5 for sketches (vs 0.07 for photos) because sketch features are more distinct

2. **BYOL (Bootstrap Your Own Latent)**
   - **How it works**: Uses two networks (online and target) where online predicts target's representation
   - **No negative pairs needed**: More stable training
   - **EMA updates**: Target network updated via exponential moving average
   - **Advantages**: Simpler than SimCLR, no tuning temperature
   - **Design choice**: EMA decay=0.996 provides smooth updates

#### Stage 2: Few-Shot Meta-Learning

**Purpose**: Learn to classify new classes with minimal examples using pretrained embeddings.

**Why Few-Shot Learning?**
- Traditional supervised learning fails with limited data
- Meta-learning enables rapid adaptation to new classes
- Episodic training simulates real-world few-shot scenarios

**Algorithms Implemented**:

1. **Prototypical Networks**
   - **How it works**: 
     - Compute prototype (mean embedding) for each class from support samples
     - Classify queries based on distance to nearest prototype
   - **Distance metrics**: Euclidean or Cosine
   - **Advantages**: 
     - Simple and interpretable
     - Computationally efficient
     - Works well in practice
   - **Design choice**: Default to Euclidean distance as it works better for sketch embeddings

2. **Matching Networks**
   - **How it works**:
     - Uses attention mechanism over support set
     - Compares query to all support samples
     - Weighted combination based on similarity
   - **Advantages**:
     - More flexible than Prototypical
     - Can capture complex relationships
     - Better for classes with high intra-class variance
   - **Design choice**: Optional attention module can be disabled for faster inference

3. **Relation Networks**
   - **How it works**:
     - Learns comparison metric with neural network
     - Predicts relation scores between query and support
   - **Advantages**:
     - Most flexible - metric is learned
     - Can capture non-linear relationships
   - **Trade-off**: More parameters, slower training

## Data Processing Pipeline

### Sketch-Specific Augmentations

**Design Philosophy**: Augmentations must preserve sketch structure while providing variation.

**Implemented Augmentations**:

1. **Geometric Transformations**:
   - Rotation (±20°): Sketches can be drawn at angles
   - Translation (±15%): Position invariance
   - Scaling (0.85-1.15): Size invariance
   - Horizontal flip: Many objects are symmetric

2. **Stroke-Level Augmentations**:
   - **Random Stroke Dropout**: Simulates incomplete sketches (dropout_prob=0.05-0.1)
   - **Random Stroke Thickness**: Varies pen width (±2 pixels)
   - Preserves overall sketch structure

3. **Contrast/Brightness**:
   - Simulates different drawing styles
   - Limited to maintain sketch visibility

**Why NOT standard image augmentations?**
- Color jitter: Not useful for mostly grayscale sketches
- Cutout: Destroys sketch structure too much
- Mixup: Doesn't make sense for sketches

### Dataset Handling

**TU-Berlin Dataset**:
- 250 classes → Split: 200 training, 50 testing
- Ensures test classes are completely unseen
- 60 samples per class for training, 10 for validation, 10 for testing
- Images resized to 224x224 (standard size)

**QuickDraw Dataset**:
- 345 categories available
- Default: 50 categories for faster experimentation
- 28x28 images upscaled to 224x224
- Split: 70% train, 15% val, 15% test

## Episodic Training Strategy

### Episode Structure

Each training episode consists of:
- **Support Set**: N classes × K samples (N-way K-shot)
- **Query Set**: N classes × Q samples (Q queries per class)
- **Episode Labels**: Remapped to [0, N-1] for the episode

**Example** (5-way 5-shot):
```
Support Set: 5 classes × 5 samples = 25 images
Query Set: 5 classes × 15 samples = 75 images
Task: Classify 75 queries into 5 classes using 25 support examples
```

### Why Episodic Training?

1. **Simulates Test Conditions**: Training episodes mimic real few-shot scenarios
2. **Meta-Learning**: Model learns to learn from small datasets
3. **Class Diversity**: Each episode has different classes, forcing generalization
4. **Prevents Overfitting**: Model can't memorize specific classes

## Training Hyperparameters

### Self-Supervised Learning

**Optimized for Sketches**:
- **Batch Size**: 128-256 (larger = better negative samples for SimCLR)
- **Learning Rate**: 0.001 (with cosine annealing)
- **Epochs**: 200 (SSL needs more epochs than supervised)
- **Optimizer**: Adam (more stable than SGD for SSL)
- **Temperature (SimCLR)**: 0.5 (higher than images due to sketch distinctiveness)

### Few-Shot Learning

**Optimized for Meta-Learning**:
- **Episodes per Epoch**: 1000 (enough diversity)
- **Learning Rate**: 0.001 (with step decay)
- **Epochs**: 50-100 (few-shot converges faster)
- **Optimizer**: Adam
- **N-way**: 5 (standard few-shot benchmark)
- **N-shot**: 1, 5, or 10 (1-shot is hardest)

## Evaluation Protocol

### Rigorous Testing

1. **Unseen Classes**: Test on completely different classes
2. **Multiple Episodes**: 600 episodes for statistical significance
3. **Confidence Intervals**: Report 95% CI for reliability
4. **Multiple Configurations**: Test 1-shot, 5-shot, 10-shot

### Metrics

- **Accuracy**: Primary metric (percentage correct)
- **Confidence Interval**: Reliability measure (±X%)
- **Per-Episode Accuracy**: Distribution of performance

### Baseline Comparisons

1. **Random**: ~20% (5-way)
2. **Supervised**: Train classifier on training classes, test on new
3. **From-Scratch Few-Shot**: No SSL pretraining
4. **Transfer Learning**: ImageNet pretrained → sketch finetuning

## Design Decisions and Trade-offs

### 1. Two-Stage vs End-to-End

**Choice**: Two-stage (SSL pretraining → Few-shot training)

**Rationale**:
- ✅ SSL creates better embeddings than end-to-end
- ✅ Can reuse pretrained encoder for multiple few-shot configurations
- ✅ More interpretable - can analyze embeddings separately
- ❌ Slower to train (two stages)

### 2. SimCLR vs BYOL

**Choice**: Support both, default to SimCLR

**Rationale**:
- SimCLR: Simpler, one hyperparameter (temperature)
- BYOL: More stable, no negative pairs needed
- Both work well - user choice depends on preference

### 3. Prototypical vs Matching Networks

**Choice**: Default to Prototypical

**Rationale**:
- ✅ Simpler and faster
- ✅ Works very well in practice
- ✅ More interpretable (class prototypes)
- Matching Networks: Better for complex cases but slower

### 4. Grayscale vs RGB

**Choice**: Grayscale (1 channel)

**Rationale**:
- Sketches are inherently grayscale
- Reduces computation (3× fewer parameters in first layer)
- Prevents model from learning spurious color patterns

### 5. Image Size: 224×224

**Choice**: 224×224 (can be configured)

**Rationale**:
- Standard size for pretrained models
- Good balance between detail and computation
- Can reduce to 128×128 for faster training

## Modular Design

### Extensibility

The framework is designed to be easily extended:

1. **New Encoders**: Add to `models/backbone.py`
2. **New SSL Methods**: Add to `models/contrastive.py`
3. **New Few-Shot Algorithms**: Add to `models/few_shot.py`
4. **New Datasets**: Add to `data/datasets.py`
5. **New Augmentations**: Add to `data/transforms.py`

### Factory Pattern

Used throughout for flexibility:
```python
encoder = get_encoder('sketch_cnn')
ssl_model = get_ssl_model('simclr', encoder=encoder)
fs_model = get_few_shot_model('prototypical', encoder=encoder)
```

## Performance Optimizations

1. **Data Loading**: Multi-worker data loading with pin_memory
2. **Mixed Precision**: Optional AMP support (not included by default)
3. **Batch Processing**: Efficient episode batching
4. **Checkpointing**: Save only best models to save space

## Future Improvements

Potential enhancements:
1. **More SSL Methods**: MoCo, SwAV, SimSiam
2. **Meta-Learning**: MAML, Reptile for faster adaptation
3. **Data Augmentation**: AutoAugment for sketches
4. **Temporal Modeling**: Use stroke sequences (temporal data)
5. **Multi-Modal**: Combine sketches with text descriptions
6. **Architecture Search**: NAS for optimal encoder
7. **Continual Learning**: Update model as new classes arrive

## Summary

This framework combines:
- ✅ State-of-the-art self-supervised learning (SimCLR, BYOL)
- ✅ Proven few-shot learning algorithms (Prototypical, Matching)
- ✅ Sketch-specific augmentations and preprocessing
- ✅ Rigorous evaluation on unseen classes
- ✅ Modular, extensible design
- ✅ Well-documented and easy to use

The result is a comprehensive, production-ready framework for few-shot sketch recognition that achieves strong performance on benchmark datasets while remaining flexible and easy to extend.

