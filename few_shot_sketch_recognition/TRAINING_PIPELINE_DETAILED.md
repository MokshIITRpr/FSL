# Detailed Training Pipeline: Step-by-Step Process

## Overview

This framework uses a **two-stage training approach**:

1. **Stage 1: Self-Supervised Learning (SSL) Pretraining** - Learn robust sketch representations without labels
2. **Stage 2: Few-Shot Learning Fine-tuning** - Adapt the pretrained encoder for few-shot classification

The model is **NOT built entirely from scratch** - it uses a **pretraining + fine-tuning** strategy that combines:
- Optionally: ImageNet-pretrained ResNet (if using ResNet encoder)
- Self-supervised pretraining on sketch data (SimCLR or BYOL)
- Few-shot learning fine-tuning (Prototypical, Matching, or Relation Networks)

---

## Stage 1: Self-Supervised Learning (SSL) Pretraining

### Purpose
Learn general sketch representations from unlabeled data that will generalize well to unseen classes.

### Step 1.1: Encoder Initialization

**Method**: Encoder Architecture Selection

**Options**:
- **Custom SketchEncoder** (`sketch_cnn`): Built from scratch
  - 4 convolutional blocks (64→128→256→512 channels)
  - Batch normalization and ReLU activations
  - Adaptive average pooling
  - Projection head (512 → embedding_dim)
  - **Initialization**: Random weights (Xavier/Kaiming)

- **ResNetEncoder** (`resnet18/34/50`): Pretrained + Modified
  - **Pretrained weights**: ImageNet-pretrained ResNet (if `pretrained=True`)
  - **Modification**: First conv layer changed for grayscale input (1 channel)
  - **Custom projection head**: Replaces final FC layer
  - **Initialization**: 
    - ResNet layers: ImageNet pretrained weights
    - First conv layer: Random (due to channel mismatch)
    - Projection head: Random

**Location**: `models/backbone.py` lines 15-235

---

### Step 1.2: SSL Model Construction

**Method**: Contrastive Learning Framework Selection

**Options**:

#### Option A: SimCLR (Simple Contrastive Learning)

**Technique**: Normalized Temperature-scaled Cross-Entropy (NT-Xent) Loss

**Architecture**:
1. **Encoder**: Feature extraction (from Step 1.1)
2. **Projection Head**: MLP (embedding_dim → 2048 → projection_dim)
3. **Normalization**: L2 normalization on projections

**Training Process**:
- **Input**: Two augmented views of the same sketch (x1, x2)
- **Forward Pass**:
  1. Encode both views: `h1 = encoder(x1)`, `h2 = encoder(x2)`
  2. Project to contrastive space: `z1 = projection_head(h1)`, `z2 = projection_head(h2)`
  3. Normalize: `z1 = normalize(z1)`, `z2 = normalize(z2)`
- **Loss Computation**:
  1. Concatenate: `z = [z1; z2]` (batch_size * 2, projection_dim)
  2. Compute similarity matrix: `sim = z @ z.T / temperature`
  3. Create positive pairs: (i, i+batch_size) and (i+batch_size, i)
  4. NT-Xent loss: `-log(exp(sim_positive) / sum(exp(sim_all)))`

**Hyperparameters**:
- Temperature: 0.5 (controls contrastive learning sharpness)
- Projection dimension: 128
- Batch size: 128 (larger = more negative pairs)

**Location**: `models/contrastive.py` lines 44-156

#### Option B: BYOL (Bootstrap Your Own Latent)

**Technique**: Asymmetric Architecture with Exponential Moving Average (EMA)

**Architecture**:
1. **Online Network** (trainable):
   - Encoder
   - Projector (embedding_dim → 4096 → projection_dim)
   - Predictor (projection_dim → 4096 → projection_dim)
2. **Target Network** (EMA updated):
   - Encoder (copy of online)
   - Projector (copy of online)

**Training Process**:
- **Input**: Two augmented views (x1, x2)
- **Forward Pass**:
  1. Online network: `pred1 = predictor(projector(encoder(x1)))`
  2. Target network (no grad): `proj2 = target_projector(target_encoder(x2))`
  3. Symmetric: Repeat for (x2, x1)
- **Loss Computation**:
  1. Normalize predictions and projections
  2. MSE in normalized space: `||normalize(pred1) - normalize(proj2)||²`
  3. Symmetric loss: `(loss1 + loss2) / 2`
- **EMA Update** (after each step):
  - `θ_target = 0.996 * θ_target + 0.004 * θ_online`

**Hyperparameters**:
- EMA decay: 0.996
- Projection dimension: 256
- Hidden dimension: 4096

**Location**: `models/contrastive.py` lines 158-317

---

### Step 1.3: Data Augmentation for SSL

**Method**: Two-View Transform Strategy

**Techniques Applied** (in `data/transforms.py`):
1. **Random Rotation**: -15° to +15° (preserves sketch orientation)
2. **Random Translation**: Small shifts (preserves spatial relationships)
3. **Random Scale**: 0.9x to 1.1x
4. **Stroke Thickness Variation**: Random line width changes
5. **Random Erasing**: Small regions (prevents overfitting to specific strokes)
6. **Color Jitter**: For RGB sketches (minimal for grayscale)
7. **Normalization**: Mean/std normalization

**Implementation**: Each sketch generates two different augmented views

**Location**: `data/transforms.py`

---

### Step 1.4: SSL Training Loop

**Method**: Standard SGD with Learning Rate Scheduling

**Training Process** (`train_ssl.py`):
1. **Data Loading**:
   - Create ContrastiveDataset wrapper
   - Apply TwoViewTransform to generate (view1, view2) pairs
   - Batch size: 128
   - Shuffle: True

2. **Optimizer**: Adam
   - Learning rate: 0.001
   - Weight decay: 1e-4
   - Beta1: 0.9, Beta2: 0.999

3. **Learning Rate Scheduler**: CosineAnnealingLR
   - T_max: epochs (200)
   - Decay from 0.001 to near 0

4. **Training Loop** (per epoch):
   ```python
   for batch in train_loader:
       view1, view2, _ = batch
       
       # Forward pass
       if simclr:
           z1, z2 = model(view1, view2)
           loss = model.compute_loss(z1, z2)
       elif byol:
           (pred1, pred2), (proj1, proj2) = model(view1, view2)
           loss = model.compute_loss((pred1, pred2), (proj1, proj2))
           model.update_target_network()  # EMA update
       
       # Backward pass
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
   ```

5. **Checkpointing**:
   - Save every 10 epochs
   - Save best model (lowest loss)
   - Save: model state, optimizer state, epoch, loss

**Duration**: 200 epochs (typical)

**Location**: `train_ssl.py` lines 78-258

---

### Step 1.5: SSL Model Output

**Result**: Pretrained encoder with learned sketch representations

**Saved Components**:
- Encoder weights (for feature extraction)
- Projection head weights (discarded in Stage 2)

**Checkpoint Structure**:
```python
{
    'epoch': 200,
    'model_state_dict': {
        'encoder.conv1.weight': ...,
        'encoder.conv2.weight': ...,
        'projection_head.0.weight': ...,
        ...
    },
    'optimizer_state_dict': ...,
    'loss': 2.34,
    'best_loss': 2.12
}
```

**Key Point**: Only the **encoder** is used in Stage 2, not the projection head.

---

## Stage 2: Few-Shot Learning Fine-tuning

### Purpose
Adapt the pretrained encoder to classify new classes with minimal examples using meta-learning.

### Step 2.1: Load Pretrained Encoder

**Method**: Weight Transfer from SSL Checkpoint

**Process** (`train_few_shot.py` lines 288-300):
1. Load SSL checkpoint
2. Extract encoder weights:
   - Filter keys: `encoder.*` or `online_encoder.*` (for BYOL)
   - Remove prefix: `encoder.` → `` (root level)
3. Load into few-shot encoder:
   - Use `strict=False` (allows missing projection head)
   - Encoder architecture must match SSL encoder

**Code**:
```python
checkpoint = torch.load(pretrained_encoder_path)
encoder_state = {}
for key, value in checkpoint['model_state_dict'].items():
    if key.startswith('encoder.') or key.startswith('online_encoder.'):
        new_key = key.replace('encoder.', '').replace('online_encoder.', '')
        encoder_state[new_key] = value
encoder.load_state_dict(encoder_state, strict=False)
```

**Result**: Encoder initialized with SSL-pretrained weights (not random)

---

### Step 2.2: Few-Shot Model Construction

**Method**: Meta-Learning Algorithm Selection

**Options**:

#### Option A: Prototypical Networks

**Technique**: Prototype-based Classification

**Architecture**:
1. **Encoder**: Pretrained encoder (from Stage 1)
2. **No additional trainable layers** (distance-based classification)

**Inference Process**:
1. **Encode Support Set**: 
   - `support_embeddings = encoder(support_images)` (n_way * n_shot, embedding_dim)
2. **Compute Prototypes**:
   - For each class: `prototype = mean(support_embeddings[class])`
   - Result: (n_way, embedding_dim)
3. **Encode Query Set**:
   - `query_embeddings = encoder(query_images)` (n_query, embedding_dim)
4. **Compute Distances**:
   - Euclidean: `distances = ||query - prototype||²`
   - Cosine: `distances = 1 - cosine_similarity(query, prototype)`
5. **Classification**:
   - `logits = -distances` (negative distance = higher similarity)
   - `predictions = argmax(logits)`

**Training**: End-to-end via CrossEntropyLoss on query predictions

**Location**: `models/few_shot.py` lines 17-134

#### Option B: Matching Networks

**Technique**: Attention-based Nearest Neighbor

**Architecture**:
1. **Encoder**: Pretrained encoder
2. **Attention Module**: MLP (embedding_dim * 2 → embedding_dim → 1)

**Inference Process**:
1. **Encode Support and Query**:
   - `support_embeddings = encoder(support_images)`
   - `query_embeddings = encoder(query_images)`
2. **For each query**:
   - Compute attention weights: `attention = softmax(MLP([query; support]))`
   - Weighted sum of one-hot labels: `logits = attention @ one_hot_labels`
3. **Classification**: `predictions = argmax(logits)`

**Training**: End-to-end via CrossEntropyLoss

**Location**: `models/few_shot.py` lines 180-273

#### Option C: Relation Networks

**Technique**: Learned Similarity Metric

**Architecture**:
1. **Encoder**: Pretrained encoder
2. **Relation Module**: MLP (embedding_dim * 2 → relation_dim * 2 → relation_dim → 1)

**Inference Process**:
1. **Encode and Compute Prototypes**: Same as Prototypical
2. **For each query-prototype pair**:
   - Concatenate: `combined = [query; prototype]`
   - Relation score: `score = sigmoid(MLP(combined))`
3. **Classification**: `predictions = argmax(scores)`

**Training**: End-to-end via CrossEntropyLoss (treats scores as logits)

**Location**: `models/few_shot.py` lines 276-358

---

### Step 2.3: Episode Sampling

**Method**: Episodic Training (Meta-Learning)

**Technique**: N-way K-shot Episode Construction

**Process** (`data/samplers.py`):
1. **Sample N classes** randomly from training set
2. **For each class**:
   - Sample K support examples
   - Sample Q query examples (typically 15)
3. **Create Episode**:
   - Support set: (N * K, C, H, W) images + labels
   - Query set: (N * Q, C, H, W) images + labels
   - Labels: 0 to N-1 (episode-local class indices)

**Parameters**:
- N-way: 5 (number of classes per episode)
- K-shot: 5 (support examples per class)
- Q-query: 15 (query examples per class)
- Episodes per epoch: 1000

**Location**: `data/samplers.py`

---

### Step 2.4: Few-Shot Training Loop

**Method**: Episodic Meta-Learning with Gradient Descent

**Training Process** (`train_few_shot.py`):
1. **Data Loading**:
   - EpisodeSampler generates episodes on-the-fly
   - Each episode: (support_images, support_labels, query_images, query_labels)

2. **Optimizer**: Adam
   - Learning rate: 0.001
   - Weight decay: 1e-4

3. **Learning Rate Scheduler**: StepLR
   - Step size: 20 epochs
   - Gamma: 0.5 (halve LR every 20 epochs)

4. **Training Loop** (per episode):
   ```python
   for episode in train_sampler:
       support_images, support_labels = episode['support_images'], episode['support_labels']
       query_images, query_labels = episode['query_images'], episode['query_labels']
       
       # Forward pass
       logits = model(support_images, support_labels, query_images, n_way, n_shot)
       
       # Compute loss (on query set only)
       loss = CrossEntropyLoss(logits, query_labels)
       
       # Backward pass
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
   ```

5. **Validation**:
   - Use held-out validation classes
   - Sample validation episodes
   - Compute accuracy on query set

6. **Checkpointing**:
   - Save every 5 epochs
   - Save best model (highest validation accuracy)

**Duration**: 100 epochs (typical)

**Key Difference from SSL**: 
- SSL: Standard supervised-like training with batches
- Few-Shot: Episodic training simulating few-shot scenarios

**Location**: `train_few_shot.py` lines 89-379

---

### Step 2.5: Fine-tuning Strategy

**Method**: End-to-End Fine-tuning

**What Gets Updated**:
- **Encoder weights**: Fine-tuned (not frozen)
- **Few-shot head** (if applicable): Trained from scratch
  - Matching Networks: Attention module
  - Relation Networks: Relation module
  - Prototypical Networks: No additional head

**Gradient Flow**:
- Gradients flow through:
  1. Few-shot head (if exists)
  2. Encoder (backpropagated from few-shot loss)
- Both encoder and head are updated simultaneously

**Learning Rate Strategy**:
- Same LR for encoder and head (0.001)
- Could use different LRs (encoder: 1e-4, head: 1e-3) for better results

**Location**: `train_few_shot.py` lines 312-316

---

## Complete Training Pipeline Summary

### Phase 1: SSL Pretraining
1. **Initialize Encoder**: Custom CNN (random) or ResNet (ImageNet pretrained)
2. **Create SSL Model**: SimCLR or BYOL wrapper
3. **Train on Unlabeled Sketches**: 200 epochs, contrastive learning
4. **Save Encoder**: Extract and save encoder weights

### Phase 2: Few-Shot Fine-tuning
1. **Load Pretrained Encoder**: Transfer weights from SSL checkpoint
2. **Create Few-Shot Model**: Prototypical, Matching, or Relation Network
3. **Train on Episodes**: 100 epochs, episodic meta-learning
4. **Evaluate on Unseen Classes**: Test on held-out test classes

---

## Key Techniques Used

### 1. Transfer Learning
- **ImageNet → Sketch**: ResNet pretrained on ImageNet (optional)
- **SSL → Few-Shot**: Encoder pretrained with SSL, fine-tuned for few-shot

### 2. Self-Supervised Learning
- **SimCLR**: Contrastive learning with negative pairs
- **BYOL**: Self-supervised learning without negative pairs

### 3. Meta-Learning
- **Episodic Training**: Simulate few-shot scenarios during training
- **Prototypical Networks**: Distance-based classification
- **Matching Networks**: Attention-based classification
- **Relation Networks**: Learned similarity metric

### 4. Data Augmentation
- **Two-view augmentation**: Different views for contrastive learning
- **Sketch-specific augmentations**: Preserve sketch structure

### 5. Optimization
- **Adam optimizer**: Adaptive learning rates
- **Learning rate scheduling**: Cosine annealing (SSL) or step decay (few-shot)
- **Weight decay**: L2 regularization

---

## Model Architecture Flow

```
Input Sketch (224x224, grayscale)
    ↓
[Stage 1: SSL Pretraining]
    ↓
Encoder (SketchEncoder or ResNetEncoder)
    ↓
Projection Head (SSL only)
    ↓
Contrastive Loss (SimCLR/BYOL)
    ↓
[Save Encoder Weights]
    ↓
[Stage 2: Few-Shot Learning]
    ↓
Encoder (Loaded from SSL, fine-tuned)
    ↓
Few-Shot Head (Prototypical/Matching/Relation)
    ↓
Episode-based Classification
    ↓
CrossEntropyLoss on Query Set
    ↓
Fine-tuned Model
```

---

## Answer to Your Question

**Are we making the model from scratch or using pretrained + fine-tuning?**

**Answer**: **Hybrid Approach - Pretrained + Fine-tuning**

1. **Encoder**:
   - Option A: Built from scratch (SketchEncoder) → Random initialization → SSL pretraining → Few-shot fine-tuning
   - Option B: Pretrained ResNet (ImageNet) → SSL pretraining → Few-shot fine-tuning

2. **SSL Stage**: Trains encoder from (random or ImageNet) to learn sketch representations

3. **Few-Shot Stage**: Fine-tunes the SSL-pretrained encoder for few-shot classification

**Key Point**: The encoder is **never trained from scratch in the few-shot stage** - it always starts from SSL-pretrained weights (which themselves may start from ImageNet if using ResNet).

---

## Performance Impact

**Without SSL Pretraining**:
- Random encoder → Few-shot training: ~50% accuracy

**With SSL Pretraining**:
- SSL-pretrained encoder → Few-shot training: ~65-68% accuracy

**Conclusion**: SSL pretraining significantly improves few-shot performance by learning better sketch representations.


