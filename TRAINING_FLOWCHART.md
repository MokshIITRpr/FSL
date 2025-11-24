# Training Pipeline Flowchart

## Quick Visual Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: SSL PRETRAINING                     │
└─────────────────────────────────────────────────────────────────┘

Step 1: Encoder Initialization
┌─────────────────────────────────────────────────────────────┐
│  Option A: Custom SketchEncoder                             │
│  • Built from scratch                                       │
│  • Random initialization (Xavier/Kaiming)                   │
│  • 4 conv blocks: 64→128→256→512 channels                  │
└─────────────────────────────────────────────────────────────┘
                          OR
┌─────────────────────────────────────────────────────────────┐
│  Option B: ResNetEncoder                                    │
│  • ImageNet pretrained ResNet (18/34/50)                    │
│  • First layer modified for grayscale (1 channel)           │
│  • Projection head: Random initialization                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
Step 2: SSL Model Construction
┌─────────────────────────────────────────────────────────────┐
│  Option A: SimCLR                                           │
│  • Encoder + Projection Head (MLP)                          │
│  • NT-Xent Loss (contrastive)                               │
│  • Temperature: 0.5                                         │
└─────────────────────────────────────────────────────────────┘
                          OR
┌─────────────────────────────────────────────────────────────┐
│  Option B: BYOL                                             │
│  • Online Network: Encoder + Projector + Predictor          │
│  • Target Network: EMA copy                                 │
│  • MSE Loss (no negatives)                                  │
│  • EMA decay: 0.996                                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
Step 3: Data Augmentation
┌─────────────────────────────────────────────────────────────┐
│  Two-View Transform                                         │
│  • Random rotation (-15° to +15°)                           │
│  • Random translation                                       │
│  • Random scale (0.9x to 1.1x)                              │
│  • Stroke thickness variation                               │
│  • Random erasing                                           │
└─────────────────────────────────────────────────────────────┘
                          ↓
Step 4: SSL Training (200 epochs)
┌─────────────────────────────────────────────────────────────┐
│  For each batch:                                            │
│  1. Generate two augmented views (x1, x2)                   │
│  2. Forward: z1, z2 = model(x1, x2)                         │
│  3. Compute contrastive loss                                │
│  4. Backward pass + optimizer.step()                        │
│  5. (BYOL only) Update target network via EMA               │
│                                                             │
│  Optimizer: Adam (lr=0.001, weight_decay=1e-4)             │
│  Scheduler: CosineAnnealingLR                               │
│  Batch size: 128                                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
Step 5: Save Encoder
┌─────────────────────────────────────────────────────────────┐
│  Checkpoint saved:                                          │
│  • Encoder weights (encoder.*)                              │
│  • Projection head weights (discarded later)                │
│  • Best model: lowest loss                                  │
└─────────────────────────────────────────────────────────────┘

═════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 2: FEW-SHOT LEARNING                     │
└─────────────────────────────────────────────────────────────────┘

Step 1: Load Pretrained Encoder
┌─────────────────────────────────────────────────────────────┐
│  1. Load SSL checkpoint                                      │
│  2. Extract encoder weights (filter 'encoder.*' keys)       │
│  3. Remove 'encoder.' prefix                                 │
│  4. Load into few-shot encoder (strict=False)               │
│                                                             │
│  Result: Encoder initialized with SSL-pretrained weights    │
└─────────────────────────────────────────────────────────────┘
                          ↓
Step 2: Few-Shot Model Construction
┌─────────────────────────────────────────────────────────────┐
│  Option A: Prototypical Networks                            │
│  • Encoder (pretrained)                                     │
│  • No additional layers                                     │
│  • Distance-based: Euclidean or Cosine                      │
└─────────────────────────────────────────────────────────────┘
                          OR
┌─────────────────────────────────────────────────────────────┐
│  Option B: Matching Networks                                │
│  • Encoder (pretrained)                                     │
│  • Attention Module (MLP, trained from scratch)             │
│  • Attention-based classification                           │
└─────────────────────────────────────────────────────────────┘
                          OR
┌─────────────────────────────────────────────────────────────┐
│  Option C: Relation Networks                                │
│  • Encoder (pretrained)                                     │
│  • Relation Module (MLP, trained from scratch)              │
│  • Learned similarity metric                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
Step 3: Episode Sampling
┌─────────────────────────────────────────────────────────────┐
│  For each episode:                                          │
│  1. Sample N classes randomly (N-way, e.g., 5)              │
│  2. Sample K support examples per class (K-shot, e.g., 5)   │
│  3. Sample Q query examples per class (Q-query, e.g., 15)   │
│                                                             │
│  Support: (N*K, C, H, W) + labels                          │
│  Query: (N*Q, C, H, W) + labels                            │
│  Episodes per epoch: 1000                                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
Step 4: Few-Shot Training (100 epochs)
┌─────────────────────────────────────────────────────────────┐
│  For each episode:                                          │
│  1. Encode support: support_emb = encoder(support_images)   │
│  2. Encode query: query_emb = encoder(query_images)         │
│  3. Compute prototypes/logits (method-specific)             │
│  4. Compute loss: CrossEntropyLoss(logits, query_labels)    │
│  5. Backward pass (updates encoder + head)                  │
│  6. optimizer.step()                                        │
│                                                             │
│  Optimizer: Adam (lr=0.001, weight_decay=1e-4)             │
│  Scheduler: StepLR (step=20, gamma=0.5)                    │
│  Episodes per epoch: 1000                                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
Step 5: Evaluation
┌─────────────────────────────────────────────────────────────┐
│  Test on unseen classes:                                    │
│  • Sample test episodes (600 episodes)                      │
│  • Compute accuracy on query set                            │
│  • Report: mean accuracy ± 95% CI                           │
│                                                             │
│  Expected: ~65-68% accuracy (5-way 5-shot)                  │
└─────────────────────────────────────────────────────────────┘

═════════════════════════════════════════════════════════════════

## Data Flow Diagram

```
INPUT: Sketch Images (224x224, grayscale)
    │
    ├─[STAGE 1: SSL PRETRAINING]───────────────────────────────┐
    │                                                           │
    │  Augmentation: x → (x1, x2)                              │
    │      │                                                    │
    │      ├─→ x1 ──┐                                          │
    │      │        │                                          │
    │      └─→ x2 ──┤                                          │
    │               │                                          │
    │  Encoder: (x1, x2) → (h1, h2)                           │
    │      │                                                    │
    │  Projection: (h1, h2) → (z1, z2)                        │
    │      │                                                    │
    │  Loss: Contrastive Loss (SimCLR/BYOL)                    │
    │      │                                                    │
    │  Update: Encoder + Projection Head                       │
    │      │                                                    │
    │  Output: Pretrained Encoder Weights                      │
    │      │                                                    │
    └──────────────────────────────────────────────────────────┘
    │
    ├─[STAGE 2: FEW-SHOT LEARNING]─────────────────────────────┐
    │                                                           │
    │  Load: Pretrained Encoder                                │
    │      │                                                    │
    │  Episode: (support_images, query_images)                 │
    │      │                                                    │
    │  Encode Support: support_emb = encoder(support_images)   │
    │      │                                                    │
    │  Encode Query: query_emb = encoder(query_images)         │
    │      │                                                    │
    │  Few-Shot Head:                                          │
    │      │                                                    │
    │      ├─ Prototypical:                                    │
    │      │   prototypes = mean(support_emb per class)        │
    │      │   logits = -distance(query_emb, prototypes)       │
    │      │                                                    │
    │      ├─ Matching:                                        │
    │      │   attention = softmax(MLP([query; support]))      │
    │      │   logits = attention @ one_hot_labels             │
    │      │                                                    │
    │      └─ Relation:                                        │
    │          prototypes = mean(support_emb per class)        │
    │          scores = MLP([query; prototypes])               │
    │          logits = scores                                 │
    │      │                                                    │
    │  Loss: CrossEntropyLoss(logits, query_labels)            │
    │      │                                                    │
    │  Update: Encoder (fine-tuned) + Few-Shot Head            │
    │      │                                                    │
    │  Output: Fine-tuned Few-Shot Model                       │
    │                                                           │
    └──────────────────────────────────────────────────────────┘
    │
    └─→ OUTPUT: Predictions for Unseen Classes
```

═════════════════════════════════════════════════════════════════

## Component Architecture

### SSL Model (SimCLR Example)
```
Input: Sketch (1, 224, 224)
    │
    ├─ Encoder (SketchEncoder)
    │   ├─ Conv Block 1: 1 → 64 channels
    │   ├─ Conv Block 2: 64 → 128 channels
    │   ├─ Conv Block 3: 128 → 256 channels
    │   ├─ Conv Block 4: 256 → 512 channels
    │   ├─ Global Avg Pool: (512, 1, 1)
    │   └─ Flatten: (512,)
    │
    ├─ Projection Head (SSL only)
    │   ├─ Linear: 512 → 2048
    │   ├─ BatchNorm + ReLU
    │   ├─ Linear: 2048 → 128
    │   └─ Normalize: L2 norm
    │
    └─ Output: z (128,)
```

### Few-Shot Model (Prototypical Example)
```
Support Set: (N*K, 1, 224, 224)
    │
    ├─ Encoder (Pretrained from SSL)
    │   └─ Same as SSL encoder (without projection head)
    │
    └─ Support Embeddings: (N*K, 512)
        │
        ├─ Compute Prototypes
        │   └─ Mean per class: (N, 512)
        │
Query Set: (N*Q, 1, 224, 224)
    │
    ├─ Encoder (Same pretrained encoder)
    │
    └─ Query Embeddings: (N*Q, 512)
        │
        ├─ Compute Distances
        │   └─ Euclidean/Cosine: (N*Q, N)
        │
        └─ Logits: -distances
            │
            └─ Predictions: argmax(logits)
```

═════════════════════════════════════════════════════════════════

## Training Timeline

```
Time →
│
├─[0-200 epochs] SSL Pretraining
│  ├─ Epoch 1-50:   Loss decreases rapidly (8.0 → 4.0)
│  ├─ Epoch 50-150: Loss decreases slowly (4.0 → 2.5)
│  └─ Epoch 150-200: Loss plateaus (2.5 → 2.0)
│
├─[Checkpoint] Save best encoder
│
├─[0-100 epochs] Few-Shot Fine-tuning
│  ├─ Epoch 1-20:   Accuracy increases rapidly (50% → 60%)
│  ├─ Epoch 20-60:  Accuracy increases slowly (60% → 65%)
│  └─ Epoch 60-100: Accuracy plateaus (65% → 68%)
│
└─[Evaluation] Test on unseen classes: ~65-68% accuracy
```

═════════════════════════════════════════════════════════════════

## Key Decisions at Each Step

### Stage 1: SSL Pretraining
1. **Encoder Choice**: Custom CNN (from scratch) vs ResNet (ImageNet pretrained)
2. **SSL Method**: SimCLR (contrastive) vs BYOL (no negatives)
3. **Augmentation Strength**: Light vs Medium vs Heavy
4. **Training Duration**: 100 vs 200 vs 300 epochs

### Stage 2: Few-Shot Learning
1. **Few-Shot Algorithm**: Prototypical vs Matching vs Relation
2. **Distance Metric**: Euclidean vs Cosine (Prototypical only)
3. **N-way K-shot**: 5-way 5-shot vs 10-way 1-shot vs others
4. **Fine-tuning Strategy**: Full fine-tuning vs frozen encoder

### Hyperparameters
- **Learning Rate**: 0.001 (standard), can tune 1e-4 to 1e-3
- **Batch Size**: 128 (SSL), 1000 episodes/epoch (few-shot)
- **Temperature**: 0.5 (SimCLR), can tune 0.3 to 0.7
- **EMA Decay**: 0.996 (BYOL), can tune 0.99 to 0.999


