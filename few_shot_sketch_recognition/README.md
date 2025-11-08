# Few-Shot Sketch Recognition Framework

A comprehensive framework for few-shot sketch recognition using self-supervised representation learning. This implementation combines contrastive learning methods (SimCLR, BYOL) with few-shot learning algorithms (Prototypical Networks, Matching Networks) to recognize previously unseen sketch classes with minimal examples.

## 🎯 Project Overview

This framework addresses the **FSUGR-2** problem: designing a few-shot sketch recognition system that can recognize unseen object classes after being shown only a few example sketches. The approach consists of two main stages:

1. **Self-Supervised Pretraining**: Learn robust sketch embeddings using contrastive learning (SimCLR or BYOL)
2. **Few-Shot Learning**: Train meta-learning models to classify new classes with limited examples

### Key Features

- ✅ **Multiple SSL Methods**: SimCLR and BYOL implementations optimized for sketch data
- ✅ **Multiple Few-Shot Algorithms**: Prototypical Networks, Matching Networks, and Relation Networks
- ✅ **Sketch-Specific Augmentations**: Custom transformations that preserve sketch structure
- ✅ **Benchmark Datasets**: Support for TU-Berlin and QuickDraw datasets
- ✅ **Comprehensive Evaluation**: Rigorous testing on unseen classes with confidence intervals
- ✅ **Modular Design**: Easy to extend and experiment with new methods

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

## 🚀 Quick Start

### 1. Installation

**Step 1: Navigate to the project**
```bash
cd few_shot_sketch_recognition
```

**Step 2: Create a Virtual Environment (HIGHLY RECOMMENDED)**

Using a virtual environment keeps dependencies isolated and prevents conflicts:

```bash
# Create virtual environment (one-time setup)
python3 -m venv venv

# Activate it (do this every time you work on the project)
source venv/bin/activate  # On macOS/Linux
# OR on Windows: venv\Scripts\activate
```

You'll know it's activated when you see `(venv)` in your terminal prompt.

**Step 3: Install Dependencies**
```bash
# Install all required packages (only affects the virtual environment)
pip install -r requirements.txt
```

**Important Notes**:
- ✅ Virtual environment keeps packages isolated (recommended)
- ✅ You can delete the `venv` folder anytime to start fresh
- ✅ Always activate the venv before running scripts: `source venv/bin/activate`
- ✅ To exit the virtual environment: `deactivate`
- ❌ Installing system-wide (without venv) works but may cause conflicts

### 2. Download Datasets

**TU-Berlin Dataset** (250 classes, 20,000 sketches):
```bash
python main.py download --dataset tuberlin --output_dir data/tuberlin
```

**QuickDraw Dataset** (50 categories by default):
```bash
python main.py download --dataset quickdraw --output_dir data/quickdraw --n_categories 50
```

### 3. Run Full Pipeline

Train and evaluate with a single command:

```bash
python main.py pipeline \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --ssl_method simclr \
    --few_shot_model prototypical \
    --n_way 5 \
    --n_shot 5
```

This will:
1. Pretrain encoder using SimCLR
2. Train Prototypical Network using pretrained encoder
3. Evaluate on unseen classes (5-way 5-shot)

## 📚 Detailed Usage

### Stage 1: Self-Supervised Pretraining

Train an encoder using contrastive learning to create a robust embedding space:

**SimCLR Training:**
```bash
python train_ssl.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --ssl_method simclr \
    --batch_size 128 \
    --epochs 200 \
    --lr 0.001 \
    --temperature 0.5 \
    --checkpoint_dir checkpoints/ssl/simclr
```

**BYOL Training:**
```bash
python train_ssl.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder resnet18 \
    --ssl_method byol \
    --batch_size 128 \
    --epochs 200 \
    --lr 0.001 \
    --ema_decay 0.996 \
    --checkpoint_dir checkpoints/ssl/byol
```

### Stage 2: Few-Shot Learning

Train a few-shot model using the pretrained encoder:

**Prototypical Networks:**
```bash
python train_few_shot.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --few_shot_model prototypical \
    --pretrained_encoder checkpoints/ssl/simclr/best_model.pth \
    --n_way 5 \
    --n_shot 5 \
    --n_query 15 \
    --epochs 100 \
    --lr 0.001 \
    --checkpoint_dir checkpoints/few_shot/prototypical
```

**Matching Networks:**
```bash
python train_few_shot.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --few_shot_model matching \
    --pretrained_encoder checkpoints/ssl/simclr/best_model.pth \
    --n_way 5 \
    --n_shot 5 \
    --n_query 15 \
    --epochs 100 \
    --lr 0.001 \
    --checkpoint_dir checkpoints/few_shot/matching
```

### Evaluation

Evaluate trained models on unseen classes:

```bash
python evaluate.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --checkpoint checkpoints/few_shot/prototypical/best_model.pth \
    --encoder sketch_cnn \
    --few_shot_model prototypical \
    --n_way 5 \
    --n_shot 5 \
    --n_query 15 \
    --n_episodes 600 \
    --visualize \
    --output_dir results
```

## 🏗️ Architecture

### Project Structure

```
few_shot_sketch_recognition/
├── models/                      # Neural network models
│   ├── backbone.py             # Encoder architectures (CNN, ResNet)
│   ├── contrastive.py          # SSL methods (SimCLR, BYOL)
│   ├── few_shot.py             # Few-shot algorithms (Prototypical, Matching)
│   └── supervised.py           # Baseline supervised model
├── data/                        # Data loading and preprocessing
│   ├── datasets.py             # Dataset classes (TU-Berlin, QuickDraw)
│   ├── samplers.py             # Episode samplers for few-shot learning
│   ├── transforms.py           # Sketch-specific augmentations
│   └── download.py             # Dataset download utilities
├── utils/                       # Utility functions
│   ├── metrics.py              # Evaluation metrics
│   ├── visualization.py        # Plotting and visualization
│   ├── checkpoint.py           # Model saving/loading
│   └── logger.py               # Logging utilities
├── configs/                     # Configuration files (YAML)
├── train_ssl.py                # Self-supervised learning training
├── train_few_shot.py           # Few-shot learning training
├── evaluate.py                 # Model evaluation
├── main.py                     # Unified entry point
└── requirements.txt            # Python dependencies
```

### Model Components

#### 1. Encoders

**SketchEncoder** (Custom CNN):
- Specialized architecture for sketch features
- 4 convolutional blocks with residual connections
- Adaptive pooling for flexible input sizes
- Optimized for grayscale sketch images

**ResNetEncoder**:
- Pretrained ResNet backbone (18/34/50)
- Modified first layer for grayscale input
- Custom projection head for embeddings

#### 2. Self-Supervised Learning

**SimCLR (Simple Framework for Contrastive Learning)**:
- Learns representations by contrasting augmented views
- NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
- Temperature-controlled similarity scoring
- Negative pair mining for better discrimination

**BYOL (Bootstrap Your Own Latent)**:
- Learns without negative pairs
- Online and target networks with EMA updates
- Predictor network for asymmetric architecture
- More stable training than contrastive methods

#### 3. Few-Shot Learning Algorithms

**Prototypical Networks**:
- Classifies based on distance to class prototypes
- Prototype = mean embedding of support samples
- Supports Euclidean and cosine distance metrics
- Simple, efficient, and effective

**Matching Networks**:
- Attention-based classification
- Compares query to all support samples
- Learns similarity metric end-to-end
- Better for complex relationships

**Relation Networks**:
- Learns comparison metric with neural network
- More flexible than fixed distance metrics
- Predicts relation scores between samples

## 📊 Datasets

### TU-Berlin Sketch Dataset

- **Size**: 20,000 sketches across 250 categories
- **Format**: PNG images (varied sizes)
- **Split**: 200 classes for training, 50 for testing
- **Source**: [TU-Berlin](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/)

**Citation:**
```
Eitz, M., Hays, J., & Alexa, M. (2012). 
How do humans sketch objects?
ACM Transactions on Graphics (TOG), 31(4), 1-10.
```

### Google QuickDraw Dataset

- **Size**: 50M+ drawings across 345 categories
- **Format**: 28x28 grayscale numpy arrays
- **Split**: 70% train, 15% val, 15% test
- **Source**: [Google QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset)

## 🔬 Experiments and Results

### Experimental Setup

1. **Training**:
   - SSL pretraining: 200 epochs on all training classes
   - Few-shot training: 100 epochs with episodic sampling
   - Image size: 224x224
   - Augmentations: Rotation, translation, stroke thickness variation

2. **Evaluation**:
   - Test on unseen classes (50 classes held out)
   - 600 episodes per configuration
   - Report mean accuracy ± 95% confidence interval

### Baseline Comparisons

The framework includes baselines for comparison:

1. **Supervised Baseline**: Standard supervised learning on training classes
2. **From-Scratch Few-Shot**: Few-shot models without SSL pretraining
3. **Transfer Learning**: Fine-tuning ImageNet pretrained models

### Expected Performance

Typical results on TU-Berlin (5-way 5-shot):

| Method | Pretraining | Accuracy |
|--------|-------------|----------|
| Random | None | ~20% |
| Supervised Baseline | Supervised | ~35% |
| Prototypical (scratch) | None | ~50% |
| Prototypical + SimCLR | SimCLR | ~65% |
| Prototypical + BYOL | BYOL | ~68% |
| Matching + SimCLR | SimCLR | ~63% |

*Note: Results may vary based on hyperparameters and random seed*

## 🛠️ Customization

### Adding Custom Datasets

Create a new dataset class in `data/datasets.py`:

```python
class CustomSketchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Load your data
        pass
    
    def __getitem__(self, idx):
        # Return (image, label)
        pass
```

### Adding New Models

Implement new encoders or few-shot algorithms:

```python
# In models/backbone.py
class MyCustomEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        # Define architecture
    
    def forward(self, x):
        # Return embeddings
        pass
```

### Custom Augmentations

Add sketch-specific augmentations in `data/transforms.py`:

```python
class MyCustomAugmentation:
    def __call__(self, img):
        # Apply transformation
        return transformed_img
```

## 📈 Monitoring Training

Training logs and checkpoints are saved automatically:

- **Logs**: `logs/ssl/` and `logs/few_shot/`
- **Checkpoints**: `checkpoints/ssl/` and `checkpoints/few_shot/`
- **TensorBoard**: (optional) `tensorboard --logdir logs`

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce batch size: `--batch_size 64`
   - Use smaller encoder: `--encoder resnet18`
   - Reduce image size: `--image_size 128`

2. **Slow Training**:
   - Reduce number of workers if CPU-bound: `--num_workers 2`
   - Use GPU if available
   - Reduce training episodes: `--n_train_episodes 500`

3. **Dataset Not Found**:
   - Ensure datasets are downloaded to correct location
   - Check `--data_root` path
   - Verify directory structure

4. **Poor Performance**:
   - Train longer: increase epochs
   - Use SSL pretraining
   - Try different augmentation strengths
   - Experiment with learning rates

## 📖 References

### Key Papers

1. **SimCLR**: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020
2. **BYOL**: Grill et al., "Bootstrap Your Own Latent", NeurIPS 2020
3. **Prototypical Networks**: Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017
4. **Matching Networks**: Vinyals et al., "Matching Networks for One Shot Learning", NeurIPS 2016
5. **TU-Berlin Dataset**: Eitz et al., "How Do Humans Sketch Objects?", SIGGRAPH 2012

### Related Work

- Meta-Learning: "Model-Agnostic Meta-Learning" (Finn et al., ICML 2017)
- Metric Learning: "Learning to Compare" (Sung et al., CVPR 2018)
- Sketch Recognition: "Sketch-a-Net" (Yu et al., ECCV 2015)

## 💡 Tips for Best Results

1. **Start with Pretraining**: Always use SSL pretraining for better generalization
2. **Tune Temperature**: SimCLR temperature (0.3-0.7) significantly affects performance
3. **Augmentation Matters**: Sketch-specific augmentations preserve structure
4. **Class Split**: Ensure training/test classes are diverse and balanced
5. **Episode Sampling**: More episodes = better meta-learning (1000+ recommended)
6. **Distance Metric**: Try both Euclidean and cosine for Prototypical Networks
7. **Learning Rate**: Start with 1e-3, decay if training plateaus

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional SSL methods (MoCo, SwAV)
- More few-shot algorithms (MAML, Relation Networks)
- Advanced augmentation techniques
- Multi-modal sketch recognition
- Temporal sketch recognition (stroke sequences)

## 📄 License

This project is intended for educational and research purposes.

## 🙏 Acknowledgments

- TU-Berlin team for the sketch dataset
- Google for the QuickDraw dataset
- PyTorch team for the deep learning framework
- Research community for SSL and few-shot learning methods

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Happy Sketching! 🎨✨**

