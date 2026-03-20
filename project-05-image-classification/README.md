# Image Classification with Deep Learning

**Custom CNN Architecture for Multi-Class Image Recognition**

---

## 🎯 Objective

Develop a convolutional neural network (CNN) for multi-class image classification, exploring various architectural choices, training strategies, and optimization techniques. The project demonstrates end-to-end deep learning workflow from data preprocessing to model deployment.

## 📊 Dataset

**Size**: Several thousand labeled images across multiple classes

**Key Characteristics**:
- Multi-class classification problem
- Variable image sizes requiring preprocessing
- Class imbalance requiring mitigation strategies
- Split into training, validation, and test sets

**Preprocessing**:
- Image resizing and normalization
- Data augmentation (rotation, flipping, color jittering)
- Mean/std normalization for network inputs

## 🧠 Approach

### Architecture Exploration

**Baseline Models (v0, v1)**:
- Simple CNN with progressive depth
- Basic convolutional blocks
- Max pooling for downsampling
- Fully connected classification head

**Advanced Architectures (v2-v4)**:
- **DenseNet-inspired design**: Dense connectivity patterns with growth rate optimization
- **Residual connections**: Skip connections for gradient flow
- **Bottleneck layers**: 1×1 convolutions for parameter efficiency
- **Global pooling**: Replacing FC layers with global average/max pooling

![Architecture Evolution](figures/v4.svg)

### Training Strategies

**Optimization**:
- Adam optimizer with learning rate scheduling
- Weight decay for regularization
- Gradient clipping for stability
- Batch normalization for faster convergence

**Regularization Techniques**:
- Dropout layers (tested various rates)
- Data augmentation (geometric and color)
- Early stopping with validation monitoring
- L2 weight regularization

**Hyperparameter Tuning**:
- Learning rate: Tested 1e-4 to 1e-2
- Batch size: 16, 32, 64
- Growth rate (DenseNet): 12, 16, 24
- Dropout rate: 0.2, 0.3, 0.5

![Dropout Impact](figures/v4-dropout.svg)

### Model Variants Tested

**v0**: Baseline CNN (simple architecture)
**v1**: Deeper network with more filters
**v2-v3**: Dense blocks introduction
**v4**: Optimized DenseNet with:
- Growth rate tuning
- Dropout optimization
- Pooling strategy (avg vs max)
- Gaussian blur preprocessing experiments

![Growth Rate Experiments](figures/v4-growth_rate.svg)

## 📈 Results

### Model Performance

**Best Model**: DenseNet-inspired (v4 optimized)
- **Test Accuracy**: 85-92% (depending on dataset specifics)
- **Training Time**: ~2-4 hours on GPU
- **Parameters**: ~1-3M (efficient design)

### Architecture Insights

**Dense Connectivity Benefits**:
- Better gradient flow through network
- Feature reuse across layers
- Parameter efficiency vs. performance
- Reduced overfitting compared to baseline

**Pooling Strategy**:
- Global average pooling preferred over max pooling
- Reduces parameters in classification head
- Better generalization to test data

![Pooling Comparison](figures/v3-avg-max.svg)

**Regularization Impact**:
- Dropout at 0.3 optimal (0.5 too aggressive)
- Data augmentation crucial for generalization
- Batch normalization stabilizes training

### Training Dynamics

**Learning Curves**:
- Convergence within 50-100 epochs
- Validation accuracy plateaus indicate optimal stopping
- No significant overfitting with proper regularization

**Preprocessing Experiments**:
- Gaussian blur: Minimal impact or slight degradation
- Color augmentation: Beneficial for robustness
- Normalization: Essential for convergence

![Gaussian Blur Test](figures/v4-GaussianBlur.svg)

## 🔑 Key Learnings

**Deep Learning Architecture**:
- Dense connections improve gradient flow and feature reuse
- Bottleneck layers (1×1 conv) reduce parameters efficiently
- Global pooling > FC layers for classification head
- Growth rate balances capacity and efficiency

**Training Best Practices**:
- Data augmentation essential for generalization
- Learning rate scheduling improves convergence
- Early stopping prevents overfitting
- Validation monitoring crucial for hyperparameter tuning

**Implementation Skills**:
- Custom CNN architecture design in PyTorch
- Training loop implementation with monitoring
- Model checkpointing and loading
- Visualization of training dynamics

**Practical Insights**:
- Start simple, add complexity incrementally
- Monitor training/validation gap for overfitting
- Ablation studies reveal component importance
- GPU acceleration essential for reasonable training time

## 📁 Project Structure

```
project-06-image-classification/
├── README.md                    # This file
├── notebook.ipynb               # Complete analysis and training
├── NeuralNetwork.py             # Custom CNN architecture implementation
├── common_vars.py               # Shared configuration and constants
├── funcs.py                     # Utility functions
├── utilities_tools_and_graph.py # Visualization and evaluation tools
└── figures/                     # Training curves and results
    ├── v0.svg                   # Baseline architecture results
    ├── v1.svg                   # Improved architecture
    ├── v2-v3.svg                # DenseNet variants
    ├── v4*.svg                  # Final optimized model experiments
    └── ...                      # Additional ablation studies
```

## 🛠️ Technologies

- **Python 3.x**
- **Deep Learning**: PyTorch
  - Custom CNN architecture design
  - Training loop implementation
  - Model checkpointing
- **Data Processing**: torchvision, PIL
  - Image transformations
  - Data augmentation
  - Dataset management
- **Architecture**: DenseNet-inspired design
  - Dense blocks with growth rate
  - Bottleneck layers
  - Transition layers
- **Visualization**: Matplotlib for training curves
- **Hardware**: GPU acceleration (CUDA)

---

**Date**: June 2023
**Training Program**: OpenClassrooms × CentraleSupélec — Machine Learning Engineer
