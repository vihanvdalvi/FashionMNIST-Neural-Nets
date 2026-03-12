# Fashion-MNIST Classification with PyTorch

This project demonstrates building and training feed-forward neural networks on the **Fashion-MNIST** dataset using **PyTorch**. It includes data loading, preprocessing, model training, evaluation, and top-3 class prediction for sample images.

---

## Project Overview

The **Fashion-MNIST** dataset contains grayscale 28x28 images across 10 clothing categories.  
This project implements a pipeline that:

- Loads and normalizes training and test data
- Builds two neural network architectures: **baseline** and **deeper**
- Trains models using **stochastic gradient descent with momentum**
- Evaluates model performance using **accuracy and average loss**
- Generates **top-3 class predictions** for selected test images

---

## Features Implemented

### Data Pipeline
- Loads Fashion-MNIST automatically with `torchvision`
- Converts images to tensors and normalizes them
- Uses `DataLoader` for batch processing

### Model Architectures
- **Baseline MLP:** Flatten → 128 → 64 → 10  
- **Deeper MLP:** Flatten → 256 → 128 → 64 → 32 → 10  
- **LeakyReLU** activations for hidden layers

### Training
- **Loss function:** Cross-entropy  
- **Optimizer:** SGD with momentum  
- Epoch-wise reporting of **training accuracy and loss**

### Evaluation
- Reports **test accuracy**
- Optional **average test loss**
- Inference performed **without gradient tracking**

### Prediction Utility
- Prints **top-3 predicted classes** with probabilities for a chosen image index

---

## Tech Stack
- **Python**  
- **PyTorch**  
- **Torchvision**

---

## How to Run

1. Install dependencies:
```bash
pip install torch torchvision
