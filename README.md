# Image-Classification-with-Fully-Connected-Neural-Networks

This project implements an image classification pipeline using two different fully connected feed-forward neural network architectures: one with 2 layers and one with 3 layers. The goal is to classify RGB images (cloudy, shine, sunrise) by learning from their pixel features.

## Dataset

The dataset contains 908 images across 3 categories:
- **cloudy**
- **shine**
- **sunrise**

Each image is resized to **32x32x3** and converted into a flat feature vector (1x3072). The dataset is split as follows:
- **Training set**: 50%
- **Validation set**: 25%
- **Test set**: 25%

## Features

Each image is represented by its pixel intensity values. Features are extracted by flattening the resized image arrays.

## Model Architectures

### 1. 2-Layer Feed-Forward Network
- **Input**: 3072-dim flattened image vector
- **Hidden Layer**: 128 neurons, ReLU, He initialization
- **Dropout**: 0.3
- **Output Layer**: 10-class softmax
- **Optimizer**: Adam or SGD
- **Loss**: SparseCategoricalCrossentropy

### 2. 3-Layer Feed-Forward Network
- **Input**: 3072-dim
- **Hidden Layers**: 128 and 64 neurons, ReLU, He initialization
- **Dropout**: 0.3 each
- **Output Layer**: 10-class softmax
- **Optimizer**: Adam or SGD
- **Loss**: SparseCategoricalCrossentropy

## Training Strategy

- Batch Gradient Descent (using full dataset as batch)
- Training and validation run for 20 epochs
- Models are saved every 5 epochs
- Accuracy and loss curves are plotted to monitor overfitting

## Results Summary

- **2-layer network**: Simpler and more effective on this dataset
- **3-layer network**: Slightly overfits and is slower to converge
- Dropout and He initialization help mitigate overfitting

## Requirements

Install required libraries using:
```bash
pip install numpy opencv-python matplotlib tensorflow keras
