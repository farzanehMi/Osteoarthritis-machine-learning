# Knee Osteoarthritis Classification using Transfer Learning

## Overview

This project focuses on the classification of knee osteoarthritis (OA) severity from knee X-ray images using deep learning and transfer learning techniques.

A pretrained ResNet18 convolutional neural network was fine-tuned to classify osteoarthritis severity into five grades:

- Healthy
- Doubtful
- Minimal
- Moderate
- Severe

The main goal of this project was to investigate how different transfer learning strategies affect model performance in medical image classification.

---

## Dataset

The dataset consists of knee X-ray images categorized into five osteoarthritis severity levels.

Classes:

| Class | Description |
|---|---|
| 0 | Healthy |
| 1 | Doubtful |
| 2 | Minimal |
| 3 | Moderate |
| 4 | Severe |

The dataset was divided into:

- Training set
- Validation set
- Test set

---

## Methodology

### Data Preprocessing

Images were processed using PyTorch torchvision transforms:

- Resizing images to 224 × 224 pixels
- Conversion to tensors
- ImageNet normalization

### Model Architecture

A pretrained **ResNet18** model was used as the backbone.

Transfer learning strategies were investigated by modifying which layers were trainable:

- Frozen feature extractor with trainable classifier
- Fine-tuning deeper convolutional layers
- Full comparison of different training strategies

---

## Experiments

Four different experiments were performed:

| Experiment | Strategy |
|---|---|
| Experiment 1 | Baseline transfer learning with frozen backbone |
| Experiment 2 | Class-weighted training to address class imbalance |
| Experiment 3 | Fine-tuning deeper layers of ResNet18 |
| Experiment 4 | Fine-tuning strategy optimization |

Each experiment was evaluated using:

- Accuracy
- Macro F1-score
- Weighted F1-score
- Confusion Matrix
- Classification Report

---

## Results

Four experiments were conducted to compare different transfer learning strategies.

| Experiment | Accuracy | Macro F1-score |
|---|---:|---:|
| Experiment 1 | 43.56% | 0.281 |
| Experiment 2 | 31.98% | 0.308 |
| Experiment 3 | 49.40% | 0.527 |
| Experiment 4 | 50.11% | 0.559 |

The best performance was achieved by Experiment 4 using fine-tuning of the pretrained ResNet18 model.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Scikit-learn
- NumPy
- Matplotlib
- Google Colab

---

## How to Run

1. Clone this repository.

2. Install required libraries:

```bash
pip install -r requirements.txt
```

3. Open the notebook:
Knee_OA_TransferLearning.ipynb

4. Run the cells sequentially

## Future Improvements

Possible future improvements include:

Data augmentation
Larger pretrained architectures
Hyperparameter optimization
Cross-validation
Explainable AI methods such as Grad-CAM
Training with multiple random seeds

## Author

Farzaneh M
