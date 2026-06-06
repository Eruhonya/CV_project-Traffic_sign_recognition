# 🚦 Traffic Sign Recognition — CV Project

A computer vision project for **traffic sign classification** using multiple CNN architectures trained and evaluated in Google Colab.

---

## 📌 Overview

This project compares four CNN-based models for traffic sign recognition, analyzing their accuracy, recall, and overfitting behavior across train/test splits.

| Model | Train Recall | Test Recall | Overfit Gap |
|-------|-------------|-------------|-------------|
| Basis | 98.8% | 86.7% | 12.3% |
| LeNet | 99.8% | 88.8% | 11.0% |
| LeNet_pp | 99.3% | 90.4% | 8.9% |
| **Custom** | **99.8%** | **95.2%** | **4.6%** ✅ |

> **Best result:** Custom architecture achieved **95.2% test recall** with the lowest overfitting gap of only 4.6%.

---

## 🧠 Models

### 1. Basis
A simple baseline CNN used as a reference point. High training recall (98.8%) but significant overfit (12.3%).

### 2. LeNet
Classic LeNet architecture adapted for traffic sign classification. Slightly better test performance than Basis.

### 3. LeNet_pp (LeNet++)
An improved version of LeNet with additional regularization/augmentation. Reduces overfitting to 8.9%.

### 4. Custom ⭐
A custom-designed CNN architecture. Best generalization with 95.2% test recall and minimal overfit (4.6%).

---

## 📁 Project Structure

```
traffic-sign-recognition/
│
├── notebooks/
│   └── traffic_sign_classification.ipynb   # Main Colab notebook
│
├── models/
│   ├── basis.py          # Baseline CNN
│   ├── lenet.py          # LeNet architecture
│   ├── lenet_pp.py       # LeNet++ architecture
│   └── custom.py         # Custom CNN architecture
│
├── datasets/
│   ├── dataset_1/        # Dataset 1
│   ├── dataset_2/        # Dataset 2
│   └── dataset_3/        # Dataset 3
│
├── results/
│   └── recall_comparison.png   # Train/Test/Overfit chart
│
└── README.md
```

---

## 🗂️ Datasets

The project uses **3 datasets** of traffic signs for training and evaluation. Data is split into train/test sets for each experiment.

> Datasets are not included in this repository due to size. Download links or instructions below.

<!-- TODO: Add dataset download links or Kaggle/Drive links here -->

---

## 🚀 Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebooks/traffic_sign_classification.ipynb)

### Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/traffic-sign-recognition.git
cd traffic-sign-recognition
pip install -r requirements.txt
```

### Dependencies

```
tensorflow>=2.x   # or pytorch
numpy
matplotlib
opencv-python
scikit-learn
pandas
```

---

## 📊 Results

![Recall Comparison](results/recall_comparison.png)

The chart above shows **Recall** for Train and Test sets, along with the **Overfit %** for each model.

Key findings:
- All models achieve near-perfect training recall (~99%)
- The **Custom model** generalizes best with the smallest train-test gap
- LeNet++ shows a good balance between complexity and generalization

---

## 🛠️ Tech Stack

- **Platform:** Google Colab
- **Framework:** TensorFlow / Keras
- **Language:** Python 3
- **Visualization:** Matplotlib

---

## 📈 Future Work

- [ ] Add data augmentation pipeline
- [ ] Try transfer learning (MobileNet, EfficientNet)
- [ ] Deploy as a web app or mobile app
- [ ] Expand dataset coverage (more sign types)

---

## 👤 Author

<!-- Add your name/GitHub profile here -->

---

## 📄 License

This project is licensed under the MIT License.
