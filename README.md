# Stress Detection from Physiological Signals using HDAO, SMOTE, and Ensemble Learning

This repository presents a robust machine learning pipeline for detecting stress from wearable biosensor data. It introduces a novel oversampling technique called HDAO (Hypersphere Density-Aware Oversampling), compares it with SMOTE and a no-oversampling baseline, and evaluates multiple traditional classifiers along with a stacked ensemble model.

---

## Overview

Physiological signals such as ECG, EDA, EMG, and others carry important information about an individual’s affective and cognitive state. Automatic stress detection using these signals is challenging due to class imbalance, noise, and variability across individuals.

This project implements and compares three data balancing strategies:

1. No Oversampling (raw imbalanced data)
2. SMOTE Oversampling (interpolation-based)
3. HDAO Oversampling (proposed method using hypersphere-based sampling)

These are combined with multiple classification models including traditional ML algorithms and a custom stacked ensemble.

---

## Repository Structure

| File                     | Description                                                             |
|--------------------------|-------------------------------------------------------------------------|
| `stress-classifier.ipynb`| Jupyter notebook containing the full pipeline and experiments           |
| `requirements.txt`       | Required Python libraries and dependencies                              |
| `README.md`              | Project documentation (this file)                                       |

---

## Dataset

We recommend using the WESAD dataset (Wearable Stress and Affect Detection) which contains multimodal physiological data collected via a chest-worn device. It includes:

- ECG (Electrocardiogram)
- EDA (Electrodermal Activity)
- EMG (Electromyogram)
- RESP (Respiration)
- ACC (Accelerometer)
- TEMP (Temperature)

Labels:  
- 0: Baseline  
- 1: Stress  
- 2: Amusement

Prior to using this repository, preprocess the raw WESAD signals into per-window statistical and domain-specific features, and assign the appropriate labels.

---

## Methodology

### 1. Feature Extraction

- The physiological signals are segmented into fixed-size windows (e.g., 30 seconds with 50% overlapping)
- Time-domain, frequency-domain, and morphological features are extracted
- Each row in the final dataset represents a window with a corresponding `stress_label`

### 2. Oversampling Methods

**(a) No Oversampling**  
- Direct training on the imbalanced dataset

**(b) SMOTE**  
- Synthetic samples generated through interpolation between k-nearest neighbors

**(c) HDAO (Proposed)**  
- Outlier removal using Local Outlier Factor (LOF)
- PCA projection (optional) to reduce dimensionality
- Local density estimation using k-nearest neighbor distances
- Density-weighted centroid computation
- Gaussian sampling from class-specific covariance matrix within a learned hypersphere radius

### 3. Classifiers Used

**Traditional Classifiers:**
- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)

**Custom Ensemble Model:**
- Base Learners: LDA, KNN, Logistic Regression (with scaling)
- Meta Learner: XGBoost (no scaling required)

### 4. Evaluation Metrics

All classifiers are evaluated on:
- Accuracy
- Precision, Recall, and F1-score
- Confusion Matrix
- ROC Curve and AUC

---

## Experimental Design

Each classifier is trained under three different data conditions:

| Oversampling Strategy | Classifiers Tested                    |
|------------------------|--------------------------------------|
| None                  | LR, DT, SVM, RF, KNN, NB, ANN         |
| SMOTE                 | LR, DT, SVM, RF, KNN, NB, ANN         |
| HDAO                  | LR, DT, SVM, RF, KNN, NB, ANN         |

Based on instance-based, generative and discriminative nature of classifiers, a custom ensemble model is made having LR, KNN and LDA as base learners and then feeding the probabilistic feature vector to XGBoost classifier. 

The same train-test split (80:20) is used for all experiments to ensure comparability. Oversampling applied only on training set to see the performance on unseen real world data. 
The HDAO + Ensemble model consistently yields the best results across multiple metrics.

