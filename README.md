# Signal Processing–Based Synthetic Augmentation for Credit Card Fraud Detection

A research-inspired fraud detection project implementing **IAAFT-S (Iterative Amplitude Adjusted Fourier Transform – Surrogate)** techniques for generating synthetic fraud transactions from highly imbalanced credit card datasets.

The project applies:
- Fourier-transform-based signal processing,
- spectral amplitude preservation,
- iterative surrogate generation,
- synthetic fraud augmentation,

to improve fraud detection robustness in machine learning systems.

---

# Project Motivation

Credit card fraud detection suffers from:
- extreme class imbalance,
- scarcity of fraud samples,
- difficulty learning minority fraud patterns.

Traditional oversampling methods often fail to preserve the statistical structure of fraud data.

This project explores whether:
> signal-processing-based surrogate generation can synthesize realistic fraud samples while preserving important statistical characteristics of original fraudulent transactions.

---

# Research Inspiration

This implementation is inspired by the research paper:

> **Signal Processing on Graphs for Improving Automatic Credit Card Fraud Detection**

The original paper proposes:
- IAAFT-based surrogate generation,
- graph signal processing methods,
- synthetic augmentation for fraud detection.

This project implements a simplified **IAAFT-S-inspired surrogate generation pipeline** using classical Fourier transforms.

---

# Dataset

## Source
Kaggle Credit Card Fraud Detection Dataset

## Dataset Details
- Transaction records containing anonymized PCA-transformed features
- Features used:
  - V1 to V28
  - Amount
- Total signal dimension:
  - 29-dimensional transaction vectors
- Highly imbalanced fraud classification problem

---

# Core Signal Processing Concepts Used

## Fast Fourier Transform (FFT)
Fraud transaction vectors are transformed into frequency domain representations.

## Amplitude Spectrum Preservation
The surrogate generation process preserves the original signal amplitude spectrum.

## Phase Randomization
Signal phases are randomized through time-domain shuffling.

## Inverse FFT
Synthetic transaction profiles are reconstructed using inverse Fourier transforms.

## Iterative Rank Adjustment
IAAFT-S iteratively adjusts generated signals to preserve statistical similarity with original fraud samples.

---

# Project Workflow

```text
Raw Credit Card Transactions
        ↓
Feature Extraction (V1–V28 + Amount)
        ↓
Data Standardization
        ↓
Fraud Sample Isolation
        ↓
IAAFT-S Surrogate Generation
        ↓
Synthetic Fraud Dataset Creation
        ↓
Combine Real + Synthetic Fraud Data
        ↓
Random Forest Model Training
        ↓
ROC & False Alarm Evaluation
```

---

# Data Preprocessing

The notebook:
- loads the credit card fraud dataset,
- extracts V1–V28 + Amount features,
- standardizes features using StandardScaler,
- splits data into train/test sets,
- isolates training fraud samples for surrogate generation.

---

# IAAFT-S Surrogate Generation

The project implements a custom `iaaft_s()` function.

The algorithm:

1. Computes FFT of original fraud signal
2. Extracts target amplitude spectrum
3. Initializes surrogate using shuffled signal values
4. Iteratively:
   - adjusts Fourier amplitudes,
   - preserves randomized phase structure,
   - rank-orders values to match original distribution
5. Produces realistic synthetic fraud transaction vectors

---

# Synthetic Fraud Augmentation

For every training fraud transaction:
- one synthetic surrogate fraud sample is generated,
- synthetic frauds are appended to the training set,
- minority-class representation is improved.

The notebook also compares:
- original fraud statistics,
- synthetic fraud statistics,
- visual transaction-profile similarity.

---

# Machine Learning Model

## Random Forest Classifier

Two models are trained:

### Baseline Model
- trained only on original imbalanced data

### Augmented Model
- trained on real + IAAFT-S synthetic fraud samples

---

# Evaluation Metrics

The project evaluates:

- ROC-AUC
- False Alarm Probability (PFA)
- Fraud Detection Probability (PD)
- Low-false-alarm operational performance
- Statistical similarity between real and synthetic frauds

---

# Statistical Validation

The notebook compares:
- mean values,
- standard deviations,
- min/max ranges,

between:
- original fraud samples,
- synthetic surrogate fraud samples.

This validates that generated surrogates preserve important statistical properties.

---

# Visualizations Included

## Transaction Signal Visualization
Displays one real fraud transaction as a 29-dimensional signal vector.

## Fourier Spectrum Analysis
Visualizes:
- amplitude spectrum,
- phase spectrum.

## Synthetic vs Original Comparison
Compares generated surrogate fraud profiles against original fraud transactions.

## ROC Curve Analysis
Compares:
- baseline Random Forest,
- augmented Random Forest.

## False Alarm Trade-off Analysis
Evaluates fraud detection performance under strict operational false-alarm constraints.

---

# Key Observations

- IAAFT-S successfully generated statistically realistic synthetic fraud samples.
- Synthetic augmentation preserved important spectral characteristics of fraud transactions.
- Augmented Random Forest models achieved fraud detection performance comparable to baseline models.
- The project demonstrated how signal processing concepts can be integrated into machine learning workflows for imbalanced fraud detection.

---

# Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- FFT / Signal Processing
- Random Forest

---



---

# Installation

Clone repository:

```bash
git clone https://github.com/Pranathi3101/IAAFT-Based-Signal-Processing-for-Credit-Card-Fraud-Detection.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
IAAFT-Based Signal Processing for Credit Card Fraud Detection.ipynb
```

Run all cells sequentially.

---

# Future Improvements

- Graph Signal Processing implementation
- ISSG surrogate generation
- Graph Fourier Transform (GFT)
- XGBoost / LightGBM augmentation evaluation
- Advanced anomaly detection architectures
- Real-time fraud scoring systems

---

# Skills Demonstrated

- Signal Processing
- Fourier Analysis
- Synthetic Data Augmentation
- Fraud Detection
- Imbalanced Classification
- Machine Learning
- Research Paper Implementation
- Statistical Signal Analysis

