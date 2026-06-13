# EMG Hand Gesture Recognition (NinaPro)

This repository contains machine learning models for recognizing hand gestures from Electromyography (EMG) signals using a custom dataset collected by Ecem and Selman.

## 📋 Overview

This project implements and compares multiple machine learning classifiers to recognize hand gestures from 2-channel EMG signals. 

### Hand Gestures Recognized (6 Movements)

1. Point 
2. Tip
3. Rock
4. Closed
5. Cylindrical
6. Open

## 🏗️ Project Structure

```
custom_data/
├── custom_data_preprocessing.py      # Preprocessing pipeline
├── custom_data_wo_cyl.py             # Script for analysis without the "cylindrical" gesture
├── svm.py                            # SVM-based classification
├── custom_dataset.csv                # Custom EMG feature dataset
├── data/                             # Raw MATLAB data from Ecem and Selman
├── outputs/                          # Confusion Matrix of Model outputs
└── README.md                         # This documentation file
```

## 📊 Data

Kullanılan dosyalar:


- `ecem_data_1.1.mat`, `ecem_data_1.2.mat` ,`selman_data_1.mat`, `selman_data_2.mat`: Raw EMG data from trials by Ecem and Selman.

### Signal Preprocessing & Feature Extraction (MATLAB)
Raw sensor data is resampled to 1000 Hz and divided into 150 ms windows. The following Time-Domain Features are extracted for each of the 8 channels, yielding a **32-dimensional** feature vector per window:
* **RMS (Root Mean Square):** Total muscle power produced.
* **MAV (Mean Absolute Value):** Signal amplitude/intensity level.
* **WL (Waveform Length):** Signal complexity.
* **SSC (Slope Sign Changes):** Frequency characteristic estimator

### Data Files Generated

- `custom_dataset.csv`: Feature-extracted data table.

## 🤖 Machine Learning Models

### 1. Support Vector Machine (SVM)
- **Kernel**: RBF (Radial Basis Function)
- **Parameters**: C=10, gamma='scale'
- Effective for high-dimensional EMG feature spaces

### 2. Random Forest Classifier
- **Estimators**: 100 trees
- **Max depth**: None (unlimited)
- Provides feature importance insights

### 3. MEET (Mixture of Experts Extra Trees)
- **Base estimator**: Extra Trees Classifier
- **Strategy**: One-Vs-One multiclass approach
- Ensemble of binary classifiers for improved generalization

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn


## 📈 Expected Output

All scripts will output:
- **Accuracy Score**: Overall classification accuracy percentage
- **Classification Report**: Precision, recall, and F1-score per gesture
- **Confusion Matrix Heatmap**: Visualization of classification performance
- **Model Comparison Summary**: Relative performance of all models

## 🔄 Data Processing 

run custom_data_preprocessing.py
These script:
- Load raw EMG signals
- Apply preprocessing (filtering, segmentation)
- Extract statistical features from sliding windows
- Save results to CSV files

## 📚 Feature Engineering

EMG features extracted per channel include:
- Mean Absolute Value (MAV)
- Variance (VAR)
- Zero Crossing Rate (ZCR)
- Slope Sign Changes (SSC)

## 🎯 Performance Metrics

- **Train-Test Split**: 80% train, 20% test (within-subject)
- **Feature Scaling**: StandardScaler applied to normalize features
- **Evaluation Metrics**:
  - Accuracy
  - Precision, Recall, F1-score
  - Normalized Confusion Matrix


## 📊 Performance Results (Random Split)

Following training tests conducted with data from 2 different subjects, the overall accuracy rates of the models are as follows:

| Model Type | Accuracy Rate | Confused Classes |
| :--- | :--- | :--- |
| **SVM (RBF Kernel)** | 71.19% | CYCLINDRICAL and CLOSED |
| **Random Forest** | 71.63% |  CYCLINDRICAL and CLOSED |
| **MEET (Extra Trees + OvO)** | 70.93% |  CYCLINDRICAL and CLOSED  |


Accuracy when the CYLINDRICAL class is removed from the training set:

| Model Type | Accuracy Rate | Confused Classes |
| :--- | :--- | :--- |
| **SVM (RBF Kernel)** | 71.19% |  |
| **Random Forest** | 71.63% |  |
| **MEET (Extra Trees + OvO)** | 70.93% |   |

*(Detailed Confusion Matrices and classification reports are available in the project's visual assets directory.)*

---

## 📝 Notes
- All code documentation within the scripts is in Turkish.
- Features are scaled only on training data to prevent data leakage.
- Visuals are available in svm.png, rf.png, meet.png, and the *_wo_cyl.png files.
- custom_data_wo_cyl.py is prepared for scenarios that exclude the "cyl" gesture.
- Random state fixed (22) for reproducibility.


## 🔗 References

- MEET: Mixture of Experts Extra Tree-Based sEMG Hand Gesture Identification : https://arxiv.org/pdf/2405.09562

## 👤 Author

Burak Talha Güçlü

This project is part of a thesis work on EMG-based hand gesture recognition.

## 📄 License

Please refer to project documentation for licensing information.


---

**Last Updated**: 2026-06-07
