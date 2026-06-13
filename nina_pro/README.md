# EMG Hand Gesture Recognition (NinaPro)

This repository contains machine learning models for recognizing hand gestures from Electromyography (EMG) signals using the NinaPro dataset.

## 📋 Overview

This project implements and compares multiple machine learning classifiers to recognize hand gestures from 8-channel EMG signals. The models are trained on NinaPro dataset movements and evaluated using both within-subject and cross-subject validation approaches.

### Hand Gestures Recognized (7 Movements)

1. **B1**: Thumb Up 
2. **B2**: Scissors
3. **B5**: Open
4. **B6**: Closed
5. **B7**: Point
6. **C5**: Cylindrical
7. **C14**: Pinch

## 🏗️ Project Structure

```
nina_pro/
├── S2_S5.py                           # Within-subject training (Subjects 2 & 5)
├── S2_S5_S1_S10/                      # Confusion Matrix of Model outputs 
├── train_test.py                      # Cross-subject model training and testing
├── main.py                            # Within-subject training with MEET (Mixture of Experts) model implementation (Subjects 1, 2, 5, 10)
├── ninapro_to_mydata_S2_S5.m          # MATLAB data preprocessing (Subjects 2 & 5)
├── Train_S2_S5_Test_S1_S10.m          # MATLAB training/testing setup
├── db5_data/                          # Processed and Unprocessed EMG feature data (CSV files)
├── hand_moves.png                     # Hand gesture visualization
└── README.md                          # This file
```

## 📊 Data

The EMG signals are from the **NinaPro Database (DB5)** containing:
- **8 EMG channels** from forearm sensors
- **Extracted features** (mean absolute value, variance, zero crossing rate, etc.)
- **7 hand movement classes** from NinaPro standard protocol

### Signal Preprocessing & Feature Extraction (MATLAB)
Raw sensor data is resampled to 1000 Hz and divided into 150 ms windows. The following Time-Domain Features are extracted for each of the 8 channels, yielding a **32-dimensional** feature vector per window:
* **RMS (Root Mean Square):** Total muscle power produced.
* **MAV (Mean Absolute Value):** Signal amplitude/intensity level.
* **WL (Waveform Length):** Signal complexity.
* **SSC (Slope Sign Changes):** Frequency characteristic estimator

### Data Files Generated

- `emg_features_7moves_8ch.csv` - Within-subject features (Subjects 2 & 5)
- `emg_train_S2_S5.csv` - Training features (cross-subject setup)
- `emg_test_S1_S10.csv` - Testing features (cross-subject setup)

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

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nina_pro
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

3. Ensure EMG feature CSV files are present in `db5_data/` directory

### Running the Models

#### Within-Subject Validation
Train and test on the same subject (Subjects 2 & 5):
```bash
python S2_S5.py
```

#### Cross-Subject Validation
Train on subjects 2 & 5, test on subjects 1 & 10:
```bash
python train_test.py
```

#### MEET Model
Run the advanced MEET ensemble method:
```bash
python meet.py
```

## 📈 Expected Output

All scripts will output:
- **Accuracy Score**: Overall classification accuracy percentage
- **Classification Report**: Precision, recall, and F1-score per gesture
- **Confusion Matrix Heatmap**: Visualization of classification performance
- **Model Comparison Summary**: Relative performance of all models

## 🔄 Data Processing (MATLAB)

If you need to regenerate the feature data from raw NinaPro signals:

1. **For within-subject features**:
```matlab
ninapro_to_mydata
```

2. **For cross-subject features**:
```matlab
Train_S2_S5_Test_S1_S10
```

These scripts:
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

Following training tests conducted with data from 4 different subjects, the overall accuracy rates of the models are as follows:

| Model Type | Accuracy Rate | Confused Classes |
| :--- | :--- | :--- |
| **SVM (RBF Kernel)** | 87.65% | B1, B2, C14 |
| **Random Forest** | 92.88% | Partially B1 and C14 |
| **MEET (Extra Trees + OvO)** | **> 94.74%** | **Most stable class separation** |

*(Detailed Confusion Matrices and classification reports are available in the project's visual assets directory.)*

---

## 📝 Notes

- All text in scripts is in **Turkish** for documentation purposes
- Features are scaled only on training data to prevent data leakage
- Cross-subject validation tests generalization to unseen subjects
- Random state fixed (22) for reproducibility

## 🔗 References

- NinaPro Database: https://www.ninapro.org/
- Dataset DB5: Upper limb kinematic and EMG signals
- NinaPro Protocol: Standard hand movement gestures
- MEET: Mixture of Experts Extra Tree-Based sEMG Hand Gesture Identification : https://arxiv.org/pdf/2405.09562

## 👤 Author

Burak Talha Güçlü

This project is part of a thesis work on EMG-based hand gesture recognition.

## 📄 License

Please refer to project documentation for licensing information.


---

**Last Updated**: 2026-05-29
