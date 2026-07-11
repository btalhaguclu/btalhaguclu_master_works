# EMG-Based Hand Gesture Recognition

Machine-learning experiments for recognising hand gestures from surface EMG
(sEMG) signals, developed as part of an MSc thesis. The project covers two
datasets and compares classical models (SVM, Random Forest, MEET) with a 1D CNN.

- **NinaPro DB5** — public benchmark, 8-channel Myo-armband EMG, 7 gestures.
- **Custom dataset** — 2-channel EMG recorded from two subjects (Ecem & Selman), 6 gestures.

---

## Repository structure

```
.
├── src/
│   ├── emg_common.py              # Shared utilities (seeding, paths, features, plotting)
│   ├── ninapro/
│   │   ├── preprocess.py          # Raw .mat -> features CSV + raw .npy windows
│   │   ├── train_classical.py     # SVM / RF / MEET on 4 subjects
│   │   ├── within_subject.py      # SVM / RF, random split over S2+S5
│   │   ├── cross_subject.py       # Train on S2,S5 -> test on unseen S1,S10
│   │   └── train_cnn.py           # 1D CNN on raw EMG windows
│   └── custom/
│       ├── preprocess.py          # Raw .mat -> custom_dataset.csv
│       ├── train_classical.py     # SVM / RF / MEET (6 gestures)
│       ├── train_classical_wo_cyl.py  # Same, cylindrical gesture removed
│       ├── train_cnn.py           # 1D CNN (6 gestures)
│       └── train_cnn_wo_cyl.py    # 1D CNN (5 gestures)
├── data/
│   ├── ninapro/                   # NinaPro DB5 .mat, feature CSVs, raw .npy
│   └── custom/                    # Ecem/Selman .mat + custom_dataset.csv
├── matlab/                        # Original MATLAB feature-extraction scripts
├── outputs/
│   ├── ninapro/                   # Confusion matrices (auto-saved by scripts)
│   └── custom/
├── docs/
│   └── images/                    # Gesture reference figures
├── requirements.txt
├── .gitignore
└── .gitattributes
```

---

## Gestures

**NinaPro DB5 (7):** Thumb up (B1), Scissors (B2), Open (B5), Closed (B6),
Point (B7), Cylindrical (C5), Pinch (C14).

**Custom (6):** point, tip, rock, closed, cylindrical, open.

---

## Signal processing & features

Raw EMG is segmented into **150 ms sliding windows**. For NinaPro, the 200 Hz
signal is first resampled to 1000 Hz. Four classic **time-domain features** are
computed per channel:

| Feature | Meaning |
| :-- | :-- |
| RMS | Root Mean Square — muscle power |
| MAV | Mean Absolute Value — signal intensity |
| WL  | Waveform Length — signal complexity |
| SSC | Slope Sign Changes — frequency-content estimator |

This yields **32 features/window** for NinaPro (8 channels × 4) and
**8 features/window** for the custom 2-channel dataset (2 channels × 4).

---

## Models

- **SVM** — RBF kernel, `C=10`, `gamma='scale'`.
- **Random Forest** — 100 trees.
- **MEET** — *Mixture of Experts Extra Trees*: One-vs-One ensemble of
  Extra-Trees experts ([Xiong et al., 2024](https://arxiv.org/pdf/2405.09562)).
- **1D CNN** — convolutional network trained on standardised raw windows
  (NinaPro) or feature vectors (custom), with early stopping and class weighting.

---

## Getting started

```bash
# 1. Install dependencies (a virtual environment is recommended)
pip install -r requirements.txt

# 2. (Optional) regenerate features from raw .mat files
python src/ninapro/preprocess.py
python src/custom/preprocess.py

# 3. Train and evaluate — run any script from the repo root
python src/ninapro/train_classical.py
python src/ninapro/cross_subject.py
python src/custom/train_classical.py
python src/custom/train_cnn.py
```

Every script resolves data paths relative to the repository root, so it can be
run from anywhere. Confusion matrices are written automatically to `outputs/`.

---

## Reported results

Results below are from the reference runs committed to the repository
(random 80/20 split unless noted).

**NinaPro DB5 — 4 subjects**

| Model | Accuracy |
| :-- | :-- |
| SVM (RBF) | 87.65% |
| Random Forest | 92.88% |
| **MEET (Extra Trees + OvO)** | **> 94.74%** |

**Custom dataset — all 6 gestures**

| Model | Accuracy | Main confusion |
| :-- | :-- | :-- |
| SVM (RBF) | 71.19% | cylindrical ↔ closed |
| Random Forest | 71.63% | cylindrical ↔ closed |
| MEET | 70.93% | cylindrical ↔ closed |

The cylindrical gesture is the dominant source of error on the 2-channel data;
`*_wo_cyl.py` variants isolate that effect.

---

## Notes

- Random seed is fixed project-wide (**22**) for reproducibility.
- Feature scaling (`StandardScaler`) is always fit on the training split only,
  to prevent data leakage; the cross-subject script fits on the training
  subjects and only transforms the unseen ones.

## References

- NinaPro Database — https://www.ninapro.org/
- MEET: *Mixture of Experts Extra Tree-Based sEMG Hand Gesture Identification* — https://arxiv.org/pdf/2405.09562

## Author

**Burak Talha Güçlü** — MSc thesis work on EMG-based hand gesture recognition.
