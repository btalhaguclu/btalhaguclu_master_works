"""
emg_common.py
=============
Shared utilities for the EMG hand-gesture recognition project.

This module centralises the code that used to be copy-pasted across every
training script (confusion-matrix plotting, time-domain feature extraction,
random-seed handling and file-path resolution). Importing from here keeps the
individual scripts short and guarantees that every experiment uses the same
seed, the same feature definitions and the same output conventions.

Usage (from any script under src/):

    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
    from emg_common import (set_seed, DATA_DIR, OUTPUT_DIR,
                            compute_td_features, evaluate_model, NINAPRO_NAMES)
"""

from __future__ import annotations

import os
import random
import pathlib

import numpy as np

# Use a non-interactive backend so scripts run headless (e.g. on a server or in
# CI) and always *save* their figures instead of blocking on a GUI window.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 22  # Fixed project-wide seed (Edirne plate number) for reproducibility.


def set_seed(seed: int = SEED) -> int:
    """Seed Python, NumPy and (if available) TensorFlow for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # TensorFlow is only needed by the CNN scripts.
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
    return seed


# --------------------------------------------------------------------------- #
# Project paths (resolved relative to the repo root, so scripts work from any
# working directory).
# --------------------------------------------------------------------------- #
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"


def output_path(*parts: str) -> pathlib.Path:
    """Build a path under outputs/, creating parent directories as needed."""
    path = OUTPUT_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Gesture label maps
# --------------------------------------------------------------------------- #
# NinaPro DB5 subset used in this project (label -> readable name).
NINAPRO_NAMES = {
    0: "Thumb up (B1)",
    1: "Scissors (B2)",
    2: "Open (B5)",
    3: "Closed (B6)",
    4: "Point (B7)",
    5: "Cylindrical (C5)",
    6: "Pinch (C14)",
}

# Custom (Ecem & Selman) 2-channel dataset (label -> readable name).
CUSTOM_NAMES = {
    0: "point",
    1: "tip",
    2: "rock",
    3: "closed",
    4: "cylindrical",
    5: "open",
}


def names_for_labels(name_map: dict, labels) -> list:
    """Return readable names for the labels actually present in the data."""
    return [name_map[int(lbl)] for lbl in labels]


# --------------------------------------------------------------------------- #
# Time-domain feature extraction
# --------------------------------------------------------------------------- #
def compute_td_features(segment: np.ndarray):
    """Return the four classic time-domain sEMG features for a 1-D window.

    RMS  - Root Mean Square      (muscle power)
    MAV  - Mean Absolute Value   (signal intensity)
    WL   - Waveform Length       (signal complexity)
    SSC  - Slope Sign Changes    (frequency-content estimator)
    """
    segment = np.asarray(segment, dtype=float)
    rms = np.sqrt(np.mean(segment ** 2))
    mav = np.mean(np.abs(segment))
    wl = np.sum(np.abs(np.diff(segment)))
    diff = np.diff(segment)
    ssc = int(np.sum((diff[:-1] * diff[1:]) < 0))
    return rms, mav, wl, ssc


def feature_columns(n_channels: int) -> list:
    """Column names for an n-channel time-domain feature table (+ 'Label')."""
    cols = []
    for ch in range(1, n_channels + 1):
        cols += [f"RMS_ch{ch}", f"MAV_ch{ch}", f"WL_ch{ch}", f"SSC_ch{ch}"]
    cols.append("Label")
    return cols


# --------------------------------------------------------------------------- #
# Evaluation & plotting
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(y_true, y_pred, model_name, target_names,
                          labels=None, out_file=None, title=None):
    """Draw a row-normalised confusion matrix and save it to `out_file`."""
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title(title or f"{model_name} - Normalized Confusion Matrix", fontsize=15)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if out_file is not None:
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        print(f"  -> confusion matrix saved to {out_file}")
    plt.close()


def evaluate_model(model, X_test, y_test, model_name, target_names,
                   labels=None, out_file=None):
    """Predict, print accuracy + classification report, and save the CM.

    Returns the accuracy (float) so callers can build a comparison summary.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {model_name} ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, labels=labels,
                                target_names=target_names, zero_division=0))
    plot_confusion_matrix(y_test, y_pred, model_name, target_names,
                          labels=labels, out_file=out_file)
    return acc


def print_summary(results: dict):
    """Pretty-print an accuracy comparison table and name the winner."""
    print("\n" + "=" * 50)
    print("SUMMARY COMPARISON")
    print("=" * 50)
    for name, acc in results.items():
        print(f"{name:<28} {acc * 100:6.2f}%")
    best = max(results, key=results.get)
    print(f"\nBest model: {best} ({results[best] * 100:.2f}%)")
