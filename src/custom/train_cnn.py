"""
Custom (Ecem & Selman) dataset - 1D CNN on RAW EMG windows (5 classes; 'tip' excluded).

Trains on the raw (150 x 2) windows saved by preprocess.py, using the SAME
raw-signal CNN architecture as the NinaPro CNN (emg_common.build_raw_cnn) -
only the channel count (2) and class count (5) differ. This makes the CNN
consistent across both datasets: in both cases the network learns from the raw
signal instead of from hand-crafted features.

Run:  python src/custom/preprocess.py   # first, to create X_custom_raw.npy
      python src/custom/train_cnn.py
"""
import sys
import pathlib

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from emg_common import (set_seed, SEED, DATA_DIR, output_path,
                        plot_confusion_matrix, standardize_windows,
                        build_raw_cnn, CUSTOM_NAMES)

set_seed()

# --------------------------------------------------------------------------- #
# 1. Load raw windows
# --------------------------------------------------------------------------- #
custom_dir = DATA_DIR / "custom"
try:
    X_raw = np.load(custom_dir / "X_custom_raw.npy")   # (n, 150, 2)
    y = np.load(custom_dir / "y_custom_raw.npy")
    print(f"Raw data loaded: {X_raw.shape}")
except FileNotFoundError:
    print("ERROR: raw .npy files not found. Run preprocess.py first.")
    sys.exit(1)

labels = np.sort(np.unique(y))
target_names = [CUSTOM_NAMES[int(i)] for i in labels]   # 5 gestures (no 'tip')

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=SEED, stratify=y)

# 2. Standardise raw windows (fit on train only)
X_train, X_test = standardize_windows(X_train, X_test)
print(f"Train: {X_train.shape}   Test: {X_test.shape}")

# 3. Model (shared raw-signal architecture)
model = build_raw_cnn(input_shape=X_train.shape[1:], n_classes=len(labels))

class_weights = dict(enumerate(compute_class_weight(
    "balanced", classes=np.unique(y_train), y=y_train)))
callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
]

model.fit(X_train, y_train, epochs=80, batch_size=64, validation_split=0.1,
          class_weight=class_weights, callbacks=callbacks, verbose=1)

# 4. Evaluate
y_pred = np.argmax(model.predict(X_test), axis=1)
acc = accuracy_score(y_test, y_pred)
print(f"\nCustom Raw 1D CNN accuracy (5 classes): {acc * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=target_names,
                            zero_division=0))
plot_confusion_matrix(y_test, y_pred, "Custom 1D CNN (raw, 5 classes)",
                      target_names, out_file=output_path("custom", "CNN.png"))
