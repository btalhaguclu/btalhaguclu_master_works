"""
Custom (Ecem & Selman) dataset - 1D CNN on RAW windows WITHOUT 'tip' and 'cylindrical' (4 classes).

Loads the raw (150 x 2) windows, drops the cylindrical class (label 3 after
'tip' removal), re-encodes the remaining labels to 0..3 and trains the same
raw-signal CNN (emg_common.build_raw_cnn) as the 5-class version and NinaPro.

Run:  python src/custom/preprocess.py   # first, to create X_custom_raw.npy
      python src/custom/train_cnn_wo_cyl.py
"""
import sys
import pathlib

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from emg_common import (set_seed, SEED, DATA_DIR, output_path,
                        plot_confusion_matrix, standardize_windows,
                        build_raw_cnn, CUSTOM_NAMES)

set_seed()

CYL_LABEL = 3  # cylindrical label after 'tip' removal (0..4 renumbering)

custom_dir = DATA_DIR / "custom"
try:
    X_raw = np.load(custom_dir / "X_custom_raw.npy")
    y_raw = np.load(custom_dir / "y_custom_raw.npy")
except FileNotFoundError:
    print("ERROR: raw .npy files not found. Run preprocess.py first.")
    sys.exit(1)

# drop cylindrical
keep = y_raw != CYL_LABEL
X_raw, y_raw = X_raw[keep], y_raw[keep]
kept_labels = np.sort(np.unique(y_raw))          # [0, 1, 2, 4]
target_names = [CUSTOM_NAMES[int(i)] for i in kept_labels]  # point, rock, closed, open
y = LabelEncoder().fit_transform(y_raw)          # 0,1,2,4 -> 0,1,2,3
print(f"Raw data (no cyl): {X_raw.shape}   classes: {target_names}")

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=SEED, stratify=y)

X_train, X_test = standardize_windows(X_train, X_test)

model = build_raw_cnn(input_shape=X_train.shape[1:], n_classes=len(kept_labels))

class_weights = dict(enumerate(compute_class_weight(
    "balanced", classes=np.unique(y_train), y=y_train)))
callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
]

model.fit(X_train, y_train, epochs=80, batch_size=64, validation_split=0.1,
          class_weight=class_weights, callbacks=callbacks, verbose=1)

y_pred = np.argmax(model.predict(X_test), axis=1)
acc = accuracy_score(y_test, y_pred)
print(f"\nCustom Raw 1D CNN accuracy (4 classes, no tip/cyl): {acc * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=target_names,
                            zero_division=0))
plot_confusion_matrix(y_test, y_pred, "Custom 1D CNN (raw, 4 classes)",
                      target_names, out_file=output_path("custom", "CNN_wo_cyl.png"))
