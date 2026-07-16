"""
NinaPro DB5 - 1D CNN on raw EMG windows.

Loads the raw (150 x 8) windows saved by preprocess.py and trains a 1D CNN.

Improvements over the original script:
  * fixed, project-wide random seed (22) instead of 42 -> reproducible & consistent
  * EarlyStopping + ReduceLROnPlateau with best-weight restoration
    -> stops before overfitting and usually improves the test score
  * class weighting -> compensates for uneven window counts per gesture
  * the confusion matrix is saved to outputs/ninapro/ instead of only shown

Run:  python src/ninapro/train_cnn.py
"""
import sys
import pathlib

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from emg_common import (set_seed, SEED, DATA_DIR, output_path,
                        plot_confusion_matrix, standardize_windows,
                        build_raw_cnn, NINAPRO_NAMES)

set_seed()

print("=" * 50)
print("NINAPRO DB5 RAW 1D CNN")
print("=" * 50)

# --------------------------------------------------------------------------- #
# 1. Load raw windows
# --------------------------------------------------------------------------- #
nina_dir = DATA_DIR / "ninapro"
try:
    X_raw = np.load(nina_dir / "X_ninapro_raw.npy")
    y_raw = np.load(nina_dir / "y_ninapro_raw.npy")
    print(f"Raw data loaded: {X_raw.shape}")
except FileNotFoundError:
    print("ERROR: raw .npy files not found. Run preprocess.py first.")
    sys.exit(1)

target_names = [NINAPRO_NAMES[i] for i in range(7)]
y = LabelEncoder().fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=SEED, stratify=y)

# --------------------------------------------------------------------------- #
# 2. Standardise raw windows (fit on train only)
# --------------------------------------------------------------------------- #
X_train, X_test = standardize_windows(X_train, X_test)
print(f"Train: {X_train.shape}   Test: {X_test.shape}")

# --------------------------------------------------------------------------- #
# 3. Model (shared raw-signal architecture; same as the custom CNN)
# --------------------------------------------------------------------------- #
model = build_raw_cnn(input_shape=X_train.shape[1:], n_classes=len(target_names))

class_weights = dict(enumerate(compute_class_weight(
    "balanced", classes=np.unique(y_train), y=y_train)))

callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
]

model.fit(X_train, y_train, epochs=60, batch_size=64, validation_split=0.1,
          class_weight=class_weights, callbacks=callbacks, verbose=1)

# --------------------------------------------------------------------------- #
# 4. Evaluate
# --------------------------------------------------------------------------- #
y_pred = np.argmax(model.predict(X_test), axis=1)
acc = accuracy_score(y_test, y_pred)
print(f"\nNinaPro Raw 1D CNN accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=target_names,
                            zero_division=0))
plot_confusion_matrix(y_test, y_pred, "NinaPro 1D CNN", target_names,
                      out_file=output_path("ninapro", "CNN_4sbj.png"))
