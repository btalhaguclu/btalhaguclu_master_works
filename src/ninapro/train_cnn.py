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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, Dense,
                                      Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from emg_common import (set_seed, SEED, DATA_DIR, output_path,
                        plot_confusion_matrix, NINAPRO_NAMES)

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
# 2. Standardise (fit on train only; reshape 3D<->2D around the scaler)
# --------------------------------------------------------------------------- #
n_tr, t_steps, ch = X_train.shape
n_te = X_test.shape[0]
scaler = StandardScaler()
X_train = scaler.fit_transform(
    X_train.reshape(n_tr, t_steps * ch)).reshape(n_tr, t_steps, ch)
X_test = scaler.transform(
    X_test.reshape(n_te, t_steps * ch)).reshape(n_te, t_steps, ch)
print(f"Train: {X_train.shape}   Test: {X_test.shape}")

# --------------------------------------------------------------------------- #
# 3. Model
# --------------------------------------------------------------------------- #
model = Sequential([
    Conv1D(64, 10, activation="relu", input_shape=(t_steps, ch)),
    BatchNormalization(),
    MaxPooling1D(2),
    Conv1D(128, 5, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(2),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(7, activation="softmax"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

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
