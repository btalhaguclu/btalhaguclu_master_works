"""
Custom (Ecem & Selman) dataset - 1D CNN WITHOUT the 'cylindrical' class (5 classes).

Drops label 4 (cylindrical), re-encodes the remaining labels to 0..4 and trains
the same 1D CNN. Same reproducibility/early-stopping/class-weight improvements
as train_cnn.py. Confusion matrix saved to outputs/custom/.

Run:  python src/custom/train_cnn_wo_cyl.py
"""
import sys
import pathlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from emg_common import (set_seed, SEED, DATA_DIR, output_path,
                        plot_confusion_matrix, CUSTOM_NAMES)

set_seed()

CYL_LABEL = 4

df = pd.read_csv(DATA_DIR / "custom" / "custom_dataset.csv")
df = df[df["Label"] != CYL_LABEL]

X = df.drop("Label", axis=1).values
y_raw = df["Label"].values
y = LabelEncoder().fit_transform(y_raw)  # 0,1,2,3,5 -> 0,1,2,3,4
target_names = [CUSTOM_NAMES[i] for i in [0, 1, 2, 3, 5]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)[..., np.newaxis]
X_test = scaler.transform(X_test)[..., np.newaxis]
n_features = X_train.shape[1]

model = Sequential([
    Conv1D(32, 3, activation="relu", input_shape=(n_features, 1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(5, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

class_weights = dict(enumerate(compute_class_weight(
    "balanced", classes=np.unique(y_train), y=y_train)))
callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
]

model.fit(X_train, y_train, epochs=80, batch_size=32, validation_split=0.1,
          class_weight=class_weights, callbacks=callbacks, verbose=1)

y_pred = np.argmax(model.predict(X_test), axis=1)
acc = accuracy_score(y_test, y_pred)
print(f"\n1D CNN accuracy (5 classes, no cyl): {acc * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=target_names,
                            zero_division=0))
plot_confusion_matrix(y_test, y_pred, "Custom 1D CNN (5 classes)", target_names,
                      out_file=output_path("custom", "CNN_wo_cyl.png"))
