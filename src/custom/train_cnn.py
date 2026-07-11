"""
Custom (Ecem & Selman) dataset - 1D CNN over the 8 feature values (all 6 classes).

Improvements over the original: project-wide seed (22), EarlyStopping +
ReduceLROnPlateau with best-weight restoration, class weighting, and the
confusion matrix is saved to outputs/custom/.

Run:  python src/custom/train_cnn.py
"""
import sys
import pathlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

df = pd.read_csv(DATA_DIR / "custom" / "custom_dataset.csv")
X = df.drop("Label", axis=1).values
y = df["Label"].values
target_names = [CUSTOM_NAMES[i] for i in range(6)]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# reshape to (samples, features, 1) for Conv1D
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
n_features = X_train.shape[1]

model = Sequential([
    Conv1D(32, 3, activation="relu", input_shape=(n_features, 1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(6, activation="softmax"),
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
print(f"\n1D CNN accuracy (6 classes): {acc * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=target_names,
                            zero_division=0))
plot_confusion_matrix(y_test, y_pred, "Custom 1D CNN (6 classes)", target_names,
                      out_file=output_path("custom", "CNN.png"))
