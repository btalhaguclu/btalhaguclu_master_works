"""
Custom (Ecem & Selman) dataset - classical models WITHOUT the 'cylindrical' class.

The cylindrical gesture (label 4) is the main source of confusion in this
2-channel dataset. This variant drops it and re-runs SVM / RF / MEET on the
remaining 5 gestures. Confusion matrices are saved under outputs/custom/.

Run:  python src/custom/train_classical_wo_cyl.py
"""
import sys
import pathlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.multiclass import OneVsOneClassifier

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from emg_common import (set_seed, SEED, DATA_DIR, output_path,
                        evaluate_model, print_summary,
                        CUSTOM_NAMES, names_for_labels)

set_seed()

CYL_LABEL = 4

csv_path = DATA_DIR / "custom" / "custom_dataset.csv"
df = pd.read_csv(csv_path)
df = df[df["Label"] != CYL_LABEL]
print(f"Dataset loaded (cylindrical removed): {df.shape}")

X = df.drop("Label", axis=1)
y = df["Label"]
labels = np.sort(y.unique())
target_names = names_for_labels(CUSTOM_NAMES, labels)
print(f"Classes: {target_names}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

results = {}

svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED)
svm.fit(X_train, y_train)
results["SVM (RBF)"] = evaluate_model(
    svm, X_test, y_test, "SVM (no cyl)", target_names, labels,
    output_path("custom", "svm_wo_cyl.png"))

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
rf.fit(X_train, y_train)
results["Random Forest"] = evaluate_model(
    rf, X_test, y_test, "Random Forest (no cyl)", target_names, labels,
    output_path("custom", "rf_wo_cyl.png"))

meet = OneVsOneClassifier(
    ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
    n_jobs=-1)
meet.fit(X_train, y_train)
results["MEET (ExtraTrees+OvO)"] = evaluate_model(
    meet, X_test, y_test, "MEET (no cyl)", target_names, labels,
    output_path("custom", "meet_wo_cyl.png"))

print_summary(results)
