"""
NinaPro DB5 - classical models (SVM / Random Forest / MEET).

Trains and compares three classical classifiers on the 8-channel, 4-subject
time-domain feature table produced by preprocess.py. Confusion matrices are
saved under outputs/ninapro/.

Run:  python src/ninapro/train_classical.py
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from emg_common import (set_seed, SEED, DATA_DIR, output_path,
                        evaluate_model, print_summary,
                        NINAPRO_NAMES, names_for_labels)

set_seed()

# --------------------------------------------------------------------------- #
# 1. Load data
# --------------------------------------------------------------------------- #
csv_path = DATA_DIR / "ninapro" / "emg_features_7moves_8ch_4sbj.csv"
try:
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {df.shape} from {csv_path.name}")
except FileNotFoundError:
    print(f"ERROR: '{csv_path}' not found. Run preprocess.py first.")
    sys.exit(1)

X = df.drop("Label", axis=1)
y = df["Label"]

labels = np.sort(y.unique())
target_names = names_for_labels(NINAPRO_NAMES, labels)
print(f"Detected classes: {target_names}\n")

# --------------------------------------------------------------------------- #
# 2. Split + scale (scaler fit on TRAIN only to avoid data leakage)
# --------------------------------------------------------------------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------------------------------- #
# 3. Train & evaluate each model
# --------------------------------------------------------------------------- #
results = {}

svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED)
svm.fit(X_train, y_train)
results["SVM (RBF)"] = evaluate_model(
    svm, X_test, y_test, "SVM", target_names, labels,
    output_path("ninapro", "SVM_4sbj.png"))

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
rf.fit(X_train, y_train)
results["Random Forest"] = evaluate_model(
    rf, X_test, y_test, "Random Forest", target_names, labels,
    output_path("ninapro", "RF_4sbj.png"))

# MEET: One-vs-One ensemble of Extra-Trees "experts" (Xiong et al., 2024).
meet = OneVsOneClassifier(
    ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
    n_jobs=-1)
meet.fit(X_train, y_train)
results["MEET (ExtraTrees+OvO)"] = evaluate_model(
    meet, X_test, y_test, "MEET (Extra Trees + OvO)", target_names, labels,
    output_path("ninapro", "MEET_4sbj.png"))

# k-Nearest Neighbours: simple distance-based baseline.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
results["KNN (k=5)"] = evaluate_model(
    knn, X_test, y_test, "KNN (k=5)", target_names, labels,
    output_path("ninapro", "KNN_4sbj.png"))

# Multi-Layer Perceptron: a small feed-forward neural network.
mlp = MLPClassifier(hidden_layer_sizes=(128,), activation="relu",
                    max_iter=300, random_state=SEED)
mlp.fit(X_train, y_train)
results["MLP (1x128)"] = evaluate_model(
    mlp, X_test, y_test, "MLP", target_names, labels,
    output_path("ninapro", "MLP_4sbj.png"))

print_summary(results)
