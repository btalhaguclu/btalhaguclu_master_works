"""
NinaPro DB5 - cross-subject validation.

Trains on Subjects 2 & 5 and tests on the completely unseen Subjects 1 & 10,
measuring how well the models generalise to new people. The scaler is fit on
the training subjects only. Confusion matrices are saved under outputs/ninapro/.

Run:  python src/ninapro/cross_subject.py
"""
import sys
import pathlib

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from emg_common import (set_seed, SEED, DATA_DIR, output_path,
                        evaluate_model, print_summary,
                        NINAPRO_NAMES, names_for_labels)

set_seed()

ninapro_dir = DATA_DIR / "ninapro"
try:
    train_df = pd.read_csv(ninapro_dir / "emg_train_S2_S5.csv")
    test_df = pd.read_csv(ninapro_dir / "emg_test_S1_S10.csv")
    print(f"Train (S2, S5): {train_df.shape}   Test (S1, S10): {test_df.shape}")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

X_train = train_df.drop("Label", axis=1)
y_train = train_df["Label"]
X_test = test_df.drop("Label", axis=1)
y_test = test_df["Label"]

labels = np.sort(train_df["Label"].unique())
target_names = names_for_labels(NINAPRO_NAMES, labels)

# Fit scaler on TRAIN subjects only; the unseen subjects are only transformed.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

results = {}

svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED)
svm.fit(X_train, y_train)
results["SVM (RBF)"] = evaluate_model(
    svm, X_test, y_test, "SVM (cross-subject)", target_names, labels,
    output_path("ninapro", "SVM_cross_subject.png"))

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
rf.fit(X_train, y_train)
results["Random Forest"] = evaluate_model(
    rf, X_test, y_test, "Random Forest (cross-subject)", target_names, labels,
    output_path("ninapro", "RF_cross_subject.png"))

print_summary(results)
