"""
NinaPro DB5 - within-subject validation (Subjects 2 & 5).

Random 80/20 split over the combined S2+S5 feature table. Trains SVM and
Random Forest and saves confusion matrices under outputs/ninapro/.

Run:  python src/ninapro/within_subject.py
"""
import sys
import pathlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from emg_common import (set_seed, SEED, DATA_DIR, output_path,
                        evaluate_model, print_summary,
                        NINAPRO_NAMES, names_for_labels)

set_seed()

csv_path = DATA_DIR / "ninapro" / "emg_features_7moves_8ch.csv"
try:
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {df.shape} from {csv_path.name}")
except FileNotFoundError:
    print(f"ERROR: '{csv_path}' not found.")
    sys.exit(1)

X = df.drop("Label", axis=1)
y = df["Label"]
labels = np.sort(y.unique())
target_names = names_for_labels(NINAPRO_NAMES, labels)
print(f"Detected classes: {target_names}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

results = {}

svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED)
svm.fit(X_train, y_train)
results["SVM (RBF)"] = evaluate_model(
    svm, X_test, y_test, "SVM", target_names, labels,
    output_path("ninapro", "SVM_2sbj.png"))

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
rf.fit(X_train, y_train)
results["Random Forest"] = evaluate_model(
    rf, X_test, y_test, "Random Forest", target_names, labels,
    output_path("ninapro", "RF_2sbj.png"))

print_summary(results)
