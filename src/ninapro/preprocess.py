"""
NinaPro DB5 (8-channel) preprocessing.

Reads the raw .mat files from data/ninapro/, resamples each gesture repetition
from 200 Hz to 1000 Hz, splits it into 150 ms sliding windows, and produces:

  * emg_features_7moves_8ch_4sbj.csv  - 32 time-domain features/window (SVM/RF/MEET)
  * X_ninapro_raw.npy / y_ninapro_raw.npy  - raw windows (150 x 8) for the 1D CNN

All inputs and outputs live in data/ninapro/ (paths resolved via emg_common).

Run:  python src/ninapro/preprocess.py
"""
import sys
import pathlib

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from emg_common import DATA_DIR, compute_td_features, feature_columns

NINA_DIR = DATA_DIR / "ninapro"
WINDOW_SIZE = 150   # samples @ 1000 Hz  = 150 ms
STEP_SIZE = 30      # 120 ms overlap
UPSAMPLE = 5        # 200 Hz -> 1000 Hz


def extract_windows_8ch(emg, stim, rep, target_mov):
    """Return all 150x8 windows for a given movement id, upsampled to 1000 Hz."""
    num_reps = int(np.max(rep))
    windows = []
    for r in range(1, num_reps + 1):
        idx = np.where((stim == target_mov) & (rep == r))[0]
        if len(idx) == 0:
            continue
        sig = emg[idx, :8]  # first 8 channels (Myo Armband)
        new_len = len(sig) * UPSAMPLE
        if new_len <= WINDOW_SIZE:
            continue
        sig_1000 = signal.resample(sig, new_len, axis=0)
        start = 0
        while start + WINDOW_SIZE <= len(sig_1000):
            windows.append(sig_1000[start:start + WINDOW_SIZE, :])
            start += STEP_SIZE
    if windows:
        return np.stack(windows)  # (n_windows, 150, 8)
    return np.empty((0, WINDOW_SIZE, 8))


def main():
    print("=" * 50)
    print("NINAPRO DB5 (8-CHANNEL) PREPROCESSING")
    print("=" * 50)

    subjects = ["S1", "S2", "S5", "S10"]
    # (name, exercise file, movement id, label 0-6)
    movements = [
        ("B_1", "E2", 1, 0), ("B_2", "E2", 2, 1), ("B_5", "E2", 5, 2),
        ("B_6", "E2", 6, 3), ("B_7", "E2", 7, 4),
        ("C_5", "E3", 5, 5), ("C_14", "E3", 14, 6),
    ]

    feature_rows, X_raw, y_raw = [], [], []

    for subj in subjects:
        f_e2 = NINA_DIR / f"{subj}_E2_A1.mat"
        f_e3 = NINA_DIR / f"{subj}_E3_A1.mat"
        if not f_e2.exists() or not f_e3.exists():
            print(f"WARNING: .mat files for {subj} not found. Skipping.")
            continue
        print(f"\n--- Processing subject {subj} ---")
        data = {"E2": sio.loadmat(f_e2), "E3": sio.loadmat(f_e3)}

        for _, ex, mov_id, label in movements:
            emg = data[ex]["emg"]
            stim = data[ex]["restimulus"].flatten()
            rep = data[ex]["repetition"].flatten()

            windows = extract_windows_8ch(emg, stim, rep, mov_id)
            if windows.shape[0] == 0:
                continue

            X_raw.append(windows)
            y_raw.extend([label] * windows.shape[0])

            for w in windows:  # 32 time-domain features per window
                row = []
                for ch in range(8):
                    row.extend(compute_td_features(w[:, ch]))
                row.append(label)
                feature_rows.append(row)

    if not feature_rows:
        print("No data extracted! Check the .mat files in data/ninapro/.")
        sys.exit(1)

    # 1. Feature CSV (classical models)
    df = pd.DataFrame(feature_rows, columns=feature_columns(8))
    csv_path = NINA_DIR / "emg_features_7moves_8ch_4sbj.csv"
    df.to_csv(csv_path, index=False)

    # 2. Raw windows (CNN)
    X_np = np.vstack(X_raw)
    y_np = np.array(y_raw)
    np.save(NINA_DIR / "X_ninapro_raw.npy", X_np)
    np.save(NINA_DIR / "y_ninapro_raw.npy", y_np)

    print("\nDone.")
    print(f"  -> {csv_path.name}  {df.shape}  (SVM/RF/MEET)")
    print(f"  -> X_ninapro_raw.npy  {X_np.shape}  (1D CNN)")
    print(f"  -> y_ninapro_raw.npy  {y_np.shape}  (1D CNN)")


if __name__ == "__main__":
    main()
