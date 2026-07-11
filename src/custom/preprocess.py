"""
Custom (Ecem & Selman) 2-channel dataset preprocessing.

Reads the raw .mat trials from data/custom/, detects the muscle-activation
onset on channel 1, extracts 150 ms sliding windows and computes the four
time-domain features per channel (8 features/window). Writes:

  data/custom/custom_dataset.csv

Run:  python src/custom/preprocess.py
"""
import sys
import pathlib

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from emg_common import DATA_DIR, compute_td_features

CUSTOM_DIR = DATA_DIR / "custom"


def find_onset(sig, window_size, threshold):
    """Index of the first window whose mean rectified amplitude exceeds threshold."""
    for start in range(len(sig) - window_size + 1):
        if np.mean(np.abs(sig[start:start + window_size])) > threshold:
            return start
    return None


def process_mat_file(path, labels, params, rows):
    mat = sio.loadmat(path)
    for label_idx, label_name in enumerate(labels):
        k1, k2 = f"{label_name}_ch1", f"{label_name}_ch2"
        if k1 not in mat or k2 not in mat:
            continue
        ch1, ch2 = mat[k1], mat[k2]
        for r in range(ch1.shape[0]):
            sig1, sig2 = ch1[r, :], ch2[r, :]
            onset = find_onset(sig1, params["iemg_window_size"],
                               params["iemg_threshold"])
            if onset is None:
                continue
            pos = onset
            while pos + params["window_size"] <= params["total_samples"]:
                w1 = sig1[pos:pos + params["window_size"]]
                w2 = sig2[pos:pos + params["window_size"]]
                rows.append([*compute_td_features(w1),
                             *compute_td_features(w2), label_idx])
                pos += params["step_size"]


def main():
    params = {
        "fs": 1000,
        "total_samples": 6000,
        "iemg_window_size": int(0.04 * 1000),
        "iemg_threshold": 435,
        "window_size": int(0.150 * 1000),
        "overlap_size": int(0.12 * 1000),
    }
    params["step_size"] = params["window_size"] - params["overlap_size"]

    labels = ["point", "tip", "rock", "closed", "cyl", "open"]
    files = ["ecem_data_1.1.mat", "ecem_data_1.2.mat",
             "selman_data_1.mat", "selman_data_2.mat"]

    rows = []
    for fname in files:
        path = CUSTOM_DIR / fname
        if not path.exists():
            print(f"WARNING: '{path}' missing, skipping.")
            continue
        print(f"Reading: {fname}")
        process_mat_file(path, labels, params, rows)

    if not rows:
        print("No windows extracted. Check the source files in data/custom/.")
        return

    columns = ["RMS_ch1", "MAV_ch1", "WL_ch1", "SSC_ch1",
               "RMS_ch2", "MAV_ch2", "WL_ch2", "SSC_ch2", "Label"]
    df = pd.DataFrame(rows, columns=columns)
    out_path = CUSTOM_DIR / "custom_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDone. {len(df)} windows saved -> {out_path.name}")


if __name__ == "__main__":
    main()
