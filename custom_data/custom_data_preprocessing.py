import os
import scipy.io as sio
import numpy as np
import pandas as pd


def find_onset(signal, window_size, threshold):
    for start in range(len(signal) - window_size + 1):
        window = signal[start:start + window_size]
        if np.mean(np.abs(window)) > threshold:
            return start
    return None


def compute_features(segment):
    rms = np.sqrt(np.mean(segment**2))
    mav = np.mean(np.abs(segment))
    wl = np.sum(np.abs(np.diff(segment)))
    ssc = np.sum((np.diff(segment[:-1]) * np.diff(segment[1:])) < 0)
    return rms, mav, wl, ssc


def process_mat_file(filename, labels, params, output_rows):
    mat = sio.loadmat(filename)
    for label_idx, label_name in enumerate(labels):
        ch1_key = f"{label_name}_ch1"
        ch2_key = f"{label_name}_ch2"
        if ch1_key not in mat or ch2_key not in mat:
            continue

        data_ch1 = mat[ch1_key]
        data_ch2 = mat[ch2_key]
        for row_idx in range(data_ch1.shape[0]):
            sig1 = data_ch1[row_idx, :]
            sig2 = data_ch2[row_idx, :]
            onset = find_onset(sig1, params["iemg_window_size"], params["iemg_threshold"])
            if onset is None:
                continue

            pos = onset
            while pos + params["window_size"] <= params["total_samples"]:
                win1 = sig1[pos:pos + params["window_size"]]
                win2 = sig2[pos:pos + params["window_size"]]

                rms_1, mav_1, wl_1, ssc_1 = compute_features(win1)
                rms_2, mav_2, wl_2, ssc_2 = compute_features(win2)

                output_rows.append([
                    rms_1, mav_1, wl_1, ssc_1,
                    rms_2, mav_2, wl_2, ssc_2,
                    label_idx,
                ])
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
    files = [
        "ecem_data_1.1.mat",
        "ecem_data_1.2.mat",
        "selman_data_1.mat",
        "selman_data_2.mat",
    ]

    rows = []
    for filename in files:
        if not os.path.exists(filename):
            print(f"Uyarı: '{filename}' yok, atlıyorum.")
            continue
        print(f"Dosya okunuyor: {filename}")
        process_mat_file(filename, labels, params, rows)

    if not rows:
        print("Hiç pencere çıkarılmadı. Kaynak dosyaları kontrol edin.")
        return

    columns = [
        "RMS_ch1", "MAV_ch1", "WL_ch1", "SSC_ch1",
        "RMS_ch2", "MAV_ch2", "WL_ch2", "SSC_ch2",
        "Label",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv("custom_dataset.csv", index=False)

    print(f"\nVeri işleme tamamlandı. Toplam {len(df)} pencere kaydedildi.")
    print("custom_dataset.csv oluşturuldu.")


if __name__ == "__main__":
    main()
