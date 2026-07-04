import os
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy import signal

def extract_trials_8ch(emg_data, stim_data, rep_data, target_mov):
    """
    Belirli bir hareketin tekrarlarını bulur, 200Hz'den 1000Hz'e yükseltir (resample)
    ve 150ms'lik pencerelere böler.
    """
    num_reps = int(np.max(rep_data))
    window_size = 150
    step_size = 30
    
    windows_list = []
    
    for rep in range(1, num_reps + 1):
        # O harekete ve tekrara ait indeksleri bul
        idx = np.where((stim_data == target_mov) & (rep_data == rep))[0]
        
        if len(idx) > 0:
            sig = emg_data[idx, :8] # Sadece ilk 8 kanalı (Myo Armband) al
            
            # NinaPro verisi 200 Hz'dir. Bunu 1000 Hz'e (5 katına) upsample ediyoruz.
            new_length = len(sig) * 5 
            if new_length > window_size:
                sig_1000 = signal.resample(sig, new_length, axis=0)
                
                # Pencereleme işlemi
                start_idx = 0
                while (start_idx + window_size) <= len(sig_1000):
                    end_idx = start_idx + window_size
                    windows_list.append(sig_1000[start_idx:end_idx, :])
                    start_idx += step_size
                    
    if len(windows_list) > 0:
        return np.stack(windows_list) # Boyut: (Pencere_Sayısı, 150, 8)
    else:
        return np.empty((0, window_size, 8))

print("="*50)
print("NINAPRO DB5 (8 KANAL) PYTHON VERİ İŞLEYİCİ")
print("="*50)

subjects = ['S1', 'S2', 'S5', 'S10']

# [Hareket_Adı, Dosya_Tipi, Mov_ID, Label(0-6)]
features_info = [
    ('B_1',  'E2', 1,  0),
    ('B_2',  'E2', 2,  1),
    ('B_5',  'E2', 5,  2),
    ('B_6',  'E2', 6,  3),
    ('B_7',  'E2', 7,  4),
    ('C_5',  'E3', 5,  5),
    ('C_14', 'E3', 14, 6)
]

final_dataset = []
X_raw_all = []
y_raw_all = []

for subj in subjects:
    print(f"\n--- İşlenen Denek: {subj} ---")
    
    file_E2 = f"{subj}_E2_A1.mat"
    file_E3 = f"{subj}_E3_A1.mat"
    
    if not os.path.exists(file_E2) or not os.path.exists(file_E3):
        print(f"UYARI: {subj} için .mat dosyaları bulunamadı. Atlanıyor...")
        continue
        
    try:
        data_E2 = sio.loadmat(file_E2)
        data_E3 = sio.loadmat(file_E3)
    except Exception as e:
        print(f"Dosya okuma hatası: {e}")
        continue
        
    for feat_name, file_type, mov_id, label in features_info:
        # Doğru dosyadan verileri çekiyoruz (Sinyalleri 1D diziye düzleştiriyoruz)
        if file_type == 'E2':
            emg = data_E2['emg']
            stim = data_E2['restimulus'].flatten()
            rep = data_E2['repetition'].flatten()
        else:
            emg = data_E3['emg']
            stim = data_E3['restimulus'].flatten()
            rep = data_E3['repetition'].flatten()
            
        # Pencereleri Çıkar
        windows_8ch = extract_trials_8ch(emg, stim, rep, mov_id)
        
        num_windows = windows_8ch.shape[0]
        if num_windows == 0:
            continue
            
        # 1. CNN İÇİN HAM VERİLERİ (RAW) BİRİKTİR
        X_raw_all.append(windows_8ch)
        y_raw_all.extend([label] * num_windows)
        
        # 2. RF/SVM İÇİN ZAMAN UZAYI ÖZNİTELİKLERİNİ ÇIKAR
        for i in range(num_windows):
            window = windows_8ch[i]
            row_features = []
            
            for ch in range(8):
                ch_data = window[:, ch]
                rms_val = np.sqrt(np.mean(ch_data**2))
                mav_val = np.mean(np.abs(ch_data))
                wl_val = np.sum(np.abs(np.diff(ch_data)))
                diff_ch = np.diff(ch_data)
                ssc_val = np.sum((diff_ch[:-1] * diff_ch[1:]) < 0)
                
                row_features.extend([rms_val, mav_val, wl_val, ssc_val])
                
            row_features.append(label)
            final_dataset.append(row_features)

# =========================================================
# KAYDETME İŞLEMLERİ
# =========================================================

# 1. CSV Olarak Kaydet (Geleneksel Modeller İçin)
col_names = []
for ch in range(1, 9):
    col_names.extend([f"RMS_ch{ch}", f"MAV_ch{ch}", f"WL_ch{ch}", f"SSC_ch{ch}"])
col_names.append("Label")

df = pd.DataFrame(final_dataset, columns=col_names)
csv_filename = "emg_features_7moves_8ch_4sbj.csv"
df.to_csv(csv_filename, index=False)

# 2. NPY Olarak Kaydet (1D CNN İçin)
if len(X_raw_all) > 0:
    X_raw_numpy = np.vstack(X_raw_all)
    y_raw_numpy = np.array(y_raw_all)
    
    np.save("X_ninapro_raw.npy", X_raw_numpy)
    np.save("y_ninapro_raw.npy", y_raw_numpy)
else:
    print("Hiç veri çıkarılamadı!")
    exit()

print("\n✅ İşlem başarıyla tamamlandı!")
print(f"-> {csv_filename} oluşturuldu (RF/SVM için).")
print(f"-> X_ninapro_raw.npy oluşturuldu {X_raw_numpy.shape} (1D CNN için).")
print(f"-> y_ninapro_raw.npy oluşturuldu {y_raw_numpy.shape} (1D CNN için).")