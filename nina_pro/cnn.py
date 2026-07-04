import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

def plot_normalized_confusion_matrix(y_true, y_pred, model_name, target_names):
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'{model_name} - Karmaşıklık Matrisi (NinaPro DB5 - 8 Kanal)', fontsize=14)
    plt.xlabel('Tahmin Edilen (Predicted)', fontsize=12)
    plt.ylabel('Gerçek Sınıf (Actual)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

print("="*50)
print("NINAPRO DB5 HAM VERİ (RAW) İÇİN 1D CNN")
print("="*50)

# =========================================================
# 1. VERİ YÜKLEME
# =========================================================
try:
    X_raw = np.load('X_ninapro_raw.npy') 
    y_raw = np.load('y_ninapro_raw.npy') 
    print(f"✅ NinaPro Ham Veri Yüklendi! Boyut: {X_raw.shape}")
except FileNotFoundError:
    print("❌ HATA: 'X_ninapro_raw.npy' dosyası bulunamadı.")
    print("Lütfen önce veri hazırlama betiğini (ninapro_prepare.py) çalıştırın.")
    exit()

# NinaPro için 7 Hedef Sınıf (Sıralama çıkarma scriptindekiyle aynıdır)
target_names = ['B1 (Thumb up)', 'B2 (Scissors)', 'B5 (Open)', 'B6 (Closed)', 'B7 (Point)', 'C5 (Cylindrical)', 'C14 (Pinch)']

le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)

# 2. Eğitim ve Test Bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded)

# =========================================================
# 3. ZAMAN SERİSİ (3D) İÇİN ÖLÇEKLENDİRME (KRİTİK ADIM)
# =========================================================
print("Veriler Standartlaştırılıyor...")
num_samples_train, time_steps, channels = X_train.shape  # channels = 8
num_samples_test = X_test.shape[0]

# Ölçeklendirme işlemi için geçici olarak 2D'ye düzleştirme
X_train_reshaped = X_train.reshape((num_samples_train, time_steps * channels))
X_test_reshaped = X_test.reshape((num_samples_test, time_steps * channels))

scaler = StandardScaler()
X_train_scaled_2d = scaler.fit_transform(X_train_reshaped)
X_test_scaled_2d = scaler.transform(X_test_reshaped)

# Yeniden 3D formata (150, 8) çevirme
X_train_scaled = X_train_scaled_2d.reshape((num_samples_train, time_steps, channels))
X_test_scaled = X_test_scaled_2d.reshape((num_samples_test, time_steps, channels))

print(f"Eğitim Seti Boyutu: {X_train_scaled.shape}")
print(f"Test Seti Boyutu: {X_test_scaled.shape}")

# =========================================================
# 4. 8 KANAL İÇİN GELİŞMİŞ 1D CNN MİMARİSİ
# =========================================================
print("\n--- 1D CNN Modeli Eğitiliyor ---")

model = Sequential([
    # 1. Evrişim Bloğu (Giriş: 150 zaman adımı, 8 kanal)
    Conv1D(filters=64, kernel_size=10, activation='relu', input_shape=(150, 8)),
    BatchNormalization(), 
    MaxPooling1D(pool_size=2),
    
    # 2. Evrişim Bloğu 
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    
    Flatten(),
    
    # NinaPro'nun yüksek boyutlu verisini işlemek için kapasite artırıldı
    Dense(256, activation='relu'),
    Dropout(0.5), 
    
    Dense(128, activation='relu'),
    Dropout(0.3),
    
    # ÇIKTI KATMANI: 7 SINIF
    Dense(7, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Eğitimi Başlat
history = model.fit(X_train_scaled, y_train, 
                    epochs=30, # NinaPro verisi hacimli olduğu için 30 epoch genellikle yeterlidir
                    batch_size=64, 
                    validation_split=0.1, 
                    verbose=1)

# =========================================================
# 5. DEĞERLENDİRME
# =========================================================
print("\n--- NinaPro Raw 1D CNN Test Seti Performansı ---")
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)

cnn_acc = accuracy_score(y_test, y_pred)
print(f"\nNinaPro Raw 1D CNN Doğruluk Oranı: %{cnn_acc * 100:.2f}")
print(classification_report(y_test, y_pred, target_names=target_names))

plot_normalized_confusion_matrix(y_test, y_pred, "NinaPro 1D CNN", target_names)