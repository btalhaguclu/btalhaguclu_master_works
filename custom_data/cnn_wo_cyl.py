import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def plot_normalized_confusion_matrix(y_true, y_pred, model_name, target_names):
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'{model_name} - Karmaşıklık Matrisi (5 Sınıf)', fontsize=14)
    plt.xlabel('Tahmin Edilen (Predicted)', fontsize=12)
    plt.ylabel('Gerçek Sınıf (Actual)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

print("="*50)
print("1D CNN İÇİN VERİ HAZIRLIĞI (5 SINIF - 'cyl' HARİÇ)")
print("="*50)

# 1. Veriyi Yükle ve 'cyl' sınıfını (Label 4) filtrele
df = pd.read_csv('custom_dataset.csv')
df_filtered = df[df['Label'] != 4]

X = df_filtered.drop('Label', axis=1).values
y = df_filtered['Label'].values

# 2. Etiketleri Keras için kodla (0, 1, 2, 3, 5 -> 0, 1, 2, 3, 4)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
target_names = ['point (0)', 'tip (1)', 'rock (2)', 'closed (3)', 'open (5)']

# 3. Eğitim ve Test Bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded)

# 4. Ölçeklendirme (Standartlaştırma)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Keras CNN için Yeniden Şekillendirme (Reshaping)
X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# =========================================================
# 1D CNN MİMARİSİNİN KURULUMU
# =========================================================
print("\n--- 1D CNN Modeli Eğitiliyor (5 Sınıf) ---")

model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(8, 1)),
    MaxPooling1D(pool_size=2),
    
    Flatten(),
    
    Dense(64, activation='relu'),
    Dropout(0.3), 
    
    # Çıktı Katmanı (Artık 5 sınıf olduğu için 5 nöron)
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Eğitimi Başlat
history = model.fit(X_train_cnn, y_train, 
                    epochs=30, 
                    batch_size=32, 
                    validation_split=0.1, 
                    verbose=1)

# =========================================================
# DEĞERLENDİRME VE SONUÇLAR
# =========================================================
print("\n--- 1D CNN Test Seti Performansı (5 Sınıf) ---")
y_pred_prob = model.predict(X_test_cnn)
y_pred = np.argmax(y_pred_prob, axis=1)

cnn_acc = accuracy_score(y_test, y_pred)
print(f"\n1D CNN Doğruluk Oranı (5 Sınıf): %{cnn_acc * 100:.2f}")
print(classification_report(y_test, y_pred, target_names=target_names))

plot_normalized_confusion_matrix(y_test, y_pred, "1D CNN (wo_cyl)", target_names)