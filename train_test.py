import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_normalized_confusion_matrix(y_true, y_pred, model_name, labels, target_names):
    cm_normalized = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    
    plt.title(f'{model_name} - Cross-Subject Validation Matrix', fontsize=16)
    plt.xlabel('Tahmin Edilen (S1 & S10)', fontsize=12)
    plt.ylabel('Gerçek Sınıf (S1 & S10)', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# 1. BAĞIMSIZ EĞİTİM VE TEST DOSYALARINI YÜKLEME
try:
    train_df = pd.read_csv('db5_data/emg_train_S2_S5.csv')
    test_df = pd.read_csv('db5_data/emg_test_S1_S10.csv')
    print(f"Eğitim Verisi (S2, S5) Boyutu: {train_df.shape}")
    print(f"Test Verisi (S1, S10) Boyutu: {test_df.shape}")
except FileNotFoundError as e:
    print(f"Dosya okuma hatası: {e}")
    exit()

# 2. X VE Y OLARAK AYIRMA
X_train = train_df.drop('Label', axis=1)
y_train = train_df['Label']

X_test = test_df.drop('Label', axis=1)
y_test = test_df['Label']


random_seed = 22 #Edirne

# --- HAREKET İSİMLERİ ---
all_target_names = {
    0: 'Thumb up (B1)', 
    1: 'Scissors (B2)', 
    2: 'Open (B5)', 
    3: 'Closed(B6)', 
    4: 'Point (B7)',
    5: 'Cylindrical (C5)',
    6: 'Pinch (C14)'
}

# Modellerin hangi etiketlerle çalıştığını dinamik tespit etme
existing_labels = np.sort(train_df['Label'].unique())
current_target_names = [all_target_names[lbl] for lbl in existing_labels]

# 3. ÖZELLİK ÖLÇEKLEME (ÇOK ÖNEMLİ: Sadece Train üzerinde 'fit' edilir)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Yeni kişilerin verisi sadece dönüştürülür

print("\n" + "="*50)
print("BAĞIMSIZ DENEK MODELLERİ EĞİTİLİYOR...")
print("="*50 + "\n")

# =========================================================
# MODEL 1: SVM
# =========================================================
print("--- 1. Destek Vektör Makineleri (SVM) ---")
svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=random_seed)
svm_model.fit(X_train_scaled, y_train)

svm_predictions = svm_model.predict(X_test_scaled)

svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"Bağımsız SVM Doğruluk Oranı (Accuracy): %{svm_accuracy * 100:.2f}\n")
print(classification_report(y_test, svm_predictions, labels=existing_labels, target_names=current_target_names))

plot_normalized_confusion_matrix(y_test, svm_predictions, "SVM", existing_labels, current_target_names)

# =========================================================
# MODEL 2: RANDOM FOREST
# =========================================================
print("\n--- 2. Rastgele Orman (Random Forest) ---")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=random_seed)
rf_model.fit(X_train_scaled, y_train)

rf_predictions = rf_model.predict(X_test_scaled)

rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Bağımsız Random Forest Doğruluk Oranı (Accuracy): %{rf_accuracy * 100:.2f}\n")
print(classification_report(y_test, rf_predictions, labels=existing_labels, target_names=current_target_names))

plot_normalized_confusion_matrix(y_test, rf_predictions, "Random Forest", existing_labels, current_target_names)