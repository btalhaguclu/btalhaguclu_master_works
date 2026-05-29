import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_normalized_confusion_matrix(y_true, y_pred, model_name, labels, target_names):
    cm_normalized = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    
    plt.title(f'{model_name} - Normalized Confusion Matrix', fontsize=16)
    plt.xlabel('Tahmin Edilen (Predicted)', fontsize=12)
    plt.ylabel('Gerçek Sınıf (Actual)', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# 1. VERİYİ YÜKLEME
file_path = 'db5_data/emg_features_7moves_8ch.csv'
# file_path = 'db5_data/emg_features_7moves_8ch_4sbj.csv'

try:
    df = pd.read_csv(file_path)
    print(f"Veri seti başarıyla yüklendi. Boyut: {df.shape}")
except FileNotFoundError:
    print(f"Hata: '{file_path}' bulunamadı. Lütfen dosyanın aynı dizinde olduğundan emin olun.")
    exit()

# 2. ÖZNİTELİKLER (X) VE ETİKETLERİ (Y) AYIRMA
X = df.drop('Label', axis=1)
y = df['Label']

# --- NİNAPRO LİTERATÜRÜNE UYGUN HAREKET İSİMLERİ ---
all_target_names = {
    0: 'Thumb up (B1)', 
    1: 'Scissors (B2)', 
    2: 'Open (B5)', 
    3: 'Closed(B6)', 
    4: 'Point (B7)',
    5: 'Cylindrical (C5)',
    6: 'Pinch (C14)'
}

existing_labels = np.sort(df['Label'].unique())
current_target_names = [all_target_names[lbl] for lbl in existing_labels]

print(f"\nAlgılanan Sınıflar: {current_target_names}\n")

# 3. VERİYİ EĞİTİM VE TEST OLARAK BÖLME
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 4. ÖZELLİK ÖLÇEKLEME (FEATURE SCALING)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("="*50)
print("MODELLER EĞİTİLİYOR VE TEST EDİLİYOR")
print("="*50 + "\n")

# =========================================================
# MODEL 1: SVM
# =========================================================
print("--- 1. Destek Vektör Makineleri (SVM) ---")
svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)

svm_predictions = svm_model.predict(X_test_scaled)

svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Doğruluk Oranı (Accuracy): %{svm_accuracy * 100:.2f}\n")
print(classification_report(y_test, svm_predictions, labels=existing_labels, target_names=current_target_names))

plot_normalized_confusion_matrix(y_test, svm_predictions, "SVM", existing_labels, current_target_names)

# =========================================================
# MODEL 2: RANDOM FOREST
# =========================================================
print("\n--- 2. Rastgele Orman (Random Forest) ---")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf_model.fit(X_train_scaled, y_train)

rf_predictions = rf_model.predict(X_test_scaled)

rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Doğruluk Oranı (Accuracy): %{rf_accuracy * 100:.2f}\n")
print(classification_report(y_test, rf_predictions, labels=existing_labels, target_names=current_target_names))

plot_normalized_confusion_matrix(y_test, rf_predictions, "Random Forest", existing_labels, current_target_names)

# =========================================================
# MODEL 3: MEET YAKLAŞIMI (One-Vs-One Extra Trees)
# =========================================================
print("\n--- 3. MEET (Mixture of Experts Extra Trees) ---")
# Makaledeki gibi "ikili sınıflara (class of two) odaklanan uzmanlar" oluşturmak için OneVsOneClassifier
# ve her bir uzman için ExtraTreesClassifier kullanıyoruz.
meet_expert = ExtraTreesClassifier(n_estimators=100, max_depth=None, random_state=42)
meet_model = OneVsOneClassifier(meet_expert)

meet_model.fit(X_train_scaled, y_train)

meet_predictions = meet_model.predict(X_test_scaled)

meet_accuracy = accuracy_score(y_test, meet_predictions)
print(f"MEET Doğruluk Oranı (Accuracy): %{meet_accuracy * 100:.2f}\n")
print("MEET Sınıflandırma Raporu:")
print(classification_report(y_test, meet_predictions, labels=existing_labels, target_names=current_target_names))

plot_normalized_confusion_matrix(y_test, meet_predictions, "MEET (Extra Trees + OvO)", existing_labels, current_target_names)

# =========================================================
# SONUÇ KARŞILAŞTIRMASI
# =========================================================
print("\n" + "="*50)
print("ÖZET KARŞILAŞTIRMA")
print("="*50)
print(f"SVM Doğruluk:           %{svm_accuracy*100:.2f}")
print(f"Random Forest Doğruluk: %{rf_accuracy*100:.2f}")
print(f"MEET Doğruluk:          %{meet_accuracy*100:.2f}")

best_acc = max(svm_accuracy, rf_accuracy, meet_accuracy)
if best_acc == meet_accuracy:
    print("\nSonuç: Makaleden uyarlanan MEET modeli en yüksek performansı gösterdi!")
elif best_acc == rf_accuracy:
    print("\nSonuç: Random Forest en yüksek performansı gösterdi.")
else:
    print("\nSonuç: SVM en yüksek performansı gösterdi.")