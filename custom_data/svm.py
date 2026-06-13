import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


random_seed = 22 #Edirne

def plot_normalized_confusion_matrix(y_true, y_pred, model_name, target_names):
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'{model_name} - Karmaşıklık Matrisi ', fontsize=14)
    plt.xlabel('Tahmin Edilen (Predicted)', fontsize=12)
    plt.ylabel('Gerçek Sınıf (Actual)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

print("="*50)
print("VERİ YÜKLENİYOR VE MODELLER EĞİTİLİYOR")
print("="*50)

# CSV dosyasını oku
df = pd.read_csv('custom_dataset.csv')
print(f"Veri seti başarıyla yüklendi. Toplam satır: {df.shape[0]}")

X = df.drop('Label', axis=1)
y = df['Label']

target_names = ['point (0)', 'tip (1)', 'rock (2)', 'closed (3)', 'cyl (4)', 'open (5)']

# Ortak Havuz (Random Split) - %80 Eğitim, %20 Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_seed, stratify=y)

# Özellik Ölçekleme (StandardScaler) - SVM için hayati önem taşır!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================================
# MODEL 1: DESTEK VEKTÖR MAKİNELERİ (SVM)
# =========================================================
print("\n--- 1. Destek Vektör Makineleri (SVM) ---")
print("Lütfen bekleyin... (80.000+ satır için SVM eğitimi uzun sürebilir)")

# C=10 ve kernel='rbf' sEMG için genelde en iyi sonuçları verir
svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=random_seed)
svm_model.fit(X_train_scaled, y_train)

svm_preds = svm_model.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, svm_preds)
print(f"SVM Doğruluk Oranı: %{svm_acc * 100:.2f}")
print(classification_report(y_test, svm_preds, target_names=target_names))
plot_normalized_confusion_matrix(y_test, svm_preds, "SVM", target_names)


# =========================================================
# MODEL 2: RASTGELE ORMAN (RANDOM FOREST)
# =========================================================
print("\n--- 2. Rastgele Orman (Random Forest) ---")
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_seed)
rf_model.fit(X_train_scaled, y_train)

rf_preds = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"Random Forest Doğruluk Oranı: %{rf_acc * 100:.2f}")
print(classification_report(y_test, rf_preds, target_names=target_names))
plot_normalized_confusion_matrix(y_test, rf_preds, "Random Forest", target_names)


# =========================================================
# MODEL 3: MEET (Mixture of Experts Extra Trees)
# =========================================================
print("\n--- 3. MEET (Mixture of Experts Extra Trees) ---")
meet_expert = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=random_seed)
meet_model = OneVsOneClassifier(meet_expert, n_jobs=-1)

meet_model.fit(X_train_scaled, y_train)

meet_preds = meet_model.predict(X_test_scaled)
meet_acc = accuracy_score(y_test, meet_preds)
print(f"MEET Doğruluk Oranı: %{meet_acc * 100:.2f}")
print(classification_report(y_test, meet_preds, target_names=target_names))
plot_normalized_confusion_matrix(y_test, meet_preds, "MEET (Extra Trees + OvO)", target_names)


# =========================================================
# ÖZET
# =========================================================
print("\n" + "="*50)
print("ÖZET KARŞILAŞTIRMA TABLOSU")
print("="*50)
print(f"SVM Doğruluk:           %{svm_acc*100:.2f}")
print(f"Random Forest Doğruluk: %{rf_acc*100:.2f}")
print(f"MEET Doğruluk:          %{meet_acc*100:.2f}")