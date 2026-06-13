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

random_seed = 22


def plot_confusion_matrix(y_true, y_pred, title, labels, names):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=names, yticklabels=names)
    plt.title(title, fontsize=14)
    plt.xlabel('Tahmin', fontsize=12)
    plt.ylabel('Gerçek', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def load_data(path):
    df = pd.read_csv(path)
    print(f"Veri yüklendi: {df.shape[0]} satır bulundu.")
    return df


def train_model(model, name, X_train, X_test, y_train, y_test, labels, target_names):
    print(f"\n{name} eğitiliyor...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} doğruluk: %{acc * 100:.2f}")
    print(classification_report(y_test, preds, labels=labels, target_names=target_names))
    plot_confusion_matrix(y_test, preds, f"{name} - Karmaşıklık Matrisi", labels, target_names)
    return acc


def main():
    df = load_data('custom_dataset.csv')
    df = df[df['Label'] != 4].copy()
    print(f"'cyl' sınıfı çıkarıldı. Kalan satır: {df.shape[0]}")

    X = df.drop('Label', axis=1)
    y = df['Label']

    labels = np.sort(y.unique())
    target_names = {0: 'point (0)', 1: 'tip (1)', 2: 'rock (2)', 3: 'closed (3)', 5: 'open (5)'}
    target_names = [target_names[label] for label in labels]
    print(f"Kullanılacak sınıflar: {target_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = [
        ('SVM', SVC(kernel='rbf', C=10, gamma='scale', random_state=random_seed)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_seed)),
        ('MEET', OneVsOneClassifier(ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=random_seed), n_jobs=-1)),
    ]

    results = {}
    for name, model in models:
        results[name] = train_model(model, name, X_train, X_test, y_train, y_test, labels, target_names)

    print('\n' + '=' * 50)
    print('Sonuçlar')
    print('=' * 50)
    for name, score in results.items():
        print(f"{name}: %{score * 100:.2f}")


if __name__ == '__main__':
    main()
