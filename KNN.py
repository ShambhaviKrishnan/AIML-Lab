import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. DATA PREPROCESSING
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. HYPERPARAMETER TUNING (KNN)
param_grid = {'n_neighbors': np.arange(1, 21)}
knn_gscv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
knn_gscv.fit(X_train_scaled, y_train)
best_k = knn_gscv.best_params_['n_neighbors']

# 3. TRAIN MODELS & EVALUATE
knn = KNeighborsClassifier(n_neighbors=best_k).fit(X_train_scaled, y_train)
nb = GaussianNB().fit(X_train_scaled, y_train)

knn_preds = knn.predict(X_test_scaled)
nb_preds = nb.predict(X_test_scaled)

# 4. VISUAL REPRESENTATION
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: KNN Tuning Curve
k_values = np.arange(1, 21)
mean_scores = knn_gscv.cv_results_['mean_test_score']
axes[0, 0].plot(k_values, mean_scores, marker='o', linestyle='--', color='teal')
axes[0, 0].axvline(best_k, color='orange', linestyle=':', label=f'Best K={best_k}')
axes[0, 0].set_title('KNN Tuning: Accuracy vs K-Value')
axes[0, 0].set_xlabel('K Value')
axes[0, 0].set_ylabel('CV Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Model Accuracy Comparison
acc_knn = accuracy_score(y_test, knn_preds)
acc_nb = accuracy_score(y_test, nb_preds)
axes[0, 1].bar(['KNN', 'Naive Bayes'], [acc_knn, acc_nb], color=['#5dade2', '#ec7063'], width=0.6)
axes[0, 1].set_title('Final Test Accuracy Comparison')
axes[0, 1].set_ylim(0.8, 1.05)
for i, v in enumerate([acc_knn, acc_nb]):
    axes[0, 1].text(i, v + 0.01, f"{v:.2%}", ha='center', fontweight='bold')

# Plot 3: KNN Confusion Matrix
sns.heatmap(confusion_matrix(y_test, knn_preds), annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=iris.target_names, yticklabels=iris.target_names)
axes[1, 0].set_title(f'KNN (k={best_k}) Confusion Matrix')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

# Plot 4: Naive Bayes Confusion Matrix
sns.heatmap(confusion_matrix(y_test, nb_preds), annot=True, fmt='d', cmap='Reds', ax=axes[1, 1],
            xticklabels=iris.target_names, yticklabels=iris.target_names)
axes[1, 1].set_title('Naive Bayes Confusion Matrix')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# 5. PRINT METRICS
print(f"Optimal K for KNN: {best_k}")
print("\n--- KNN Classification Report ---")
print(classification_report(y_test, knn_preds, target_names=iris.target_names))
print("\n--- Naive Bayes Classification Report ---")
print(classification_report(y_test, nb_preds, target_names=iris.target_names))
