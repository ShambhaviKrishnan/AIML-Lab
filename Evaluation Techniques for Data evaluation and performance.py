import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. LOAD DATASET
wine = load_wine()
X, y = wine.data, wine.target

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. STANDARDIZATION & NORMALIZATION
# Standardization: Mean=0, StdDev=1 (Best for SVM, Logistic Regression)
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)

# Normalization: Range [0, 1] (Best for KNN, Neural Networks)
norm_scaler = MinMaxScaler()
X_train_norm = norm_scaler.fit_transform(X_train)
X_test_norm = norm_scaler.transform(X_test)

# 3. EVALUATION (Using KNN - sensitive to scaling)
def evaluate_model(X_tr, X_te, title):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr, y_train)
    y_pred = knn.predict(X_te)

    print(f"--- {title} ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.2f}")
    print(f"Recall:    {recall_score(y_test, y_pred, average='macro'):.2f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred, average='macro'):.2f}\n")

# Compare Results
evaluate_model(X_train, X_test, "Original (No Scaling)")
evaluate_model(X_train_std, X_test_std, "After Standardization")
evaluate_model(X_train_norm, X_test_norm, "After Normalization")
