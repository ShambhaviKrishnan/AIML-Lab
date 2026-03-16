from google.colab import files
uploaded = files.upload()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Step 1: Load Dataset
# ----------------------------
df = pd.read_csv("StudentsPerformance (4).csv")  # use the exact uploaded file name

# Optional: see data and columns
print("Columns:", df.columns.tolist())
print(df.head())

# ----------------------------
# Step 2: Create Pass/Fail Column
# ----------------------------
df["Pass"] = ((df["math score"] + df["reading score"] + df["writing score"]) / 3 >= 50).astype(int)

# Features and Target
X = df[["math score", "reading score", "writing score"]]
y = df["Pass"]

# ----------------------------
# Step 3: Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# ----------------------------
# Step 4: Train Decision Tree
# ----------------------------
model = DecisionTreeClassifier(max_depth=3, random_state=1)
model.fit(X_train, y_train)

# ----------------------------
# Step 5: Make Predictions
# ----------------------------
y_pred = model.predict(X_test)

# ----------------------------
# Step 6: Accuracy
# ----------------------------
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# ----------------------------
# Step 7: Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(6,5))  # bigger figure
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Fail","Pass"],
            yticklabels=["Fail","Pass"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ----------------------------
# Step 8: Decision Tree Visualization
# ----------------------------
plt.figure(figsize=(12,8))  # wider and taller for clarity
plot_tree(model,
          feature_names=X.columns,
          class_names=["Fail","Pass"],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree")
plt.show()

# Optional: Save Decision Tree as PNG for your report
plt.savefig("DecisionTree.png", dpi=300, bbox_inches='tight')

# ----------------------------
# Step 9: Precision, Recall, F1-score
# ----------------------------
print("\nPrecision, Recall, F1-score:")
print(classification_report(y_test, y_pred))

# ----------------------------
# Step 10: Overfitting vs Underfitting
# ----------------------------
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("\nTraining Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

if train_acc > test_acc:
    print("Model may be Overfitting")
elif train_acc < test_acc:
    print("Model may be Underfitting")
else:
    print("Model is Balanced")
