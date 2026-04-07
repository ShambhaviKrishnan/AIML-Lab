import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load Data (64 dimensions - 8x8 image pixels)
digits = load_digits()
X, y = digits.data, digits.target
X_std = StandardScaler().fit_transform(X)

# 2. Performance Comparison (Before vs After)
def get_accuracy(data, target):
    xt, xv, yt, yv = train_test_split(data, target, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(xt, yt)
    return accuracy_score(yv, model.predict(xv))

acc_orig = get_accuracy(X_std, y)
pca_test = PCA(n_components=10) # Reduce 64D to 10D
acc_pca = get_accuracy(pca_test.fit_transform(X_std), y)

print(f"Accuracy with 64 features: {acc_orig:.4f}")
print(f"Accuracy with 10 PCA components: {acc_pca:.4f}")

# 3. Apply PCA for Visualization
pca_2d = PCA(n_components=2).fit_transform(X_std)
pca_3d = PCA(n_components=3).fit_transform(X_std)

# 4. 2D Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=y, cmap='jet', s=10, alpha=0.7)
plt.title("2D PCA Projection")
plt.colorbar(label='Digit Value')

# 5. 3D Visualization
ax = plt.subplot(1, 2, 2, projection='3d')
sc = ax.scatter(pca_3d[:, 0], pca_3d[:, 1], pca_3d[:, 2], c=y, cmap='jet', s=10)
ax.set_title("3D PCA Projection")
plt.show()
