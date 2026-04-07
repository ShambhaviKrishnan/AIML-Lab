from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load Dataset
data = load_digits()
X, y = data.data, data.target

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. AdaBoost
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)
ada_acc = accuracy_score(y_test, ada.predict(X_test))

# 4. Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))

# 5. Voting Classifier
dt = DecisionTreeClassifier()
lr = LogisticRegression(max_iter=1000)

voting_hard = VotingClassifier(
    estimators=[('dt', dt), ('lr', lr)],
    voting='hard'
)
voting_hard.fit(X_train, y_train)
hard_acc = accuracy_score(y_test, voting_hard.predict(X_test))

voting_soft = VotingClassifier(
    estimators=[('dt', dt), ('lr', lr)],
    voting='soft'
)
voting_soft.fit(X_train, y_train)
soft_acc = accuracy_score(y_test, voting_soft.predict(X_test))

# 6. Stacking
stack = StackingClassifier(
    estimators=[('dt', dt), ('lr', lr)],
    final_estimator=LogisticRegression()
)
stack.fit(X_train, y_train)
stack_acc = accuracy_score(y_test, stack.predict(X_test))

# 7. Results
print(f"AdaBoost Accuracy: {ada_acc:.4f}")
print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")
print(f"Hard Voting Accuracy: {hard_acc:.4f}")
print(f"Soft Voting Accuracy: {soft_acc:.4f}")
print(f"Stacking Accuracy: {stack_acc:.4f}")
