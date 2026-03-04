import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__solver": ["lbfgs"]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Evaluate best model
y_pred = best_model.predict(X_test)
print("Best params:", grid.best_params_)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# Save the tuned pipeline
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Load and predict
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

sample = [[5.1, 3.5, 1.4, 0.2]]
pred = loaded_model.predict(sample)[0]
print("Prediction index:", pred)
print("Prediction label:", data.target_names[pred])