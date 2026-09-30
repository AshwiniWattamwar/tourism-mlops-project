
import pandas as pd
import joblib
import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load Train and Test Data

X_train = pd.read_csv("Xtrain.csv")
X_test = pd.read_csv("Xtest.csv")

y_train = pd.read_csv("ytrain.csv")
y_test = pd.read_csv("ytest.csv")

print("Train and Test Data Loaded Successfully")

# Identify Numerical and Categorical Columns

num_cols = X_train.select_dtypes(
    exclude="object"
).columns

cat_cols = X_train.select_dtypes(
    include="object"
).columns

# Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            num_cols
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            cat_cols
        )
    ]
)

# Model

rf = RandomForestClassifier(
    random_state=42
)

# Pipeline

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf)
    ]
)

# Hyperparameter Tuning

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="accuracy"
)

grid.fit(
    X_train,
    y_train.values.ravel()
)

print("Model Training Completed")

# Prediction

y_pred = grid.predict(X_test)

# Accuracy

acc = accuracy_score(
    y_test,
    y_pred
)

# MLflow Logging

mlflow.set_experiment(
    "Tourism_Project"
)

with mlflow.start_run():

    mlflow.log_params(
        grid.best_params_
    )

    mlflow.log_metric(
        "accuracy",
        acc
    )

print("MLflow Logging Completed")

# Evaluation

print("\nBest Parameters:")
print(grid.best_params_)

print("\nAccuracy:")
print(acc)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# Save Best Model

joblib.dump(
    grid.best_estimator_,
    "tourism_project/deployment/best_model.pkl"
)

print("\nBest Model Saved Successfully")
