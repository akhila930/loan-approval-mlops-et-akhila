"""
Loan Approval Prediction - Model Training with MLflow

This module:
1. Loads the raw loan dataset
2. Splits the data into training and testing sets
3. Builds preprocessing pipeline
4. Trains Logistic Regression and Random Forest
5. Evaluates both models
6. Tracks parameters, metrics and models using MLflow
7. Saves trained models and preprocessor locally
8. Selects the best model based on F1-score
"""

import os
import joblib
import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/raw/loan_data.csv"
MODEL_DIR = "models"

MLFLOW_DB = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Loan Approval Prediction"

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_COLUMN = "Loan_Status"

NUMERICAL_FEATURES = [
    "Dependents",
    "Applicant_Income",
    "Coapplicant_Income",
    "Loan_Amount",
    "Loan_Term",
    "Credit_History",
    "Age",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Education",
    "Employment_Status",
    "Property_Area",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

def create_preprocessor():
    """
    Creates the preprocessing pipeline.

    Numerical features:
        - Missing values -> median
        - Standardization

    Categorical features:
        - Missing values -> most frequent
        - One-hot encoding
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained classification model.

    Returns:
        Dictionary containing evaluation metrics.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    return metrics


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(model_name, metrics):
    """
    Prints model evaluation metrics.
    """

    print(f"\n{model_name}:")

    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")


# ============================================================
# TRAIN AND TRACK MODEL
# ============================================================

def train_and_track_model(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    model_params,
):
    """
    Trains one model and tracks it using MLflow.

    Returns:
        trained model
        metrics
    """

    print(f"\nTraining {model_name}...")

    # --------------------------------------------------------
    # Fit model
    # --------------------------------------------------------

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Evaluate model
    # --------------------------------------------------------

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print_metrics(
        model_name,
        metrics,
    )

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=model_name
    ):

        # ----------------------------------------------------
        # Log model parameters
        # ----------------------------------------------------

        params = {
            "model_type": model_name,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
            "num_features": X_train.shape[1],
        }

        params.update(model_params)

        mlflow.log_params(params)

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metrics(metrics)

        # ----------------------------------------------------
        # Log model
        # ----------------------------------------------------
        #
        # MLflow 3.x uses skops serialization by default.
        # Random Forest contains sklearn.tree._tree.Tree,
        # which must explicitly be trusted when logging.
        #
        # We only trust this specific type because the model
        # was trained locally by us from our own source code.
        # ----------------------------------------------------

        if model_name == "Random Forest":

            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ],
            )

        else:

            mlflow.sklearn.log_model(
                model,
                name="model",
            )

        # ----------------------------------------------------
        # Log preprocessor as an artifact
        # ----------------------------------------------------

        preprocessor_path = os.path.join(
            MODEL_DIR,
            "preprocessor.joblib",
        )

        joblib.dump(
            preprocessor,
            preprocessor_path,
        )

        mlflow.log_artifact(
            preprocessor_path,
            artifact_path="preprocessor",
        )

        print(
            f"✓ MLflow tracking completed for {model_name}"
        )

    return model, metrics


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_models():

    print("=" * 60)
    print("LOAN APPROVAL MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(MLFLOW_DB)

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    # --------------------------------------------------------
    # Load raw dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print(
        f"✓ Loaded raw data: {df.shape}"
    )

    # --------------------------------------------------------
    # Prepare target
    # --------------------------------------------------------

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .map(
            {
                "Approved": 1,
                "Rejected": 0,
            }
        )
    )

    # Check target conversion
    if df[TARGET_COLUMN].isnull().any():

        raise ValueError(
            "Target column contains invalid values after mapping."
        )

    # --------------------------------------------------------
    # Prepare features and target
    # --------------------------------------------------------

    X = df[FEATURE_COLUMNS].copy()

    y = df[TARGET_COLUMN].copy()

    print("\nTarget distribution:")
    print(y.value_counts())

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"\n✓ Training samples: {len(X_train)}"
    )

    print(
        f"✓ Testing samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Create preprocessing pipeline
    # --------------------------------------------------------

    preprocessor = create_preprocessor()

    # --------------------------------------------------------
    # FIT PREPROCESSOR ONLY ON TRAINING DATA
    # --------------------------------------------------------
    #
    # This prevents data leakage.
    #
    # The test dataset is transformed using the preprocessing
    # learned only from the training dataset.
    # --------------------------------------------------------

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    print(
        "✓ Preprocessing completed"
    )

    print(
        f"✓ Training features: {X_train_processed.shape[1]}"
    )

    # --------------------------------------------------------
    # Save fitted preprocessor
    # --------------------------------------------------------

    preprocessor_path = os.path.join(
        MODEL_DIR,
        "preprocessor.joblib",
    )

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    logistic_regression = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    logistic_model, logistic_metrics = train_and_track_model(
        model_name="Logistic Regression",
        model=logistic_regression,
        X_train=X_train_processed,
        X_test=X_test_processed,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        model_params={
            "max_iter": 1000,
        },
    )

    # --------------------------------------------------------
    # Save Logistic Regression model
    # --------------------------------------------------------

    logistic_model_path = os.path.join(
        MODEL_DIR,
        "logistic_regression.joblib",
    )

    joblib.dump(
        logistic_model,
        logistic_model_path,
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    random_forest_model, random_forest_metrics = (
        train_and_track_model(
            model_name="Random Forest",
            model=random_forest,
            X_train=X_train_processed,
            X_test=X_test_processed,
            y_train=y_train,
            y_test=y_test,
            preprocessor=preprocessor,
            model_params={
                "n_estimators": 200,
                "max_depth": 10,
            },
        )
    )

    # --------------------------------------------------------
    # Save Random Forest model
    # --------------------------------------------------------

    random_forest_model_path = os.path.join(
        MODEL_DIR,
        "random_forest.joblib",
    )

    joblib.dump(
        random_forest_model,
        random_forest_model_path,
    )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    if (
        logistic_metrics["f1_score"]
        >= random_forest_metrics["f1_score"]
    ):

        best_model = logistic_model

        best_model_name = "Logistic Regression"

        best_model_metrics = logistic_metrics

    else:

        best_model = random_forest_model

        best_model_name = "Random Forest"

        best_model_metrics = random_forest_metrics

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    best_model_path = os.path.join(
        MODEL_DIR,
        "loan_approval_model.joblib",
    )

    joblib.dump(
        best_model,
        best_model_path,
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 60)

    print(
        f"BEST MODEL: {best_model_name}"
    )

    print("=" * 60)

    print("\nBest model metrics:")

    for metric_name, metric_value in (
        best_model_metrics.items()
    ):

        print(
            f"  {metric_name}: {metric_value:.4f}"
        )

    print("\nSaved models:")

    print(
        f"  ✓ {logistic_model_path}"
    )

    print(
        f"  ✓ {random_forest_model_path}"
    )

    print(
        f"  ✓ {best_model_path}"
    )

    print(
        f"  ✓ {preprocessor_path}"
    )

    print("\nMLflow experiment:")

    print(
        f"  ✓ {EXPERIMENT_NAME}"
    )

    print("\nTraining completed successfully.")

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    train_models()