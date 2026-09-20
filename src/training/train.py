import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROCESSED_DATA_PATH = Path(
    "data/processed/processed_data.csv"
)

MODEL_DIR = Path("models")

LOGISTIC_MODEL_PATH = (
    MODEL_DIR / "logistic_regression.joblib"
)

RANDOM_FOREST_MODEL_PATH = (
    MODEL_DIR / "random_forest.joblib"
)

BEST_MODEL_PATH = (
    MODEL_DIR / "loan_approval_model.joblib"
)


TARGET_COLUMN = "Loan_Status"


# ---------------------------------------------------------
# Load processed data
# ---------------------------------------------------------

def load_data():

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: "
            f"{PROCESSED_DATA_PATH}"
        )

    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        ),
    }

    return metrics


# ---------------------------------------------------------
# Train models
# ---------------------------------------------------------

def train_models():

    print("=" * 60)
    print("LOAN APPROVAL MODEL TRAINING")
    print("=" * 60)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    X, y = load_data()

    print(f"✓ Loaded processed data: {X.shape}")
    print(f"✓ Target distribution:")
    print(y.value_counts())

    # -----------------------------------------------------
    # Train/test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(
        f"✓ Training samples: {len(X_train)}"
    )

    print(
        f"✓ Testing samples: {len(X_test)}"
    )

    # -----------------------------------------------------
    # Create models
    # -----------------------------------------------------

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    # -----------------------------------------------------
    # Train Logistic Regression
    # -----------------------------------------------------

    print("\nTraining Logistic Regression...")

    logistic_model.fit(
        X_train,
        y_train
    )

    logistic_metrics = evaluate_model(
        logistic_model,
        X_test,
        y_test
    )

    print("\nLogistic Regression:")
    
    for metric, value in logistic_metrics.items():
        print(
            f"  {metric}: {value:.4f}"
        )

    # -----------------------------------------------------
    # Train Random Forest
    # -----------------------------------------------------

    print("\nTraining Random Forest...")

    random_forest_model.fit(
        X_train,
        y_train
    )

    random_forest_metrics = evaluate_model(
        random_forest_model,
        X_test,
        y_test
    )

    print("\nRandom Forest:")

    for metric, value in random_forest_metrics.items():
        print(
            f"  {metric}: {value:.4f}"
        )

    # -----------------------------------------------------
    # Save individual models
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        logistic_model,
        LOGISTIC_MODEL_PATH
    )

    joblib.dump(
        random_forest_model,
        RANDOM_FOREST_MODEL_PATH
    )

    print(
        "\n✓ Logistic Regression saved"
    )

    print(
        "✓ Random Forest saved"
    )

    # -----------------------------------------------------
    # Select best model based on F1 score
    # -----------------------------------------------------

    if (
        random_forest_metrics["f1_score"]
        >= logistic_metrics["f1_score"]
    ):

        best_model = random_forest_model
        best_model_name = "Random Forest"
        best_metrics = random_forest_metrics

    else:

        best_model = logistic_model
        best_model_name = "Logistic Regression"
        best_metrics = logistic_metrics

    # -----------------------------------------------------
    # Save best model
    # -----------------------------------------------------

    joblib.dump(
        best_model,
        BEST_MODEL_PATH
    )

    print(
        f"\n✓ Selected model: {best_model_name}"
    )

    print(
        f"✓ Best model saved to: "
        f"{BEST_MODEL_PATH}"
    )

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)

    return {
        "logistic_regression": logistic_metrics,
        "random_forest": random_forest_metrics,
        "best_model": best_model_name,
        "best_metrics": best_metrics,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    train_models()