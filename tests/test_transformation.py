"""
Tests for the feature engineering and preprocessing stage.
"""

import pandas as pd

from src.training.train import (
    create_preprocessor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


DATA_PATH = "data/raw/loan_data.csv"


def test_preprocessor_can_be_created():
    """Check that the preprocessing pipeline can be created."""

    preprocessor = create_preprocessor()

    assert preprocessor is not None


def test_preprocessor_transforms_data():
    """Check that preprocessing produces valid numerical output."""

    df = pd.read_csv(DATA_PATH)

    feature_columns = (
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
    )

    X = df[feature_columns].copy()

    preprocessor = create_preprocessor()

    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed is not None

    assert len(X_transformed) == len(X)

    assert X_transformed.shape[1] > 0


def test_preprocessed_data_contains_no_nan():
    """Check that preprocessing handles missing values."""

    df = pd.read_csv(DATA_PATH)

    feature_columns = (
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
    )

    X = df[feature_columns].copy()

    preprocessor = create_preprocessor()

    X_transformed = preprocessor.fit_transform(X)

    assert not pd.isna(X_transformed).any()


def test_expected_feature_count():
    """
    Check that the current dataset produces
    the expected number of engineered features.
    """

    df = pd.read_csv(DATA_PATH)

    feature_columns = (
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
    )

    X = df[feature_columns].copy()

    preprocessor = create_preprocessor()

    X_transformed = preprocessor.fit_transform(X)

    # Current dataset should produce 19 features
    assert X_transformed.shape[1] == 19