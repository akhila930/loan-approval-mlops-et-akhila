# FastAPI application will be implemented in the API phase.
"""
FastAPI service for Loan Approval Prediction.

Endpoints:
    GET  /
    GET  /health
    POST /predict
    GET  /model-info
    GET  /metrics
"""

import os
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/loan_approval_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Loan Approval Prediction API",
    description=(
        "MLOps REST API for predicting loan approval "
        "using applicant information."
    ),
    version="1.0.0",
)


# ============================================================
# PROMETHEUS MONITORING
# ============================================================

Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

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


# ============================================================
# REQUEST SCHEMA
# ============================================================

class LoanApplication(BaseModel):
    """
    Input data required for loan approval prediction.
    """

    Gender: Literal["Male", "Female"] = "Male"

    Married: Literal["Yes", "No"] = "No"

    Dependents: int = Field(
        default=0,
        ge=0,
    )

    Education: Literal[
        "Graduate",
        "Not Graduate",
    ] = "Graduate"

    Employment_Status: Literal[
        "Employed",
        "Self-Employed",
        "Unemployed",
    ] = "Employed"

    Applicant_Income: float = Field(
        ...,
        ge=0,
    )

    Coapplicant_Income: float = Field(
        default=0,
        ge=0,
    )

    Loan_Amount: float = Field(
        ...,
        ge=0,
    )

    Loan_Term: int = Field(
        ...,
        gt=0,
    )

    Credit_History: Literal[0, 1] = 1

    Property_Area: Literal[
        "Urban",
        "Semiurban",
        "Rural",
    ] = "Urban"

    Age: float = Field(
        ...,
        ge=18,
    )


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class PredictionResponse(BaseModel):
    """
    Prediction response returned by the API.
    """

    prediction: int

    loan_status: str

    approval_probability: float

    rejection_probability: float

    model: str


# ============================================================
# LOAD MODEL
# ============================================================

model = None
preprocessor = None


def load_model_artifacts():
    """
    Load the trained model and preprocessing pipeline.
    """

    global model
    global preprocessor

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )


# Load artifacts when application starts
load_model_artifacts()


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    """
    Basic API information.
    """

    return {
        "service": "Loan Approval Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "prediction": "/predict",
        "metrics": "/metrics",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    """
    Health check endpoint used by Docker/Kubernetes.
    """

    model_loaded = model is not None
    preprocessor_loaded = preprocessor is not None

    if model_loaded and preprocessor_loaded:

        return {
            "status": "healthy",
            "model_loaded": True,
            "preprocessor_loaded": True,
        }

    raise HTTPException(
        status_code=503,
        detail={
            "status": "unhealthy",
            "model_loaded": model_loaded,
            "preprocessor_loaded": preprocessor_loaded,
        },
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():
    """
    Returns information about the currently loaded model.
    """

    return {
        "model": "Loan Approval Classifier",
        "model_file": MODEL_PATH,
        "preprocessor_file": PREPROCESSOR_PATH,
        "features": (
            NUMERICAL_FEATURES
            + CATEGORICAL_FEATURES
        ),
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(application: LoanApplication):
    """
    Generate a loan approval prediction.
    """

    try:

        # ----------------------------------------------------
        # Convert request to DataFrame
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [
                application.model_dump()
            ]
        )

        # ----------------------------------------------------
        # Apply preprocessing
        # ----------------------------------------------------

        processed_data = preprocessor.transform(
            input_data
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = int(
            model.predict(
                processed_data
            )[0]
        )

        # ----------------------------------------------------
        # Prediction probabilities
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            processed_data
        )[0]

        rejection_probability = float(
            probabilities[0]
        )

        approval_probability = float(
            probabilities[1]
        )

        # ----------------------------------------------------
        # Convert prediction to business label
        # ----------------------------------------------------

        if prediction == 1:
            loan_status = "Approved"
        else:
            loan_status = "Rejected"

        return PredictionResponse(
            prediction=prediction,
            loan_status=loan_status,
            approval_probability=round(
                approval_probability,
                4,
            ),
            rejection_probability=round(
                rejection_probability,
                4,
            ),
            model="loan_approval_model",
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# LOCAL DEVELOPMENT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )