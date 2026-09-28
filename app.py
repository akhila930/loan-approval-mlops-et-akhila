from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from prometheus_fastapi_instrumentator import Instrumentator


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "loan_approval_model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"

FRONTEND_DIR = BASE_DIR / "frontend"


# --------------------------------------------------
# Load ML artifacts
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Loan Approval Prediction API",
    description="MLOps-based Loan Approval Prediction System",
    version="1.0.0",
)


# --------------------------------------------------
# Frontend static files
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)


# --------------------------------------------------
# Prometheus monitoring
# --------------------------------------------------

Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class LoanApplication(BaseModel):

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
        default=5000,
        ge=0,
    )

    Coapplicant_Income: float = Field(
        default=0,
        ge=0,
    )

    Loan_Amount: float = Field(
        default=150,
        ge=0,
    )

    Loan_Term: int = Field(
        default=360,
        gt=0,
    )

    Credit_History: Literal[0, 1] = 1

    Property_Area: Literal[
        "Urban",
        "Semiurban",
        "Rural",
    ] = "Urban"

    Age: float = Field(
        default=30,
        ge=18,
    )


# --------------------------------------------------
# Frontend
# --------------------------------------------------

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(
        FRONTEND_DIR / "index.html"
    )


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
    }


# --------------------------------------------------
# Model information
# --------------------------------------------------

@app.get("/model-info")
def model_info():

    return {
    "model": "loan_approval_model",
    "model_file": MODEL_PATH.name,
        "preprocessor_file": PREPROCESSOR_PATH.name,
        "features": ["Gender", "Married", "Dependents", "Education", "Employment_Status", "Applicant_Income", "Coapplicant_Income", "Loan_Amount", "Loan_Term", "Credit_History", "Property_Area", "Age"],
    "algorithm": type(model).__name__,
    "service": "Loan Approval Prediction API",
    "version": "1.0.0",
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(application: LoanApplication):

    try:

        input_data = pd.DataFrame(
            [
                application.model_dump()
            ]
        )

        transformed_data = preprocessor.transform(
            input_data
        )

        prediction = model.predict(
            transformed_data
        )[0]

        probabilities = model.predict_proba(
            transformed_data
        )[0]

        approval_probability = float(
            probabilities[1]
        )

        rejection_probability = float(
            probabilities[0]
        )

        loan_status = (
            "Approved"
            if int(prediction) == 1
            else "Rejected"
        )

        return {
            "prediction": int(prediction),
            "loan_status": loan_status,
            "approval_probability": round(
                approval_probability,
                4,
            ),
            "rejection_probability": round(
                rejection_probability,
                4,
            ),
            "model": "loan_approval_model",
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# --------------------------------------------------
# Prometheus metrics
# --------------------------------------------------

@app.get("/metrics-info")
def metrics_info():

    return {
        "metrics_endpoint": "/metrics",
        "monitoring": "Prometheus",
    }
